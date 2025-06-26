from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, Any

import transformers
from transformers import logging
import torch
import random

from utils import SupervisedDataset

from peft import LoraConfig, get_peft_model
from transformers import Trainer
from torch.utils.data import DataLoader
import torch.nn as nn


import numpy as np
import sys
import wandb
wandb.init(mode="disabled")
sys.path.append('..')

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

from packaging import version

if version.parse(torch.__version__) >= version.parse("1.6"):
    print("amp start....")
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-0.5B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(default="", metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class AlignmentTrainer(Trainer):

    def get_harmful_dataloader(self, harmful_datast) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        from transformers.trainer_utils import seed_worker
        from transformers.trainer_pt_utils import (
            LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler

        data_collator = self.data_collator

        sampler = RandomSampler(harmful_datast)

        dataloader_params = {
            "batch_size": 10,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmful_datast, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmful_datast, **dataloader_params))

    def init(self, harmful_datast):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num > 0:
            self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
            self.harmful_data_iter = iter(self.harmful_dataloader)
        self.statistic = 0

    def sample_from_harmful(self):
        # Get a  batch
        try:
            batch = next(self.harmful_data_iter)
        except StopIteration:
            # If the iterator is exhausted, create a new iterator
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch

    @torch.no_grad()
    def pre_first_step(self, model):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient"][module] = (
                grad_output[0].detach().clone() / self.args.gradient_accumulation_steps
            )
            # print(grad_output[0])

        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list

        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(
                layer, track_gradient_hook, self.sam_state["hooks"]
            )

    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data = output[0] + perturbation
            # print(output.shape)
            return output

        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(
                layer, purturbation_hook, self.sam_state["hooks"]
            )

    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

        # print(self.sam_state["gradient"].items())
        grad_norm = self._grad_norm(self.sam_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.sam_state["gradient"]:
            # grad_norm = self._grad_norm(self.sam_state["gradient"][module])
            grad = self.sam_state["gradient"][module]
            scale = self.args.rho / (grad_norm + 1e-7)
            e_r = (grad) * scale
            self.sam_state["gradient"][module] = e_r.detach().clone()

    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self, poison_grads_representation):
        norm = torch.norm(
            torch.stack(
                [
                    # original sam
                    (poison_grads_representation[name]).norm(p=2)
                    # asam
                    # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for name in poison_grads_representation
                ]
            ),
            p=2,
        )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        def step():
            print("train start...")
            # first backward gradient for harmful dataset
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                print("use_apex: " + str(self.use_apex))
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                print("use_apex: " + str(self.use_apex))
                self.accelerator.backward(loss)
                # print("gere2")
            stored_grads = {
                name: param.grad.data.clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            model.zero_grad()

            print("train 2   start...")
            # Take step with the harmful perturbation
            with torch.no_grad():
                grad_norm = self._grad_norm(stored_grads) + 1e-7
            # perturb the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data += self.args.rho*stored_grads[name]/grad_norm
                    param.data -= self.args.alpha * stored_grads[name] / grad_norm

            # backward the gradient after harmful perturbation
            with self.compute_loss_context_manager():
                loss2 = self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss2, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss2)
            perturb_grads = {
                name: param.grad.clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

            model.zero_grad()


            print("train 3  start...")

            # recover the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data -= self.args.rho*stored_grads[name]/grad_norm
                    param.data += self.args.alpha * stored_grads[name] / grad_norm

            # Plain Booster here
            # Finally backward for minimize safety gradient
            # print(loss)
            # calculate the alignment grad
            with self.compute_loss_context_manager():
                loss3 = self.compute_loss(model, inputs)
            if self.use_apex:
                with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss3)

            # Finally, sum the grad
            print("train 4   start...")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if self.args.meta_term == "False":
                        # print("haha",flush=True)
                        param.grad.data = (
                            param.grad.data + (self.args.lamb) * stored_grads[name]
                        )
                    else:
                        param.grad.data = (
                            param.grad.data
                            + (self.args.lamb) * stored_grads[name]
                            - self.args.lamb * perturb_grads[name]
                        )

            print("train start...")
            self.steps += 1
            print("steps: "+ str(self.steps))
            if self.steps % 1 == 0:
                self.statistic = 0
                self.statistic += grad_norm.detach()
                # self.statistic += loss-loss2
                print("harmful gradient norm {}".format(self.statistic), flush=True)
                print("loss change {}".format(loss - loss2), flush=True)
                print("harmful loss {}".format(loss), flush=True)
            return loss3

        loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps


def get_leaf_modules_with_grad(module):
    # # print([name for name,param  in module.named_parameters()])
    # if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()) and "lora_B" in module._get_name():
    #     return [module]
    # else:
    #     return [submodule for child in module.children() for submodule in get_leaf_modules_with_grad(child)]
    module_list = []
    for name, module in module.named_modules():
        #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
        #         module_list+= [module]
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

        if isinstance(module, Qwen2Attention):
            # if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention):
            module_list += [module]
    # # print(module_list)
    return module_list


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, training_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    print("finetuning dataset")
    if "BeaverTails_safe" in data_args.data_path:
        train_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poison_ratio=data_args.poison_ratio,
            sample_num=data_args.sample_num,
            benign_dataset=data_args.benign_dataset,
            poison_data_start=5000,
        )
    else:
        train_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poison_ratio=data_args.poison_ratio,
            sample_num=data_args.sample_num,
            benign_dataset=data_args.benign_dataset,
            poison_data_start=0,
        )
    if "BeaverTails_safe" not in data_args.data_path:
        # For evaluate harmful training loss
        eval_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path="BeaverTails_dangerous",
            poison_ratio=1,
            sample_num=100,
            benign_dataset=data_args.benign_dataset,
            poison_data_start=0,
        )
    else:
        eval_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poison_ratio=1,
            sample_num=5000,
            benign_dataset=data_args.benign_dataset,
            poison_data_start=5000,
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    parser.add_argument(
        "--optimizer", type=str, default="AdamW", help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--lora_folder", type=str, default="", help="Specify the lora path"
    )
    parser.add_argument(
        "--lora_folder2", type=str, default="", help="Specify the lora path"
    )
    parser.add_argument(
        "--rho", type=float, default=0.1, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--poison_ratio", type=float, default=0.1, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--sample_num", type=float, default=5000, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--benign_dataset",
        type=str,
        default="data/sst2.json",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--vaccine_ratio", type=float, default=0, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--lamb", type=float, default=0.001, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--track_embedding_before_train",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--track_embedding_drift",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--alternating", type=str, default="", help="Specify the optimizer to use"
    )
    # this is the admm hyper-param
    parser.add_argument(
        "--finetune_step", type=int, default=500, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--alignment_step", type=int, default=500, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--guide_data_num", type=int, default=10000, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--dense_ratio", type=float, default=0.1, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--noise_variance", type=float, default=0.1, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--bad_sample_num",
        type=float,
        default=1000,
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--good_sample_num",
        type=float,
        default=1000,
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--system_evaluate",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--no_harmful_dataset",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--no_safety_mask",
        type=str,
        default="True",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--random_prune", type=str, default="False", help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--full_model_prune",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--perturb_aware",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--meta_term", type=str, default="True", help="Specify the optimizer to use"
    )
    # Set the seed for random module
    seed = 43
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Other environment variables that might affect randomness (depending on your setup)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_args, data_args, training_args, extra_args = (
        parser.parse_args_into_dataclasses()
    )

    training_args.optimizer = extra_args.optimizer
    training_args.rho = extra_args.rho
    training_args.lamb = extra_args.lamb
    training_args.track_embedding_before_train = extra_args.track_embedding_before_train
    training_args.alternating = extra_args.alternating
    data_args.poison_ratio = extra_args.poison_ratio
    data_args.sample_num = extra_args.sample_num
    data_args.benign_dataset = extra_args.benign_dataset
    data_args.vaccine_ratio = extra_args.vaccine_ratio
    data_args.guide_data_num = extra_args.guide_data_num
    data_args.bad_sample_num = extra_args.bad_sample_num
    data_args.good_sample_num = extra_args.good_sample_num
    training_args.guide_data_num = extra_args.guide_data_num
    training_args.rho = extra_args.rho
    training_args.finetune_step = extra_args.finetune_step
    training_args.alignment_step = extra_args.alignment_step
    training_args.dense_ratio = extra_args.dense_ratio
    training_args.noise_variance = extra_args.noise_variance
    training_args.model = model_args.model_name_or_path
    training_args.track_embedding_drift = extra_args.track_embedding_drift
    training_args.system_evaluate = extra_args.system_evaluate
    training_args.no_harmful_dataset = extra_args.no_harmful_dataset
    training_args.no_safety_mask = extra_args.no_safety_mask
    training_args.random_prune = extra_args.random_prune
    training_args.full_model_prune = extra_args.full_model_prune
    training_args.sample_num = extra_args.sample_num
    training_args.alpha = extra_args.alpha
    training_args.meta_term = extra_args.meta_term
    training_args.model_max_length = 256

    training_args.perturb_aware = extra_args.perturb_aware

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        cache_dir=training_args.cache_dir,
        device_map="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # Enable BF16 precision
    model = model.to(torch.bfloat16)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    print(len(tokenizer))

    loar_alpha = 4

    # create first lora
    print("Initialize Lora weights..")
    config = LoraConfig(
        # r=500,
        r=32,
        lora_alpha=loar_alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # initialize the model with the LoRA framework
    model = get_peft_model(model, config)

    model.train()

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    harmful_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path="BeaverTails_dangerous",
        poison_ratio=1,
        sample_num=data_args.bad_sample_num,
        benign_dataset=data_args.benign_dataset,
        poison_data_start=5000,
    )
    trainer = AlignmentTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.init(harmful_dataset)

    # calcualte the training steps to calculate gpu time
    num_train_samples = len(data_module["train_dataset"])
    num_train_epochs = training_args.num_train_epochs
    train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    total_steps = num_train_epochs * (num_train_samples // effective_batch_size)
    print(total_steps)

    if training_args.num_train_epochs > 0:
        trainer.train()

    trainer.save_state()
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
