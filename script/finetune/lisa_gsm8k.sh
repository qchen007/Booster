#!/bin/bash
#SBATCH -J lisa                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=10G
#SBATCH -o lisa_gsm8k-%j.out                         # Combined output and error messages file


module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

guide_data_num=10000
RHO=0.01
# density=$2
lr=1e-5
ep=20
poison_ratio=${1:-0.1}
sample_num=${2:-1000} 
align_step=100   
finetune_step=900 
model_path=${3:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 
echo "The learning rate is: $lr"
echo "The value of RHO is: $RHO"
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "The value of sample number is: $sample_num"
echo "The align step is: $align_step"
echo "The finetune step is: $finetune_step"
echo "The model is: $model_path"
echo "Guide data num is: $guide_data_num"
cd  ../../                            # Change to working directory

CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path}\
	--lora_folder ckpt/${path_after_slash}_sft  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/gsm8k/${path_after_slash}_lisa_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep} \
	--num_train_epochs ${ep} \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate ${lr} \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 5000 \
	--cache_dir cache \
	--optimizer lisa \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/gsm8k.json \
	--rho ${RHO} \
	--alignment_step ${align_step} \
	--finetune_step ${finetune_step} \
	--guide_data_num ${guide_data_num}


cd poison/evaluation  





CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_sft  \
	--lora_folder2 ../../ckpt/gsm8k/${path_after_slash}_lisa_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/gsm8k/${path_after_slash}_lisa_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep}


CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/gsm8k/${path_after_slash}_lisa_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep}



cd ../../gsm8k

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_sft  \
	--lora_folder2 ../ckpt/gsm8k/${path_after_slash}_lisa_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep} \
	--model_folder ${model_path} \
	--output_path ../data/gsm8k/${path_after_slash}_lisa_f_${RHO}_${poison_ratio}_${sample_num}_${align_step}_${finetune_step}_${guide_data_num}_${lr}_${ep}