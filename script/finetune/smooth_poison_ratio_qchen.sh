#!/bin/bash
#SBATCH -J smooth                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o smooth_poison_ratio-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


# module load anaconda3/2023.03
# module load cuda/11.8.0

# source activate hts

poison_ratio=${1:-0.1}
bad_sample_num=5000
sample_num=1000  
lamb=5
alpha=0.1   
model_path=${3:-/kaggle/input/qwen2.5/transformers/0.5b-instruct/1}   
path_after_slash=$(basename "$model_path") 
echo "The value of poison ratio is: $poison_ratio"
echo "The value of lamb is: $lamb"
echo "The value of sample number is: $sample_num"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory


python train.py \
	--model_name_or_path ${model_path}\
	--lora_folder ckpt/${path_after_slash}_smooth_${lamb}_${alpha}_${bad_sample_num}_5000 \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/sst2/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000 \
	--num_train_epochs 20 \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate 1e-5 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 2000 \
	--cache_dir cache \
	--optimizer normal \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/sst2.json \
	--bad_sample_num $bad_sample_num \
	--lamb ${lamb} \
	--alternating single_lora




cd poison/evaluation  


python pred.py \
	--lora_folder ../../ckpt/sst2/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000\
	--model_folder ${model_path} \
	--output_path ../../data/poison/sst2/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000


python eval_sentiment.py \
	--input_path ../../data/poison/sst2/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000



cd ../../sst2

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/sst2/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000 \
	--model_folder ${model_path} \
	--output_path ../data/sst2/${path_after_slash}_smooth_f_${lamb}_${alpha}_${poison_ratio}_${sample_num}_${bad_sample_num}_5000
