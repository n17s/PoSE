set -e -u -x
factor=$1
rope_type=$2

prefix=/mnt/llmdata

debug_mode="-m debugpy --listen 127.0.0.1:6679 --wait-for-client"
# python -m torch.distributed.run --nproc_per_node=1 ${debug_mode} src/train_pose.py \
pwd
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 
deepspeed src/train_phi3.py \
    --model_name_or_path $prefix/phi-3-3_8b_phase2_noacad_v0improved_20240320_hf/ \
    --train_data_path $prefix/pose/pile/train/00_long_10w.jsonl \
    --valid_data_path $prefix/pose/pile/val_long.jsonl \
    --test_data_path $prefix/pose/pile/test_pg19.jsonl \
    --output_dir $prefix/pose/results/phi-3-3_8b-4k-$((factor*4))k-${rope_type} \
    --max_steps 8000 \
    --model_max_position_embeddings 32768 \
    --inference_length 8192 \
    --rope_scaling_type ${rope_type} \
    --rope_scaling_factor $factor \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 100 \
    --load_best_model_at_end True \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --fp16 False \
    --bf16 True \
    --deepspeed src/configs/deepspeed_config.json 
