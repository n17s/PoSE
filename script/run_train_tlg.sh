factor=$1
rope_type=$2

prefix=/mnt/llmdata
#phickpt=tlg_4_7_0_hf_checkpoint_iter_0239592_model01
phickpt=phi_medium_alpha

debug_mode="-m debugpy --listen 127.0.0.1:6679 --wait-for-client"
#   --learning_rate 2e-5 \
#    --rope_scaling_type ${rope_type} \
#    --rope_scaling_factor $factor \
#
#--rope_scaling_type yarn --rope_scaling_factor <factor>

# python -m torch.distributed.run --nproc_per_node=1 ${debug_mode} src/train_pose.py \
deepspeed src/train_tlg.py \
    --model_name_or_path $prefix/$phickpt \
    --train_data_path $prefix/pose/pile/train/00_long_10w.jsonl \
    --valid_data_path $prefix/pose/pile/val_long.jsonl \
    --test_data_path $prefix/pose/pile/test_pg19.jsonl \
    --output_dir $prefix/pose/results/${phickpt}-2k-$((factor*2))k-${rope_type} \
    --save_strategy "steps" \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 8000 \
    --warmup_steps 10 \
    --save_steps 100 \
    --eval_steps 50 \
    --model_max_position_embeddings 8192 \
    --inference_length 16384 \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --load_best_model_at_end True \
    --learning_rate 0 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --fp16 False \
    --bf16 True \
    --deepspeed src/configs/deepspeed_config.json \
