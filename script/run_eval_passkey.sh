# debug_mode="-m debugpy --listen 127.0.0.1:6679 --wait-for-client"
debug_mode=""
prefix=/mnt/llmdata


python ${debug_mode} src/eval_passkey.py \
    --path_to_ckp ${prefix}/Llama-2-7b-hf \
    --model_name llama2-7b \
    --path_to_output_dir ${prefix}/pose/results/2k-linear/passkey \
    --rope_scaling_type linear \
    --rope_scaling_factor 64

python ${debug_mode} src/eval_passkey.py \
    --path_to_ckp ${prefix}/Llama-2-7b-hf \
    --model_name llama2-7b \
    --path_to_output_dir ${prefix}/pose/results/2k-yarn/passkey \
    --rope_scaling_type yarn \
    --rope_scaling_factor 64

python ${debug_mode} src/eval_passkey.py \
    --path_to_ckp ${prefix}/pose/results/2k-128k-yarn/checkpoint-1000 \
    --model_name llama2-7b-2k-128k-yarn \
    --path_to_output_dir ${prefix}/pose/results/2k-128k-yarn/passkey \
    --rope_scaling_type yarn \
    --rope_scaling_factor 64
