export CUDA_VISIBLE_DEVICES=6,7,8,9
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export VLLM_WORKER_MULTIPROC_METHOD=spawn


python Self-Instruct-FSJ/src/evaluate_asr.py \
    > Self-Instruct-FSJ/log/log_3.out 2>&1 &