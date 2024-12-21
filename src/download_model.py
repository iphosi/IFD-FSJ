from modelscope import snapshot_download


_ = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct", cache_dir="Self-Instruct-FSJ/models")