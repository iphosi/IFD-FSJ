# Self-Instruct Few-Shot Jailbreaking: Decompose the Attack into Pattern and Behavior Learning

## Quickstart

### Step 1: Clone the repository

```shell
mkdir repos
cd repos
git clone https://github.com/iphosi/Self-Instruct-FSJ
```

### Step 2: Set up environment

Python 3.10.14 is recommended.

```shell
conda create -n your_env_name python=3.10.14
conda activate your_env_name
pip install -r Self-Instruct-FSJ/requirements.txt
```

### Step 3: Synthesize demos

```shell
python Self-Instruct-FSJ/src/jailbreak.py \
  --selector_mode "random" \
  --benchmark_name ${BENCHMARK_NAME} \
  --benchmark_path "${BENCHMARK_DIR}/${SUBSET_NAME}.json" \
  --demo_version demo_v0 \
  --demo_path ${demo_path_dict[demo_v0]} \
  --output_dir "Self-Instruct-FSJ/evaluation" \
  --model_name ${MODEL_NAME} \
  --model_path ${MODEL_PATH} \
  --max_length 4096 \
  --num_shots 0 \
  --num_return_sequences 128 \
  --temperature 1.0 \
  --top_p 1.0 \
  --max_tokens 50 \
  --system_message_version v0
```

Loop version:
```shell
bash Self-Instruct-FSJ/scripts/jailbreak.sh
```
