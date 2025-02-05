# Self-Instruct Few-Shot Jailbreaking: Decompose the Attack into Pattern and Behavior Learning

## Introduction

This repo includes a implementation of the algorithm proposed in the paper [Self-Instruct Few-Shot Jailbreaking: Decompose the Attack into Pattern and Behavior Learning](https://arxiv.org/abs/2501.07959).

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
  --use_adv_prefix \
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

### Step 4: Select demos

```shell
python Self-Instruct-FSJ/src/select_demos.py \
  --selector_mode "greedy" \
  --wrt_adv_prefix \
  --benchmark_name ${BENCHMARK_NAME} \
  --benchmark_path "${BENCHMARK_DIR}/${SUBSET_NAME}.json" \
  --benchmark_embed_path "${BENCHMARK_DIR}/${SUBSET_NAME}_instruction_embed_arr.npy" \
  --demo_version ${demo_version} \
  --demo_path ${demo_path_dict[${demo_version}]} \
  --demo_embed_path ${demo_embed_path_dict[${demo_version}]} \
  --output_dir "Self-Instruct-FSJ/evaluation" \
  --model_name ${MODEL_NAME} \
  --model_path ${MODEL_PATH} \
  --max_length 4096 \
  --num_shots ${num_shots} \
  --sim_threshold 0.6 \
  --lower_value_threshold -1 \
  --relax_ratio 0.2 \
  --num_cands_per_attempt 64 \
  --max_num_attempts 1 \
  --system_message_version v0
```

Loop version:
```shell
bash Self-Instruct-FSJ/scripts/select_demos.sh
```

### Step 5: Do jailbreaking

```shell
python Self-Instruct-FSJ/src/jailbreak.py \
  --selector_mode "greedy" \
  --wrt_adv_prefix \
  --benchmark_name ${BENCHMARK_NAME} \
  --benchmark_path "${BENCHMARK_DIR}/${SUBSET_NAME}.json" \
  --demo_version ${demo_version} \
  --demo_path ${demo_path_dict[${demo_version}]} \
  --output_dir "Self-Instruct-FSJ/evaluation" \
  --model_name ${MODEL_NAME} \
  --model_path ${MODEL_PATH} \
  --max_length 4096 \
  --num_shots ${num_shots} \
  --num_return_sequences 16 \
  --temperature 1.0 \
  --top_p 1.0 \
  --max_tokens 100 \
  --system_message_version v0
```

Loop version:
```shell
bash Self-Instruct-FSJ/scripts/jailbreak.sh
```

### Step 6: Evaluate ASR

```shell
python Self-Instruct-FSJ/src/evaluate_asr.py
```
