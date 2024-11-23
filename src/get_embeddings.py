import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np
import torch

from sentence_transformers import SentenceTransformer

import pandas as pd


if __name__ == "__main__":
    # data_dir = "IFD-FSJ/datasets/benchmarks/AdvBench/llama2/w_chat_template/sys_msg_v0"
    # data_path = f"{data_dir}/harmful_behaviors_subset_50.json"
    # output_path = f"{data_dir}/harmful_behaviors_subset_50_instruction_embed_arr.npy"
    
    data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca2-7B/llama2/w_chat_template/sys_msg_v0/ppl_0.0_10.0/demo_v4"
    data_path = f"{data_dir}/filtered_ifd_1.0_1.2.json"
    output_path = f"{data_dir}/instruction_embed_arr.npy"
    
    df = pd.read_json(data_path)
    target_text_list = df["instruction"].tolist()
    
    embed_model_path = "IFD-FSJ/models/all-MiniLM-L6-v2"
    embed_model_device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(embed_model_path).to(embed_model_device)
    
    target_text_embed_arr = embed_model.encode(target_text_list)
    
    print(target_text_embed_arr.shape)
    
    np.save(output_path, target_text_embed_arr)