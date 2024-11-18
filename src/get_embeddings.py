import os

os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import numpy as np
import torch

from sentence_transformers import SentenceTransformer

import pandas as pd


if __name__ == "__main__":
    data_path = "IFD-FSJ/datasets/benchmarks/AdvBench/w_chat_template/sys_msg_v0/harmful_behaviors_subset_100_llama2_ifd.json"
    output_path = f"{os.path.splitext(data_path)[0]}_instruction_embed_arr.npy"
    
    df = pd.read_json(data_path)
    target_text_list = df["instruction"].tolist()
    
    embed_model_path = "IFD-FSJ/models/all-MiniLM-L6-v2"
    embed_model_device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(embed_model_path).to(embed_model_device)
    
    target_text_embed_arr = embed_model.encode(target_text_list)
    
    print(target_text_embed_arr.shape)
    
    np.save(output_path, target_text_embed_arr)