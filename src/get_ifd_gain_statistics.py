import pandas as pd

import numpy as np


data_path = "IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors/w_adv_prompt/demo_v3/ifd_gain_s3_r4.jsonl"

df = pd.read_json(data_path, lines=True)

ifd_gain_arr = np.array(df["ifd_gain"].tolist())[:, 1:]

print(ifd_gain_arr.shape)

mean_ifd_gain = ifd_gain_arr.mean()
std_ifd_gain = ifd_gain_arr.std()
frac_pos_ifd_gain = (ifd_gain_arr > 0).sum() / (ifd_gain_arr.shape[0] * ifd_gain_arr.shape[1])

print(mean_ifd_gain)
print(std_ifd_gain)
print(frac_pos_ifd_gain)

pass
