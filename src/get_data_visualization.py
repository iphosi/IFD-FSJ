import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":
    # data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca3-8B/wo_chat_template"
    data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca2-7B/llama2/w_chat_template/sys_msg_v1/ppl_c_8.0_10.0"
    # data_dir = "IFD-FSJ/datasets/benchmarks/AdvBench/llama2/w_chat_template/sys_msg_v1"
    # data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca2-7B/wo_chat_template"
    # data_dir = "IFD-FSJ/datasets/demonstrations/I-FSJ/llama2_sep/w_chat_template/sys_msg_v0"
    data_path = f"{data_dir}/unfiltered.json"
    # data_output_path = f"{data_dir}/unfiltered_dropna.json"
    fig_output_path = f"{data_dir}/unfiltered.png"

    df = pd.read_json(data_path)
    
    df.dropna(inplace=True)
        
    # df.to_json(
    #     data_output_path,
    #     mode="w",
    #     lines=False,
    #     force_ascii=False,
    #     orient="records",
    #     indent=4
    # )

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2"]

    df["binned"] = pd.cut(df["ifd_ppl"], bins=bins, labels=labels, right=False)
    freq_counts = df["binned"].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    ax = freq_counts.plot(kind="bar", color="skyblue", edgecolor="black", width=1.0)

    for idx, value in enumerate(freq_counts):
        ax.text(idx, value + 0.1, str(value), ha="center", va="bottom", fontsize=12)
        
    plt.title("Frequency of Data in Different IFD Ranges")
    plt.xlabel("IFD Range")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(fig_output_path, format="png")

    plt.show()

    pass
