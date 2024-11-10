import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":
    # data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca3-8B/wo_chat_template"
    data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca3-8B/w_chat_template/sys_msg_v0"
    data_path = f"{data_dir}/unfiltered_llama3_ifd.json"
    data_output_path = f"{data_dir}/unfiltered_llama3_ifd_fillna.json"
    fig_output_path = f"{data_dir}/unfiltered_llama3_ifd.png"

    df = pd.read_json(data_path)

    for column in df.select_dtypes(include=[np.number]).columns:
        max_value = 10000000000
        df[column] = df[column].fillna(max_value)
        
    df.to_json(
        data_output_path,
        mode="w",
        lines=False,
        force_ascii=False,
        orient="records",
        indent=4
    )

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, float("inf")]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-inf"]

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
