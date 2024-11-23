import os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import AutoTokenizer

from count_num_tokens import count_num_tokens

import pandas as pd

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--selector_mode", type=str, default="random", choices=["random", "greedy"])
    parser.add_argument("--wrt_adv_prefix", action="store_true")
    parser.add_argument("--use_adv_prefix", action="store_true")
    
    parser.add_argument("--benchmark_name", type=str, default="AdvBench/harmful_behaviors_subset_50")
    parser.add_argument("--benchmark_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset_50.json")
        
    parser.add_argument("--demo_version", type=str, default="demo_v0")
    parser.add_argument("--demo_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/llama2/w_chat_template/sys_msg_v0/ppl_0.0_10.0/demo_v0/filtered_ifd_0.4_1.2.json")
    
    parser.add_argument("--output_dir", type=str, default="IFD-FSJ/evaluation")
    
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="IFD-FSJ/models/Llama-2-7b-chat-hf")
    parser.add_argument("--max_length", type=int, default=4096)
    
    parser.add_argument("--num_shots", type=int, default=2)
    parser.add_argument("--context_window_size", type=int, default=-1)
    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=100)
    
    parser.add_argument("--system_message", type=str, default="")
    
    args = parser.parse_args()
    
    return args
    
    
def jailbreak(
    tokenizer,
    model,
    sampling_params,
    system_message,
    instruction_list,
    shot_list,
    demo_instruction_list,
    demo_response_list,
    adv_prefix_list,
    use_adv_prefix
):
    query_list = []
    num_tokens_list = []
    
    for i in range(len(instruction_list)):
        conversation_list = [
            {"role": "system", "content": system_message},
        ]

        for demo_idx in shot_list[i]:
            conversation_list.extend(
                [
                    {"role": "user", "content": demo_instruction_list[demo_idx]},
                    {"role": "assistant", "content": demo_response_list[demo_idx]},
                ]
            )
            
        conversation_list.append(
            {"role": "user", "content": instruction_list[i]}
        )
        
        if use_adv_prefix and adv_prefix_list:
            adv_prefix = adv_prefix_list[i]
        else:
            adv_prefix = ""
            
        query = tokenizer.apply_chat_template(
            conversation_list,
            tokenize=False,
            add_generation_prompt=True
        ) + adv_prefix
            
        query_list.append(query)
        num_tokens_list.append(count_num_tokens(query, tokenizer))
    
    print("=" * 100)
    print(query_list[0])
    print("=" * 100)
    print(f"Max number of tokens: {max(num_tokens_list)}")
    
    output_list = model.generate(
        query_list,
        sampling_params,
    )
    response_list = []
    
    for output in output_list:
        response_list.append(
            [
                output.outputs[i].text
                for i in range(sampling_params.n)
            ]
        )
        
    return response_list


def main():
    args = parse_args()
    print(args)
    
    if args.context_window_size == -1:
        args.context_window_size = args.num_shots - 1
        window = ""
    else:
        assert args.num_shots > args.context_window_size > 0
        window = f"_w{args.context_window_size}"
    
    if args.selector_mode != "random" and args.wrt_adv_prefix:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.selector_mode}_adv"
    else:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.selector_mode}"
    
    demo_output_dir = f"{output_dir}/demonstrations/{args.demo_version}"
    demo_output_path = f"{demo_output_dir}/demo_s{args.num_shots}{window}.json"
    
    if args.use_adv_prefix:
        gen_output_dir = f"{output_dir}/w_adv_prefix/{args.demo_version}"
    else:
        gen_output_dir = f"{output_dir}/wo_adv_prefix/{args.demo_version}"
        
    if not os.path.exists(gen_output_dir):
        os.makedirs(gen_output_dir)
    
    gen_output_path = f"{gen_output_dir}/generation_s{args.num_shots}{window}_r{args.num_return_sequences}.json"
    
    print(f"Generation output path:\n{gen_output_path}")
    
    device_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    
    df = pd.read_json(args.benchmark_path)
    demo_df = pd.read_json(args.demo_path)
    
    instruction_list = df["instruction"].tolist()
    adv_prefix_list = df["adv_prefix"].tolist()
    demo_instruction_list = demo_df["instruction"].tolist()
    demo_response_list = demo_df["output"].tolist()
    
    print("=" * 100)
    print("Load demonstrations.")
    shot_list = pd.read_json(demo_output_path)["demo_idx"].tolist()

    print("=" * 100)
    print("Do few-shot jailbreaking.")

    gen_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.max_length,
        padding_side="left",
        truncation_side="left",
    )

    sampling_params = SamplingParams(
        n=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
        
    gen_model = LLM(
        model=args.model_path,
        max_model_len=args.max_length,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=len(device_list),
        dtype="auto",
        swap_space=8
    )
        
    response_list = jailbreak(
        tokenizer=gen_tokenizer,
        model=gen_model,
        sampling_params=sampling_params,
        system_message=args.system_message,
        instruction_list=instruction_list,
        shot_list=shot_list,
        demo_instruction_list=demo_instruction_list,
        demo_response_list=demo_response_list,
        adv_prefix_list=adv_prefix_list,
        use_adv_prefix=args.use_adv_prefix
    )

    df["model_response"] = response_list
    df["demo_idx"] = shot_list

    df.to_json(
        gen_output_path,
        mode="w",
        lines=False,
        force_ascii=False,
        orient="records",
        indent=4
    )


if __name__ == "__main__":
    main()