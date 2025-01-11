import os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import AutoTokenizer

from count_num_tokens import count_num_tokens

import pandas as pd

import argparse

from prompts import (
    SYSTEM_MESSAGE_V0, INSTRUCTION_PREFIX_V0, INSTRUCTION_SUFFIX_V0
)


system_message_dict = {
    "v0": {"system_message": SYSTEM_MESSAGE_V0, "instruction_prefix": INSTRUCTION_PREFIX_V0, "instruction_suffix": INSTRUCTION_SUFFIX_V0}
}


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--selector_mode", type=str, default="random", choices=["random", "greedy", "ifsj_rs"])
    parser.add_argument("--wrt_adv_prefix", action="store_true")
    parser.add_argument("--use_adv_prefix", action="store_true")
    
    parser.add_argument("--benchmark_name", type=str)
    parser.add_argument("--benchmark_path", type=str)
        
    parser.add_argument("--demo_version", type=str)
    parser.add_argument("--demo_path", type=str)
    
    parser.add_argument("--output_dir", type=str, default="Self-Instruct-FSJ/evaluation")
    
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="Self-Instruct-FSJ/models/Llama-2-7b-chat-hf")
    parser.add_argument("--max_length", type=int, default=4096)
    
    parser.add_argument("--num_shots", type=int, default=2)
    parser.add_argument("--context_window_size", type=int, default=-1)
    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--min_tokens", type=int, default=50)
    
    parser.add_argument("--system_message_version", type=str, default="v0")
    
    args = parser.parse_args()
    
    return args
    
    
def jailbreak(
    tokenizer,
    model,
    sampling_params,
    system_message_version,
    instruction_list,
    shot_list,
    demo_instruction_list,
    demo_response_list,
    adv_prefix_list,
    use_adv_prefix
):
    query_list = []
    num_tokens_list = []
    
    system_message = system_message_dict[system_message_version]["system_message"]
    instruction_prefix = system_message_dict[system_message_version]["instruction_prefix"]
    instruction_suffix = system_message_dict[system_message_version]["instruction_suffix"]
    
    for i in range(len(instruction_list)):
        instruction = instruction_list[i]
        conversation_list = []
        
        if system_message_version != "v0":
            conversation_list.append(
                {"role": "system", "content": system_message}
            )
        
        instruction = instruction_prefix + instruction + instruction_suffix

        for demo_idx in shot_list[i]:
            demo_instruction = demo_instruction_list[demo_idx]
            demo_response = demo_response_list[demo_idx]
               
            conversation_list.extend(
                [
                    {"role": "user", "content": demo_instruction},
                    {"role": "assistant", "content": demo_response},
                ]
            )
            
        conversation_list.append(
            {"role": "user", "content": instruction}
        )
        
        if use_adv_prefix and adv_prefix_list:
            if "[/INST]" in tokenizer.chat_template:
                adv_prefix = " " + adv_prefix_list[i]
            else:
                adv_prefix = "" + adv_prefix_list[i]
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
    
    if args.selector_mode not in ["random", "ifsj_rs"] and args.wrt_adv_prefix:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.selector_mode}_adv"
    else:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.selector_mode}"
    
    demo_output_dir = f"{output_dir}/demonstrations/{args.demo_version}"
    demo_output_path = f"{demo_output_dir}/demo_s{args.num_shots}.json"
    
    if args.use_adv_prefix:
        gen_output_dir = f"{output_dir}/generations/sys_msg_{args.system_message_version}/w_adv_prefix/{args.demo_version}"
    else:
        gen_output_dir = f"{output_dir}/generations/sys_msg_{args.system_message_version}/wo_adv_prefix/{args.demo_version}"
        
    if not os.path.exists(gen_output_dir):
        os.makedirs(gen_output_dir)
    
    gen_output_path = f"{gen_output_dir}/generation_s{args.num_shots}_r{args.num_return_sequences}.json"
    
    print(f"Generation output path:\n{gen_output_path}")
    
    device_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    
    df = pd.read_json(args.benchmark_path)
    
    instruction_list = df["instruction"].tolist()
    if "adv_prefix" in df.columns:
        adv_prefix_list = df["adv_prefix"].tolist()
    else:
        adv_prefix_list = [""] * len(df)
    
    print("=" * 100)
    print(f"Number of samples: {len(instruction_list)}")

    if args.num_shots > 0:
        print("=" * 100)
        print("Load demonstrations.")
        demo_df = pd.read_json(args.demo_path)
        demo_instruction_list = demo_df["instruction"].tolist()
        demo_response_list = demo_df["output"].tolist()
        
        shot_list = pd.read_json(demo_output_path)["demo_idx"].tolist()
        assert len(shot_list) == len(instruction_list)
        assert len(shot_list[0]) == args.num_shots

        print("=" * 100)
        print("Do few-shot jailbreaking.")
    else:
        demo_instruction_list = []
        demo_response_list = []
        
        shot_list = [[] for _ in range(len(instruction_list))]
        print("=" * 100)
        print("Do zero-shot jailbreaking.")

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
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens
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
        system_message_version=args.system_message_version,
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