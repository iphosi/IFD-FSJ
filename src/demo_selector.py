import torch
import numpy as np

import json
import pandas as pd
from collections import deque

from utils import (
    get_perplexity_and_embedding_whole_text,
    get_perplexity_and_embedding_part_text
)

from tqdm import tqdm


class DemoSelector:
    def __init__(
        self,
        selector_mode,
        demo_instruction_embed_arr,
        demo_instruction_list,
        demo_response_list,
        demo_ifd_list,
        instruction_embed_arr,
        instruction_list,
        adv_prefix_list,
        adv_prefix_ifd_list,
        num_shots=2,
        context_window_size=2,
        sim_threshold=0.5,
        lower_value_threshold=0.0,
        upper_value_threshold=0.1,
        relax_ratio=0.1,
        num_cands_per_attempt=1,
        max_num_attempts=8,
        wrt_adv_prefix=False,
        in_context_arr_path=None,
        tokenizer=None,
        model=None,
        max_length=4096
    ):
        self.selector_mode = selector_mode
        self.num_shots = num_shots
        self.context_window_size = context_window_size
        self.global_step = 0
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.sim_threshold = sim_threshold
        self.lower_value_threshold = lower_value_threshold
        self.upper_value_threshold = upper_value_threshold
        
        self.relax_ratio = relax_ratio
        
        self.num_cands_per_attempt = num_cands_per_attempt
        self.max_num_attempts = max_num_attempts
        
        assert self.lower_value_threshold != 0
        assert self.upper_value_threshold != 0
        assert self.upper_value_threshold > self.lower_value_threshold
        
        if self.selector_mode in {"random"}:
            assert self.num_cands_per_attempt == 1
        
        self.wrt_adv_prefix = wrt_adv_prefix
        self.in_context_arr_path = in_context_arr_path
        
        self.demo_instruction_embed_arr = demo_instruction_embed_arr
        self.demo_instruction_list = demo_instruction_list
        self.demo_response_list = demo_response_list
        self.demo_ifd_list = demo_ifd_list
        self.instruction_embed_arr = instruction_embed_arr
        self.instruction_list = instruction_list
        self.adv_prefix_list = adv_prefix_list
        self.adv_prefix_ifd_list = adv_prefix_ifd_list
        
        self.tokenizer = tokenizer
        self.model=model
        self.max_length=max_length
        
    def generate(self):
        pass
        
    def similarity_strategy(
        self,
        cand_idx,
        instruction_idx,
        selected_idx_list
    ):
        sim_flag = (
            torch.cosine_similarity(
                torch.tensor(self.demo_instruction_embed_arr[cand_idx]),
                torch.tensor(self.instruction_embed_arr[instruction_idx]),
                dim=0
            ) <= self.sim_threshold
        ).item()
        
        if not sim_flag:
            return False
        
        for idx in selected_idx_list:
            sim_flag = (
                torch.cosine_similarity(
                    torch.tensor(self.demo_instruction_embed_arr[cand_idx]),
                    torch.tensor(self.demo_instruction_embed_arr[idx]),
                    dim=0
                ) <= self.sim_threshold
            ).item()
            
            if not sim_flag:
                return False
            
        return True
        
    def rejection_strategy(
        self,
        cand_idx_list,
        shot_idx,
        lower_value_threshold,
        upper_value_threshold,
        in_context_ifd_arr,
        history_conversation_list
    ):
        max_cand_value = float("-inf")
        selected_row = None
        
        if shot_idx < len(in_context_ifd_arr) - 1:
            response = history_conversation_list[-1]["content"]
            ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(self.tokenizer, self.model, response, self.max_length)
        else:
            ppl_out_alone, loss_out_alone = None, None
        
        for k in range(len(cand_idx_list)):
            cand_idx = cand_idx_list[k]
            cand_instruction = self.demo_instruction_list[cand_idx]
            cand_response = self.demo_response_list[cand_idx]
            cand_ifd = self.demo_ifd_list[cand_idx]
            cand_row = [0] * len(in_context_ifd_arr)
            cand_row[shot_idx] = cand_ifd
            
            if shot_idx < len(in_context_ifd_arr) - 1:
                top_conversation_list = [
                    {"role": "user", "content": cand_instruction},
                    {"role": "assistant", "content": cand_response}
                ]
                
                conversation = [{"role": "system", "content": ""}]
                conversation.extend(top_conversation_list)
                conversation.extend(list(history_conversation_list))
                
                whole_text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                instruction = self.tokenizer.apply_chat_template(
                    conversation[:-1],
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                instruction_input_ids = self.tokenizer.encode(instruction, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
                instruction_len = instruction_input_ids.shape[1]

                ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(self.tokenizer, self.model, whole_text, response, self.max_length)

                ifd = ppl_out_condition / ppl_out_alone
                cand_row[-1] = ifd
                cand_value = 1 - ifd / in_context_ifd_arr[shot_idx + 1][-1]
            else:
                cand_value = cand_ifd
                
            if upper_value_threshold >= cand_value > max_cand_value:
                max_cand_value = cand_value
                selected_row = cand_row
                
                cand_idx_list[0], cand_idx_list[k] = cand_idx_list[k], cand_idx_list[0]
                
        if max_cand_value > lower_value_threshold or shot_idx == len(in_context_ifd_arr) - 1:
            in_context_ifd_arr[shot_idx] = selected_row
            return True
        else:
            return False
      
    def greedy_strategy(
        self,
        cand_idx_list,
        shot_idx,
        lower_value_threshold,
        in_context_ifd_arr,
        history_conversation_list
    ):
        max_cand_value = float("-inf")
        selected_row = None
        
        if shot_idx < len(in_context_ifd_arr) - 1:
            response = history_conversation_list[-1]["content"]
            ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(self.tokenizer, self.model, response, self.max_length)
        else:
            ppl_out_alone, loss_out_alone = None, None
        
        for k in range(len(cand_idx_list)):
            cand_idx = cand_idx_list[k]
            cand_instruction = self.demo_instruction_list[cand_idx]
            cand_response = self.demo_response_list[cand_idx]
            cand_ifd = self.demo_ifd_list[cand_idx]
            cand_row = [0] * len(in_context_ifd_arr)
            cand_row[shot_idx] = cand_ifd
            
            if shot_idx < len(in_context_ifd_arr) - 1:
                top_conversation_list = [
                    {"role": "user", "content": cand_instruction},
                    {"role": "assistant", "content": cand_response}
                ]
                
                conversation = [{"role": "system", "content": ""}]
                conversation.extend(top_conversation_list)
                conversation.extend(list(history_conversation_list))
                
                whole_text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                instruction = self.tokenizer.apply_chat_template(
                    conversation[:-1],
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                instruction_input_ids = self.tokenizer.encode(instruction, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
                instruction_len = instruction_input_ids.shape[1]

                ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(self.tokenizer, self.model, whole_text, response, self.max_length)

                ifd = ppl_out_condition / ppl_out_alone
                cand_row[-1] = ifd
                cand_value = 1 - ifd / in_context_ifd_arr[shot_idx + 1][-1]
            else:
                cand_value = cand_ifd
                
            if cand_value > max_cand_value:
                max_cand_value = cand_value
                selected_row = cand_row
                
                cand_idx_list[0], cand_idx_list[k] = cand_idx_list[k], cand_idx_list[0]
                
        if max_cand_value > lower_value_threshold or shot_idx == len(in_context_ifd_arr) - 1:
            in_context_ifd_arr[shot_idx] = selected_row
            return True
        else:
            return False
         
    def demo_selection(self):
        shot_list = []
        
        for i in tqdm(range(len(self.instruction_list))):
            selected_idx_list = deque([])
            selected_ifd_list = deque([])
            history_conversation_list = deque([])
            curr_lower_value_threshold = self.lower_value_threshold
            curr_upper_value_threshold = self.upper_value_threshold
            
            if self.wrt_adv_prefix:
                history_conversation_list.extend(
                    [
                        {"role": "user", "content": self.instruction_list[i]},
                        {"role": "assistant", "content": self.adv_prefix_list[i]}
                    ]
                )
                in_context_ifd_arr = [[0] * (self.num_shots + 1) for _ in range(self.num_shots + 1)]
                in_context_ifd_arr[-1][-1] = self.adv_prefix_ifd_list[i]
            else:
                in_context_ifd_arr = [[0] * self.num_shots for _ in range(self.num_shots)]
                
            j = 0
            
            while j < self.num_shots:
                num_attempts = 0
                selected = False
                shot_idx = self.num_shots - j - 1
                
                history = set()
                
                while not selected and num_attempts < self.max_num_attempts:
                    cand_idx_list = np.random.randint(len(self.demo_instruction_list), size=self.num_cands_per_attempt).tolist()
                    
                    k = 0
                    
                    while k < len(cand_idx_list):
                        sim_flag = self.similarity_strategy(
                            cand_idx=cand_idx_list[k],
                            instruction_idx=i,
                            selected_idx_list=selected_idx_list
                        )
                        
                        if cand_idx_list[k] in history or not sim_flag:
                            cand_idx_list.pop(k)
                        else:
                            history.add(cand_idx_list[k])
                            k += 1
                    
                    if cand_idx_list:
                        if self.selector_mode == "random":
                            selected = True
                        elif self.selector_mode == "greedy":
                            selected = self.greedy_strategy(
                                cand_idx_list=cand_idx_list,
                                shot_idx=shot_idx,
                                lower_value_threshold=curr_lower_value_threshold,
                                in_context_ifd_arr=in_context_ifd_arr,
                                history_conversation_list=history_conversation_list
                            )
                        elif self.selector_mode == "rejection":
                            selected = self.rejection_strategy(
                                cand_idx_list=cand_idx_list,
                                shot_idx=shot_idx,
                                lower_value_threshold=curr_lower_value_threshold,
                                upper_value_threshold=curr_upper_value_threshold,
                                in_context_ifd_arr=in_context_ifd_arr,
                                history_conversation_list=history_conversation_list
                            )
                        else:
                            raise NotImplementedError
                        
                    if selected:
                        break

                    num_attempts += 1
                    self.global_step += 1
                    
                if selected:
                    curr_lower_value_threshold += self.relax_ratio * abs(self.lower_value_threshold)
                    curr_lower_value_threshold = min(
                        self.lower_value_threshold,
                        curr_lower_value_threshold
                    )
                                        
                    cand_idx = cand_idx_list[0]
                    
                    selected_idx_list.appendleft(cand_idx)
                    selected_ifd_list.appendleft(self.demo_ifd_list[cand_idx])
                    history_conversation_list.appendleft(
                        {"role": "assistant", "content": self.demo_response_list[cand_idx]}
                    )
                    history_conversation_list.appendleft(
                        {"role": "user", "content": self.demo_instruction_list[cand_idx]}
                    )
                    j += 1
                elif self.selector_mode == "greedy":
                    curr_lower_value_threshold -= self.relax_ratio * abs(self.lower_value_threshold)
                    
                    print("=" * 100)
                    print(f"Number of selected demos: {len(selected_idx_list)}.")
                    print("Max number of attempts is reached.")
                    print("Relax the constraint and restart.")
                    print(f"Current lower value threshold: {curr_lower_value_threshold}.")
                elif self.selector_mode == "rejection":
                    curr_lower_value_threshold -= self.relax_ratio * abs(self.lower_value_threshold)
                    curr_upper_value_threshold += self.relax_ratio * abs(self.upper_value_threshold)

                    print("=" * 100)
                    print(f"Number of selected demos: {len(selected_idx_list)}.")
                    print("Max number of attempts is reached.")
                    print("Relax the constraint and restart.")
                    print(f"Current lower value threshold: {curr_lower_value_threshold}.")
                    print(f"Current upper value threshold: {curr_upper_value_threshold}.")
                else:
                    raise NotImplementedError
                
            assert len(selected_idx_list) == self.num_shots
            
            shot_list.append(list(selected_idx_list))
        
            if self.in_context_arr_path:
                output_dict = {
                    "demo_idx": list(selected_idx_list),
                    "in_context_ifd": in_context_ifd_arr,
                    "orig_ifd": list(selected_ifd_list) + [self.adv_prefix_ifd_list[i]]
                }
                
                with open(self.in_context_arr_path, "a") as f:
                    f.write(json.dumps(output_dict) + "\n")
            
        return shot_list