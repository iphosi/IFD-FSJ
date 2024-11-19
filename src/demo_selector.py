import torch
import numpy as np

import json
import pandas as pd
from collections import deque

from get_in_context_ifd import get_in_context_ifd_arr_row

from tqdm import tqdm


class DemoSelector:
    def __init__(
        self,
        selector_mode,
        num_shots,
        sim_threshold,
        ifd_drop_threshold,
        relax_ratio,
        max_num_attempts,
        demo_instruction_embed_arr,
        demo_instruction_list,
        demo_response_list,
        demo_ifd_list,
        instruction_embed_arr,
        instruction_list,
        adv_prompt_list,
        adv_prompt_ifd_list,
        compute_adv_prompt_ifd=False,
        in_context_ifd_path=None,
        tokenizer=None,
        model=None,
        max_length=4096
    ):
        self.selector_mode = selector_mode
        self.num_shots = num_shots
        
        self.sim_threshold = sim_threshold
        self.ifd_drop_threshold = ifd_drop_threshold
        self.relax_ratio = relax_ratio
        self.max_num_attempts = max_num_attempts
        
        assert self.ifd_drop_threshold != 0
        
        self.compute_adv_prompt_ifd = compute_adv_prompt_ifd
        self.in_context_ifd_path = in_context_ifd_path
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        
        self.demo_instruction_embed_arr = demo_instruction_embed_arr
        self.demo_instruction_list = demo_instruction_list
        self.demo_response_list = demo_response_list
        self.demo_ifd_list = demo_ifd_list
        self.instruction_embed_arr = instruction_embed_arr
        self.instruction_list = instruction_list
        self.adv_prompt_list = adv_prompt_list
        self.adv_prompt_ifd_list = adv_prompt_ifd_list
        
        self.tokenizer = tokenizer
        self.model=model
        self.max_length=max_length
        
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
        
    def ifd_rejection_strategy(
        self,
        shot_idx,
        ifd_drop_threshold,
        in_context_ifd_arr,
        cand_instruction,
        cand_response,
        cand_ifd,
        history_conversation_list
    ):
        top_conversation_list = [
            {"role": "user", "content": cand_instruction},
            {"role": "assistant", "content": cand_response}
        ]
        row = get_in_context_ifd_arr_row(
            tokenizer=self.tokenizer,
            model=self.model,
            max_length=self.max_length,
            history_conversation_list=list(history_conversation_list),
            top_conversation_list=top_conversation_list,
            top_conversation_ifd=cand_ifd
        )
        row = [0] * (len(in_context_ifd_arr) - len(row)) + row
        
        if shot_idx < len(in_context_ifd_arr) - 1:
            ifd_drop_arr = np.array(in_context_ifd_arr[shot_idx + 1]) - np.array(row)
            cand_value = ifd_drop_arr[shot_idx + 1:].sum()
            ifd_drop_flag = (ifd_drop_arr[shot_idx + 1:] >= ifd_drop_threshold).all() and cand_value > 0
            
            if ifd_drop_flag:
                in_context_ifd_arr[shot_idx] = row
                
            return ifd_drop_flag
        else:
            in_context_ifd_arr[shot_idx] = row
            return True
        
    def demo_selection(self):
        shot_list = []
        
        for i in tqdm(range(len(self.instruction_list))):
            selected_idx_list = deque([])
            selected_ifd_list = deque([])
            history_conversation_list = deque([])
            curr_ifd_drop_threshold = self.ifd_drop_threshold
            
            if self.compute_adv_prompt_ifd:
                history_conversation_list.extend(
                    [
                        {"role": "user", "content": instruction_list[i]},
                        {"role": "assistant", "content": adv_prompt_list[i]}
                    ]
                )
                in_context_ifd_arr = [[0] * (self.num_shots + 1) for _ in range(self.num_shots + 1)]
                in_context_ifd_arr[-1][-1] = self.adv_prompt_ifd_list[i]
            else:
                in_context_ifd_arr = [[0] * self.num_shots for _ in range(self.num_shots)]
                
            j = 0
            
            while j < self.num_shots:
                num_attempts = 0
                selected = False
                shot_idx = self.num_shots - j - 1
                
                history = set()
                
                while not selected and num_attempts < self.max_num_attempts:
                    cand_idx = np.random.randint(len(self.demo_instruction_list))
                    
                    if cand_idx in history:
                        continue
                    
                    history.add(cand_idx)
                    
                    sim_flag = self.similarity_strategy(
                        cand_idx=cand_idx,
                        instruction_idx=i,
                        selected_idx_list=selected_idx_list
                    )
                    
                    if sim_flag:
                        if self.selector_mode == "random":
                            selected = True
                        elif self.selector_mode == "ifd_rejection":
                            selected = self.ifd_rejection_strategy(
                                shot_idx=shot_idx,
                                ifd_drop_threshold=curr_ifd_drop_threshold,
                                in_context_ifd_arr=in_context_ifd_arr,
                                cand_instruction=self.demo_instruction_list[cand_idx],
                                cand_response=self.demo_response_list[cand_idx],
                                cand_ifd=self.demo_ifd_list[cand_idx],
                                history_conversation_list=history_conversation_list,
                            )
                        else:
                            raise NotImplementedError
                        
                    if selected:
                        break

                    num_attempts += 1
                    
                if selected:
                    curr_ifd_drop_threshold += self.relax_ratio * abs(self.ifd_drop_threshold)
                    
                    selected_idx_list.appendleft(cand_idx)
                    selected_ifd_list.appendleft(self.demo_ifd_list[cand_idx])
                    history_conversation_list.appendleft(
                        {"role": "assistant", "content": self.demo_response_list[cand_idx]}
                    )
                    history_conversation_list.appendleft(
                        {"role": "user", "content": self.demo_instruction_list[cand_idx]}
                    )
                    j += 1
                elif self.selector_mode == "ifd_rejection":
                    curr_ifd_drop_threshold -= self.relax_ratio * abs(self.ifd_drop_threshold)
                    
                    print("=" * 100)
                    print(f"Number of selected demos: {len(selected_idx_list)}.")
                    print("Max number of attempts is reached.")
                    print("Relax the ifd drop constraint and restart.")
                    print(f"Current ifd drop threshold: {curr_ifd_drop_threshold}.")
                else:
                    raise NotImplementedError
                
            assert len(selected_idx_list) == self.num_shots
            
            shot_list.append(list(selected_idx_list))
        
            if self.in_context_ifd_path:
                output_dict = {
                    "demo_idx": list(selected_idx_list),
                    "orig_ifd": list(selected_ifd_list),
                    "adv_prompt_ifd": self.adv_prompt_ifd_list[i],
                    "in_context_ifd": in_context_ifd_arr
                }
                
                with open(self.in_context_ifd_path, "a") as f:
                    f.write(json.dumps(output_dict) + "\n")
            
        return shot_list