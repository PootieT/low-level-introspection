import pdb

import torch
from transformers import DataCollatorForSeq2Seq


class LoRAGatingCollator(DataCollatorForSeq2Seq):
    """
    Our preprocessed dataset contains additional feature "frozen_input_len"
    which isn't padded correctly by default collator, so here we have to
    subclass and manually pad it with additional 0s
    """
    def __call__(self, features):
        input_lens = [f.pop("frozen_input_len") for f in features]
        unpadded_full_prompt_lens = [len(f["input_ids"]) for f in features]
        batch = super().__call__(features)
        seq_len = batch["input_ids"].shape[1]

        padded_masks = []
        for input_len, full_prompt_len in zip(input_lens, unpadded_full_prompt_lens):
            mask = [0] * input_len + [1] * (full_prompt_len - input_len)
            pad_len = seq_len - len(mask)
            padded_masks.append(mask + [0] * pad_len)

        batch["lora_token_mask"] = torch.tensor(padded_masks,dtype=torch.float32)
        return batch