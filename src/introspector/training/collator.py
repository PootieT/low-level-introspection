import pdb

import torch
from transformers import DataCollatorForSeq2Seq


class LoRAGatingCollator(DataCollatorForSeq2Seq):
    """
    Our preprocessed dataset contains additional feature "frozen_input_len"
    which isn't padded correctly by default collator, so here we have to
    subclass and manually pad it with additional 0s
    """

    def __init__(self, *args, for_prediction=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.for_prediction = for_prediction
        if for_prediction:
            self.tokenizer.padding_side = "left"

    def __call__(self, features):
        # Enforce pad side per call: the mask branch below assumes left-pad for
        # prediction and right-pad for training. Gemma-2's tokenizer defaults to
        # padding_side="left", so relying on the tokenizer default (or a one-time
        # __init__ set on a shared tokenizer) misaligns the gate mask with the
        # padded input_ids. Set it here so it is correct regardless of tokenizer
        # default or collator construction order.
        self.tokenizer.padding_side = "left" if self.for_prediction else "right"
        # Gate-OFF training builds examples WITHOUT frozen_input_len (see
        # make_preprocess(return_frozen_input_len=disable_adapter_in_input) in
        # trainers/train_sae_detector.py). No key => no gating: emit a plain batch.
        if not features or "frozen_input_len" not in features[0]:
            return super().__call__(features)
        input_lens = [f.pop("frozen_input_len") for f in features]
        unpadded_full_prompt_lens = [len(f["input_ids"]) for f in features]
        batch = super().__call__(features)
        seq_len = batch["input_ids"].shape[1]

        padded_masks = []
        for input_len, full_prompt_len in zip(input_lens, unpadded_full_prompt_lens):
            mask = [0] * input_len + [1] * (full_prompt_len - input_len)
            pad_len = seq_len - len(mask)
            if self.for_prediction:
                padded_masks.append([0] * pad_len + mask)
            else:
                padded_masks.append(mask + [0] * pad_len)

        batch["lora_token_mask"] = torch.tensor(padded_masks, dtype=torch.float32)
        return batch