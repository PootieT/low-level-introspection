import operator

import torch
from datasets import Dataset
from nnsight.modeling.language import LanguageModel
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.model import replace_model_with_gated_version, replace_lora_with_gated_version
from src.introspector.utils import get_quantization_config
from train_introspector import INTROSPECTION_SEPARATOR

MODEL_NAME = "meta-llama/Llama-3.2-1B"
LORA_PATH = "./test_artifacts/random_lora_model"
INPUTS = [
    "hello my name is dog",
    "the grey fox jumps over the lazy dog",
    "what's up my guy",
    "introspection research is cool",
    "you don't believe models can introspect?",
    "fruits banana apple love quizzes"
]
LAYER=10
SIGNAL_INDEX=24


def get_random_lora_model() -> PeftModel:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # wrap model in lora config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights = False # this ensures lora_B is not initialized to be zero
    )
    model = get_peft_model(model, peft_config)
    # model.save_pretrained(LORA_PATH)
    return model

def test_unchanged_signal_with_gated_lora():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {"additional_special_tokens": [INTROSPECTION_SEPARATOR]}
    num_added = tokenizer.add_special_tokens(special_tokens)

    model = LanguageModel(
        MODEL_NAME,
        device_map="auto",
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True)
    )
    frozen_signals = []
    for input_seq in INPUTS:
        with torch.no_grad(), model.trace(input_seq):
            mlp_layer_path = resolve_mlp_signal_path(model, LAYER)
            getter = operator.attrgetter(mlp_layer_path)
            # assuming dimensions are [bs, token, h], output dim is [bs, h]
            frozen_signal = getter(model)[0, -1, SIGNAL_INDEX].detach().save()
            frozen_signals.append(frozen_signal)

    # now pass the lora model
    gated_model = get_random_lora_model()
    # gated_model = PeftModel.from_pretrained(model.model, LORA_PATH)
    # monkey patch forward to take in mask
    replace_model_with_gated_version(gated_model)
    # modify lora linear so it takes in mask and applies across tokens
    replace_lora_with_gated_version(gated_model)

    gated_model = LanguageModel(
        gated_model,
        tokenizer=tokenizer,
        # device_map="auto",
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True)
    )
    gated_signals = []
    for i, input_seq in enumerate(INPUTS):
        tokens = tokenizer(input_seq, return_tensors="pt").to("mps")
        tokens["lora_token_mask"] = torch.zeros_like(tokens["input_ids"])
        with torch.no_grad(), gated_model.trace(tokens):
            mlp_layer_path = resolve_mlp_signal_path(gated_model, LAYER)
            getter = operator.attrgetter(mlp_layer_path)
            # assuming dimensions are [bs, token, h], output dim is [bs, h]
            gated_signal = getter(gated_model)[0, -1, SIGNAL_INDEX].detach().save()
            gated_signals.append(gated_signal)

    for frozen_signal, gated_signal in zip(frozen_signals, gated_signals):
        assert torch.allclose(frozen_signal, gated_signal)

    # counterfactual: if mask is 1, then gated signals should be different
    ctf_gated_signals = []
    for i, input_seq in enumerate(INPUTS):
        tokens = tokenizer(input_seq, return_tensors="pt").to("mps")
        tokens["lora_token_mask"] = torch.ones_like(tokens["input_ids"])
        with torch.no_grad(), gated_model.trace(tokens):
            mlp_layer_path = resolve_mlp_signal_path(gated_model, LAYER)
            getter = operator.attrgetter(mlp_layer_path)
            # assuming dimensions are [bs, token, h], output dim is [bs, h]
            ctf_gated_signal = getter(gated_model)[0, -1, SIGNAL_INDEX].detach().save()
            ctf_gated_signals.append(ctf_gated_signal)

    for frozen_signal, ctf_gated_signal in zip(frozen_signals, ctf_gated_signals):
        assert not torch.allclose(frozen_signal, ctf_gated_signal)

    print("LoraGate test passed!")

if __name__ == "__main__":
    test_unchanged_signal_with_gated_lora()