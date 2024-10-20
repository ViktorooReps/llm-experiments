from typing import Type

from transformers import GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer

from .modeling_base import GPTBase
from .modelling_llama_long_context import FlexLlama, FlexLlamaForCausalLM

REGISTERED_MODELS: dict[str, tuple[Type[PreTrainedModel], Type[PreTrainedTokenizer]]] = {
    'gpt-base': (GPTBase, GPT2Tokenizer),
    'flex-llama': (FlexLlama, GPT2Tokenizer),
}

REGISTERED_LM_MODELS: dict[str, tuple[Type[PreTrainedModel], Type[PreTrainedTokenizer]]] = {
    'flex-llama': (FlexLlamaForCausalLM, GPT2Tokenizer),
}
