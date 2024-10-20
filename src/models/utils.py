import torch
from .llama import Llama, RMSNorm
from .llama_pos import LlamaWithAbsolutePositions
from .modelling_llama_long_context import FlexLlama
from .modeling_base import GPTBase, LayerNorm


BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """ Return the right model """
    if args.model == 'base':
        model = GPTBase(args)
        return model
    elif args.model == 'llama2':
        model = Llama(args)
        return model
    elif args.model == 'llama2-pos':
        model = LlamaWithAbsolutePositions(args)
        return model
    elif args.model == 'llama2-long-context':
        model = FlexLlama(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
