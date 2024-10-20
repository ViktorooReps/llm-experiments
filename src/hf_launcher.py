import argparse
import inspect
import logging
from argparse import ArgumentParser
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Type, Literal, Any, Iterable, get_type_hints

import humanize
from transformers import HfArgumentParser, AutoConfig, PretrainedConfig, AutoModel, AutoTokenizer, \
    Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from src import insure_all_registered
from src.data.utils import get_config_type, get_data, Dataset

insure_all_registered()


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict[str, str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    config: PretrainedConfig,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    # update config with correct values for special tokens

    config.vocab_size = len(tokenizer)
    for token, token_value in special_tokens_dict.items():
        setattr(config, token, tokenizer.encode(token_value)[0])


def collect_init_arguments(cls: Type, skip_params: Iterable[str] = tuple()) -> set[tuple[str, Type, Any]]:
    """
    Collects the typed arguments of all __init__ methods in the class hierarchy.
    """
    args = []

    skip_params = set(skip_params)

    # Traverse the class hierarchy
    for current_cls in cls.__mro__:
        if current_cls is object:
            continue  # Ignore the base 'object' class

        init_method = current_cls.__init__
        sig = inspect.signature(init_method)
        type_hints = get_type_hints(init_method)

        for param_name, param in sig.parameters.items():
            if param_name in skip_params or param_name not in type_hints:
                continue

            param_type = type_hints[param_name]
            param_default = param.default if param.default is not inspect.Parameter.empty else None
            args.append((param_name, param_type, param_default))

    return set(args)


def create_dataclass_from_hierarchy(cls: Type, skip_params: Iterable[str] = tuple()) -> Type:
    """
    Creates a dataclass with the typed arguments of all __init__ methods in the class hierarchy.
    """
    init_args = collect_init_arguments(cls, skip_params=skip_params)

    DataclassType = make_dataclass(
        cls_name=f"{cls.__name__}DataClass",
        fields=init_args,
        bases=(),
        namespace={},
    )

    return DataclassType


@dataclass
class ExtraTrainingArguments:
    force_restart: bool = field(default=False, metadata={"help": "Ignore checkpoints and restart training"})
    tokenizer_source: Literal['hf', 'tiktoken'] = field(default='hf', metadata={"help": "Source of tokenizer to load"})
    tokenizer_name: str = field(default=None, metadata={
        "help": "Name or path for pretrained tokenizer (in case a new model is instantiated)"
    })
    validation_dataset_cutoff: int = field(default=None, metadata={"help": "Cutoff for validation (in examples)"})
    training_dataset_cutoff: int = field(default=None, metadata={"help": "Cutoff for training (in examples)"})


def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-base',
        help='The Hugging Face model name (e.g., "bert") or path. '
             'See `--model <model name or path> --help` for model-specific arguments.',
        required=False
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='slimpajama',
        help='Dataset name. See `--dataset <dataset name>` for dataset-specific arguments',
        required=False
    )

    args, unknown = parser.parse_known_args()

    model_name = args.model
    dataset_name = args.dataset

    pretrained = False
    try:
        config_class: Type[PretrainedConfig] = AutoConfig.from_pretrained(model_name).__class__
        pretrained = True
    except (ValueError, OSError):
        logging.warning(f'{model_name} is not a checkpoint or model on the hub, initializing empty weights!')
        config_class: Type[PretrainedConfig] = AutoConfig.for_model(model_name).__class__

    model_config_dataclass = create_dataclass_from_hierarchy(config_class, skip_params=('self', 'model_type', 'kwargs'))
    dataset_config_dataclass = get_config_type(dataset_name)

    desc = f"\n\nModel Configuration Help:\n{inspect.getdoc(config_class)}"

    config_parser = HfArgumentParser(
        [
            model_config_dataclass, dataset_config_dataclass,
            Seq2SeqTrainingArguments, ExtraTrainingArguments
        ],
        description=desc,
        formatter_class=argparse.RawTextHelpFormatter
    )
    model_args, dataset_args, training_args, extra_args, unknown = config_parser.parse_args_into_dataclasses(
        unknown, return_remaining_strings=True
    )
    training_args: Seq2SeqTrainingArguments
    extra_args: ExtraTrainingArguments

    if len(unknown):
        logging.warning(f'Ignoring unrecognized arguments: {", ".join(unknown)}')

    if pretrained:
        config = config_class.from_pretrained(model_name, **vars(model_args))
        tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    else:
        config = config_class.from_dict(vars(model_args))
        model = AutoModel.from_config(config=config)

        # tokenizer can only be loaded from pretrained

        if extra_args.tokenizer_source == 'hf':
            tokenizer = AutoTokenizer.from_pretrained(extra_args.tokenizer_name, config=config)
        elif extra_args.tokenizer_source == 'tiktoken':
            raise NotImplementedError
            # preload_tiktoken(extra_args.tokenizer_name, tokenizer_dir)
            # tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, config=config)
        else:
            logging.error(f'Unrecognized tokenizer source: {extra_args.tokenizer_source}')
            return

    # TODO: bandaid solution, think of something better
    tokenizer.model_max_length = config.sequence_length

    new_specials = {'pad_token': '<|pad|>', 'eos_token': '<|eos|>'}
    if getattr(config, 'add_sink', False):
        new_specials['sink_token'] = '<|sink|>'
    if getattr(config, 'add_blocks', False):
        new_specials['block_end_token'] = '<|block_end|>'

    smart_tokenizer_and_embedding_resize(new_specials, tokenizer, model, config)

    print(config)
    print(tokenizer)
    print(model)

    data = get_data(tokenizer, dataset_name, dataset_args)
    train_dataset = Dataset(
        data['train'],
        sequence_length=tokenizer.model_max_length,
        cutoff=extra_args.training_dataset_cutoff
    )
    validation_dataset = Dataset(
        data['validation'],
        sequence_length=tokenizer.model_max_length,
        cutoff=extra_args.validation_dataset_cutoff
    )

    print(f'Training data: {train_dataset}, {humanize.intword(train_dataset.n_tokens)} tokens')
    print(f'Validation data: {validation_dataset}, {humanize.intword(validation_dataset.n_tokens)} tokens')

    checkpoint_detected = False
    output_dir_path = Path(training_args.output_dir)

    if output_dir_path.exists():
        for checkpoint in output_dir_path.glob(f'{PREFIX_CHECKPOINT_DIR}*'):
            if checkpoint.is_dir():
                checkpoint_detected = True
                logging.warning(f'Found existing checkpoint!')
                break

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )
    trainer.train(resume_from_checkpoint=checkpoint_detected and not extra_args.force_restart)

    trainer.save_model()


if __name__ == "__main__":
    main()
