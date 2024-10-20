from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoConfig, AutoModel, AutoTokenizer
from .models import REGISTERED_MODELS

for model_name, (model_class, tokenizer_class) in REGISTERED_MODELS.items():
    AutoConfig.register(model_name, model_class.config_class)
    AutoModel.register(model_class.config_class, model_class)
    AutoTokenizer.register(model_class.config_class, tokenizer_class)


def insure_all_registered():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / 'config.json'

        for name in REGISTERED_MODELS:
            with open(temp_path, "w") as f:
                f.write('{' + f'"model_type": "{name}"' + '}')
            auto_config = AutoConfig.from_pretrained(temp_path)
            assert isinstance(auto_config, REGISTERED_MODELS[name][0].config_class)
