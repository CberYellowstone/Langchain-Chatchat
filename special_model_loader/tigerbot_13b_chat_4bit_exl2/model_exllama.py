import os
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Optional, Union

import torch
import transformers
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
from torch.nn import CrossEntropyLoss
from transformers import (
    GenerationConfig,
    LlamaTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    TextIteratorStreamer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from var_dump import var_dump

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


class Exllamav2HF(PreTrainedModel):
    def __init__(self, config: ExLlamaV2Config):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlamaV2(config)
        split = None
        self.ex_model.load(split)

        self.generation_config = GenerationConfig()

        self.ex_cache = ExLlamaV2Cache(self.ex_model)
        self.past_seq = None

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get("use_cache", True)
        labels = kwargs.get("labels", None)
        past_key_values = kwargs.get("past_key_values", None)

        if len(args) > 0:
            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            ex_cache = self.ex_cache_negative
        else:
            input_ids = kwargs["input_ids"]
            is_negative = False
            past_seq = self.past_seq
            ex_cache = self.ex_cache

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)
        reset = True

        # Make the forward call
        if labels is None:
            if past_seq is not None:
                min_length = min(past_seq.shape[0], seq_tensor.shape[0])
                indices = torch.nonzero(
                    ~torch.eq(past_seq[:min_length], seq_tensor[:min_length])
                )
                if len(indices) > 0:
                    longest_prefix = indices[0].item()
                else:
                    longest_prefix = min_length

                if longest_prefix > 0:
                    reset = False
                    ex_cache.current_seq_len = longest_prefix
                    if len(seq_tensor) - longest_prefix > 1:
                        self.ex_model.forward(
                            seq_tensor[longest_prefix:-1].view(1, -1),
                            ex_cache,
                            preprocess_only=True,
                        )

            if reset:
                ex_cache.current_seq_len = 0
                if len(seq_tensor) > 1:
                    self.ex_model.forward(
                        seq_tensor[:-1].view(1, -1),
                        ex_cache,
                        preprocess_only=True,
                    )

            logits = self.ex_model.forward(seq_tensor[-1:].view(1, -1), ex_cache).to(
                input_ids.device
            )
        else:
            ex_cache.current_seq_len = 0
            logits = self.ex_model.forward(
                seq_tensor.view(1, -1), ex_cache, last_id_only=False
            )

        if is_negative:
            self.past_seq_negative = seq_tensor
        else:
            self.past_seq = seq_tensor

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=seq if use_cache else None,
            loss=loss,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        assert (
            len(model_args) == 0 and len(kwargs) == 0
        ), "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        config = ExLlamaV2Config()
        config.model_dir = str(pretrained_model_name_or_path)
        config.prepare()

        # config.max_seq_len = shared.args.max_seq_len
        # config.scale_pos_emb = shared.args.compress_pos_emb
        # config.scale_alpha_value = shared.args.alpha_value

        return Exllamav2HF(config)


def _get_model(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = Exllamav2HF.from_pretrained(model)
    model.eval()
    return model


def load_model(
    model_path: str,
    max_generate_length: int = 2048,
):
    # print(f"loading model: {model_path}...")
    model = _get_model(model_path)

    tokenizer = LlamaTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        model_max_length=max_generate_length,
        padding_side="left",
        truncation_side="left",
        padding=True,
        truncation=True,
    )
    # if (
    #     tokenizer.model_max_length is None
    #     or tokenizer.model_max_length > max_generate_length
    # ):
    #     tokenizer.model_max_length = max_generate_length

    generation_config = GenerationConfig.from_pretrained(model_path)
    # generation_config.max_new_tokens = max_generate_length

    return model, tokenizer, generation_config


def generate_stream(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoModelForCausalLM,
    params: dict,
    generation_config: transformers.GenerationConfig,
    device: Any,
):
    def eval_generate(**args):
        with torch.inference_mode(mode=True):
            model.eval()
            model.generate(**args)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )
    generation_config.temperature = params["temperature"]

    generation_kwargs = generation_config.to_dict()
    generation_kwargs["streamer"] = streamer

    input_text = params["prompt"]

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        # max_length=generation_config.max_length - generation_config.max_new_tokens,
        truncation=True,
    )
    prompt_tokens = len(tokenizer.encode(params["prompt"]))

    for k, v in inputs.items():
        generation_kwargs[k] = v.to(device)

    thread = Thread(target=eval_generate, kwargs=generation_kwargs)
    thread.start()

    generated_tokens = 0
    text = ""
    for output in streamer:
        text += output
        generated_tokens += 1

        yield {
            "text": text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
            },
            "finish_reason": None,
        }
    yield {
        "text": text,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated_tokens,
            "total_tokens": prompt_tokens + generated_tokens,
        },
        "finish_reason": "stop",
    }
