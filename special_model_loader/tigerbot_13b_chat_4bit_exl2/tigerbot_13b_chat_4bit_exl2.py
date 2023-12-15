"""
A model worker that executes the model.
"""
import json
import uuid
from typing import List, Optional

import torch
from fastchat.constants import SERVER_ERROR_MSG, ErrorCode
from fastchat.serve.base_model_worker import BaseModelWorker, app
from fastchat.utils import build_logger

from .model_exllama import generate_stream, load_model

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")


class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        device: str,
        no_register: bool,
        conv_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )
        self.call_ct = 0

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.context_len = 2048
        self.model, self.tokenizer, self.generation_config = load_model(
            model_path, self.context_len
        )
        self.device = device

        self.generation_config.do_sample = True

        self.generate_stream_func = generate_stream
        if not no_register:
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        try:
            for output in self.generate_stream_func(
                self.model, self.tokenizer, params, self.generation_config, self.device
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())

    def count_token(self, params):
        prompt = params["prompt"]
        try:
            input_ids = self.tokenizer(prompt).input_ids
            input_echo_len = len(input_ids)
        except TypeError:
            input_echo_len = self.tokenizer.num_tokens(prompt)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret
