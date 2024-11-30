#!/usr/bin/env python
# coding=utf-8

# Apache v2 license
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import transformers
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import (
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)
from transformers.trainer_utils import is_main_process

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.utils import set_seed

import args
import utils

# 初始化日志系统
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"

def main():
    parser = HfArgumentParser(
        (
            args.ModelArguments,
            args.DataArguments,
            GaudiTrainingArguments,
            args.FinetuneArguments,
        )
    )

    # 如果传入 json 文件，则参数从 json 文件中解析，
    # 否则解析命令行参数
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, finetune_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            finetune_args,
        ) = parser.parse_args_into_dataclasses()

    # 日志训练参数信息
    b16 = training_args.fp16 or training_args.bf16
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {b16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # 初始化随机数种子
    set_seed(training_args.seed)

    # 加载模型和分词器
    model, tokenizer = utils.create_model(model_args, training_args, finetune_args)

    # 加载数据
    raw_datasets = utils.create_dataset(data_args)
    column_names = raw_datasets["train"].features
    tokenized_datasets = raw_datasets.map(
        utils.preprocess_func,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        fn_kwargs={"tokenizer": tokenizer, "data_args": data_args},
        remove_columns=column_names,
    )
    if data_args.dataset_concatenation:
        max_seq_length = data_args.max_seq_length
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]
        tokenized_datasets["train"] = utils.concatenate_data(
            train_dataset, max_seq_length
        )
        tokenized_datasets["validation"] = utils.concatenate_data(
            val_dataset, max_seq_length
        )

    train_dataset = tokenized_datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    eval_dataset = tokenized_datasets["validation"]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False
    )
    logger.info(
        "Using data collator of type {}".format(data_collator.__class__.__name__)
    )

    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    # 初始化训练器
    # TODO: 添加评估流程
    trainer = GaudiTrainer(
        model=model,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        # eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics if training_args.do_eval else None,
        # preprocess_logits_for_metrics=(
        #     preprocess_logits_for_metrics if training_args.do_eval else None
        # ),
    )
    print(trainer)

    # Solution for https://github.com/huggingface/peft/blob/v0.6.2/README.md#caveats (1)
    if (
        training_args.fsdp
        and training_args.fsdp_config["auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
    ):
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(
            model
        )

    logger.info("=> 开始训练...")
    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    if data_args.save_last_ckpt:
        trainer.save_model()


if __name__ == "__main__":
    main()
