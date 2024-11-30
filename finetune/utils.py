import copy
import os

import datasets
import peft
import torch
import transformers


PROMPT = r"""你是一个广告文案大师，请根据下面的描述帮我写一段广告

### 描述：{content}
### 广告："""

def create_model(model_args, training_args, finetune_args):
    if not os.path.isabs(model_args.model_name_or_path):
        raise ValueError("当前模型路径只支持本地路径以绝对路径的形式传入")

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    model_dtype = torch.bfloat16 if training_args.bf16 else None
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        device_map=(
            training_args.device.type if model_args.load_meta_device else None
        )
    )
    if finetune_args.peft_type == "lora":
        peft_config = peft.LoraConfig(
            r=finetune_args.lora_rank,
            lora_alpha=finetune_args.lora_alpha,
            lora_dropout=finetune_args.lora_dropout,
            target_modules=finetune_args.lora_target_modules,
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
        )
        model = peft.get_peft_model(model, peft_config)
        if training_args.bf16 and finetune_args.peft_type != "ia3":
            model = model.to(torch.bfloat16)
        model.print_trainable_parameters()
    return model, tokenizer

def create_prompts(examples):
    prompts = {}
    prompts["source"] = []
    prompts["target"] = []
    for example in examples:
        content = example.get("content", None)
        summary = example.get("summary", None)
        if content and summary:
            prompts["source"].append(PROMPT.format(content=content))
            prompts["target"].append(summary)
    return prompts

def create_dataset(data_args):
    if data_args.dataset_name is not None:
        raise ValueError("当前暂不支持从 huggingface hub 上下载数据集")
    
    extension = "json"
    raw_datasets = datasets.load_dataset(
        extension,
        data_files={
            "train": data_args.train_file,
            "validation": data_args.validation_file
        }
    )

    for key in raw_datasets:
        prompts = create_prompts(raw_datasets[key])
        columns_to_be_removed = list(raw_datasets[key].features.keys())
        raw_datasets[key] = raw_datasets[key].add_column("prompt_sources", prompts["source"])
        raw_datasets[key] = raw_datasets[key].add_column("prompt_targets", prompts["target"])
        raw_datasets[key] = raw_datasets[key].remove_columns(columns_to_be_removed)
    
    return raw_datasets

# TODO: 优化初始化数据集函数
def preprocess_func(examples, tokenizer, data_args):
    max_seq_length = data_args.max_seq_length
    keys = list(examples.data.keys())
    st = [s + t for s, t in zip(examples[keys[0]], examples[keys[1]])]
    examples_tokenized = tokenizer(
        st,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors=None,
    )
    for i in range(len(examples_tokenized["input_ids"])):
        if (
            examples_tokenized["input_ids"][i][-1] != tokenizer.eos_token_id
            and len(examples_tokenized["input_ids"][i]) < data_args.max_seq_length
            and examples_tokenized
        ):
            examples_tokenized["input_ids"][i].append(tokenizer.eos_token_id)
            examples_tokenized["attention_mask"][i].append(1)
    examples_tokenized["labels"] = copy.deepcopy(examples_tokenized["input_ids"])
    examples_tokenized["input_id_len"] = [len(result) for result in examples_tokenized["input_ids"]]

    return examples_tokenized

def concatenate_data(dataset, max_seq_length):
    concatenated_dataset = {}
    for column in dataset.features:
        concatenated_data = [
            item for sample in dataset[column] for item in sample
        ]
        reshaped_data = [
            concatenated_data[i * max_seq_length : (i + 1) * max_seq_length]
            for i in range(len(concatenated_data) // max_seq_length)
        ]
        concatenated_dataset[column] = reshaped_data
    return datasets.Dataset.from_dict(concatenated_dataset)

if __name__ == "__main__":
    from args import (
        DataArguments,
        ModelArguments,
        FinetuneArguments
    )
    from transformers import TrainingArguments
    data_args = DataArguments(
        train_file="datasets/train.json",
        validation_file="datasets/dev.json",
        max_seq_length=768
    )
    model_args = ModelArguments(model_name_or_path="/Users/lin/codes/habana_train/models",)
    training_args = TrainingArguments(bf16=True, output_dir=".output")
    finetune_args = FinetuneArguments()

    model, tokenizer = create_model(model_args, training_args, finetune_args)
    raw_datasets = create_dataset(data_args)
    column_names = raw_datasets["train"].features
    tokenized_datasets = raw_datasets.map(
        preprocess_func,
        batched=True,
        load_from_cache_file=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "data_args": data_args
        },
        remove_columns=column_names
    )
    print(tokenized_datasets["train"][0])
    