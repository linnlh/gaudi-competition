from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer
    we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    use_cache: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not the model should return the last key/values attentions (not used by all models)."
                "Only relevant if `config.is_decoder=True`."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "When set to True, it will benefit LLM loading time and RAM consumption."
            )
        },
    )
    attn_softmax_bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to run attention softmax layer in bf16 precision for fine-tuning. The current support is limited to Llama only."
            )
        },
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use Habana flash attention for fine-tuning. The current support is limited to Llama only."
            )
        },
    )
    flash_attention_recompute: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable recompute in Habana flash attention for fine-tuning."
                " It is applicable only when use_flash_attention is True."
            )
        },
    )
    flash_attention_causal_mask: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable causal mask in Habana flash attention for fine-tuning."
                " It is applicable only when use_flash_attention is True."
            )
        },
    )
    flash_attention_fp8: bool = field(
        default=False,
        metadata={"help": ("Whether to enable flash attention in FP8.")},
    )
    use_fused_rope: bool = field(
        default=True,
        metadata={
            "help": ("Whether to use Habana fused-rope for fine-tuning. The current support is limited to Llama only.")
        },
    )
    load_meta_device: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to load the model to the device instead of the host, so it can reduce the host RAM usage."
                "https://huggingface.co/blog/accelerate-large-models"
            )
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    validation_split_percentage: Optional[int] = field(
        default=0,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep in memory the loaded dataset. Defaults to False."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    dataset_seed: int = field(
        default=42,
        metadata={
            "help": "Seed to use in dataset processing, different seeds might yield different datasets. This seed and the seed in training arguments are not related"
        },
    )
    dataset_cache_directory: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory where the processed dataset will be saved. If path exists, try to load processed dataset from this path."
        },
    )
    dataset_concatenation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to concatenate the sentence for more efficient training."},
    )
    sql_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to have a SQL style prompt"},
    )
    chat_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to have a chat style prompt."},
    )
    save_last_ckpt: bool = field(
        default=True, metadata={"help": "Whether to save checkpoint at the end of the training."}
    )
    instruction_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the column in the dataset that describes the task that the model should perform. By "
            "default, the 'instruction' column is used for non-SQL prompts and the 'question' column is used for SQL prompts."
        },
    )
    input_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the column in the dataset that optionally provides context or input for the task. By "
            "default, the 'input' column is used for non-SQL prompts and the 'context' column is used for SQL prompts."
        },
    )
    output_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the column in the dataset with the answer to the instruction. By default, the "
            "'output' column is used for non-SQL prompts and the 'answer' column is used for SQL prompts."
        },
    )

@dataclass
class FinetuneArguments:
    """
    Arguments of finetune we are going to apply on the model.
    """

    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank parameter in the LoRA method."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Alpha parameter in the LoRA method."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout parameter in the LoRA method."},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA/AdaLoRA method."},
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    adalora_init_r: int = field(
        default=12,
        metadata={"help": "Initial AdaLoRA rank"},
    )
    adalora_target_r: int = field(
        default=4,
        metadata={"help": "Target AdaLoRA rank"},
    )
    adalora_tinit: int = field(
        default=50,
        metadata={"help": "Number of warmup steps for AdaLoRA wherein no pruning is performed"},
    )
    adalora_tfinal: int = field(
        default=100,
        metadata={
            "help": "Fix the resulting budget distribution and fine-tune the model for tfinal steps when using AdaLoRA"
        },
    )
    adalora_delta_t: int = field(
        default=10,
        metadata={"help": "Interval of steps for AdaLoRA to update rank"},
    )
    adalora_orth_reg_weight: float = field(
        default=0.5,
        metadata={"help": "Orthogonal regularization weight for AdaLoRA"},
    )
    peft_type: str = field(
        default="lora",
        metadata={
            "help": ("The PEFT type to use."),
            "choices": ["lora", "ia3", "adalora", "llama-adapter", "vera", "ln_tuning"],
        },
    )
    ia3_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the IA3 method."},
    )
    feedforward_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target feedforward modules for the IA3 method."},
    )
    adapter_layers: int = field(
        default=30,
        metadata={"help": "Number of adapter layers (from the top) in llama-adapter"},
    )
    adapter_len: int = field(
        default=10,
        metadata={"help": "Number of adapter tokens to insert in llama-adapter"},
    )
    vera_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the vera method."},
    )
    ln_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the ln method."},
    )