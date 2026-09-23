from __future__ import annotations

"""QLoRA training pipeline for model fine-tuning."""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset, load_dataset
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )

except ImportError:
    torch = None
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None
    TrainingArguments = Trainer = DataCollatorForLanguageModeling = None
    Dataset = load_dataset = LoraConfig = get_peft_model = None
    prepare_model_for_kbit_training = TaskType = None

from src.data_parsers.parser_factory import ParserFactory, DataSourceConfig, prepare_text_dataset


@dataclass
class TrainingConfig:
    """Configuration for QLoRA training."""
    # Model settings
    base_model_id: str = "meta-llama/Llama-2-7b-hf"
    adapter_name: str = "adapter"
    output_dir: str = "./training_output"

    # QLoRA settings
    quantization_bit: int = 4
    quantization_type: str = "nf4"  # 'nf4' or 'fp4'
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    max_seq_length: int = 512
    packing: bool = True

    # Data settings
    data_source: Optional[str] = None
    data_source_type: str = "local"
    data_source_config: Optional[DataSourceConfig] = None
    train_split: float = 0.9
    seed: int = 42

    # System settings
    local_files_only: bool = True
    gpu_ids: Optional[List[int]] = None
    max_storage_gb: float = 1024.0  # 1 TiB; model-download quota is enforced by the repository client


class QLoRAPipeline:
    """QLoRA training pipeline for efficient model fine-tuning."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize the QLoRA training pipeline.

        Args:
            config: Training configuration. If None, uses defaults.
        """
        self.config = config or TrainingConfig()
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._trainer: Optional[Trainer] = None
        self._training_stats: Dict[str, Any] = {}

    def prepare_data(self) -> Path:
        """Prepare training data from configured source.

        Returns:
            Path to the prepared training data directory.
        """
        output_dir = Path(self.config.output_dir) / "prepared_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        source = self.config.data_source_config
        if source is None and self.config.data_source:
            source = DataSourceConfig(self.config.data_source_type, self.config.data_source)
        if source is None:
            raise ValueError("Select a training data source")
        raw_dir = Path(self.config.output_dir) / "source_data"
        if raw_dir.exists() and any(raw_dir.iterdir()):
            raise ValueError("Use a new output directory for each training run")
        stats = ParserFactory.parse_data(source, raw_dir)
        if stats.get("errors") or not stats.get("files_processed"):
            raise RuntimeError(f"Data preparation failed: {stats.get('errors', [])}")
        stats["records"] = prepare_text_dataset(raw_dir, output_dir / "train.jsonl")
        self._training_stats["data_preparation"] = stats

        return output_dir

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the base model with QLoRA quantization.

        Returns:
            Tuple of (model, tokenizer).
        """
        if torch is None:
            raise RuntimeError("Install requirements-training.txt to enable QLoRA training")
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA training requires a supported NVIDIA CUDA GPU")
        if self.config.quantization_bit != 4:
            raise ValueError("This pipeline supports 4-bit QLoRA only")
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.quantization_type,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id,
            trust_remote_code=False,
            local_files_only=self.config.local_files_only,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
            local_files_only=self.config.local_files_only,
        )

        # Prepare model for k-bit training
        self._model = prepare_model_for_kbit_training(self._model)

        return self._model, self._tokenizer

    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration for the model.

        Returns:
            LoraConfig for PEFT.
        """
        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    def prepare_dataset(self, data_dir: Path) -> Dataset:
        """Prepare dataset from prepared data directory.

        Args:
            data_dir: Path to prepared data directory.

        Returns:
            HuggingFace Dataset.
        """
        # Load dataset from directory
        dataset = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")

        # Split into train/validation
        dataset = dataset.train_test_split(
            test_size=1 - self.config.train_split,
            seed=self.config.seed
        )

        return dataset["train"], dataset["test"]

    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize examples for training.

        Args:
            examples: Dictionary of examples.

        Returns:
            Tokenized examples.
        """
        return self._tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.max_seq_length,
        )

    def train(self) -> Dict[str, Any]:
        """Execute the QLoRA training pipeline.

        Returns:
            Dictionary containing training statistics and results.
        """
        if not 0 < self.config.train_split < 1:
            raise ValueError("train_split must be between zero and one")
        if self.config.num_train_epochs < 1:
            raise ValueError("Training requires at least one epoch")
        if Path(self.config.adapter_name).name != self.config.adapter_name or self.config.adapter_name in ("", ".", ".."):
            raise ValueError("Adapter name must be a single directory name")
        # Prepare data
        data_dir = self.prepare_data()

        # Load model
        model, tokenizer = self.load_model()

        # Prepare dataset
        train_dataset, eval_dataset = self.prepare_dataset(data_dir)

        # Tokenize
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        tokenized_eval = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )

        # Create LoRA config
        lora_config = self.create_lora_config()

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        self._model = model
        model.print_trainable_parameters()

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            eval_steps=self.config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        # Create trainer
        self._trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        # Train
        self._trainer.train()

        # Save adapter
        adapter_output_dir = Path(self.config.output_dir) / self.config.adapter_name
        adapter_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_output_dir))
        tokenizer.save_pretrained(str(adapter_output_dir))

        # Collect training statistics
        trainable, total = model.get_nb_trainable_parameters()
        self._training_stats = {
            **self._training_stats,
            "status": "completed",
            "base_model": self.config.base_model_id,
            "adapter_path": str(adapter_output_dir),
            "epochs_completed": self.config.num_train_epochs,
            "train_samples": len(tokenized_train),
            "eval_samples": len(tokenized_eval),
            "trainable_parameters": trainable,
            "total_parameters": total,
        }

        return self._training_stats

    def save_training_state(self, output_path: Optional[str] = None) -> str:
        """Save training state to file.

        Args:
            output_path: Optional path to save state. Defaults to output_dir/training_state.json.

        Returns:
            Path to saved state file.
        """
        if output_path is None:
            output_path = Path(self.config.output_dir) / "training_state.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": {
                "base_model_id": self.config.base_model_id,
                "adapter_name": self.config.adapter_name,
                "quantization_bit": self.config.quantization_bit,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "num_train_epochs": self.config.num_train_epochs,
                "learning_rate": self.config.learning_rate,
            },
            "stats": self._training_stats,
        }

        with open(output_path, "w") as f:
            json.dump(state, f, indent=2)

        return str(output_path)

    def load_adapter(self, adapter_path: str) -> None:
        """Load a trained adapter.

        Args:
            adapter_path: Path to the adapter directory.
        """
        if self._model is None:
            self.load_model()

        from peft import PeftModel
        self._model = PeftModel.from_pretrained(
            self._model,
            adapter_path,
            local_files_only=self.config.local_files_only,
        )

    def generate(self, prompt: str, max_length: int = 256) -> str:
        """Generate text using the trained model.

        Args:
            prompt: Input prompt.
            max_length: Maximum generation length.

        Returns:
            Generated text.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() or train() first."
            )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length
        )

        if torch is not None and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
        )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary of training statistics.
        """
        return self._training_stats.copy()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
