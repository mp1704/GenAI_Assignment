import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import psutil
import GPUtil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FineTuningConfig:
    """Configuration class for fine-tuning parameters"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.model_name = "microsoft/DialoGPT-medium"
        self.dataset_name = "tatsu-lab/alpaca"
        self.output_dir = "./results"
        self.max_seq_length = 512
        self.num_train_epochs = 3
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.logging_steps = 10
        self.save_steps = 500
        self.eval_steps = 500
        self.save_total_limit = 3
        self.fp16 = True
        self.gradient_checkpointing = True
        
        # LoRA configuration
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Quantization
        self.use_4bit = True
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = True
        
        # Monitoring
        self.use_wandb = True
        self.wandb_project = "llm-fine-tuning"
        self.wandb_run_name = None
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


class ResourceMonitor:
    """Monitor system resources during training"""
    
    @staticmethod
    def get_gpu_memory_usage():
        """Get GPU memory usage information"""
        if not torch.cuda.is_available():
            return {}
        
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            gpu = GPUtil.getGPUs()[i]
            gpu_info[f"gpu_{i}"] = {
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                "temperature": gpu.temperature,
                "utilization": gpu.load * 100
            }
        return gpu_info
    
    @staticmethod
    def get_system_info():
        """Get system resource information"""
        memory = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
    
    @staticmethod
    def log_resource_usage():
        """Log current resource usage"""
        gpu_info = ResourceMonitor.get_gpu_memory_usage()
        system_info = ResourceMonitor.get_system_info()
        
        logger.info(f"System Resources: {system_info}")
        if gpu_info:
            logger.info(f"GPU Resources: {gpu_info}")


class DataProcessor:
    """Handle data loading and preprocessing"""
    
    def __init__(self, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def load_dataset(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace hub or local files"""
        try:
            if os.path.exists(dataset_name):
                # Load local dataset
                if dataset_name.endswith('.json'):
                    import pandas as pd
                    df = pd.read_json(dataset_name)
                    dataset = Dataset.from_pandas(df)
                else:
                    dataset = load_dataset(dataset_name, split=split)
            else:
                # Load from HuggingFace hub
                dataset = load_dataset(dataset_name, split=split)
            
            logger.info(f"Loaded dataset with {len(dataset)} examples")
            return dataset
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_function(self, examples):
        """Preprocess examples for training"""
        # Customize this based on your data format
        if "instruction" in examples and "output" in examples:
            # Alpaca-style format
            inputs = [
                f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                for inst, out in zip(examples["instruction"], examples["output"])
            ]
        elif "text" in examples:
            # Simple text format
            inputs = examples["text"]
        else:
            raise ValueError("Unsupported data format")
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training"""
        # Remove unnecessary columns
        column_names = dataset.column_names
        dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=column_names,
            desc="Preprocessing dataset"
        )
        
        return dataset


class FineTuningPipeline:
    """Main fine-tuning pipeline"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.data_processor = None
        
        # Setup output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize monitoring
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config.__dict__
            )
    
    def setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration"""
        if not self.config.use_4bit:
            return None
        
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )
        
        return bnb_config
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Setup quantization
        bnb_config = self.setup_quantization_config()
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Setup LoRA configuration and apply to model"""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def prepare_data(self):
        """Load and prepare training data"""
        logger.info("Preparing training data")
        
        self.data_processor = DataProcessor(
            self.tokenizer, 
            self.config.max_seq_length
        )
        
        # Load datasets
        train_dataset = self.data_processor.load_dataset(
            self.config.dataset_name, "train"
        )
        
        # Try to load validation split
        try:
            eval_dataset = self.data_processor.load_dataset(
                self.config.dataset_name, "validation"
            )
        except:
            # Split training data if no validation set
            logger.info("No validation split found, splitting training data")
            split_dataset = train_dataset.train_test_split(test_size=0.1)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        
        # Preprocess datasets
        self.train_dataset = self.data_processor.prepare_dataset(train_dataset)
        self.eval_dataset = self.data_processor.prepare_dataset(eval_dataset)
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Evaluation samples: {len(self.eval_dataset)}")
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb" if self.config.use_wandb else None,
        )
    
    def train(self):
        """Execute training"""
        logger.info("Starting training")
        ResourceMonitor.log_resource_usage()
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        # Start training
        try:
            self.trainer.train()
            logger.info("Training completed successfully")
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            ResourceMonitor.log_resource_usage()
    
    def save_model(self):
        """Save the fine-tuned model"""
        logger.info("Saving model")
        
        # Save the model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save configuration
        self.config.save_config(f"{self.config.output_dir}/config.json")
        
        logger.info(f"Model saved to {self.config.output_dir}")
    
    def evaluate(self):
        """Evaluate the model"""
        logger.info("Evaluating model")
        
        # Run evaluation
        eval_results = self.trainer.evaluate()
        
        # Log results
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save evaluation results
        with open(f"{self.config.output_dir}/eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results
    
    def run_pipeline(self):
        """Run the complete fine-tuning pipeline"""
        try:
            self.load_model_and_tokenizer()
            self.setup_lora()
            self.prepare_data()
            self.train()
            self.evaluate()
            self.save_model()
            
            logger.info("Fine-tuning pipeline completed successfully")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            if self.config.use_wandb:
                wandb.finish()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="microsoft/DialoGPT-medium",
        help="Model name or path"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="tatsu-lab/alpaca",
        help="Dataset name or path"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = FineTuningConfig(args.config)
    
    # Override with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.dataset_name:
        config.dataset_name = args.dataset_name
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Initialize and run pipeline
    pipeline = FineTuningPipeline(config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 