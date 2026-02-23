import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling
import transformers
from peft import LoraConfig, get_peft_model, PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from trl import SFTTrainer
from datasets import Dataset


MODEL_NAME = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "./gemma_lora"

fsdp_config = {
    "fsdp_transformer_layer_cls_to_wrap": ["GemmaDecoderLayer"],
    "xla": True,
    "xla_fsdp_v2": True,
    "xla_fsdp_grad_ckpt": True
}


class IdiomaticExpressionModel:
    def __init__(self, embed_dimensions=384, prefix_length=10):
        self.embed_dimensions = embed_dimensions
        self.prefix_length = prefix_length

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[DEF]", "[IDM]"]})


        gemma_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

        lora_config = LoraConfig(
            r = 8,
            lora_alpha = 16,
            target_modules = ["q_proj", "v_proj"],
            lora_dropout = 0.05,
            bias = "none",
            task_type = "CAUSAL_LM"
        )

        if not os.path.exists(SAVE_PATH):
            self.is_trained = False
            print("No saved model found, please call train()")
            gemma = get_peft_model(gemma_base, lora_config)
            self.gemma = gemma
        else:
            self.is_trained = True
            gemma = PeftModel.from_pretrained(gemma_base, "./gemma_lora").to(DEVICE)
            gemma.eval()
            self.gemma = gemma

    def train(self, dataset, batch_size = 16, LR = 2e-5, epochs = 3, retrain=False):
        if self.is_trained and not retrain:
            print("Model already trained. Use retrain=False in train() to force more training or delete the output directory")
            return

        print(f"Using device {DEVICE}")
        dataset = Dataset.from_list(dataset)

        def tokenize_function(data):
            return self.tokenizer(
                data["text"],
                padding = "max_length",
                truncation = True,
                max_length = 64,
                return_token_type_ids = True
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched = True,
            remove_columns = dataset.column_names,
            load_from_cache_file = False
        )

        trainer = SFTTrainer(
            model = self.gemma,
            train_dataset = tokenized_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=batch_size,
                num_train_epochs = epochs,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                max_steps=10,
                learning_rate=LR,
                logging_steps = 1,
                output_dir = "./gemma_lora_checkpoints",
                optim="adamw_torch_fused",
                dataloader_drop_last=True,
                disable_tqdm = False,
                remove_unused_columns = False
                # fsdp = "full_shard",
                # fsdp_config=fsdp_config
                # I'll work on implementing FSDP later
            ),
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )


        trainer.train()
        trainer.save_model("./gemma_lora")
        self.tokenizer.save_pretrained("./gemma_lora")
        print("Training Complete")
        self.is_trained = True


    def generate_idiom(self, instruction):
        message = [
            {"role": "system", "content": "Your job is to generate novel idiomatic expressions. No conversational filler or special formatting."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "A idiom that fits that definition would be:"}
        ]

        inputs = self.tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_dict = True
        ).to(DEVICE)

        # not sure why these are missing, but this does fix the issue
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

        with torch.no_grad():
            outputs = self.gemma.generate(
                **inputs,
                max_new_tokens = 64,
                do_sample=True,
                temperature=1.1,
                top_p=0.97,
                use_cache=True
            )

        response = outputs[0][inputs["input_ids"].shape[-1]:]
        decoded = self.tokenizer.decode(response, skip_special_tokens=True)

        return decoded