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
from trl import SFTTrainer, SFTConfig
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
        gemma_base.resize_token_embeddings(len(self.tokenizer))

        lora_config = LoraConfig(
            r = 16,
            lora_alpha = 32,
            target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout = 0.05,
            bias = "none",
            task_type = "CAUSAL_LM",
            modules_to_save = ["embed_tokens", "lm_head"]
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

        self.tokenizer.eos_token_id = 1
        self.gemma.config.eos_token_id = 1
        self.gemma.generation_config.eos_token_id = 1
        self.tokenizer.padding_side = "right"




    def train(self, dataset, retrain=False, resume_from_checkpoint=False):
        if self.is_trained and not retrain:
            print("Model already trained. Use retrain=False in train() to force more training or delete the output directory")
            return

        print(f"Using device {DEVICE}")
        dataset = Dataset.from_list(dataset)

        tokenized_dataset = dataset.map(
            tokenize_function,
            fn_kwargs = {"tokenizer": self.tokenizer},
            batched = True,
            remove_columns = dataset.column_names,
            load_from_cache_file = False,
            cache_file_name="./tokenized_dataset.arrow"
        )

        trainer = SFTTrainer(
            model = self.gemma,
            train_dataset = tokenized_dataset,
            args=SFTConfig(
                output_dir = "./gemma_lora_checkpoints",

                per_device_train_batch_size = 16,
                gradient_accumulation_steps = 4,
                gradient_checkpointing = True,
                bf16=True,

                learning_rate = 2e-5,
                num_train_epochs = 3,
                warmup_steps = 100,
                lr_scheduler_type="constant_with_warmup",
                optim = "adamw_torch_fused",

                logging_steps = 10,
                dataloader_num_workers = 4,
                disable_tqdm = False,

                max_length = 64,
                remove_unused_columns = False,
                packing=False,
                push_to_hub=False
            ),
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model("./gemma_lora")
        self.tokenizer.save_pretrained("./gemma_lora")
        print("Training Complete")
        self.is_trained = True


    def generate_idiom(self, definition):

        prompt = f"[DEF] {definition} [IDM]"

        inputs = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
        ).to(DEVICE)


        with torch.no_grad():
            outputs = self.gemma.generate(
                **inputs,
                min_new_tokens = 2,
                max_new_tokens = 32,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id = self.tokenizer.eos_token_id,
                pad_token_id = self.tokenizer.pad_token_id,
            )

        response = outputs[0][inputs["input_ids"].shape[-1]:]
        decoded = self.tokenizer.decode(response, skip_special_tokens=True)

        return decoded



def tokenize_function(data, tokenizer):
    text = [t + tokenizer.eos_token for t in data["text"]]

    tokenized = tokenizer(
        text,
        padding = "max_length",
        truncation = True,
        max_length = 64,
        return_token_type_ids = True,
        add_special_tokens = True
    )
    return tokenized


