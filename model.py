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

    def train(self, dataset, batch_size = 16, LR = 2e-5, epochs = 3):
        print(f"Using device {DEVICE}")
        dataset = Dataset.from_list(dataset)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # contrastive_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)

        gemma_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

        lora_config = LoraConfig(
            r = 8,
            lora_alpha = 16,
            target_modules = ["q_proj", "v_proj"],
            lora_dropout = 0.05,
            bias = "none",
            task_type = "CAUSAL_LM"
        )
        gemma = get_peft_model(gemma_base, lora_config)

        trainer = SFTTrainer(
            model = gemma,
            train_dataset = dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=batch_size,
                num_train_epochs = epochs,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                max_steps=10,
                learning_rate=LR,
                logging_steps = 1,
                output_dir = "./gemma_lora",
                optim="paged_adamw_8bit",
                dataloader_drop_last=True,
                disable_tqdm = False,
                # fsdp = "full_shard",
                # fsdp_config=fsdp_config
            ),
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )


        trainer.train()


    def generate_idiom(self, instruction):
        # load gemma
        gemma_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        gemma = PeftModel.from_pretrained(gemma_base, "./gemma_lora").to(DEVICE)
        gemma.eval()


        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        generated = gemma.generate(**instruction)
        decoded = tokenizer.decode(generated[0]["generated_text"][-1]["content"])


        return decoded