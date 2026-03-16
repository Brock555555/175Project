import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import upload_folder

from trl import SFTTrainer, SFTConfig
from datasets import Dataset


MODEL_NAME = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "gemma_lora"


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
            gemma = PeftModel.from_pretrained(gemma_base, SAVE_PATH).to(DEVICE)
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
            keep_in_memory = True
        )

        trainer = SFTTrainer(
            model = self.gemma,
            train_dataset = tokenized_dataset,
            args=SFTConfig(
                output_dir = "../gemma_lora_checkpoints",

                per_device_train_batch_size = 16,
                gradient_accumulation_steps = 4,
                gradient_checkpointing = True,
                bf16=True,

                learning_rate = 1e-5,
                num_train_epochs = 10,
                warmup_steps = 50,
                lr_scheduler_type="cosine",
                optim = "adamw_torch_fused",

                logging_steps = 10,
                dataloader_num_workers = 0,
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
        few_shot = (
            "[DEF] to waste potential [IDM] let the candle burn idle\n"
            "[DEF] to be ignored [IDM] converse with the wallflowers\n"
            "[DEF] to use an excessively strong tool [IDM] mow grass with a chainsaw\n"
            "[DEF] to overthink a problem [IDM] to find ghosts amidst shadows\n"
            "[DEF] to drive a conversation into uncomfortable subjects [IDM] to make a minefield of a molehill\n"
        )
        instruction = "Generate a novel, creative idiom for the given prompts, avoid common expressions\n\n"

        prompt = instruction + few_shot + f"[DEF] {definition} [IDM]"

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
                max_new_tokens = 20,
                do_sample=True,
                temperature=1.5,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.8,
                no_repeat_ngram_size = 3,
                eos_token_id = self.tokenizer.eos_token_id,
                pad_token_id = self.tokenizer.pad_token_id,
            )

        response = outputs[0][inputs["input_ids"].shape[-1]:]
        decoded = self.tokenizer.decode(response, skip_special_tokens=True)

        return decoded



    def upload(self):
        merged = self.gemma.merge_and_unload()

        merged.save_pretrained("merged_model")
        self.tokenizer.save_pretrained("merged_model")

        upload_folder(
            repo_id = "Notme2222/175ProjectIdiomaticExpressionGenerator",
            folder_path = "merged_model"
        )


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


