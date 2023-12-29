from argparse import Namespace
from typing import Any, Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup, Constraint
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

from data_module.preprocess import get_preprocessor


class LLMTrainer(pl.LightningModule):
    def __init__(self,
                 params: Namespace,
                 generation_constraint: Constraint=None,) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.params = params
        self.generation_constraint = generation_constraint

        self.tokenizer = AutoTokenizer.from_pretrained(self.params.model_name, 
                                              cache_dir='hf_cache')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

        self.preprocessor = get_preprocessor(params.dataset_name, 
                                            number_training_examples=params.number_training_examples)
        self.val_metric = self.preprocessor.metric()
        self.test_metric = self.preprocessor.metric()
        
        # Load the model
        if self.params.load_in_4bit:
            quantization_config = BitsAndBytesConfig( load_in_4bit=self.params.load_in_4bit,
                                                    bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_compute_dtype=torch.float16)
        else:
            quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(self.params.model_name,
                                                    quantization_config=quantization_config,
                                                    trust_remote_code=True,
                                                    offload_folder="offload",
                                                    cache_dir='hf_cache')
        # Define the PEFT model
        if self.params.use_peft:
            peft_config = LoraConfig(r=self.params.peft_lora_r,
                                    lora_alpha=self.params.peft_lora_alpha,
                                    lora_dropout=self.params.peft_lora_dropout,
                                    target_modules=["q_proj", "v_proj"],
                                    bias="none",
                                    task_type="CAUSAL_LM",)
            self.model = get_peft_model(model=model, peft_config=peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = model
        self.lr = self.params.learning_rate

    def configure_optimizers(self) -> OptimizerLRScheduler:
        num_batches = self.trainer.estimated_stepping_batches
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = bnb.optim.PagedAdamW(trainable_params, 
                                        lr=self.lr, 
                                        weight_decay=0.05,)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_training_steps=num_batches, 
                                                    num_warmup_steps=int(num_batches*0.05) if num_batches*0.05 < 50 else 50)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }}
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        if len(batch) == 3:
            input_ids, attn_mask, labels = batch.values()
        else:
            input_ids, labels = batch.values()
            attn_mask = None
        output = self.model(input_ids=input_ids,
                            attention_mask=attn_mask,
                            labels=labels,
                            return_dict=True)
        loss = output.loss

        self.log('loss', loss, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input_ids, attn_mask, labels = batch.values()
        gold_size = labels.size(-1) + 5 if labels!=None else 512
        generate_ids = self.model.generate(input_ids=input_ids, 
                                            attention_mask=attn_mask, 
                                            max_length=gold_size,
                                            num_beams=10,
                                            do_sample=False,
                                            num_return_sequences=1,
                                            constraints=self.generation_constraint,
                                            no_repeat_ngram_size=2) 
        responses = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        labels, preds = self.preprocessor.process_outputs(labels, responses, save_wrong_preds=False)

        self.val_metric.update(preds, labels)

    def on_validation_epoch_end(self):
        self.log('valid_metric', self.val_metric.compute(), prog_bar=True)
        self.val_metric.reset()

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input_ids, attn_mask, labels = batch.values()
        gold_size = labels.size(-1) + 5 if labels!=None else 512
        generate_ids = self.model.generate(input_ids=input_ids, 
                                            attention_mask=attn_mask, 
                                            max_length=gold_size,
                                            num_beams=10,
                                            do_sample=False,
                                            num_return_sequences=1,
                                            constraints=self.generation_constraint,
                                            no_repeat_ngram_size=2) 
        responses = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        labels, preds = self.preprocessor.process_outputs(labels, responses, save_wrong_preds=False)

        self.test_metric.update(preds, labels)
    
    def on_test_epoch_end(self) -> None:
        self.log('test_metric', self.test_metric.compute(), prog_bar=True)
        self.test_metric.reset() 
        
