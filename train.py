# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
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
import os
from typing import Dict, List, Optional
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
from trl import is_xpu_available, DataCollatorForCompletionOnlyLM, SFTTrainer

from arguments import ScriptArguments
from data_module.preprocess import get_preprocessor
from trainer import LLMComplexityTrainer


tqdm.pandas()

if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    output_dir = os.path.join(script_args.output_dir, script_args.dataset_name)

    # Data processing
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, 
                                              cache_dir='hf_cache')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    preprocessor = get_preprocessor(script_args.dataset_name, 
                                    number_training_examples=script_args.number_training_examples)
    train_data, val_data, test_data = preprocessor.create_datasets(tokenizer=tokenizer, 
                                                                   script_args=script_args)
    print(train_data)

    # Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, 
            load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        offload_folder="offload",
        cache_dir='hf_cache'
    )

    # Define the PEFT model
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            lora_dropout=script_args.peft_lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model=model, peft_config=peft_config)
    else:
        peft_config = None

    model.print_trainable_parameters()

    # Define metric
    def compute_metrics(val_preds):
        generate_ids, label_ids = val_preds
        responses = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        metrics = preprocessor.compute_metric(preds=responses, labels=labels, number_training_example=script_args.number_training_examples)
        print(metrics)
        with open('result.txt', 'a', encoding='utf-8') as f:
            f.write(f'{script_args.dataset_name}: {script_args.number_training_examples} \n')
            f.write(f'Metric: \n{metrics}\n')
            f.write(f'_'*10 + '\n')
        return metrics

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        group_by_length=False,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.05,
        optim="paged_adamw_32bit",
        bf16=True,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        warmup_steps=15,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        remove_unused_columns=False,
        run_name=f'sft_{script_args.dataset_name}',
        gradient_checkpointing=script_args.gradient_checkpointing,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        eval_steps=10,
        metric_for_best_model='acc',)

    # Define the Trainer
    # instruct_template = "[INST] <<SYS>>"
    # response_template = "\n[/INST]###Response: "
    # collator = DataCollatorForCompletionOnlyLM(instruction_template=instruct_template, 
    #                                            response_template=response_template, 
    #                                            tokenizer=tokenizer, 
    #                                            mlm=False)
    
    trainer = LLMComplexityTrainer(
        model=model,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field='input',
        # data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save the model
    trainer.save_model(training_args.output_dir)
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    # Free memory for merging weights
    del model
    if is_xpu_available():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, 
                                                     device_map="auto", 
                                                     torch_dtype=torch.bfloat16,
                                                     cache_dir='hf_cache')
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)



