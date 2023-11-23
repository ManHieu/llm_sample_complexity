import os
from typing import Dict, List, Optional
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, DisjunctiveConstraint
from trl import is_xpu_available

from arguments import ScriptArguments
from data_module.preprocess import get_preprocessor

tqdm.pandas()

if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    output_dir = os.path.join(script_args.output_dir, script_args.dataset_name)
    model_path = os.path.join(output_dir, 'final_merged_checkpoint')

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
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        offload_folder="offload",
        cache_dir='hf_cache'
    )
    tokenizer = AutoTokenizer.from_pretrained(output_dir, cache_dir='hf_cache')

    #Data processing
    preprocessor = get_preprocessor(script_args.dataset_name, 
                                    number_training_examples=script_args.number_training_examples)
    _, _, test_data = preprocessor.load_dataset()
    test_dataloader = DataLoader(test_data, batch_size=script_args.batch_size)
    if hasattr(preprocessor, 'posible_outputs'):
        posible_outputs = preprocessor.posible_outputs
        posible_outputs_ids = tokenizer(posible_outputs, add_special_tokens=False).input_ids
        constraints = [DisjunctiveConstraint(posible_outputs_ids)]

    preds = []
    labels = []
    for batch in tqdm(test_dataloader):
        prompt = batch['input']
        gold = batch['gold']

        gold_size = tokenizer(gold, return_tensors="pt", padding='longest').input_ids.size(-1)
        inputs = tokenizer(prompt, return_tensors="pt", padding='longest')
        generate_ids = model.generate(inputs.input_ids, 
                                      attention_mask=inputs.attention_mask, 
                                      max_length=gold_size,
                                      num_beams=10,
                                      do_sample=False,
                                      num_return_sequences=1,
                                      constraints=constraints,
                                      no_repeat_ngram_size=2)
        
        responses = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds.extend(responses)
        labels.extend(gold)
    
    metrics = preprocessor.compute_metric(preds=preds, labels=labels, number_training_example=script_args.number_training_examples)
    print(metrics)
    with open('result.txt', 'a', encoding='utf-8') as f:
        f.write(f'{script_args.dataset_name}: {script_args.number_training_examples} \n')
        f.write(f'Metric: \n{metrics}\n')
        f.write(f'_'*10 + '\n')
    
