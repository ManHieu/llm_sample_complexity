from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
import tqdm
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM 
from transformers import PreTrainedTokenizerBase

from .base_dataset import BaseDataset

PREPROCESSOR = {}


@dataclass
class LLMComplexityDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(batch,
                                padding='longest',
                                max_length=self.max_length,
                                return_tensors=self.return_tensors,)
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


class BasePreprocessor(ABC):

    def __init__(self, 
                 number_training_examples: int,) -> None:
        super().__init__()
        self.number_training_examples = number_training_examples

    @abstractmethod
    def formating_prompts_func(self, examples):
        raise NotImplementedError
    
    @abstractmethod
    def load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        raise NotImplementedError
    
    @abstractmethod
    def compute_metric(self, 
                       number_training_example, 
                       labels, 
                       preds,
                       save_wrong_preds=True):
        raise NotImplementedError
    
    def prepare_data(self, 
                        tokenizer, 
                        script_args):
        train_data, val_data, test_data = self.load_dataset()

        if script_args.colate_fn == 'constant_len':
            chars_per_token = self.chars_token_ratio(train_data, tokenizer)
            print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
            train_dataset = ConstantLengthDataset(tokenizer,
                                                train_data,
                                                dataset_text_field='input',
                                                seq_length=script_args.seq_length,
                                                infinite=True,
                                                chars_per_token=chars_per_token,
                                                eos_token_id=tokenizer.eos_token_id,)
        elif script_args.colate_fn == 'completion':
            # instruct_template = "[INST] <<SYS>>"
            response_template = "\n###Response:"
            collator = DataCollatorForCompletionOnlyLM(response_template=response_template, 
                                                    tokenizer=tokenizer, 
                                                    mlm=False)
            train_dataset = BaseDataset(train_data)
            train_dataloader = DataLoader(train_dataset, 
                                          batch_size=script_args.batch_size,
                                          collate_fn=collator,
                                          num_workers=8,
                                          shuffle=True,
                                          pin_memory=False,
                                          drop_last=True)
            
        val_collator = LLMComplexityDataCollator(self.tokenizer, 
                                                 self.max_seq_length)
        val_data = val_data.map(lambda example: self.tokenize_func_for_evaluation(examples=example, tokenizer=tokenizer, max_seq_len=script_args.seq_length),
                                batched=True, remove_columns=val_data.column_names)
        val_data.set_format(type="torch", columns=val_data.column_names)
        val_dataset = BaseDataset(val_data)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=1,
                                    collate_fn=val_collator,
                                    num_workers=8,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=True)

        test_data = test_data.map(lambda example: self.tokenize_func_for_evaluation(examples=example, tokenizer=tokenizer, max_seq_len=script_args.seq_length),
                                batched=True, remove_columns=test_data.column_names)
        test_data.set_format(type="torch", columns=test_data.column_names)
        test_dataset = BaseDataset(test_data)
        test_dataloader = DataLoader(test_dataset,
                                    batch_size=1,
                                    collate_fn=val_collator,
                                    num_workers=8,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=True)

        return train_dataloader, val_dataloader, test_dataloader
    
    @staticmethod
    def tokenize_func_for_evaluation(examples, tokenizer, max_seq_len):
        tokenized_input = tokenizer(examples['input'],
                                    padding='longest',
                                    max_length=max_seq_len,
                                    return_overflowing_tokens=False,
                                    return_length=False,)
        tokenized_output = tokenizer(examples['gold'],
                                    padding='longest',
                                    max_length=max_seq_len,
                                    return_overflowing_tokens=False,
                                    return_length=False,)
        return {'input_ids': tokenized_input['input_ids'],
                'attention_mask': tokenized_input['attention_mask'],
                'labels': tokenized_output['input_ids'],}
    
    @staticmethod
    def chars_token_ratio(dataset, tokenizer, nb_examples=400):
        """
        Estimate the average number of characters per token in the dataset.
        """
        total_characters, total_tokens = 0, 0
        for _, example in zip(range(nb_examples), iter(dataset)):
            text = example['input']
            total_characters += len(text)
            if tokenizer.is_fast:
                total_tokens += len(tokenizer(text).tokens())
            else:
                total_tokens += len(tokenizer.tokenize(text))

        return total_characters / total_tokens
    

def register_preprocess(preprocessor_class: BasePreprocessor):
    PREPROCESSOR[preprocessor_class.name] = preprocessor_class
    return preprocessor_class


def get_preprocessor(name: str,
                     number_training_examples: int
                     ) -> BasePreprocessor:
    return PREPROCESSOR[name](number_training_examples)
        

@register_preprocess
class SSTPreprocessor(BasePreprocessor):
    name = 'sst2'
    label_mapping = {0: 'The text is negative',
                    1: 'The text is positive'}
    label_to_id = {'The text is negative': 0,
                    'The text is positive': 1}
    instruction = "You should judge the overall emotional feeling of the text, not just focus on certain words. Decide if the following text expresses a positive or negative sentiment."
    posible_outputs = ['The text is negative', 'The text is positive']
    
    def formating_prompts_func(self, examples):
        output_text = []
        for i in range(len(examples['text'])):
            input_text = examples['text'][i]
            response = self.label_mapping[examples['label'][i]]
            text = f"[INST] <<SYS>> {self.instruction} <</SYS>> ###Text: {input_text} \n[/INST]\n###Response: {response}"
            output_text.append(text)
        return {'input': output_text}
    
    def formating_prompts_func_for_evluation(self, examples):
        output_text = []
        gold_response = []
        label = []
        for i in range(len(examples['text'])):
            input_text = examples['text'][i]
            response = self.label_mapping[examples['label'][i]]
            text = f"[INST] <<SYS>> {self.instruction} <</SYS>> ###Text: {input_text} \n[/INST]\n###Response:"
            full_response = f"[INST] <<SYS>> {self.instruction} <</SYS>> ###Text: {input_text} \n[/INST]\n###Response: {response}"
            output_text.append(text)
            gold_response.append(full_response)
            label.append(response)
        return {'input': output_text, 'gold': gold_response, 'label': label}
    
    def load_dataset(self):
        data = load_dataset('SetFit/sst2', cache_dir='./hf_cache')
        train_data = data['train']
        train_data = train_data.train_test_split(train_size=self.number_training_examples, seed=1741)['train']
        val_data = data['validation'].train_test_split(train_size=100, seed=1741)['train']
        test_data = data['test']

        train_data = train_data.map(self.formating_prompts_func, batched=True, load_from_cache_file=False)
        val_data = val_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False)
        test_data = test_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False)

        return train_data, val_data, test_data
    
    def compute_metric(self, 
                       number_training_example, 
                       labels, 
                       preds,
                       save_wrong_preds=True):
        labels = [item.split('\n[/INST]\n###Response:')[1].strip() for item in labels]
        preds = [item.split('\n[/INST]\n###Response:')[1].strip() for item in preds]

        labels_ids = []
        preds_ids = []
        wrong_results = {'label': [],
                         'pred': []}
        for label, pred in zip(labels, preds):
            labels_ids.append(self.label_to_id[label])
            _pred = pred[0:len(label)]
            pred_idx = self.label_to_id.get(_pred, -1)
            preds_ids.append(pred_idx)
            if _pred != label:
                wrong_results['label'].append(label)
                wrong_results['pred'].append(pred)
        if save_wrong_preds:
            dataset = Dataset.from_dict(wrong_results)
            dataset.to_json(f'Wrong_result_{self.name}_{number_training_example}.jsonl')

        labels = torch.tensor(labels_ids)
        preds = torch.tensor(preds_ids)
        return {'acc': torch.sum(labels==preds).float() / labels.numel()}
    

class ANLIPreprocessor(BasePreprocessor):
    label_mapping = {0: 'Entailment', 
                     2: 'Contradiction', 
                     1: 'Neutral'}
    label_to_id = {'Entailment': 0, 
                   'Contradiction': 2, 
                   'Neutral':1}
    instruction = "Natural Language Inference is a task which determines the logical relationship between two sentences. You will be shown two sentences - a premise and a hypothesis. You must determine whether the hypothesis is true (entailment), false (contradiction), or undetermined (neutral) based on the premise. You should focus on reasoning about the semantics and logic of the sentences to determine the relationship between them, avoid using superficial cues or correlations in making your judgment."
    posible_outputs = ['###Response: Entailment', '###Response: Contradiction', '###Response: Neutral']
    
    def formating_prompts_func(self, examples):
        output_text = []
        for i in range(len(examples['premise'])):
            premise = examples['premise'][i]
            hypothesis = examples['hypothesis'][i]
            response = self.label_mapping[examples['label'][i]]
            text = f"[INST] <<SYS>> {self.instruction} <</SYS>> ###Premise: {premise}\n###Hypothesis: {hypothesis} \n[/INST]\n###Response: {response}"
            output_text.append(text)
        return {'input': output_text}
    
    def formating_prompts_func_for_evluation(self, examples):
        output_text = []
        gold_response = []
        for i in range(len(examples['premise'])):
            premise = examples['premise'][i]
            hypothesis = examples['hypothesis'][i]
            response = self.label_mapping[examples['label'][i]]
            text = f"[INST] <<SYS>> {self.instruction} <</SYS>> ###Premise: {premise}\n###Hypothesis: {hypothesis} \n[/INST]\n###Response:"
            full_response = f"[INST] <<SYS>> {self.instruction} <</SYS>> ###Premise: {premise}\n###Hypothesis: {hypothesis} \n[/INST]\n###Response: {response}"
            output_text.append(text)
            gold_response.append(full_response)
        return {'input': output_text, 'gold': gold_response}
    
    def compute_metric(self, 
                       number_training_example, 
                       labels, 
                       preds,
                       save_wrong_preds=True):
        labels = [item.split('\n[/INST]\n###Response:')[1].strip() for item in labels]
        preds = [item.split('\n[/INST]\n###Response:')[1].strip() for item in preds]

        labels_ids = []
        preds_ids = []
        wrong_results = {'label': [],
                         'pred': []}
        for label, pred in zip(labels, preds):
            labels_ids.append(self.label_to_id[label])
            _pred = pred[0:len(label)]
            pred_idx = self.label_to_id.get(_pred, -1)
            preds_ids.append(pred_idx)
            if _pred != label:
                wrong_results['label'].append(label)
                wrong_results['pred'].append(pred)
        if save_wrong_preds:
            dataset = Dataset.from_dict(wrong_results)
            dataset.to_json(f'Wrong_result_{self.name}_{number_training_example}.jsonl')

        labels = torch.tensor(labels_ids)
        preds = torch.tensor(preds_ids)
        return {'acc': torch.sum(labels==preds).float() / labels.numel()}


@register_preprocess
class ANLIR1Preprocessor(ANLIPreprocessor):
    name = 'anli_r1'

    def load_dataset(self):
        data = load_dataset('anli', cache_dir='./hf_cache')
        train_data = data['train_r1']
        train_data = train_data.train_test_split(train_size=self.number_training_examples, seed=1741)['train']
        val_data = data['dev_r1'].train_test_split(train_size=100, seed=1741)['train']
        test_data = data['test_r1']

        train_data = train_data.map(self.formating_prompts_func, batched=True, load_from_cache_file=False, remove_columns=train_data.column_names)
        val_data = val_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False, remove_columns=val_data.column_names)
        test_data = test_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False, remove_columns=test_data.column_names)

        return train_data, val_data, test_data
    

@register_preprocess
class ANLIR2Preprocessor(ANLIPreprocessor):
    name = 'anli_r2'

    def load_dataset(self):
        data = load_dataset('anli', cache_dir='./hf_cache')
        train_data = data['train_r2']
        train_data = train_data.train_test_split(train_size=self.number_training_examples, seed=1741)['train']
        val_data = data['dev_r2'].train_test_split(train_size=100, seed=1741)['train']
        test_data = data['test_r2']

        train_data = train_data.map(self.formating_prompts_func, batched=True, load_from_cache_file=False)
        val_data = val_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False)
        test_data = test_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False)

        return train_data, val_data, test_data
    

@register_preprocess
class ANLIR2Preprocessor(ANLIPreprocessor):
    name = 'anli_r3'

    def load_dataset(self):
        data = load_dataset('anli', cache_dir='./hf_cache')
        train_data = data['train_r3']
        train_data = train_data.train_test_split(train_size=self.number_training_examples, seed=1741)['train']
        val_data = data['dev_r3'].train_test_split(train_size=100, seed=1741)['train']
        test_data = data['test_r3']

        train_data = train_data.map(self.formating_prompts_func, batched=True, load_from_cache_file=False)
        val_data = val_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False)
        test_data = test_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False)

        return train_data, val_data, test_data
    

if __name__=='__main__':
    from transformers import HfArgumentParser, AutoTokenizer
    from arguments import ScriptArguments

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    breakpoint()

    preprocessor = get_preprocessor('anli_r1', number_training_examples=50)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, 
                                              cache_dir='hf_cache')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_dataloader, val_dataloader, test_dataloader = preprocessor.prepare_data(tokenizer=tokenizer,
                                                                                  script_args=script_args)
    breakpoint()

