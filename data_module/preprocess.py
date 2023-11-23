from abc import ABC, abstractmethod
from typing import Tuple
from datasets import load_dataset, Dataset
import torch
import tqdm
from trl.trainer import ConstantLengthDataset

from data_module.base_dataset import BaseDataset

PREPROCESSOR = {}


class BasePreprocessor(ABC):

    def __init__(self, number_training_examples: int) -> None:
        super().__init__()
        self.number_training_examples = number_training_examples

    @abstractmethod
    def formating_prompts_func(self, examples):
        raise NotImplementedError
    
    @abstractmethod
    def load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        raise NotImplementedError
    
    @abstractmethod
    def compute_metric(self, labels, preds):
        raise NotImplementedError
    
    def create_datasets(self, tokenizer, script_args):
        train_data, val_data, test_data = self.load_dataset()
        chars_per_token = self.chars_token_ratio(train_data, tokenizer)
        print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

        train_dataset = ConstantLengthDataset(tokenizer,
                                            train_data,
                                            dataset_text_field='input',
                                            seq_length=script_args.seq_length,
                                            infinite=True,
                                            chars_per_token=chars_per_token,
                                            eos_token_id=tokenizer.eos_token_id,)
        
        val_data = val_data.map(lambda example: self.tokenize_func_for_evaluation(examples=example, tokenizer=tokenizer, max_seq_len=script_args.seq_length),
                                batched=True, remove_columns=val_data.column_names)
        val_data.set_format(type="torch", columns=val_data.column_names)
        val_dataset = BaseDataset(val_data)

        test_data = test_data.map(lambda example: self.tokenize_func_for_evaluation(examples=example, tokenizer=tokenizer, max_seq_len=script_args.seq_length),
                                batched=True, remove_columns=test_data.column_names)
        test_data.set_format(type="torch", columns=test_data.column_names)
        test_dataset = BaseDataset(test_data)

        return train_dataset, val_dataset, test_dataset
    
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
            text = f"[INST] <<SYS>> {self.instruction} <</SYS>> ###Text: {input_text} \n[/INST]###Response: {response}"
            output_text.append(text)
        return {'input': output_text}
    
    def formating_prompts_func_for_evluation(self, examples):
        output_text = []
        gold_response = []
        label = []
        for i in range(len(examples['text'])):
            input_text = examples['text'][i]
            response = self.label_mapping[examples['label'][i]]
            text = f"[INST] <<SYS>> {self.instruction} <</SYS>> ###Text: {input_text} \n[/INST]###Response:"
            full_response = f"[INST] <<SYS>> {self.instruction} <</SYS>> ###Text: {input_text} \n[/INST]###Response: {response}"
            output_text.append(text)
            gold_response.append(full_response)
            label.append(response)
        return {'input': output_text, 'gold': gold_response, 'label': label}
    
    def load_dataset(self):
        data = load_dataset('SetFit/sst2', cache_dir='./hf_cache')
        train_data = data['train']
        train_data = train_data.train_test_split(train_size=self.number_training_examples, seed=1741)['train']
        val_data = data['validation']
        test_data = data['test']

        train_data = train_data.map(self.formating_prompts_func, batched=True, load_from_cache_file=False)
        val_data = val_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False)
        test_data = test_data.map(self.formating_prompts_func_for_evluation, batched=True, load_from_cache_file=False)

        return train_data, val_data, test_data
    
    def compute_metric(self, number_training_example, labels, preds):
        labels = [item.split('\n[/INST]###Response:')[1].strip() for item in labels]
        preds = [item.split('\n[/INST]###Response:')[1].strip() for item in preds]

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
        
        dataset = Dataset.from_dict(wrong_results)
        dataset.to_json(f'Wrong_result_{self.name}_{number_training_example}.jsonl')

        labels = torch.tensor(labels_ids)
        preds = torch.tensor(preds_ids)
        return {'acc': torch.sum(labels==preds).float() / labels.numel()}
    
