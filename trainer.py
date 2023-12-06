from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
from datasets import Dataset
from peft import PeftConfig
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments, Constraint
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_pt_utils import nested_detach
from trl import SFTTrainer
from dataclasses import dataclass
import datasets


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


class LLMComplexityTrainer(SFTTrainer):
    def __init__(self, 
                model: Union[PreTrainedModel, nn.Module, str] = None,
                args: TrainingArguments = None,
                data_collator: Optional[DataCollator] = None,
                train_dataset: Optional[Dataset] = None,
                eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                tokenizer: Optional[PreTrainedTokenizerBase] = None,
                model_init: Optional[Callable[[], PreTrainedModel]] = None,
                compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                callbacks: Optional[List[TrainerCallback]] = None,
                optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                peft_config: Optional["PeftConfig"] = None,
                dataset_text_field: Optional[str] = None,
                packing: Optional[bool] = False,
                formatting_func: Optional[Callable] = None,
                max_seq_length: Optional[int] = None,
                infinite: Optional[bool] = False,
                num_of_sequences: Optional[int] = 1024,
                chars_per_token: Optional[float] = 3.6,
                dataset_num_proc: Optional[int] = None,
                dataset_batch_size: int = 1000,
                neftune_noise_alpha: Optional[float] = None,
                generation_constraint: Optional[Constraint]=None,
                model_init_kwargs: Optional[Dict] = None,):
        
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics, peft_config, dataset_text_field, packing, formatting_func, max_seq_length, infinite, num_of_sequences, chars_per_token, dataset_num_proc, dataset_batch_size, neftune_noise_alpha, model_init_kwargs)
        
        if max_seq_length is None:
            # to overcome some issues with broken tokenizers
            self.max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {max_seq_length}"
            )
        else:
            self.max_seq_length = max_seq_length
        
        self.generation_constraint = generation_constraint

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = LLMComplexityDataCollator(self.tokenizer, 
                                                 self.max_seq_length)
        
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        data_collator = LLMComplexityDataCollator(self.tokenizer, 
                                                 self.max_seq_length)
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(test_dataset, IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))
    
    def prediction_step(self, 
                        model: nn.Module, 
                        inputs: Dict[str, Tensor | Any], 
                        prediction_loss_only: bool, 
                        ignore_keys: List[str] | None = None) -> Tuple[Tensor | None, Tensor | None, Tensor | None]:
        
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
        gold_size = labels.size(-1) if labels!=None else self.max_seq_length

        with torch.no_grad():
            loss = None
            with self.compute_loss_context_manager():
                if isinstance(model, nn.DataParallel):
                    model = model.module
                generate_ids = model.generate(input_ids=inputs['input_ids'], 
                                            attention_mask=inputs.attention_mask, 
                                            max_length=gold_size,
                                            num_beams=10,
                                            do_sample=False,
                                            num_return_sequences=1,
                                            constraints=self.generation_constraint,
                                            no_repeat_ngram_size=2) 

        return (loss, generate_ids, labels)
