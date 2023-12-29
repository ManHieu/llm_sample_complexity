import gc
import os
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, DisjunctiveConstraint, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from arguments import ScriptArguments
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data_module.preprocess import get_preprocessor
from hyperparams_tuning_args import create_tuning_args
from models.lightning_trainer import LLMTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def train(params):
    pl.seed_everything(1741)

    preprocessor = get_preprocessor(params.dataset_name, 
                                    number_training_examples=params.number_training_examples)

    tokenizer = AutoTokenizer.from_pretrained(params.model_name, 
                                              cache_dir='hf_cache')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_dataloader, val_dataloader, test_dataloader = preprocessor.prepare_data(tokenizer=tokenizer,
                                                                                  script_args=params)
    
    if hasattr(preprocessor, 'posible_outputs'):
        posible_outputs = preprocessor.posible_outputs
        posible_outputs_ids = tokenizer(posible_outputs, add_special_tokens=False).input_ids
        constraints = [DisjunctiveConstraint(posible_outputs_ids)]
    else:
        constraints = None

    model = LLMTrainer(params, generation_constraint=constraints)

    # Keep the model with the highest F1 score.
    check_interval_step = params.save_steps
    experiment_dir = os.path.join(params.output_dir, f'{params.dataset_name}_{params.number_training_examples}')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(experiment_dir, 'checkpoints'),
                                        filename="{epoch}-{valid_metric:.2f}",
                                        monitor="valid_metric",
                                        mode="max",
                                        verbose=True,
                                        save_top_k=1)
    early_stop_callback = EarlyStopping(monitor="valid_metric",
                                        min_delta=0.01,
                                        patience=20,
                                        verbose=True,
                                        mode="max",)
    logger = TensorBoardLogger(experiment_dir)
    lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(default_root_dir=experiment_dir, 
                        max_epochs=params.num_train_epochs,
                        logger=logger,
                        accelerator='gpu', 
                        devices=params.gpus,
                        strategy='ddp_find_unused_parameters_true' if params.gpus > 1 else 'auto',
                        precision='16-mixed',
                        val_check_interval=1.0,
                        callbacks=[checkpoint_callback, lr_logger, early_stop_callback],)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    best_model_path = checkpoint_callback.best_model_path

    # Evaluate the best models on the test sample.
    # best_model = LLMTrainer.load_from_checkpoint(best_model_path)
    # trainer.test(model=best_model, dataloaders=test_dataloader, ckpt_path=best_model_path,)
    del trainer
    del model
    gc.collect()
    
    return best_model_path


if __name__=='__main__':
    hparams = create_tuning_args()

    for params in hparams.trials(10):
        print(params)
        experiment_dir = os.path.join(params.output_dir, f'{params.dataset_name}_{params.number_training_examples}', 'tmp')
        best_model_path = train(params)
        best_model = LLMTrainer.load_from_checkpoint(best_model_path)
        best_model.model.save_pretrained(experiment_dir)
        model = AutoPeftModelForCausalLM.from_pretrained(experiment_dir, 
                                                        device_map="auto", 
                                                        torch_dtype=torch.float16,
                                                        cache_dir='hf_cache')
        model = model.merge_and_unload()
        preprocessor = get_preprocessor(params.dataset_name, 
                                    number_training_examples=params.number_training_examples)

        tokenizer = AutoTokenizer.from_pretrained(params.model_name, 
                                                cache_dir='hf_cache')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

        _, _, test_data = preprocessor.process_data()
        test_dataloader = DataLoader(test_data, batch_size=params.batch_size)
        if hasattr(preprocessor, 'posible_outputs'):
            posible_outputs = preprocessor.posible_outputs
            posible_outputs_ids = tokenizer(posible_outputs, add_special_tokens=False).input_ids
            constraints = [DisjunctiveConstraint(posible_outputs_ids)]
        else:
            constraints = None

        metric = preprocessor.metric()

        preds = []
        labels = []
        for batch in tqdm(test_dataloader):
            prompt = batch['input']
            gold = batch['gold']

            gold_size = tokenizer(gold, return_tensors="pt", padding='longest').input_ids.size(-1)
            inputs = tokenizer(prompt, return_tensors="pt", padding='longest')
            generate_ids = model.generate(inputs.input_ids.cuda(), 
                                        attention_mask=inputs.attention_mask.cuda(), 
                                        max_length=gold_size+5,
                                        num_beams=10,
                                        do_sample=False,
                                        num_return_sequences=1,
                                        constraints=constraints,
                                        no_repeat_ngram_size=2)
            
            responses = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            preds.extend(responses)
            labels.extend(gold)
        
        labels, preds = preprocessor.process_outputs(labels, preds, save_wrong_preds=True)
        metric.update(preds, labels)
        score = metric.compute()
        metric.reset() 

        print(score)
        with open('result.txt', 'a', encoding='utf-8') as f:
            f.write(f'{params.dataset_name}: {params.number_training_examples} \n')
            f.write(f'Hprams: \n{params}\n')
            f.write(f'Metric: \n{score}\n')
            f.write(f'_'*10 + '\n')

        del model
        del best_model
        del metric
        shutil.rmtree(experiment_dir)
        shutil.rmtree(os.path.join(params.output_dir, f'{params.dataset_name}_{params.number_training_examples}', 'checkpoints'))
        gc.collect()


