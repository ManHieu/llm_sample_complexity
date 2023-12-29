if __name__=='__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    from transformers import HfArgumentParser, AutoTokenizer
    from arguments import ScriptArguments
    from data_module.preprocess import get_preprocessor

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    preprocessor = get_preprocessor('anli_r1', number_training_examples=50)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, 
                                              cache_dir='hf_cache')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_dataloader, val_dataloader, test_dataloader = preprocessor.prepare_data(tokenizer=tokenizer,
                                                                                  script_args=script_args)
    for batch in train_dataloader:
        breakpoint()

    for batch in val_dataloader:
        breakpoint()

