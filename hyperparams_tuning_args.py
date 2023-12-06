from test_tube import HyperOptArgumentParser


def create_tuning_args():
    parser = HyperOptArgumentParser(strategy='random_search')

    # DATA_PARAMS
    parser.add_argument('--dataset_name', type=str, help="the dataset name")
    parser.add_argument('--seq_length', type=int, default=512, help="Input sequence length")
    parser.add_argument('--number_training_examples', type=int, default=50, help="Number examples of training data")

    # MODEL_PARAMS
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help="the model name")
    parser.add_argument('--load_in_8bit', type=bool, default=False, help="load the model in 8 bits precision")
    parser.add_argument('--load_in_4bit', type=bool, default=False, help="load the model in 4 bits precision")
    parser.add_argument('--use_peft', type=bool, default=True, help="Wether to use PEFT or not to train adapters")
    parser.add_argument('--trust_remote_code', type=bool, default=False, help="Enable `trust_remote_code`")
    parser.opt_list('--peft_lora_r', type=int, default=8, tunable=True, options=[4, 8, 16, 32], help="the r parameter of the LoRA adapters")
    parser.opt_list('--peft_lora_alpha', type=int, default=16, tunable=True, options=[8, 16, 32], help="the alpha parameter of the LoRA adapters")
    parser.add_argument('--peft_lora_dropout', type=float, default=0.05, help="the dropout parameter of the LoRA adapters")

    # TRAINING_PARAMS
    parser.opt_list('--learning_rate', type=float, default=1.41e-4, tunable=True, options=[1e-5, 5e-5, 1e-4, 2e-4], help="the learning rate")
    parser.add_argument('--batch_size', type=int, default=1, help="the batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="the number of gradient accumulation steps")
    parser.opt_list('--num_train_epochs', type=int, default=20, tunable=True, options=[20, 30, 40, 50], help="the number of training epochs")
    parser.add_argument('--max_steps', type=int, default=-1, help="the number of training steps")
    parser.add_argument('--gradient_checkpointing', type=bool, default=False, help="Whether to use gradient checkpointing or no")
    
    # MISC.
    parser.add_argument('--log_with', type=str, default="tensorboard", help="use 'wandb' to log with wandb")
    parser.add_argument('--output_dir', type=str, default="output", help="the output directory")
    parser.add_argument('--logging_steps', type=int, default=1, help="the number of logging steps")
    parser.add_argument('--use_auth_token', type=bool, default=True, help="Use HF auth token to access the model")
    parser.add_argument('--save_steps', type=int, default=10, help="Number of updates steps before two checkpoint saves")
    parser.add_argument('--save_total_limit', type=int, default=2, help="Limits total number of checkpoints.")
    parser.add_argument('--push_to_hub', type=bool, default=False, help="Push the model to HF Hub")
    parser.add_argument('--hub_model_id', type=str, default=None, help="The name of the model on HF Hub")

    return parser.parse_args()