from train import run
from hyperparams_tuning_args import create_tuning_args



if __name__=='__main__':
    hparams = create_tuning_args()

    for trial in hparams.trials(10):
        print(trial)
        run(trial)


