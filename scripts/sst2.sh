python train.py --dataset_name sst2 --load_in_8bit True --number_training_examples 50
python test.py --dataset_name sst2 --load_in_8bit True --number_training_examples 50
rm -r output

python train.py --dataset_name sst2 --load_in_8bit True --number_training_examples 100
python test.py --dataset_name sst2 --load_in_8bit True --number_training_examples 100
rm -r output

python train.py --dataset_name sst2 --load_in_8bit True --number_training_examples 200
python test.py --dataset_name sst2 --load_in_8bit True --number_training_examples 200
rm -r output

python train.py --dataset_name sst2 --load_in_8bit True --number_training_examples 500
python test.py --dataset_name sst2 --load_in_8bit True --number_training_examples 500
rm -r output
