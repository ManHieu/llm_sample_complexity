# python tune.py --dataset_name rte --load_in_4bit True --number_training_examples 10
# rm -r output

python tune.py --dataset_name anli_r1 --load_in_4bit True --number_training_examples 50 
rm -r output

# python tune.py --dataset_name rte --load_in_4bit True --number_training_examples 100 
# rm -r output

# python tune.py --dataset_name rte --load_in_4bit True --number_training_examples 200 
# rm -r output

# python tune.py --dataset_name rte --load_in_4bit True --number_training_examples 500 
# rm -r output