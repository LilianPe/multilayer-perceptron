# How to use:

## Dataset

First, split the dataset to create a train set and a validation set 
with

```bash
python src/MultilayerPerceptron/splitData.py --data_path data/dataNames.csv --train_path data/train.csv --validation_path data/validation.csv
``` 

## Training

Then create and train a model using 

```bash
python src/MultilayerPerceptron/training.py --train_set data/train.csv --validation_set data/validation.csv --save_model_path model/model --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314
``` 

## Prediction

```bash
python src/MultilayerPerceptron/predict.py --model model/model.npy --prediction_set data/validation.csv 
``` 
