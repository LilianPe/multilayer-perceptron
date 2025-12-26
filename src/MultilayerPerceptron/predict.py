import argparse
from utils import BCEAccuracy


def handleArguments():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron creation parameters")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the training model")
    parser.add_argument("--validation_set", type=str, required=True, help="Path to the validation set")
    return parser

def main():
    parser = handleArguments()
    loss = BCEAccuracy()

if __name__ == "__main__":
    main()

# Voir comment load le dataset, pour l'instant, parsedata prend un training set avec le validation set alors que la on a que le validation set
# De plus, il faut normaliser la data mais on a plus le scaler, voir si il faut le save ou autre
