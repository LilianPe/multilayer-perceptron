from utils import splitDataset, parseData, trainMLP, MLPAccuracy
from MultilayerPerceptron import MultilayerPerceptron
import argparse

def handleData():
    datasetPath: str = "data/dataNames.csv"
    trainDatasetPath: str = "data/train_data.csv"
    testDatasetPath: str = "data/test_data.csv"

    splitDataset(datasetPath, trainDatasetPath, testDatasetPath)

    return parseData(trainDatasetPath, testDatasetPath)

def multilayePerceptronTest(args):
    train_X, train_Y, test_X, test_Y = handleData()

    inputLayer = train_X.shape[1]
    model: MultilayerPerceptron = MultilayerPerceptron(train_X.T, train_Y.T, args.layers)
    trainMLP(model, args.epochs, args.learning_rate)
    accuracy: int = MLPAccuracy(model, test_X.T, test_Y)
    print(f"Model Accuracy: {accuracy}%.")

def main():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron parameters")
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 24], help="Layers sizes") # V
    parser.add_argument("--epochs", type=int, default=100, help="Epochs number") # V
    parser.add_argument("--loss", type=str, default="categoricalCrossentropy",choices=["categoricalCrossentropy"], help="Loss function") # X
    parser.add_argument("--batch", type=int, default=8, help="Batch size") # X
    parser.add_argument("--learning_rate", type=int, default=0.5, help="Learning rate") # V
    multilayePerceptronTest(parser.parse_args())

if __name__ == "__main__":
    main()

# Reste a ajouter la modularitee, avec implementation softmax etc...
# Voir si besoin de faire les 2 exemples (fichier input et parametres)
# Mais je crois pas


# Le MLP handle tout seul l'input et output layer, pas pris dans les args

# Ajouter des batch pour le training au lieu de tout train d'un coup
# Juste split X en X.size() // 8 set et les envoyer 1 par 1 au training

# Pour l'instant, on ignore le parametre loss, a quoi il sert ? Pas demande d'implemente autre chose que categoricalCrossentropy
# -> A revoir quand meme pour etre sur

# Ajouter les graphes a la fin, reprendre le fonctionnement dans les notebooks
