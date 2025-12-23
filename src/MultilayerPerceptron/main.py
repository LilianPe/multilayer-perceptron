from utils import splitDataset, parseData, trainMLP, MLPAccuracy
from MultilayerPerceptron import MultilayerPerceptron
import argparse
import matplotlib.pyplot as plt

def handleData():
    datasetPath: str = "data/dataNames.csv"
    trainDatasetPath: str = "data/train_data.csv"
    testDatasetPath: str = "data/test_data.csv"

    splitDataset(datasetPath, trainDatasetPath, testDatasetPath)

    return parseData(trainDatasetPath, testDatasetPath)

def displayLossGraph(lossT, lossV):
    epochs = range(len(lossT))
    plt.figure()
    plt.plot(epochs, lossT, label="Train Loss")
    plt.plot(epochs, lossV, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def displayAccuracyGraph(accuracyT, accuracyV):
    epochs = range(len(accuracyT))
    plt.figure()
    plt.plot(epochs, accuracyT, label="Train accuracy")
    plt.plot(epochs, accuracyV, label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def multilayePerceptronTest(args):
    train_X, train_Y, validation_X, validation_Y = handleData()
    dataset = {"train_X": train_X, "train_Y": train_Y, "validation_X": validation_X, "validation_Y": validation_Y}

    inputLayer = train_X.shape[1]
    model: MultilayerPerceptron = MultilayerPerceptron(train_X.T, train_Y.T, args.layers)
    perfValues = trainMLP(model, args.epochs, args.learning_rate, args.batch_size, dataset)
    accuracy: int = MLPAccuracy(model, validation_X.T, validation_Y)
    print(f"Model Accuracy: {accuracy}%.")
    displayLossGraph(perfValues["lossT"], perfValues["lossV"])
    displayAccuracyGraph(perfValues["accuracyT"], perfValues["accuracyV"])

def main():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron parameters")
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 24], help="Layers sizes") # V
    parser.add_argument("--epochs", type=int, default=100, help="Epochs number") # V
    parser.add_argument("--loss", type=str, default="categoricalCrossentropy",choices=["categoricalCrossentropy"], help="Loss function") # X
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size") # X
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate") # V
    multilayePerceptronTest(parser.parse_args())

if __name__ == "__main__":
    main()

# Reste a ajouter la modularitee, avec implementation softmax etc...
# Voir si besoin de faire les 2 exemples (fichier input et parametres)
# Mais je crois pas


# Le MLP handle tout seul l'input et output layer, pas pris dans les args

# Pour l'instant, on ignore le parametre loss, a quoi il sert ? Pas demande d'implemente autre chose que categoricalCrossentropy
# -> A revoir quand meme pour etre sur

# Decouper le main en 3, 1 qui fait le splitData, un qui cree et entraine le mlp, puis save ses weights / bias et affiche les graphes
# , puis un qui load le fichier et donne le resultat d'une prediction   


# Le main training doit afficher cela:

# python mlp.py --dataset data_training.csv
# x_train shape : (342, 30)
# x_valid shape : (85, 30)
# epoch 01/70 - loss: 0.6882 - val_loss: 0.6788
# ...
# epoch 39/70 - loss: 0.0750 - val_loss: 0.0406
# epoch 40/70 - loss: 0.0749 - val_loss: 0.0404
# epoch 41/70 - loss: 0.0747 - val_loss: 0.0400
# ...
# epoch 70/70 - loss: 0.0640 - val_loss: 0.0474
# > saving model './saved_model.npy' to disk..

# Pour le main predict, revoir comment il marche, sujet demande d'evaluer un dataset avec logLoss, mais pas tres visuel et deja fait avant ?