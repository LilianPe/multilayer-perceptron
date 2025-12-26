from utils import splitDataset, parseData, trainMLP, MLPAccuracy, BCEAccuracy
from MultilayerPerceptron import MultilayerPerceptron
import argparse
import matplotlib.pyplot as plt

def handleData(trainDatasetPath: str, testDatasetPath: str):
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
    train_X, train_Y, validation_X, validation_Y = handleData(args.train_set, args.validation_set)
    dataset = {"train_X": train_X, "train_Y": train_Y, "validation_X": validation_X, "validation_Y": validation_Y}

    model: MultilayerPerceptron = MultilayerPerceptron(train_X.shape[1], args.layers, 2)
    perfValues = trainMLP(model, args.epochs, args.learning_rate, args.batch_size, dataset)
    model.saveModel(args.save_model_path)
    displayLossGraph(perfValues["lossT"], perfValues["lossV"])
    displayAccuracyGraph(perfValues["accuracyT"], perfValues["accuracyV"])
    accuracy: int = MLPAccuracy(model, validation_X.T, validation_Y)
    print(f"Model Accuracy: {accuracy}%.")
    accuracy: int = BCEAccuracy(model, validation_X.T, validation_Y)

def handleArguments():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron creation parameters")
    parser.add_argument("--train_set", type=str, required=True, help="Path to the training set")
    parser.add_argument("--validation_set", type=str, required=True, help="Path to the validation set")
    parser.add_argument("--save_model_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 24], help="Layers sizes") # V
    parser.add_argument("--epochs", type=int, default=100, help="Epochs number") # V
    parser.add_argument("--loss", type=str, default="categoricalCrossentropy",choices=["categoricalCrossentropy"], help="Loss function") # X
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size") # V
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate") # V
    return parser

def main():
    parser = handleArguments()
    multilayePerceptronTest(parser.parse_args())

if __name__ == "__main__":
    main()


# Pour l'instant, on ignore le parametre loss, a quoi il sert ? Pas demande d'implemente autre chose que categoricalCrossentropy
# -> A revoir quand meme pour etre sur


# Decouper le main en 3, 1 qui fait le splitData, un qui cree et entraine le mlp, puis save ses weights / bias et affiche les graphes
# , puis un qui load le fichier et donne le resultat d'une prediction   

# Actuellement, binary cross entropy utilise + sigmoide sur output, passer a categorical cross entropy + softmax sur output
# Mais pour l'instant ca fait nimp
# Probleme venait de la shape de Y

# Recheck les choses demandees dans le sujet mais ca me parait pas mal

# Reste a faire le 3eme main

# Pour le main predict, revoir comment il marche, sujet demande d'evaluer un dataset avec logLoss, mais pas tres visuel et deja fait avant ?