from utils import parseData, trainMLP, MLPAccuracy, BCEAccuracy
from MultilayerPerceptron import MultilayerPerceptron
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def handleData(trainDatasetPath: str, testDatasetPath: str, scaler):
    return parseData(trainDatasetPath, testDatasetPath, scaler)

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
    scaler = StandardScaler()
    train_X, train_Y, validation_X, validation_Y = handleData(args.train_set, args.validation_set, scaler)
    dataset = {"train_X": train_X, "train_Y": train_Y, "validation_X": validation_X, "validation_Y": validation_Y}

    model: MultilayerPerceptron = MultilayerPerceptron(train_X.shape[1], args.layers, 2)
    model.scaler_mean = scaler.mean_
    model.scaler_scale = scaler.scale_
    perfValues = trainMLP(model, args.epochs, args.learning_rate, args.batch_size, dataset, args.loss)
    model.saveModel(args.save_model_path)
    displayLossGraph(perfValues["lossT"], perfValues["lossV"])
    displayAccuracyGraph(perfValues["accuracyT"], perfValues["accuracyV"])
    # accuracy: int = MLPAccuracy(model, validation_X.T, validation_Y)
    # print(f"Model Accuracy: {accuracy}%.")
    # accuracy: int = BCEAccuracy(model, validation_X.T, validation_Y)

def handleArguments():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron creation parameters")
    parser.add_argument("--train_set", type=str, required=True, help="Path to the training set")
    parser.add_argument("--validation_set", type=str, required=True, help="Path to the validation set")
    parser.add_argument("--save_model_path", type=str, required=True, help="Path to save the model")
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 24], help="Layers sizes")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs number")
    parser.add_argument("--loss", type=str, default="categoricalCrossentropy",choices=["categoricalCrossentropy", "binaryCrossentropy"], help="Loss function")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate")
    return parser

def main():
    parser = handleArguments()
    multilayePerceptronTest(parser.parse_args())

if __name__ == "__main__":
    main()

# Maintenant, a voir si bonus
# Bonus interessants:

# - Voir ce qu'est early stoping
# - Ajouter quelques metriques + historique de celles ci (2 bonus)
# - Voir ce qu'est une optimisation fonction, et si ca vaut le coup d'en implementer une plus complexe
# - Dernier bonus parait pas tres interessant, mais a revoir