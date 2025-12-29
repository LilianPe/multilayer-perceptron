import argparse
import numpy as np
import pandas
from utils import BCEAccuracy
from sklearn.preprocessing import StandardScaler
from MultilayerPerceptron import MultilayerPerceptron

def parseDataset(datasetPath: str, scaler):
    dataset = pandas.read_csv(datasetPath, header=0)
    X = dataset.iloc[:, 2:] # La data sans id et diagnosis
    Y = dataset['Diagnosis'].map({'B': 0, 'M': 1})

    # numpy arrays :
    X = X.to_numpy()
    Y = Y.to_numpy()

    # passer Y en One hot
    Y_onehot = np.zeros((2, Y.shape[0]))
    Y_onehot[0, :] = 1 - Y
    Y_onehot[1, :] = Y

    # Normalisation des donnees
    X_scaled = scaler.transform(X)
    return X_scaled, Y_onehot

def handleArguments():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron creation parameters")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the training model")
    parser.add_argument("--prediction_set", type=str, required=True, help="Path to the validation set")
    return parser

def loadScaler(path):
    data = np.load(path, allow_pickle=True).item()
    mean = data["scaler_mean"]
    scale = data["scaler_scale"]
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale

    # nécessaire pour que sklearn accepte transform()
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler

def main():
    parser = handleArguments()
    args = parser.parse_args()
    scaler = loadScaler(args.model_path)
    model = MultilayerPerceptron.loadModel(args.model_path)
    X, Y = parseDataset(args.prediction_set, scaler)
    loss = BCEAccuracy(model, X.T, Y)
    print(f"Loss on the dataset: {loss}.")

if __name__ == "__main__":
    main()
