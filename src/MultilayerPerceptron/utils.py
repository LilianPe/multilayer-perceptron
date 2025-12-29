import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from MultilayerPerceptron import MultilayerPerceptron
from pathlib import Path

def checkPaths(dataPath: str, trainDataPath: str, testDataPath: str):
    dataPath = Path(dataPath)
    trainDataPath = Path(trainDataPath)
    testDataPath = Path(testDataPath)

    if dataPath.suffix.lower() != ".csv":
         raise ValueError(f"Invalid file extension: {dataPath}")
    if trainDataPath.suffix.lower() != ".csv":
         raise ValueError(f"Invalid file extension: {trainDataPath}")
    if testDataPath.suffix.lower() != ".csv":
         raise ValueError(f"Invalid file extension: {testDataPath}")

def splitDataset(dataPath: str, trainDataPath: str, testDataPath: str):
    checkPaths(dataPath, trainDataPath, testDataPath)
    
    data = pandas.read_csv(dataPath, header=0)

    train_df, test_df = train_test_split(data, test_size=0.4, random_state=42)

    train_df.to_csv(trainDataPath, index=False)
    test_df.to_csv(testDataPath, index=False)

def parseData(trainDataPath: str, testDataPath: str, scaler):
    train_data = pandas.read_csv(trainDataPath, header=0)
    test_data = pandas.read_csv(testDataPath, header=0)
    train_X = train_data.iloc[:, 2:] # La data sans id et diagnosis
    train_Y = train_data['Diagnosis'].map({'B': 0, 'M': 1})
    test_X = test_data.iloc[:, 2:] # La data sans id et diagnosis
    test_Y = test_data['Diagnosis'].map({'B': 0, 'M': 1})

    # numpy arrays :
    train_X = train_X.to_numpy()
    train_Y = train_Y.to_numpy()
    test_X = test_X.to_numpy()
    test_Y = test_Y.to_numpy()

    # passer Y en One hot (format pour categorical crossentropy)
    train_Y_onehot = np.zeros((2, train_Y.shape[0]))
    train_Y_onehot[0, :] = 1 - train_Y
    train_Y_onehot[1, :] = train_Y
    test_Y_onehot = np.zeros((2, test_Y.shape[0]))
    test_Y_onehot[0, :] = 1 - test_Y
    test_Y_onehot[1, :] = test_Y

    # Normalisation des donnees
    XTrain_scaled = scaler.fit_transform(train_X)
    XTest_scaled = scaler.transform(test_X)
    return XTrain_scaled, train_Y_onehot, XTest_scaled, test_Y_onehot

def updatePerformances(m, dataset, perfValues, lossFunction: str):
    if lossFunction == "categoricalCrossentropy":
        perfValues["lossT"].append(m.categoricalCrossentropy(dataset["train_X"].T, dataset["train_Y"]))
        perfValues["lossV"].append(m.categoricalCrossentropy(dataset["validation_X"].T, dataset["validation_Y"]))
    else:
        perfValues["lossT"].append(BCEAccuracy(m, dataset["train_X"].T, dataset["train_Y"]))
        perfValues["lossV"].append(BCEAccuracy(m, dataset["validation_X"].T, dataset["validation_Y"]))

    perfValues["accuracyT"].append(MLPAccuracy(m, dataset["train_X"].T, dataset["train_Y"]))
    perfValues["accuracyV"].append(MLPAccuracy(m, dataset["validation_X"].T, dataset["validation_Y"]))

def initPerfValues():
    lossT = []
    lossV = []
    accuracyT = []
    accuracyV = []
    perfValues = {"lossT": lossT, "lossV": lossV, "accuracyT": accuracyT, "accuracyV": accuracyV}
    return perfValues

def printPerformances(perfValues, epoch, n_epochs):
    width = len(str(n_epochs))
    print(f"epoch {epoch:0{width}d}/{n_epochs} - loss: {perfValues["lossT"][epoch-1]} - val_loss: {perfValues["lossV"][epoch-1]}")

def trainMLP(m: MultilayerPerceptron, epochs: int, learningRate: float, batch_size: int, dataset, lossFunction: str):
    print(f"x_train shape : {dataset["train_X"].shape}")
    print(f"x_valid shape : {dataset["validation_X"].shape}")
    
    dataset_size = dataset["train_X"].shape[0]
    perfValues = initPerfValues()
    updatePerformances(m, dataset, perfValues, lossFunction)
    for epoch in range(epochs):
        total = 0
        iteration = 0
        while(total < dataset_size):
            batch_X, batch_Y = m.getBatch(batch_size, iteration, dataset)
            activations = m.forwardPropagation(batch_X)
            gradients_dW, gradients_db = m.backPropagation(batch_X, batch_Y, activations)
            m.update(gradients_dW, gradients_db, learningRate)
            total += batch_size
            iteration += 1
        updatePerformances(m, dataset, perfValues, lossFunction)
        printPerformances(perfValues, epoch+1, epochs)
    return perfValues

def MLPAccuracy(model: MultilayerPerceptron, X, Y) :
    A = model.predict(X)
    pred_classes = np.argmax(A, axis=0)
    true_classes = np.argmax(Y, axis=0)
    accuracy = np.mean(pred_classes == true_classes) * 100
    return accuracy

def BCEAccuracy(model: MultilayerPerceptron, X, Y):
    eps = 1e-15
    A = model.predict(X)
    A_bin = A[1, :]
    A_bin = np.clip(A_bin, eps, 1 - eps)
    Y_bin = Y[1, :]
    loss = - np.mean(Y_bin * np.log(A_bin) + (1 - Y_bin) * np.log(1 - A_bin))
    return loss
