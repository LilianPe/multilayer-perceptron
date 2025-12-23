import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from MultilayerPerceptron import MultilayerPerceptron
from tqdm import tqdm

def splitDataset(dataPath: str, trainDataPath: str, testDataPath: str):
    data = pandas.read_csv(dataPath, header=0)

    train_df, test_df = train_test_split(data, test_size=0.4, random_state=42)

    train_df.to_csv(trainDataPath, index=False)
    test_df.to_csv(testDataPath, index=False)

def parseData(trainDataPath: str, testDataPath: str):
    train_data = pandas.read_csv(trainDataPath, header=0)
    test_data = pandas.read_csv(testDataPath, header=0)
    train_X = train_data.iloc[:, 2:] # La data sans id et diagnosis
    train_Y = train_data['Diagnosis'].map({'B': 0, 'M': 1})
    test_X = test_data.iloc[:, 2:] # La data sans id et diagnosis
    test_Y = test_data['Diagnosis'].map({'B': 0, 'M': 1})

    # numpy arrays :
    train_X = train_X.to_numpy()
    train_Y = train_Y.to_numpy().reshape(-1, 1)
    test_X = test_X.to_numpy()
    test_Y = test_Y.to_numpy().reshape(-1, 1)

    # Normalisation des donnees
    scaler = StandardScaler()
    XTrain_scaled = scaler.fit_transform(train_X)
    XTest_scaled = scaler.transform(test_X)
    return XTrain_scaled, train_Y, XTest_scaled, test_Y

def updatePerformances(m, dataset, perfValues):
    perfValues["lossT"].append(m.logLoss(dataset["train_X"].T, dataset["train_Y"]))
    perfValues["lossV"].append(m.logLoss(dataset["validation_X"].T, dataset["validation_Y"]))
    perfValues["accuracyT"].append(MLPAccuracy(m, dataset["train_X"].T, dataset["train_Y"]))
    perfValues["accuracyV"].append(MLPAccuracy(m, dataset["validation_X"].T, dataset["validation_Y"]))

def initPerfValues():
    lossT = []
    lossV = []
    accuracyT = []
    accuracyV = []
    perfValues = {"lossT": lossT, "lossV": lossV, "accuracyT": accuracyT, "accuracyV": accuracyV}
    return perfValues

def trainMLP(m: MultilayerPerceptron, epochs: int, learningRate: float, batch_size: int, dataset):
    perfValues = initPerfValues()
    updatePerformances(m, dataset, perfValues)
    for i in tqdm(range(epochs)):
        total = 0
        iteration = 0
        while(total < m.dataset_size):
            batch_X, batch_Y = m.getBatch(batch_size, iteration)
            activations = m.forwardPropagation(batch_X)
            gradients_dW, gradients_db = m.backPropagation(batch_X, batch_Y, activations)
            m.update(gradients_dW, gradients_db, learningRate)
            total += batch_size
            iteration += 1
        updatePerformances(m, dataset, perfValues)
    return perfValues

def MLPAccuracy(model: MultilayerPerceptron, X, Y) :
	A = model.predict(X)[0]
	accuracy: int = 0
	for i in range(0, Y.shape[0]) :
		if (A[i] > 0.5) == Y[i] :
			accuracy += 1
	return (accuracy / Y.shape[0] * 100)
