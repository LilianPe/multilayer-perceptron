from utils import splitDataset, parseData, trainPerceptron, perceptronAccuracy
from Perceptron import Perceptron


def handleData():
    datasetPath: str = "data/dataNames.csv"
    trainDatasetPath: str = "data/train_data.csv"
    testDatasetPath: str = "data/test_data.csv"

    splitDataset(datasetPath, trainDatasetPath, testDatasetPath)

    return parseData(trainDatasetPath, testDatasetPath)

def perceptronTest():
    train_X, train_Y, test_X, test_Y = handleData()

    model: Perceptron = Perceptron(train_X, train_Y)
    trainPerceptron(model)
    accuracy: int = perceptronAccuracy(model, test_X, test_Y)
    print(f"Model Accuracy: {accuracy}%.")

def main():
    perceptronTest()

if __name__ == "__main__":
    main()
