import numpy as np
from pathlib import Path


class MultilayerPerceptron :
	def __init__(self, inputLayer, hiddenLayers, n_classes):
		hiddenLayers.insert(0, inputLayer) # Input layer
		hiddenLayers.append(n_classes) # Output layer
		self.n_hlayers = len(hiddenLayers)
		self.W_array = []
		self.b_array = []

		for i in range(1, self.n_hlayers):
			self.W_array.append(np.random.randn(hiddenLayers[i], hiddenLayers[i-1]))
			self.b_array.append(np.random.randn(hiddenLayers[i], 1))


	def forwardPropagation(self, batch_X) :
		activations = []
		activations.append(batch_X)
		for i in range(0, self.n_hlayers-1):
			self.Z = self.W_array[i].dot(activations[i]) + self.b_array[i]
			if i == self.n_hlayers - 2:
				self.A = self.softmax(self.Z)
				# print(f"Softmax output: {self.A}")
			else:
				self.A = self.sigmoide(self.Z)
			activations.append(self.A)
		return activations

	def predict(self, X) :
		activations = []
		activations.append(X)
		for i in range(0, self.n_hlayers-1):
			Z = self.W_array[i].dot(activations[i]) + self.b_array[i]
			if i == self.n_hlayers - 2:
				A = self.softmax(Z)
				# print(f"Softmax output: {A}")
			else:
				A = self.sigmoide(Z)
			activations.append(A)
		return A

	def sigmoide(self, Z):
		return 1 / (1 + np.exp(-Z))

	def softmax(self, Z):
		Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
		exp_Z = np.exp(Z_shifted)
		return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

	def binaryCrossentropy(self, X, Y) :
		eps = 1e-15
		m = X.shape[1]
		A = self.predict(X)
		A_clipped = np.clip(A, eps, 1 - eps) # pour éviter log(0) (empeche les valeures d'etre en dehors de l'interval 𝐴∈[10**−15,1−10**−15]
		result = -1 / m * np.sum(Y * np.log(A_clipped) + (1 - Y) * np.log(1 - A_clipped))
		return result

	def categoricalCrossentropy(self, X, Y):
		eps=1e-15
		m = X.shape[1]
		A = self.predict(X)
		A = np.clip(A, eps, 1 - eps)
		return -1 / m * np.sum(Y * np.log(A))

	def backPropagation(self, batch_X, batch_Y, activations) :
		m = batch_X.shape[1] # Taille du dataset

		dZ = activations[self.n_hlayers - 1] - batch_Y
		gradients_dW = []
		gradients_db = []

		for i in reversed(range(1, self.n_hlayers)):
			gradients_dW.insert(0, 1/m * np.dot(dZ, activations[i-1].T))
			gradients_db.insert(0, 1/m * np.sum(dZ, axis=1, keepdims=True))
			if i > 1:
				dZ = np.dot(self.W_array[i-1].T, dZ) * activations[i-1] * (1 - activations[i-1])
		
		return gradients_dW, gradients_db
		
	def update(self, gradient_dW, gradient_db, learning_rate):
		for i in range(0, self.n_hlayers - 1):
			self.W_array[i] -= learning_rate * gradient_dW[i]
			self.b_array[i] -= learning_rate * gradient_db[i]
	
	def getBatch(self, batch_size: int, iteration: int, dataset):
		start = iteration * batch_size
		end   = start + batch_size

		X_batch = dataset["train_X"].T[:, start:end]
		Y_batch = dataset["train_Y"][:, start:end]

		return X_batch, Y_batch

	def saveModel(self, path: str):
		Path(path).parent.mkdir(parents=True, exist_ok=True)
		np.save(
			path,
			{
				"W_array": self.W_array,
				"b_array": self.b_array,
				"n_hlayers": self.n_hlayers
			},
			allow_pickle=True
		)
		print(f"> saving model '{path}.npy' to disk...")
	
	@classmethod
	def loadModel(cls, path):
		data = np.load(path, allow_pickle=True)
		
		# créer une instance sans appeler __init__
		model = cls.__new__(cls)
		
		model.W_array = data["W_array"].tolist()
		model.b_array = data["b_array"].tolist()
		model.n_hlayers = data["n_hlayers"]
		
		return model
