import numpy as np

class MultilayerPerceptron :
	def __init__(self, X, Y, layers):
		layers.insert(0, X.shape[0]) # Input layer
		layers.append(1) # Output layer
		self.n_layers = len(layers)
		self.W_array = []
		self.b_array = []

		for i in range(1, self.n_layers):
			self.W_array.append(np.random.randn(layers[i], layers[i-1]))
			self.b_array.append(np.random.randn(layers[i], 1))

		self.X = X
		self.Y = Y

	def forwardPropagation(self) :
		activations = []
		activations.append(self.X)
		for i in range(0, self.n_layers-1):
			self.Z = self.W_array[i].dot(activations[i]) + self.b_array[i]
			self.A = 1 / (1 + np.exp(-self.Z))
			activations.append(self.A)
		return activations

	def predict(self, X) :
		activations = []
		activations.append(X)
		for i in range(0, self.n_layers-1):
			Z = self.W_array[i].dot(activations[i]) + self.b_array[i]
			A = 1 / (1 + np.exp(-Z))
			activations.append(A)
		return A

	def log_loss(self) :
		eps = 1e-15 
		A_clipped = np.clip(self.A, eps, 1 - eps) # pour éviter log(0) (empeche les valeures d'etre en dehors de l'interval 𝐴∈[10**−15,1−10**−15]
		result = -1 / self.X.shape[0] * np.sum(self.Y * np.log(A_clipped) + (1 - self.Y) * np.log(1 - A_clipped))
		return result

	def backPropagation(self, activations) :
		m = self.X.shape[1] # Taille du dataset

		dZ = activations[self.n_layers - 1] - self.Y
		gradients_dW = []
		gradients_db = []

		for i in reversed(range(1, self.n_layers)):
			gradients_dW.insert(0, 1/m * np.dot(dZ, activations[i-1].T))
			gradients_db.insert(0, 1/m * np.sum(dZ, axis=1, keepdims=True))
			if i > 1:
				dZ = np.dot(self.W_array[i-1].T, dZ) * activations[i-1] * (1 - activations[i-1])
		
		return gradients_dW, gradients_db
		
	def update(self, gradient_dW, gradient_db, learning_rate):
		for i in range(0, self.n_layers - 1):
			self.W_array[i] -= learning_rate * gradient_dW[i]
			self.b_array[i] -= learning_rate * gradient_db[i]
