import numpy as np

class Perceptron :
	def __init__(self, X, Y):
		self.W = np.random.randn(X.shape[1], 1)
		self.b = np.random.randn(1)
		self.X = X
		self.Y = Y

	def model(self) :
		self.Z = self.X.dot(self.W) + self.b # Fait cela: Zi​=Xi​[0]∗W[0]+Xi​[1]∗W[1]+...+Xi​[29]∗W[29]+b
		self.A = 1 / (1 + np.exp(-self.Z))

	def predict(self, X) :
		Z = X.dot(self.W) + self.b
		A = 1 / (1 + np.exp(-Z))
		return A

	def log_loss(self) :
		self.model()
		eps = 1e-15 
		A_clipped = np.clip(self.A, eps, 1 - eps) # pour éviter log(0) (empeche les valeures d'etre en dehors de l'interval 𝐴∈[10**−15,1−10**−15]
		result = -1 / self.X.shape[0] * np.sum(self.Y * np.log(A_clipped) + (1 - self.Y) * np.log(1 - A_clipped))
		return result

	# Wn -= Somme de i allant de 1 a N (taille du dataset) et m (allant de 1 a M, nombre de features) (yi - ai) * xn -> 
	# W1 = ((y1 - a1) * x[0][0] + (y2 - a2) * x[0][1] ... + (yi - ai) * x[0][i])
	# W2 = ((y1 - a1) * x[1][0] + (y2 - a2) * x[1][1] ... + (yi - ai) * x[1][i])
	# W3 = ((y1 - a1) * x[2][0] + (y2 - a2) * x[2][1] ... + (yi - ai) * x[2][i])
	# ...
	# Wm = ((y1 - a1) * x[m][0] + (y2 - a2) * x[m][1] ... + (yi - ai) * x[m][i])
	# On retrouve le vecteur X mais transpose (de base: (N, M), ici (M, N))
	# Donc vecteur W = produit matricielle de X par (Y - A)
	def gradient_descent(self, learning_rate) :
		self.model()
		self.W -= learning_rate * (-1/self.X.shape[0] * np.dot(self.X.T, self.Y - self.A))
		self.b -= learning_rate * (-1/self.X.shape[0] * np.sum(self.Y - self.A))
