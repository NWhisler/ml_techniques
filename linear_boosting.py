import numpy as np
import matplotlib.pyplot as plt 

class Linear_Model(object):

	def __init__(self,X,Y,projection_sums,models):

		self.X = X
		self.Y = Y
		self.projection_sums = projection_sums
		self.w = np.linalg.solve(X.T.dot(X),X.T.dot(Y))
		self.models = models

	def fit(self):

		self.P = self.X.dot(self.w)

		self.projection_sums = self.projection_sums + self.P

		self.residual = self.Y - self.projection_sums

		self.rss = np.sum(self.residual**2)

		self.tss = np.sum((self.Y - np.mean(self.Y))**2)

		self.r_squared = 1 - self.rss/self.tss

class Linear_Boost(object):

	def __init__(self,X,Y,iterations=1000):

		self.X = X
		self.Y = Y
		self.iterations = iterations
		self.projection_sums = np.zeros(len(Y))
		self.models = []
		self.r_squared = []
		self.best_model = 0
		self.best_r_squared = 0

	def fit(self):

		for iteration in range(int(self.iterations)):

			if iteration == 0:

				self.model = Linear_Model(self.X,self.Y,self.projection_sums,iteration+1)

			else:

				self.model = Linear_Model(self.X,self.residual,self.projection_sums,iteration+1)

			self.model.fit()

			self.projection_sums = self.model.projection_sums
			self.residual = self.model.residual
			self.models.append(self.model.projection_sums)
			self.r_squared.append(self.model.r_squared)

			if self.model.r_squared > self.best_r_squared:

				self.best_r_squared = self.model.r_squared
				self.best_model = iteration

if __name__ == '__main__':

	samples = 100

	X = np.linspace(0,100,samples) + np.random.randn(samples)*10
	X = X.reshape(samples,1)
	Y = np.arange(0,samples) + np.random.randn(samples)*10
	Y = Y.reshape(samples,1)

	plt.scatter(X,Y)
	plt.show()

	model = Linear_Boost(X,Y)
	model.fit()

	plt.plot(model.r_squared[:model.best_model+1])
	plt.show()

	print(model.r_squared[:model.best_model+1])

	print(model.best_model)

	print(model.best_r_squared)

	plt.scatter(X,Y)
	plt.plot(X,-model.models[model.best_model])
	plt.show()