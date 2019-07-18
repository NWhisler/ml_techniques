import numpy as np 
import matplotlib.pyplot as plt 

X = np.random.randn(1000,2)
X[:50] = X[:50] + 5
X[50:] = X[50:] - 5

Y = np.array([0]*50 + [1]*50)

cluster_centers = np.vstack([X[np.random.randint(X.shape[0])],X[np.random.randint(X.shape[0])]])
cluster_labels = np.zeros(X.shape[0])
labels = [0,1]

total_distance = []
previous_distance = None
current_distance = None
count = 0

while True:

	count += 1

	for row in range(X.shape[0]):

		cluster_labels[row] = 0 if np.abs(np.sum(X[row]-cluster_centers[0])) < np.abs(np.sum(X[row]-cluster_centers[1])) else 1

	distance = 0 

	for label in labels:

		distance += np.abs(np.sum(X[cluster_labels==label]-cluster_centers[label]))
		
	total_distance.append(distance)

	for label in labels:

		idx = np.argmin(np.sum(X[cluster_labels==label]-cluster_centers[label],axis=1))
		
		if np.abs(np.sum(X[cluster_labels==label] - X[idx])) < np.abs(np.sum(X[cluster_labels==label] - cluster_centers[label])):
			
			cluster_centers[label] = X[idx]

	plt.scatter(X[:,0],X[:,1],c=cluster_labels)
	plt.scatter(cluster_centers[:,0],cluster_centers[:,1],c=[0,1])
	plt.show() 

	previous_distance = current_distance

	current_distance = distance

	cluster_centers = np.vstack([np.mean(X[cluster_labels==0],axis=0),np.mean(X[cluster_labels==1],axis=0)])

	if previous_distance and previous_distance <= current_distance:

		break

plt.scatter(X[:,0],X[:,1],c=cluster_labels)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],c=[0,1])
plt.show()

plt.plot(total_distance)
plt.show()

print(count)
print(total_distance)