import numpy as np
import matplotlib.pyplot as plt


def readFile(path):
    X = np.loadtxt(path)
    return X

'''
    --euclideanDistance--

    Input: 
        inst1 - vector of size length
        inst2 - vector of size length
        length - the length of the two vectors.

    Output: 
       the euclidean distance between the two instances.
''' 
def euclideanDistance(inst1, inst2, length):
	distance = 0
	for x in range(length):
		distance += np.power((inst1[x] - inst2[x]), 2)
	return np.sqrt(distance)

'''
    --plot--

    Input: 
        X - numpy.ndarray of samples (2 collums)

    Output: 
       plot the data on 2d graph
''' 
def plot(X):
    plt.scatter(X[:, 0], X[:, 1], marker="x", color="r")
    plt.show()

'''
    --plotCentroids--

    Input: 
        centroids - numpy.ndarray of samples (2 collums) (the centroids of kmeans return value)
        classes - classification parameter 

    Output: 
       plot the seperated data by the algorithm on 2d graph and bold their centorids.
''' 
def plotCentroid(centroids, classes):
    colors = 10*["r", "g", "c", "b", "k"]
    for i in range(len(centroids)):
	    plt.scatter(centroids[i][0], centroids[i][1], s = 150, marker = "D")

    for classification in classes:
	    color = colors[classification]
	    for features in classes[classification]:
		    plt.scatter(features[0], features[1], color = color,s = 30, marker='x')
    plt.show()

'''
    --plotCentroids--

    Input: 
        X - numpy.ndarray of samples (2 collums) (the centroids of kmeans return value)
        k - classification parameter
        max_iter - 
        tolerance -  

    Output: 
       centroids - 
       classes - 
''' 
def KMeans(X, k, max_iter, tolerance):
    centroids = []
    for i in range(k):
	    centroids.append(np.copy(X[i]))

    for i in range(max_iter):
        classes = {}
        for i in range(k):
            classes[i] = []

        #find the distance between the point and cluster; choose the nearest centroid
        for features in X:
            distances = [euclideanDistance(features, centroids[j], len(features)) for j in range(len(centroids))]
            classification = distances.index(min(distances))
            classes[classification].append(features)

        previous = np.array(centroids)
        #average the cluster datapoints to re-calculate the centroids
        for classification in classes:
            centroids[classification] = np.average(classes[classification], axis = 0)
        
        isOptimal = True
        for i in range(len(centroids)):
	        original_centroid = previous[i]
	        curr = centroids[i]
	        if np.sum((curr - original_centroid) / original_centroid * 100.0) > tolerance:
		        isOptimal = False

	    #break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
        if isOptimal:
            break
        
    return centroids, classes


if __name__ == '__main__':
    X = readFile('faithful.txt')
    k = X.shape[1]
    max_iter = 10000
    tolerance = 0.001
    plot(X)
    centroids, classes = KMeans(X, k, max_iter, tolerance)
    plotCentroid(centroids, classes)
