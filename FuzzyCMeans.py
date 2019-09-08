import numpy

points = numpy.array([[1, 2, 2], [3, 2, 2], [6, 5, 5], [6,5,1]])

class FuzzyCMeans:
    def __init__(self,inputDataFrame, numberOfClusters, fuzzyfier):
        self.inputData = inputDataFrame
        self.numberOfClusters = numberOfClusters
        self.numberOfInstances, self.numberOfFeatures = self.inputData.shape
        self.MAX_ITER = 100
        #default fuzzifier value is 2
        self.fuzzifier = fuzzyfier
        print("input data",self.inputData)
        print("number of cluster",self.numberOfClusters)
        print("number of instances",self.numberOfInstances)
        print("number of features",self.numberOfFeatures)
        print("max iter",self.MAX_ITER)


fuzzyCMeansObj = FuzzyCMeans(inputDataFrame=points, numberOfClusters=2,fuzzyfier=2)

