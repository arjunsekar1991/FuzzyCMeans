from collections import defaultdict

import numpy
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import itertools from izip
points = numpy.array ([[1, 2, 2], [3, 2, 2], [6, 5, 5], [6, 5, 1]])
from scipy.spatial import distance

#points = numpy.array ([[1, 3], [1.5, 3.2], [1.3, 2.8], [3, 1]])


class FuzzyCMeans:
    def __init__(self, inputDataFrame, numberOfClusters, fuzzyfier):
        self.inputData = inputDataFrame
        self.inputDataCopy = self.inputData
        self.numberOfClusters = numberOfClusters
        self.numberOfInstances, self.numberOfFeatures = self.inputData.shape
        self.MAX_ITER = 1000
        self.membershipMatrix = []
        self.clusterMembership = []
        # default fuzzifier value is 2
        self.fuzzifier = fuzzyfier
        self.fuzzyCentres = []
        self.clusteredData = []
        self.iterationResults = []
        # print("input data",self.inputData)
        # print("number of cluster",self.numberOfClusters)
        # print("number of instances",self.numberOfInstances)
        # print("number of features",self.numberOfFeatures)
        # print("max iter",self.MAX_ITER)

    def generateRandomInitialMembershipMatrix(self):

        for instanceCounter in range (self.numberOfInstances):
            # this loop will run till the number of instances are there in input generate those many random splits
            # this will also generate according to the number of features
            # randomNumberAsPerNumberOfFeatures = numpy.random.normal(size=self.numberOfFeatures)
            randomNumberAsPerNumberOfFeatures = numpy.random.random (size=self.numberOfClusters)
            # print("random matrix",randomNumberAsPerNumberOfFeatures)
            randomNumberAsPerNumberOfFeatures /= randomNumberAsPerNumberOfFeatures.sum ()
            # using divide by sum technique to get the matrix normalized to the sum 1

            self.membershipMatrix.append (randomNumberAsPerNumberOfFeatures)
            # print("Random Membership matrix",self.membershipMatrix)

    def calculateCentroidsBasedOnMembershipMatrix(self):
        # [list(a) for a in zip([1,2,3], [4,5,6], [7,8,9])]
        # memberShipValuesIteratedInRowWise = list(itertools(*self.membershipMatrix))
        self.fuzzyCentres = []
        memberShipValuesConvertedToTuplesColumnWise = zip (*self.membershipMatrix)
        # print(list(memberShipValuesConvertedToTuplesColumnWise))
        memberShipValuesConvertedToListColumnWise = list (memberShipValuesConvertedToTuplesColumnWise)
        # print(memberShipValuesConvertedToListColumnWise)
        for clusterCounter in range (self.numberOfClusters):
            fuzzyCentroidCoordinates = []
            # print("used for fuzzy centroid", clusterCounter,memberShipValuesConvertedToListColumnWise[clusterCounter])
            # print(memberShipValuesConvertedToListColumnWise[clusterCounter])

            squaresOfDenominator = numpy.square (memberShipValuesConvertedToListColumnWise[clusterCounter])
            # denominator is sum of squares
            denominator = numpy.sum (squaresOfDenominator)

            # print (denominator)

            listOfDataZippedBasedOnColumn = list (zip (*self.inputDataCopy))
            # print("input data ",listOfDataZippedBasedOnColumn[clusterCounter])
            # print("numerator")

            for dataColumnCounter in range (self.numberOfFeatures):
                centreCordinates = numpy.sum (numpy.multiply (squaresOfDenominator, listOfDataZippedBasedOnColumn[
                    dataColumnCounter])) / denominator
                # print("centre coordinate ",clusterCounter,"",centreCordinates)
                fuzzyCentroidCoordinates.append (centreCordinates)
            self.fuzzyCentres.append (fuzzyCentroidCoordinates)
        # print("fuzzy centroids",self.fuzzyCentres)

    def fuzzyCMeansCoreAlgorithm(self):
        self.generateRandomInitialMembershipMatrix ()
        for iteration in range (self.MAX_ITER):
            self.calculateCentroidsBasedOnMembershipMatrix ()
            self.updateMembershipValue ()
        # print ("final membership matrix",self.membershipMatrix)
        # print("fuzzy centroids",self.fuzzyCentres)

    def getClusters(self):
        # self.clusterMembership =[]
        for instanceCounter in range (self.numberOfInstances):
            # index = numpy.unravel_index(numpy.argmax(numpy.array(self.membershipMatrix), axis=None), numpy.array(self.membershipMatrix).shape)
            # self.clusterMembership.append(index)
            # print(self.membershipMatrix)
            a = numpy.array (self.membershipMatrix[instanceCounter])  # Can be of any shape\

            # i,j = numpy.unravel_index(a.argmax(), a.shape)
            # print(a.argmax())
            self.clusterMembership.append (a.argmax ())
        # index =a.argmax(axis=0)
        # print(index)
        # return index
        return self.clusterMembership

    def updateMembershipValue(self):
        # print("membership value",self.membershipMatrix)
        for dataCounter in range (self.numberOfInstances):
            inputDataPoint = self.inputData[dataCounter]
            # print("inputdata point" , inputDataPoint)
            distances = []
            for clusterCount in range (self.numberOfClusters):
                distances.append (numpy.linalg.norm (self.fuzzyCentres[clusterCount] - inputDataPoint))
            # print("distances", distances)
            for clusterCount in range (self.numberOfClusters):
                numerator = sum ([math.pow (float (distances[clusterCount] / distances[temp]), 2) for temp in
                                  range (self.numberOfClusters)])
                self.membershipMatrix[dataCounter][clusterCount] = float (1 / numerator)

    def computeSValue(self, index):
        sValue = 0.0
        for clusteredDataCount in self.clusteredData[index]:
            sValue += distance.euclidean (clusteredDataCount, self.fuzzyCentres[index])
            #print (sValue)
            norm_c = len (self.clusteredData[index])

        return sValue / norm_c

    def calculateEuclideanDistance(self, centroid, inputDataPoint):
        return numpy.linalg.norm(centroid - inputDataPoint)

    def computeRIJValue(self, idx_i, idx_j):
        RIJ = 0
        try:
            #print ("fuzzy centres", self.fuzzyCentres[idx_i], self.fuzzyCentres[idx_j])
            #print("#########")
            interClusterDistance = self.calculateEuclideanDistance (numpy.array(self.fuzzyCentres[idx_i]), numpy.array(self.fuzzyCentres[idx_j]))
            #print("*********")
            #print("inter cluster distance", interClusterDistance)
            RIJ = (self.computeSValue (idx_i) + self.computeSValue (idx_j)) / interClusterDistance
            #print ("RIJ", RIJ)
        except:
            #print("exception occured")
            return 0
        return RIJ

    def computeRValue(self, clusterCounter):
        Rij = []
        clusterIndexCombination = numpy.arange (self.numberOfClusters)
        idx_i = clusterCounter

        for idx_j, j in enumerate (clusterIndexCombination):
            if (idx_i != idx_j):
                # Rij.append((i, j))
                #print ("i ", idx_i, "j  ", idx_j)
                Rij.append (self.computeRIJValue (idx_i, idx_j))
        return max (Rij)

    def daviesBouldinIndex(self):
        print ("calculation started")
        rValue = 0.0
        for clusterCounter in range (self.numberOfClusters):
            rValue = rValue + self.computeRValue (clusterCounter)
            #print (rValue)
        rValue = float (rValue) / float (self.numberOfClusters)
        return rValue

    def plotCMeans(self, iterationResults, x_axis_label="", y_axis_label="", z_axis_label="", plot_title=""):

        fig = plt.figure ()
        colors = ['r', 'g', 'b', 'y']
        ax = fig.add_subplot (111, projection='3d')
        # print("new centroid")
        # for centroidIndex,dataPoints in iterationResults.items():
        # print(self.centroid[centroidIndex],dataPoints)

        # print("end of clustering results")
        for centroidIndex, dataPoints in iterationResults.items ():
            # print("final centroids")

            finalData = numpy.array (dataPoints)
            # print("numpy datapoints")

            # print(numpy.array(dataPoints))
            ax.scatter (finalData[:, 0], finalData[:, 1], finalData[:, 2], c=colors[centroidIndex], marker='o')
            ax.scatter (self.fuzzyCentres[centroidIndex][0], self.fuzzyCentres[centroidIndex][1],
                        self.fuzzyCentres[centroidIndex][2], c=colors[centroidIndex], marker='*');
            # ax.scatter(x, y, z, c='r', marker='o')

            ax.set_xlabel ('X Label')
            ax.set_ylabel ('Y Label')
            ax.set_zlabel ('Z Label')

        plt.show ()


fuzzyCMeansObj = FuzzyCMeans (inputDataFrame=points, numberOfClusters=2, fuzzyfier=2)
fuzzyCMeansObj.fuzzyCMeansCoreAlgorithm ()
print ("membership matrix", fuzzyCMeansObj.membershipMatrix)
print ("fuzzy centres", fuzzyCMeansObj.fuzzyCentres)
print ("input data", points)
print ("cluster membership", fuzzyCMeansObj.getClusters ())
clusteredDataInAnotherFormat = defaultdict (list)
# cluster_k = [fuzzyCMeansObj.inputData[fuzzyCMeansObj.clusterMembership[k]] for k in range(len(fuzzyCMeansObj.clusterMembership))]
# print("cluster k",cluster_k)

for hardClusterCounter in range (len (fuzzyCMeansObj.clusterMembership)):
    # print("cluster counter ",hardClusterCounter)
    # print(fuzzyCMeansObj.clusterMembership[hardClusterCounter])
    # print(fuzzyCMeansObj.inputData[hardClusterCounter])
    clusteredDataInAnotherFormat[fuzzyCMeansObj.clusterMembership[hardClusterCounter]].append (
        fuzzyCMeansObj.inputData[hardClusterCounter])
fuzzyCMeansObj.iterationResults=clusteredDataInAnotherFormat
fuzzyCMeansObj.clusteredData = [clusteredDataInAnotherFormat[k] for k in range (fuzzyCMeansObj.numberOfClusters)]
print (fuzzyCMeansObj.clusteredData)
print (fuzzyCMeansObj.daviesBouldinIndex ())
fuzzyCMeansObj.plotCMeans(fuzzyCMeansObj.iterationResults)
