import numpy
import math
#import itertools from izip
#points = numpy.array([[1, 2, 2], [3, 2, 2], [6, 5, 5], [6,5,1]])
points = numpy.array([[1, 3], [1.5,3.2], [1.3,2.8], [3,1]])

class FuzzyCMeans:
    def __init__(self,inputDataFrame, numberOfClusters, fuzzyfier):
        self.inputData = inputDataFrame
        self.inputDataCopy = self.inputData
        self.numberOfClusters = numberOfClusters
        self.numberOfInstances, self.numberOfFeatures = self.inputData.shape
        self.MAX_ITER = 1000
        self.membershipMatrix= []
        self.clusterMembership=[]
        #default fuzzifier value is 2
        self.fuzzifier = fuzzyfier
        self.fuzzyCentres =[]
        #print("input data",self.inputData)
        #print("number of cluster",self.numberOfClusters)
        #print("number of instances",self.numberOfInstances)
        #print("number of features",self.numberOfFeatures)
        #print("max iter",self.MAX_ITER)

    def generateRandomInitialMembershipMatrix(self):

        for instanceCounter in range(self.numberOfInstances):
            #this loop will run till the number of instances are there in input generate those many random splits
            #this will also generate according to the number of features
           # randomNumberAsPerNumberOfFeatures = numpy.random.normal(size=self.numberOfFeatures)
            randomNumberAsPerNumberOfFeatures = numpy.random.random(size=self.numberOfClusters)
            #print("random matrix",randomNumberAsPerNumberOfFeatures)
            randomNumberAsPerNumberOfFeatures /= randomNumberAsPerNumberOfFeatures.sum()
            #using divide by sum technique to get the matrix normalized to the sum 1

            self.membershipMatrix.append(randomNumberAsPerNumberOfFeatures)
            #print("Random Membership matrix",self.membershipMatrix)

    def calculateCentroidsBasedOnMembershipMatrix(self):
       # [list(a) for a in zip([1,2,3], [4,5,6], [7,8,9])]
       # memberShipValuesIteratedInRowWise = list(itertools(*self.membershipMatrix))
        self.fuzzyCentres = []
        memberShipValuesConvertedToTuplesColumnWise = zip(*self.membershipMatrix)
        #print(list(memberShipValuesConvertedToTuplesColumnWise))
        memberShipValuesConvertedToListColumnWise = list(memberShipValuesConvertedToTuplesColumnWise)
        #print(memberShipValuesConvertedToListColumnWise)
        for clusterCounter in range(self.numberOfClusters):
            fuzzyCentroidCoordinates = []
            #print("used for fuzzy centroid", clusterCounter,memberShipValuesConvertedToListColumnWise[clusterCounter])
            #print(memberShipValuesConvertedToListColumnWise[clusterCounter])

            squaresOfDenominator = numpy.square(memberShipValuesConvertedToListColumnWise[clusterCounter])
            #denominator is sum of squares
            denominator = numpy.sum(squaresOfDenominator)

            #print (denominator)

            listOfDataZippedBasedOnColumn = list(zip(*self.inputDataCopy))
            #print("input data ",listOfDataZippedBasedOnColumn[clusterCounter])
            #print("numerator")

            for dataColumnCounter in range(self.numberOfFeatures):
                centreCordinates = numpy.sum(numpy.multiply(squaresOfDenominator,listOfDataZippedBasedOnColumn[dataColumnCounter]))/denominator
                #print("centre coordinate ",clusterCounter,"",centreCordinates)
                fuzzyCentroidCoordinates.append(centreCordinates)
            self.fuzzyCentres.append(fuzzyCentroidCoordinates)
           # print("fuzzy centroids",self.fuzzyCentres)

    def fuzzyCMeansCoreAlgorithm(self):
        self.generateRandomInitialMembershipMatrix()
        for iteration in range(self.MAX_ITER):
            self.calculateCentroidsBasedOnMembershipMatrix()
            self.updateMembershipValue()
        #print ("final membership matrix",self.membershipMatrix)
        #print("fuzzy centroids",self.fuzzyCentres)
    def getClusters(self):
        #self.clusterMembership =[]
        for instanceCounter in range(self.numberOfInstances):

        #index = numpy.unravel_index(numpy.argmax(numpy.array(self.membershipMatrix), axis=None), numpy.array(self.membershipMatrix).shape)
        #self.clusterMembership.append(index)
        #print(self.membershipMatrix)
            a = numpy.array(self.membershipMatrix[instanceCounter])  # Can be of any shape\

            #i,j = numpy.unravel_index(a.argmax(), a.shape)
           # print(a.argmax())
            self.clusterMembership.append(a.argmax())
        #index =a.argmax(axis=0)
       # print(index)
        #return index
        return self.clusterMembership
    def updateMembershipValue(self):
        #print("membership value",self.membershipMatrix)
        for dataCounter in range(self.numberOfInstances):
            inputDataPoint = self.inputData[dataCounter]
            #print("inputdata point" , inputDataPoint)
            distances = []
            for clusterCount in range(self.numberOfClusters):
                distances.append(numpy.linalg.norm(self.fuzzyCentres[clusterCount] - inputDataPoint))
            #print("distances", distances)
            for clusterCount in range(self.numberOfClusters):
                numerator = sum([math.pow(float(distances[clusterCount] / distances[temp]), 2) for temp in range(self.numberOfClusters)])
                self.membershipMatrix[dataCounter][clusterCount] = float(1 / numerator)




fuzzyCMeansObj = FuzzyCMeans(inputDataFrame=points, numberOfClusters=2,fuzzyfier=2)
fuzzyCMeansObj.fuzzyCMeansCoreAlgorithm()
print("membership matrix",fuzzyCMeansObj.membershipMatrix)
print("fuzzy centres",fuzzyCMeansObj.fuzzyCentres)
print("cluster membership",fuzzyCMeansObj.getClusters())


