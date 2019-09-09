import numpy
#import itertools from izip
#points = numpy.array([[1, 2, 2], [3, 2, 2], [6, 5, 5], [6,5,1]])
points = numpy.array([[1, 3], [1.5,3.2], [1.3,2.8], [3,1]])

class FuzzyCMeans:
    def __init__(self,inputDataFrame, numberOfClusters, fuzzyfier):
        self.inputData = inputDataFrame
        self.numberOfClusters = numberOfClusters
        self.numberOfInstances, self.numberOfFeatures = self.inputData.shape
        self.MAX_ITER = 1
        self.membershipMatrix=[]
        #default fuzzifier value is 2
        self.fuzzifier = fuzzyfier
        self.fuzzyCentres =[]
        print("input data",self.inputData)
        print("number of cluster",self.numberOfClusters)
        print("number of instances",self.numberOfInstances)
        print("number of features",self.numberOfFeatures)
        print("max iter",self.MAX_ITER)

    def generateRandomInitialMembershipMatrix(self):

        for instanceCounter in range(self.numberOfInstances):
            #this loop will run till the number of instances are there in input generate those many random splits
            #this will also generate according to the number of features
           # randomNumberAsPerNumberOfFeatures = numpy.random.normal(size=self.numberOfFeatures)
            randomNumberAsPerNumberOfFeatures = numpy.random.random(size=self.numberOfFeatures)
            #print("random matrix",randomNumberAsPerNumberOfFeatures)
            randomNumberAsPerNumberOfFeatures /= randomNumberAsPerNumberOfFeatures.sum()
            #using divide by sum technique to get the matrix normalized to the sum 1

            self.membershipMatrix.append(randomNumberAsPerNumberOfFeatures)
            print("Random Membership matrix",self.membershipMatrix)

    def calculateCentroidsBasedOnMembershipMatrix(self):
       # [list(a) for a in zip([1,2,3], [4,5,6], [7,8,9])]
       # memberShipValuesIteratedInRowWise = list(itertools(*self.membershipMatrix))
        memberShipValuesConvertedToTuplesColumnWise = zip(*self.membershipMatrix)
        #print(list(memberShipValuesConvertedToTuplesColumnWise))
        memberShipValuesConvertedToListColumnWise = list(memberShipValuesConvertedToTuplesColumnWise)

        for clusterCounter in range(self.numberOfClusters):
            fuzzyCentroidCoordinates = []
            #print("used for fuzzy centroid", clusterCounter,memberShipValuesConvertedToListColumnWise[clusterCounter])
            print(memberShipValuesConvertedToListColumnWise[clusterCounter])

            squaresOfDenominator = numpy.square(memberShipValuesConvertedToListColumnWise[clusterCounter])
            #denominator is sum of squares
            denominator = numpy.sum(squaresOfDenominator)

            print (denominator)
            listOfDataZippedBasedOnColumn = list(zip(*self.inputData))
            print("input data ",listOfDataZippedBasedOnColumn[clusterCounter])
            print("numerator")

            for dataColumnCounter in range(self.numberOfFeatures):
                centreCordinates = numpy.sum(numpy.multiply(squaresOfDenominator,listOfDataZippedBasedOnColumn[dataColumnCounter]))/denominator
                print("centre coordinate",centreCordinates)
                fuzzyCentroidCoordinates.append(centreCordinates)
            self.fuzzyCentres.append(fuzzyCentroidCoordinates)
            print("fuzzy centroids",self.fuzzyCentres)

    def fuzzyCMeansCoreAlgorithm(self):
        self.generateRandomInitialMembershipMatrix()
        for iteration in range(self.MAX_ITER):
            self.calculateCentroidsBasedOnMembershipMatrix()



fuzzyCMeansObj = FuzzyCMeans(inputDataFrame=points, numberOfClusters=2,fuzzyfier=2)
fuzzyCMeansObj.fuzzyCMeansCoreAlgorithm()

