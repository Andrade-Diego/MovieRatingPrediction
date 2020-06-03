import numpy as np
import csv
import random
import copy

NUMUSERS = 500
NUMMOVIES = 1616

def fileReader(path):
    dataList = []
    with open(path, newline = '') as dataFile:
        data = csv.reader(dataFile, delimiter='\t')
        dataList = [[int(dataRow[0]), int(dataRow[1]), int(dataRow[2])] for dataRow in data]
    return dataList


def makeVectors(data):
    #create an array where each each user is their own array
    numpyData = np.array(copy.deepcopy(data))
    numUsers = numpyData[:,0].max()
    #get number of movies by finding biggest movieId in the data
    numMovies = numpyData[:,1].max()
    #create an array where each movie is its own array
    usersAsVectors = np.zeros((NUMUSERS, NUMMOVIES), dtype = int)

    for dataPoint in numpyData:
        usersAsVectors[dataPoint[0] - 1, dataPoint[1] - 1] = dataPoint[2]
    return usersAsVectors

#testData is the testData file as a list
#vectors is a representation of the set of users where each
#      row is a user and each element in the row is a movie
#k is the number of neighbors being considered in the prediction
def kClosestNeighbors(testData, vectors, k):
    totalAvgError = 0

    #contains array where each row is a user and each element in that row are the
    #nearest neighbors to that user
    kClosestArray = []

    for testPoint in testData:
        vectorsWithDistance = list()
        #gives userID
        user = testPoint[0]
        index = 0
        for vector in vectors:
            #checks to make sure we only consider users who have seen the movie in question
            if not (vector[testPoint[1]-1] == 0):
                #populates list with the id of the compared user and their cosine similarity to user in question
                vectorsWithDistance.append((index, np.dot(vectors[user - 1], vector)/(np.linalg.norm(vectors[user - 1])*np.linalg.norm(vector))))
            #if the user hasn't seen the movie, make this similarity 0
            elif (vector[testPoint[1]-1] == 0): #or normUser == 0 or normVector == 0):
                vectorsWithDistance.append((index, 0))
            index += 1
        #sorts movies least to greatest
        vectorsWithDistanceSorted = sorted(vectorsWithDistance, key=lambda x: x[1])
        #gets the k most similar users
        kClosest = vectorsWithDistanceSorted[len(vectorsWithDistanceSorted)-k:len(vectorsWithDistanceSorted)]
        #average the k closest users' rating for the movie to predict rating for user in question
        predictedVal = 0
        for nearVal in kClosest:
            predictedVal += vectors[nearVal[0]][testPoint[1]-1]
        predictedVal = predictedVal / k

        #calculate sum for MSE
        totalAvgError += (predictedVal - testPoint[2])**2
    #divide by size of data for MSE
    totalAvgError = totalAvgError/len(testData)
    return (predictedVal, totalAvgError)

def getBestK(data):
    #randomize data
    rData = copy.deepcopy(data)
    random.shuffle(rData)
    #establish size of bins
    testLength = 500
    dLength = len(rData)
    binLength = int((dLength - testLength)/5)
    testingData = rData[(dLength - testLength):]
    differentKs = []
    leastMSE = (999999, None)
    loopNum = 0
    for k in range(1,6):
        MSE = 0
        for i in range(0,5):
            subset = makeVectors(rData[(i * binLength) : (i * binLength + binLength)])
            MSE += kClosestNeighbors(testingData, subset, k)[1]
            loopNum += 1
            print(loopNum/25*100, "% complete")
        MSE = MSE / 5
        print ("for k = ", k, " MSE is ", MSE)
        if MSE < leastMSE[0]:
            leastMSE = (MSE, k)
    return leastMSE

if __name__ == "__main__":
    #read data as a ndarray
    trainingData = fileReader("./u1-base.base")
    testingData = fileReader("./u1-test.test")
    userVectors = makeVectors(trainingData)

    kClosestNeighborsResult = kClosestNeighbors(testingData, userVectors, 4)
    print("when K is 3, MSE = ", kClosestNeighborsResult[1])
    print("now running Cross Validation")
    crossValidationData = copy.deepcopy(trainingData)
    k = getBestK(crossValidationData)
    print("Cross Validation concluded the optimal K is ", k[1])
    kClosestNeighborsResult = kClosestNeighbors(testingData, userVectors, k[1])
    print("MSE with optimal K is ", kClosestNeighborsResult[1])
