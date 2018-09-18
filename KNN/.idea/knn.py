# -*- coding: utf-8 -*-
from numpy import *
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from itertools import chain

def createDataSet():
    dataset = read_csv('Date/ManLongSleeve.csv', header=0, index_col=None, usecols=[0, 1, 2, 3, 4, 5, 6])
    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    group = scaler.fit_transform(values)

    result = read_csv('Date/ManLongSleeve.csv', header=0, index_col=None, usecols=[7])
    values =  result.values
    labels = list(chain.from_iterable(values))

    return group, labels


def KNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]

    diff = tile(newInput, (numSamples, 1)) - dataSet
    squreDiff = diff ** 2
    squreDist = sum(squreDiff, axis=1)
    distance = squreDist ** 0.5

    sortedDistIndices = argsort(distance)

    classCount = {}
    for i in range(k):

        voteLabel = labels[sortedDistIndices[i]]

        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    for k, v in classCount.items():
        if v > maxCount:
            maxCount = v
            maxIndex = k

    return maxIndex


dataSet, labels = createDataSet()

testX = array([0.6329112,  0.75289583, 0.68539286, 0.71906376, 0.74193525, 0.6271181, 0.536232 ])
k = 3
outputLabel = KNNClassify(testX, dataSet, labels, 3)

print("Your input is:", testX, "and classified to class: ", outputLabel)

testX = array( [0.7974682, 0.92278004, 0.8426962, 0.826087, 0.881721, 0.7288141, 0.84057903])
k = 3
outputLabel = KNNClassify(testX, dataSet, labels, 3)

print("Your input is:", testX, "and classified to class: ", outputLabel)