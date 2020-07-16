import numpy as np
import csv
import math
import copy
import random

def read_csv(filename):
    data = []
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def str_to_float(dataset):
    for i in range(1,len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = float(dataset[i][j])

def minMax(dataset):
    del dataset[0]
    minMaxData = []
    for i in range(len(dataset[0]) - 1):
        col = [row[i] for row in dataset]
        minValue = min(col)
        maxValue = max(col)
        minMaxData.append( [ minValue, maxValue ] )
    return minMaxData

def normalization(dataset, minMaxData):
    for i in range(len(dataset)):
        for j in range(len( dataset[i]) - 1):
            numer = dataset[i][j] - minMaxData[j][0]
            denom = minMaxData[j][1] - minMaxData[j][0]
            dataset[i][j] = numer / denom

def crossValidation(dataset, k=5):
    dataset_copy = copy.deepcopy(dataset)
    fold_size = len(dataset) // k
    folds = []
    for i in range(k):
        fold = []
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append( dataset_copy.pop(index) )
        folds.append( fold )
    return folds

def predict(coef, row):
    y_pred = coef[0]
    for i in range(len(row) - 1):
        y_pred += coef[i+1] * row[i]
    return 1 / (1 + math.exp(-y_pred) )

def accuracyScore(actual, predicted):
    score = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            score += 1
    return score / len(actual) * 100

def evaluateAlgorithm(dataset, epochs, learning_rate, coef):
    scores = []
    folds = crossValidation(dataset)
    for fold in folds:
        train = list(folds)
        train.remove(fold)
        predictions = logisticRegression(train, fold, epochs, learning_rate, coef)
        actual = [row[-1] for row in fold]
        score = accuracyScore(actual, predictions)
        scores.append(score)
    return scores

def stochasticGradient(dataset, epochs, learning_rate, coef):
    for epoch in range(epochs):
        for i in range( len(dataset) // 2 ):
            index = random.randrange(len(dataset))
            row = dataset[index]
            y_pred = predict(coef, row)
            loss = y_pred - row[-1]
            coef[0] = coef[0] - learning_rate * loss
            for j in range( len(row) - 1 ):
                coef[j+1] = coef[j+1] - learning_rate * loss * row[j]
    return coef

def logisticRegression(train, test, epochs, learning_rate, coef):
    train = np.array(train)
    t_shape = train.shape
    train = train.reshape( t_shape[0] * t_shape[1], t_shape[2] )
    predictions = []
    coef = stochasticGradient(train, epochs, learning_rate, coef)
    for row in test:
        y_pred = predict(coef, row)
        predictions.append( round(y_pred) )
    return predictions

dataset = read_csv("dataset/diabetes.csv")
str_to_float(dataset)
minMaxData = minMax(dataset)
normalization(dataset, minMaxData)
coef = np.zeros(len(dataset[0]))
epochs = 1000
learning_rate = 0.001
scores = evaluateAlgorithm(dataset, epochs, learning_rate, coef)
print("All scores : ", scores)
print("Final score : ", sum(scores)/len(scores))
