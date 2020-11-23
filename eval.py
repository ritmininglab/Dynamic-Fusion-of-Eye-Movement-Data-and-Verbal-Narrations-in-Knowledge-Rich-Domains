from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

maxRun = 100
accuracy = np.zeros((maxRun))
split = 0.2

rawfeature = 789
featureselection = np.zeros((maxRun,rawfeature))

model = LogisticRegression(
    max_iter = 500,
    penalty='l1',
    solver='liblinear',
    C = 5.0)

task = 1
datasetidx = 1

filename = 'input'+str(datasetidx)+'.csv'
with open(filename) as f:
    ncols = len(f.readline().split(','))
Xraw = np.loadtxt(open(filename,"rb"),delimiter=",",skiprows=0,usecols=range(3,ncols))

if task==1:
    Y = np.loadtxt(open('cat'+str(datasetidx)+'.csv',"rb"),delimiter=",",skiprows=0,usecols=range(2,3))
elif task==2:
    Y = np.loadtxt(open('cor'+str(datasetidx)+'.csv',"rb"),delimiter=",",skiprows=0,usecols=range(2,3))


X = Xraw

print("Using all fields:")
for idxRun in range(0,maxRun):
    #np.random.seed(idxRun)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split, 
                                                        random_state=idxRun)
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    accuracy[idxRun] = np.sum(model_pred==Y_test)/ model_pred.shape[0]
    
print("Average Accuracy = ", np.mean(accuracy))
