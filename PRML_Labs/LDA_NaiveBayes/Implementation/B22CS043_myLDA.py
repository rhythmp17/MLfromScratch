import csv
import numpy as np
import pandas as pd

def ComputeMeanDiff(X):
    label_0_points = X[X[:, 2] == 0][:, :2]
    label_1_points = X[X[:, 2] == 1][:, :2]
    # Calculate mean of coordinates for each label
    mean_label_0 = np.mean(label_0_points, axis=0)
    mean_label_1 = np.mean(label_1_points, axis=0)
    type(mean_label_0)
    mean0 = mean_label_0.reshape(2,1)
    mean1 = mean_label_1.reshape(2,1)
    x = mean1 - mean0
    return x

def getMean(X,label):
    label_points = X[X[:, 2] == label][:, :2]

    # Calculate mean of coordinates for each label
    mean_label = np.mean(label_points, axis=0)
    
    return mean_label

def ComputeSW(X):
    s1 = np.array([[0,0],[0,0]],dtype = np.float64)
    s2 = np.array([[0,0],[0,0]],dtype = np.float64)
    tot1 = 0
    tot2 = 0
    mean1 = getMean(X,label=0)
    mean2 = getMean(X,label=1)
    for i in range(X.shape[0]):
        if float(X[i][2]) == 0:
            x = np.array([[float(X[i][0]) - mean1[0],float(X[i][1]) - mean1[1]]])
            xt = np.transpose(x)
            s1 += np.dot(xt,x)
            tot1 += 1
        else:
            x = np.array([[float(X[i][0]) - mean2[0],float(X[i][1]) - mean2[1]]])
            xt = np.transpose(x)
            s2 += np.dot(xt,x)
            tot2 += 1

    return s1/tot1 + s2/tot2

def ComputeSB(X):
    x = ComputeMeanDiff(X)
    xt = np.transpose(x)
    sb = np.dot(x,xt)

    return sb

def GetLDAProjectionVector(X):
    sw = ComputeSW(X)
    sw_i = np.linalg.inv(sw)
    mean_diff = ComputeMeanDiff(X)
    v = np.dot(sw_i,mean_diff)

    return v

def project(x,y,w):
    coordinate = np.array([[x,y]])
    coordinate = coordinate.reshape(2,1)
    w_t = np.transpose(w)
    # print(coordinate.shape)
    projection = np.dot(w_t,coordinate)

    return projection[0][0]



url = "https://raw.githubusercontent.com/anandmishra22/PRML-Spring-2023/main/programmingAssignment/PA-4/data.csv"
df = pd.read_csv(url,header = None)
data = df.values
data = data.astype(np.float64)
print(data)
print(data.shape)
# X Contains m samples each of formate (x,y) and class label 0.0 or 1.0

opt=int(input("Input your option (1-5): "))
if opt == 1:
    meanDiff=ComputeMeanDiff(data)
    print(meanDiff)
elif opt == 2:
    SW=ComputeSW(data)
    print(SW)
elif opt == 3:
    SB=ComputeSB(data)
    print(SB)
elif opt == 4:
    w=GetLDAProjectionVector(data)
    print(w)
elif opt == 5:
    x=int(input("Input x dimension of a 2-dimensional point :"))
    y=int(input("Input y dimension of a 2-dimensional point:"))
    w=GetLDAProjectionVector(data)
    print(project(x,y,w))
