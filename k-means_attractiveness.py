import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D

from sklearn import tree
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


# fn = "AP_Analytics.csv"
fn = "Attractiveness_Analytics.csv"
writeFn = fn[:-4] + "_k-means_output.csv"
print(writeFn)

df = pd.read_csv(fn, delimiter=',', quotechar='"')
df = pd.get_dummies(df)

#Garbage initialization
X = df.ix[:,:]
y = df[:]

# Removing variable for ground truth depending on which file selected
if (fn is "AP_Analytics.csv"):
    X = df.ix[:, df.columns != 'Chance of Admit ']
    y = df['Chance of Admit ']
elif (fn is "Attractiveness_Analytics.csv"):
    X = df.ix[:, df.columns != 'Attractive']
    y = df['Attractive']

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.30, random_state=30)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

inits = ['k-means++', 'random']
n_inits = [5, 10, 15, 20]
max_iters = [100, 200, 300, 500]
norms = [True, False]

def runEverything():
    elbowMethod()

    data, x, y, normx, normy = preProcessData(fn)
    results = []
    # Loop over hyperparameters
    for init in inits:
        for n_init in n_inits:
            for max_iter in max_iters:
                for norm in norms:
                    # normx = []
                    # normy = []
                    err = classify(x, y, normx, normy, init, n_init, max_iter, norm)
                    print(init, n_init, max_iter, norm, ":", err)
                    results.append([init, n_init, max_iter, norm, err])

    file = open(writeFn, "w", newline="")
    csvW = csv.writer(file)
    csvW.writerow(["Init", "n_init", "max_iter", "norm", "error"])
    csvW.writerows(results)
    print("Open your file")
    # fig.show()

    plotAttractivenessClusters(x)
    print(exit)

def plotAttractivenessClusters(x):
    fig2 = plt.figure(2)
    ax2 = Axes3D(fig2)
    labels = np.array(y)
    X = np.array(x)
    ax2.scatter(X[:, 17], X[:, 30], X[:, 38], c=labels.astype(np.float), edgecolor='k')
    ax2.set_xlabel('Heavy Makeup')
    ax2.set_ylabel('Smiling')
    ax2.set_zlabel('Young')
    plt.savefig("Cluster_Scatterplot_" + fn[:-4])
    fig2.show()
    plt.show()


def elbowMethod():
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title("Elbow Method showing optimal k for " + fn[:-4])
    plt.savefig("Graph_Elbow Method_" + fn[:-4])
    plt.show()

# For normalizing data vector
def norm(vec):
    dist = max(vec) - min(vec)
    new = []
    for item in vec:
        if dist != 0:
            new.append(item/dist)
        else:
            new.append(0)
    return new

def preProcessData(fn):
    # move file into data list
    data = []
    f = open(fn)
    csvReader = csv.reader(f,delimiter=",")
    for row in csvReader:
        data.append(row)
    data = data[1:]
    i = 0
    for col0 in data[0]:
        try:
            float(col0)
            for row in data:
                row[i] = float(row[i])
        except:
            col = []
            for row in data:
                col.append(row[i])
            le = preprocessing.LabelEncoder()
            le.fit(col)
            a = le.transform(col)
            newCol = np.array(a).tolist()
            #print(newCol)
            for row in data:
                #print(i)
                row[i] = newCol[i]
        i = i + 1
    x = []
    y = []
    for row in data:
        x.append(row[:-1])
        y.append(row[-1])
    ## normalize every vector ###
    arrx = np.array(x)
    for i in range(len(x[0])):
        arrx[:,i] = np.array(norm(np.ndarray.tolist(arrx[:,i])))
    normx = np.ndarray.tolist(arrx)
    normy = norm(y)
    return (data,x,y,normx,normy)

def classify(x, y, normx, normy, init, n_init, max_iter, norm):
    if norm:
        x = normx
        y = normy

    fitter = KMeans(n_clusters=2, init=init, n_init=n_init, max_iter=max_iter).fit(x)
    X = np.array(x)
    labels = fitter.labels_
    # print("CLASSIFYING");
    # print("clusters centers", fitter.cluster_centers_)
    # print("differences ", fitter.cluster_centers_[1] - fitter.cluster_centers_[0])
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float), edgecolor='k')
    return min(sum(abs(fitter.predict(x) - y)), sum(abs((1 - fitter.predict(x)) - y))) / len(x)


runEverything()














