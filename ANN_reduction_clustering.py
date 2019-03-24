import csv
from sklearn import tree
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from math import ceil
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import kurtosis
import time



fn = "adult data.csv"
writeFn = fn[:-4] + "_ann_rc_output.csv"
print(writeFn)

startTrainPerc = .6
trainPercInc = 1
endTrainPerc = .6


activations = ['logistic']

solvers = ['lbfgs']

numLayers = [1]

sizeLayers = [100]

layerSlopes = [1]

maxIters = [2000]

cvFolds = 3

algs = ['none','pca','ica','rp','vtresh']

clusts = ['none','k-means','EM']

def runEverything():
    data = preProcessData(fn)
    results = []
    perc = startTrainPerc
    while perc <= endTrainPerc:
        for activation in activations:
            for solver in solvers:
                    for numLayer in numLayers:
                        for sizeLayer in sizeLayers:
                            for layerSlope in layerSlopes:
                                for maxIter in maxIters:
                                    for alg in algs:
                                        for clust in clusts:
                                            start = time.time()
                                            result = [perc,activation,solver,numLayer,sizeLayer,layerSlope,maxIter,alg,clust]
                                            data = transform(data,alg,clust)
                                            trainError,testError,sumError,diffError = genDecTree(data,perc,activation,solver,numLayer,sizeLayer,layerSlope,maxIter)
                                            valError = crossValTree(data,perc,cvFolds,activation,solver,numLayer,sizeLayer,layerSlope,maxIter)
                                            result.extend([trainError,valError,testError,sumError,diffError,time.time()-start])
                                            results.append(result)
                                            print(perc,activation,solver,numLayer,sizeLayer,layerSlope,maxIter)
                    print("Progress")
            perc = perc + trainPercInc


    file = open(writeFn,"w",newline="")
    csvW = csv.writer(file)
    csvW.writerow(["Training %","Activation F","Solver","# Layers","Max Layer Size","Layer Slope","Max_Iter","alg","clust","Training Error %","Validation Error %","Testing Error %","Total Error %","Diff in Error %","s"])
    csvW.writerows(results)
    print("Open your file")

def norm(vec):
    dist = max(vec) - min(vec)
    new = []
    for item in vec:
        if dist != 0:
            new.append(item/dist)
        else:
            new.append(0)
    return new
    
def transform(data,alg,clust):
    a = np.array(data)
    x = a[:,0:-1]
    y = a[:,-1]
    if alg == 'pca':
        pca = PCA(n_components=6,whiten=True)
        x = pca.fit(x).transform(x)
        print(pca.components_)
        print(pca.explained_variance_ratio_)
    if alg == 'ica':
        kur0 = sum(kurtosis(x))
        ica = FastICA(n_components=3,whiten=False,algorithm="parallel")
        ica = ica.fit(x)
        x = ica.transform(x)
        print("kurtosis: ",sum(kurtosis(x))-kur0)
    if alg == 'rp':
        rp = GaussianRandomProjection(n_components=1)
        rp = rp.fit(x)
        x = rp.transform(x)
        print(rp.components_)
    if alg == 'vtresh':
        kb = VarianceThreshold(threshold=.04)
        x = kb.fit_transform(x)
        print(kb.variances_)

    if clust == 'k-means':
        fitter = KMeans(n_clusters=2,init='k-means++',n_init=10,max_iter=200).fit(x)
        c = np.array(fitter.predict(x))
        x = np.column_stack((x,c))
    if clust == 'EM':
        fitter = GaussianMixture(n_components=2,covariance_type="full").fit(x)
        c = np.array(fitter.predict(x))
        x = np.column_stack((x,c))
        
    data = np.column_stack((x,y))
    return data
                   
        
        
def preProcessData(fn):
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

    return data

def genDecTree(data,perc,theActivation,aSolver,numLayer,sizeLayer,layerSlope,maxIter):
    trainPerc = perc
    train = []
    x = []
    y = []
    c = 0
    while c <=trainPerc*(len(data)):
        train.append(data[c])
        x.append(data[c][:-1])
        y.append(data[c][-1])
        c = c + 1

    #print("Train set ends on row " + str(c))
    test = []
    tx = []
    ty = []
    while c < len(data):
        test.append(data[c])
        tx.append(data[c][:-1])
        ty.append(data[c][-1])
        c = c + 1

    layers = []
    for i in range(0,numLayer):
        layers.append(ceil(sizeLayer/(i+layerSlope)))
    layers = tuple(layers)

    clf = MLPClassifier(solver=aSolver,max_iter=maxIter,hidden_layer_sizes=layers,activation=theActivation)
    clf = clf.fit(x,y)
    predTrainY = clf.predict(x)
    predTestY = clf.predict(tx)

    errorTrain = sum(abs(y - predTrainY))/len(y)
    errorTest = sum(abs(ty - predTestY))/len(ty)
    sumError = (sum(abs(y - predTrainY)) + sum(abs(ty - predTestY)))/(len(y)+len(ty))
    return(errorTrain,errorTest,sumError,abs(errorTrain-errorTest))

    del clf



def crossValTree(data,perc,nFolds,theActivation,aSolver,numLayer,sizeLayer,layerSlope,maxIter):
    trainPerc = perc
    work = []
    x = []
    y = []
    c = 0
    while c <=trainPerc*(len(data)):
        work.append(data[c])
        x.append(data[c][:-1])
        y.append(data[c][-1])
        c = c + 1

    numRows = c - 1
    sizeCut = numRows//nFolds

    layers = []
    for i in range(0,numLayer):
        layers.append(ceil(sizeLayer/(i+layerSlope)))
    layers = tuple(layers)

    cuts = []
    xcuts = []
    ycuts = []
    for i in range(0,(numRows//sizeCut)):
        xcut = x[(i*sizeCut):(i+1)*sizeCut]
        ycut = y[(i*sizeCut):(i+1)*sizeCut]
        xcuts.append(xcut)
        ycuts.append(ycut)
        #print(xcut,"\n\n\n")

    errors = []
    for i in range(0,(numRows//sizeCut)):
        testx = xcuts[i]
        testy = ycuts[i]
        trainx = []
        trainy = []
        for j in range(0,(numRows//sizeCut)):
            if j!=i:
                trainx.extend(xcuts[j])
                trainy.extend(ycuts[j])
        clf = MLPClassifier(solver=aSolver,max_iter=maxIter,hidden_layer_sizes=layers,activation=theActivation)
        clf = clf.fit(trainx,trainy)
        #print(xcuts)
        #print(testx)
        predTestY = clf.predict(testx)
        errorTest = sum(abs(testy - predTestY))/len(testy)
        errors.append(errorTest)

    return (sum(errors)/len(errors))
    
runEverything()
