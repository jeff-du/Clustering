#################################
#coding=utf8
#Author: Dave
#Created Time : 2017年6月29日
#Last Modified: 2017年6月29日
#Description:
#k-means
################################# 

from numpy import *
import time
import scipy.spatial.distance as dist
import sklearn.cluster.k_means_ as km


 
 #随机生成初始的质心（ng的课说的初始方式是随机选K个点）    
def randCent(dataSet, k):
    n = len(dataSet)
    m = len(dataSet[0])
    centroids = []
    while k > 0 :
        index = int(random.uniform(0, n))  # 样本集随机挑一个，作为初始质心
        if dataSet[index] not in centroids :
            centroids.append(dataSet[index])
            k -= 1
    return centroids

     
def KMeans(dataSet, k, distMeas='correlation', times = 30):
    calcTime = 0
    n = len(dataSet)
    clusterAssment = [-1 for i in range(n)]
    centroids = randCent(dataSet, k)
    clusterChanged = True
    while clusterChanged and calcTime < times:
        calcTime += 1
        print(calcTime)
        clusterChanged = False
        
        distMatrix = dist.cdist(array(dataSet), array(centroids), distMeas)
        for index, vec in enumerate(distMatrix):
            minIndex = list(vec).index(min(vec))
            if clusterAssment[index] != minIndex:
                clusterChanged = True 
                clusterAssment[index] = minIndex

        #print (centroids)
        for cent in range(k):#recalculate centroids
            #get all the point in this cluster
            ptsInClust = [dataSet[index] for index, val in enumerate(clusterAssment) if val == cent]
            if len(ptsInClust) > 0 :
                centroids[cent] = mean(ptsInClust, axis=0) #assign centroid to mean
    return clusterAssment


if __name__ == '__main__' :
    data = [[1,10,1,1,1,1,1,1,1,1], 
            [1,20,1,1,1,1,1,1,1,1], 
            [2,5,1,1,1,1,1,1,1,1], 
            [4,2,1,1,1,1,1,1,1,1], 
            [8,2,1,1,1,1,1,1,1,1]]
    k = 2
    calcTimes = 100
    clusterAssment = KMeans(data, k, times=calcTimes)
    print(clusterAssment)



