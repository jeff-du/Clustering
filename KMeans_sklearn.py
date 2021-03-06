#######################################
# coding=utf8
# Author: Dave
# Created Time : 2017年8月20日
# Last Modified: 2017年8月20日
# Description:
# using sklearn 
# Distance:only eul
####################################### 

import sklearn.cluster as cluster
import numpy as np

def KMeans(data, n_clusters):
    kmeans = cluster.KMeans(n_clusters, init='random')
    kmeans.fit(data)
    clusteringResult = kmeans.labels_.astype(np.int)
    return clusteringResult

if __name__ == '__main__':
    data = [[1,10], 
            [1,20], 
            [2,5], 
            [4,2], 
            [8,2]]
    k = 2
    clusterAssment = KMeans(data, k)
    print(clusterAssment)