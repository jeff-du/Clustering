#######################################
#coding=utf8
#Author: Dave
#Created Time : 2017年7月4日
#Last Modified: 2017年7月4日
#Description:
#Spectral Clustering
####################################### 

import sklearn.cluster as cluster
import numpy as np

def AffinityPropagation(data):
    affinityPropagation = cluster.AffinityPropagation(damping=.9,
                                                       preference=-200)
    affinityPropagation.fit(data)
    clusteringResult = affinityPropagation.labels_.astype(np.int)
    
    return clusteringResult

if __name__ == '__main__':
    data = [[1,10], 
            [1,20], 
            [2,5], 
            [4,2], 
            [8,2]]
    k = 2
    calcTimes = 100
    clusterAssment = AffinityPropagation(data)
    print(clusterAssment)

