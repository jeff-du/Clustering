#######################################
#coding=utf8
#Author: Dave
#Created Time : 2017年7月4日
#Last Modified: 2017年7月4日
#Description:
#DBSCAN
####################################### 

import sklearn.cluster as cluster
import numpy as np

def DBSCAN(data, eps):
    dbscan = cluster.DBSCAN(eps)
    dbscan.fit(data)
    if hasattr(dbscan, 'labels_'):
        clusteringResult = dbscan.labels_.astype(np.int)
    else:
        clusteringResult = dbscan.predict(data)
    
    return clusteringResult

if __name__ == '__main__' :
    data = [[1,100], [1,200], [2,5], [4,2], [8,2]]
    eps = 15
    clusterAssment = DBSCAN(data, eps)
    print(clusterAssment)