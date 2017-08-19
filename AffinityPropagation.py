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
    if hasattr(algorithm, 'labels_'):
        clusteringResult = affinityPropagation.labels_.astype(np.int)
    else:
        clusteringResult = affinityPropagation.predict(data)
    
    return clusteringResult