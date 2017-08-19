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
import array
import scipy.spatial.distance as dist
import scipy

def SpectralClustering(data, nCluster, distMeas='correlation'):

    #distMatrix = np.array(dist.CalcDistMatrix(data, distMeas))
    # using Gaussian kernel to transform in a similarity matrix
    # n = len(distMatrix)
    # simiMatrix = [[0 for i in range(n)] for j in range(n)]
    # for i in range(n):
    #     for j in range(n):
    #         simiMatrix[i][j] = np.exp(- distMatrix[i][j] ** 2 / (2. * sigma ** 2))

    sigma = 1
    distMatrix = scipy.spatial.distance.squareform(dist.pdist(data, distMeas))
    simiArray = scipy.exp(-distMatrix / (2 * sigma ** 2))
    spectral = cluster.SpectralClustering(n_clusters=nCluster,
                                          eigen_solver='arpack',
                                          affinity='precomputed')
    spectral.fit(simiArray)
    if hasattr(spectral, 'labels_'):
        clusteringResult = spectral.labels_.astype(np.int)
    else:
        clusteringResult = spectral.predict(data)
    
    return clusteringResult

if __name__ == '__main__' :
    data = [[1,10], [1,20], [2,5], [4,2]]
    k = 2
    clusterAssment = SpectralClustering(data, k)
    print(clusterAssment)
