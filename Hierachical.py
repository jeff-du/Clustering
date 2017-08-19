#################################
#coding=utf8
#Author: Dave
#Created Time : 2017年6月29日
#Last Modified: 2017年6月29日
#Description:
#使用sklearn的层次聚类方法，具体的是ward_tree方法 
#测试数据采用iris数据，sklearn.datasets.load_iris() 
#但是发现sklearn.cluster.ward_tree方法没有看懂,实验stackoverflow里面的代码 
#http://stackoverflow.com/questions/9873840/cant-get-scipy-hierarchical-clustering-to-work
################################# 

import scipy.cluster.hierarchy as hcluster


#scipy.cluster.hierarchy.fclusterdata
#(X, t, criterion='inconsistent', metric='euclidean', depth=2, method='single', R=None)
#Parameters:
#X : (N, M) ndarray N by M data matrix with N observations in M dimensions.
#t : float The threshold to apply when forming flat clusters.
#criterion : str, optional Specifies the criterion for forming flat clusters. 
#            Valid values are ‘inconsistent’ (default), ‘distance’, 
#            or ‘maxclust’ cluster formation algorithms. 
#metric : str, optional The distance metric for calculating pairwise distances: euclidean,cosine,correlation
#depth : int, optional The maximum depth for the inconsistency calculation. 
#method : str, optional The linkage method to use (single, complete, average, weighted, median centroid, ward). 
#R : ndarray, optional The inconsistency matrix. 
#Returns:
#fclusterdata : ndarray A vector of length n. T[i] is the flat cluster number to which original observation i belongs.

def HCluster(data, k, distMeas='correlation'):
    criterion='maxclust'
    clusteringResult = hcluster.fclusterdata(data, t=k, criterion=criterion, metric=distMeas)
    return clusteringResult


if __name__ == '__main__' :
    data = [[1,10], [1,20], [2,5], [4,2], [8,2]]
    k = 2
    clusterAssment = HCluster(data, k, distMeas='correlation')
    print(clusterAssment)

