#######################################
#coding=utf8
#Author: Dave
#Created Time : 2017年7月4日
#Last Modified: 2017年7月4日
#Description:
#Different distances : euclidean, minkowski, cosine, correlation, etc
####################################### 

import numpy as np 

# calc distance of two clusters
def DistCosine(cluster1, cluster2):
    dotClusters = np.dot(cluster1, cluster2)
    sum1, sum2 = 0, 0
    for val in cluster1:
        sum1 += val ** 2
    for val in cluster2:
        sum2 += val ** 2
    
    if sum1 != 0 and sum2 != 0 :
        return 1 - dotClusters / (sum1 * sum2) ** 0.5
    else:
        return None


def DistMinkowski(point1, point2, p):
    dim = len(point1)
    disSum = 0
    for i in range(dim):
        disSum += np.power(point1[i] - point2[i], p)
    ret = np.power(disSum, 1/p)
    return ret


def DistCorrelation(point1, point2):
    avg1 = np.mean(point1)
    avg2 = np.mean(point2)

    p1 = [val - avg1 for index, val in enumerate(point1)] 
    p2 = [val - avg2 for index, val in enumerate(point2)]

    dist = DistCosine(p1, p2)
    return dist

# calc distance matrix of all the items
def CalcDistMatrix(data, distMeas = DistCorrelation):
    n = len(data)
    distMatrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j :
                distMatrix[i][j] = distMeas(data[i], data[j])
    return distMatrix


if __name__ == '__main__' :
    p1 = [1, 2, 3, 4]
    p2 = [100, 4, 3, 1]
    dist1 = DistCorrelation(p1, p2)
    dist2 = DistCosine(p1, p2)
    print(dist1)
    print(dist2)
