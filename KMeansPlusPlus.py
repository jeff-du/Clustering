#################################
#coding=utf8
#Author: Dave
#Created Time : 2017年7月04日
#Last Modified: 2017年6月04日
#Description:
#k-means-plus-plus
#k: count of clusters
################################# 

import sys
import copy
import random
import numpy as np
import scipy.spatial.distance as dist

def InitClusterCenters(points, k, distMeas='correlation'):
    n = len(points)
    clusterCenters = [[] for i in range(k)]
    
    # select the first point randomly
    # using copy 
    randSeq = random.randint(0, n-1)
    clusterCenters[0] = copy.copy(points[randSeq])

    # choose cluster center by nearest points
    distCenters = [0.0 for i in range(n)]
    for i in range(1, k):
        distMatrix = dist.cdist(np.array(points), np.array(clusterCenters[:i]), distMeas)
        maxIndex = np.where(distMatrix == np.max(distMatrix))[0][0]
        clusterCenters[i] = copy.copy(points[maxIndex])

        # 直接选择最大值 容易收到噪声的影响 因此应该选择较大值
        # 如何选择较大值： 
        # 把集合D中的每个元素D(x)想象为一根线L(x)，线的长度就是元素的值
        # 将这些线依次按照L(1)、L(2)、……、L(n)的顺序连接起来
        # 组成长线L。L(1)、L(2)、……、L(n)称为L的子线
        # 根据概率的相关知识，如果我们在L上随机选择一个点
        # 那么这个点所在的子线很有可能是比较长的子
        # 而这个子线对应的数据点就可以作为种子点
        

        # sumDist *= random.random()
        # for j, dist in enumerate(distCenters):
        #     sumDist -= dist
        #     if sumDist > 0 :
        #         continue
        #     clusterCenters[i] = copy(points[j])
        #     break

        # 距离最小值
        
    
    return clusterCenters



def KMeans(points, k, clusterCenters, distMeas, times = sys.maxsize):
    clusterChanged = True
    calcTimes = 0
    n = len(points)
    clusterAssment = [0 for i in range(n)]

    while clusterChanged and calcTimes < times:
        calcTimes += 1
        clusterChanged = False
        distMatrix = dist.cdist(np.array(points), np.array(clusterCenters), distMeas)
        for index, vec in enumerate(distMatrix):
            minIndex = list(vec).index(min(vec))
            if clusterAssment[index] != minIndex:
                clusterChanged = True 
                clusterAssment[index] = minIndex
        
        for clusterIndex in range(k):
            ptsInCluster = [points[index] for index, cluster in enumerate(clusterAssment) if cluster == clusterIndex]
            if len(ptsInCluster) > 0 :
                clusterCenters[clusterIndex] = np.mean(ptsInCluster, axis=0)
        

    return clusterCenters, clusterAssment   
    

def KMeansPlusPlus(points, k, distMeas='correlation', times = 30):
    clusterCenters = InitClusterCenters(points, k, distMeas)
    clusterCenters, clusterAssment = KMeans(points, k, clusterCenters, distMeas, times)
    return  clusterAssment


if __name__ == '__main__' :
    data = [[1,10], [1,20], [2,9], [4,1], [5,2]]
    k = 2
    calcTimes = 30
    centroids, clusterAssment = KMeansPlusPlus(data, k, times=calcTimes)
    print(centroids)
    print(clusterAssment)

