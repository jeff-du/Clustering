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
    # 直接选择最大值 容易收到噪声的影响 因此应该选择较大值
    # 如何选择较大值： 
    # 把集合D中的每个元素D(x)想象为一根线L(x)，线的长度就是元素的值
    # 将这些线依次按照L(1)、L(2)、……、L(n)的顺序连接起来
    # 组成长线L。L(1)、L(2)、……、L(n)称为L的子线
    # 根据概率的相关知识，如果我们在L上随机选择一个点
    # 那么这个点所在的子线很有可能是比较长的子
    # 而这个子线对应的数据点就可以作为种子点
    # 当K值大于2时，每个样本会有多个距离，需要取最小的那个距离作为D(x)
    distSum = np.inf
    for i in range(1, k):
        distMatrix = dist.cdist(np.array(points), np.array(clusterCenters[:i]), distMeas)  
        temp_list = [value[i-1] for value in list(distMatrix)]
        temp = np.sum(temp_list)
        if temp < distSum :
            min_index = i
            distSum = temp
        dist_list = [value[min_index-1] for value in list(distMatrix)]
        dist_random = distSum * random.random()
        for j, dist_val in enumerate(dist_list):
            dist_random -= dist_val
            if dist_random > 0 :
                continue
            clusterCenters[i] = copy.copy(points[j])
            break
        
    
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
    data = [[1,10], [1,20], [2,9], [4,1], [5,2],[9,9], [4,4], [5,5]]
    k = 3
    calcTimes = 100
    clusterAssment = KMeansPlusPlus(data, k, distMeas='euclidean', times=calcTimes)
    print(clusterAssment)

