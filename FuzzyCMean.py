################################
#Fuzzy-C-Mean Cluster Algorithm
#Typical Cluster Algorithm
#Dave 2017-06-19
#参数：
#给定同维向量数据集合points
#数目为n
#将其聚为C类
#m为权重值
#u为初始匹配度矩阵（n*C）
#采用闵式距离算法，其参数为p
#迭代终止条件为终止值e（取值范围(0，1））及终止轮次
################################

import sys
import numpy as np
import random


def fcm(points, c, m, p, e, terminateTurn = sys.maxsize):

    n = len(points)
    u = InitUMatrix(n, c)
    u1 = u
    k = 0

    while(True):
        # calc one more turn
        u2 = fcmOneTurn(points, u1, m, p)

        # max diff bewteen u1 and u2
        maxDiff = fcmMaxDiff(u1, u2)

        if maxDiff < e :
            break
        u1 = u2
        k = k + 1
        if k < terminateTurn :
            break
    
    clusteringResult = [0 for i in range(n)]
    for index, vec in enumerate(u1):
        maxIndex = vec.index(max(vec))
        clusteringResult[index] = maxIndex

    return clusteringResult


def InitUMatrix(n, c):
    ret = uMatrix = [[random.randint(0,10) for i in range(c)] for j in range(n)]
    sumRow = [0 for i in range(n)]
    for i in range(n):
        for j in range(c):
            sumRow[i] += uMatrix[i][j]
    for i in range(n):
        for j in range(c):
            ret[i][j] = uMatrix[i][j] / sumRow[i]
    return ret




def fcmOneTurn(points, u, m, p):

    n = len(points)
    c = len(u[0])

    # calc centroids of clusters
    centroids = fcmCalcCenter(points, u, m)

    # calc new u matrix
    u2 = fcmCalcU(points, centroids, m, p)

    return u2

def fcmMaxDiff(u1, u2):
    ret = 0
    n = len(u1)
    c = len(u1[0])

    for i in range(n):
        for j in range(c):
            ret = max(np.fabs(u1[i][j] - u2[i][j]), ret)
    
    return ret


def fcmCalcCenter(points, u, m):
    n = len(points)
    c = len(u[0])
    center = []

    for j in range(c):
        sum1 = 0
        sum2 = 0
        for i in range(n):
            sum1 += np.power(u[i][j], m)
            sum2 += np.dot(points[i], np.power(u[i][j], m))
        
        if sum1 != 0 :
            cj = sum2 / sum1
        else:
            cj = [0 for d in range(len(points[i]))]
        center.append(cj)
    
    return center


def fcmCalcU(points, centroids, m, p):
    n = len(points)
    c = len(centroids)
    ret = [[0 for j in range(c)] for i in range(n)]
    for i in range(n):
        for j in range(c):
            sum1 = 0
            #d1 = disMinkowski(points[i], centroids[j], p)
            d1 = disCosine(points[i], centroids[j])
            for k in range(n):
                #d2 = disMinkowski(points[k], centroids[j], p)
                d2 = disCosine(points[k], centroids[j])
                if d1 is not None and d2 is not None and d2 != 0 :
                    sum1 += np.power(d1/d2, float(2)/(float(m)-1))
            if sum1 != 0 : 
                ret[i][j] = 1/sum1
    return ret



def disMinkowski(point1, point2, p):
    dim = len(point1)
    disSum = 0
    for i in range(dim):
        disSum += np.power(point1[i] - point2[i], p)
    ret = np.power(disSum, 1/p)
    return ret

def disCosine(point1, point2):
    dotClusters = np.dot(point1, point2)
    sum1, sum2 = 0, 0
    for val in point1:
        sum1 += val ** 2
    for val in point2:
        sum2 += val ** 2
    
    if sum1 != 0 and sum2 != 0 :
        return 1 - dotClusters / (sum1 * sum2) ** 0.5
    else:
        return None


    
if __name__ == '__main__' :
    points = [[1,10], [1,20], [2,14], [4,1]]
    c = 2
    m = 2
    p = 2
    e = 0.01
    result = fcm(points, c, m, p, e)
    print(result)
