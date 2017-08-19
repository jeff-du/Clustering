#######################################
#coding=utf8
#Author: Dave
#Created Time : 2017年7月4日
#Last Modified: 2017年7月4日
#Description:
#Evaluate different clustering result
# created by different k value or different algorithms 
####################################### 

import sys

# calc max distance of one class in one cluster
def CalcMaxDist(indexVec, distMatrix):
    distMax = 0
    for i in indexVec:
        for j in indexVec:
            if i != j and distMatrix[i][j] > distMax :
                distMax = distMatrix[i][j]
    return distMax

# calc min distance of two class in one cluster
def CalcMinDist(indexVec1, indexVec2, distMatrix):
    distMin = sys.maxsize
    for i in indexVec1:
        for j in indexVec2:
            if distMatrix[i][j] < distMin :
                distMin = distMatrix[i][j]
    return distMin


# calc dbi of one cluster result to find the best class count
def CalcDBI(oneClusterResult, distMatrix, nCluster):

    clusterIndex = []
    for n in range(nCluster):
        indexVec = [index for index, val in enumerate(oneClusterResult) if val == n]
        clusterIndex.append(indexVec)
    
    dbiVal = 0
    for indexVec1 in clusterIndex:
        distMax1 = 0
        distMax2 = 0
        distMin = sys.maxsize
        for indexVec2 in clusterIndex:
            if indexVec1 != indexVec2 :
                dist1 = CalcMaxDist(indexVec1, distMatrix)
                dist2 = CalcMaxDist(indexVec2, distMatrix)
                if dist1 > distMax1 :
                    distMax1 = dist1
                if dist2 > distMax2 :
                    distMax2 = dist2
                dist3 = CalcMinDist(indexVec1, indexVec2, distMatrix)
                if dist3 < distMin :
                    distMin = dist3

        if distMin != 0 :
            dbiVal += (distMax1 + distMax2) / distMin
                
    return dbiVal/nCluster