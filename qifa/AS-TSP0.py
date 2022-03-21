import random
import numpy as np
import math
import matplotlib.pyplot as plt


# 城市坐标
C = np.array([[41, 94], [37, 84], [54, 67],[25, 62],
     [7, 64], [2, 99], [68, 58], [71, 44],
     [54, 62], [83, 69], [64, 60], [18, 54],
     [22, 60], [83, 46], [91, 38], [25, 38],
     [24, 42], [58, 69], [71, 71], [74, 78],
     [87, 76], [18, 40], [13, 40], [82, 7], [62, 32],
     [58, 35], [45, 21], [41, 26], [44, 35], [4, 50]])


#  参数初始化
m = 30  # 蚂蚁数量
alpha = 1  # 信息素权重
beta = 5  # 启发式因子权重
rho = 0.5  # 蒸发系数
G = 200  # 最大迭代次数
Q = 100  # 信息素增加强度系数
NC = 1  # 迭代次数


n=len(C[:,0]) # n表示城市个数
D=np.zeros((n,n)).astype('int') # D表示两城市间距离矩阵


for i in range(n):  # 计算出距离矩阵
     for j in range(n):
          if i != j:
               D[i, j] = math.sqrt((C[i, 0] - C[j, 0])**2 + (C[i, 1] - C[j, 1])**2)
          else:
               D[i, j] = np.spacing(1) # eps
          D[j, i] = D[i, j]


eta=1/D # 启发因子
#eta[~np.isfinite(eta)] = 0 # 主对角线inf替换为0
tau=np.ones((n,n)).astype(int) # 信息素矩阵
T=np.zeros((m,n)).astype(int) # 禁忌表，存储并记录路径的生成
R_best=np.zeros((G,n)).astype('int') # 各代最佳路线
L_best=np.inf*(np.ones((G,1)).astype(int)) # 各代最佳路线的长度

plt.figure(1)
while NC<=G:
  N = np.zeros(n)
  ##########将m只蚂蚁随机放在n个城市上###########
  for i in range(n):
    N[i]=i
  random.shuffle(N)
  N=N.astype(int)
  T[:,0] = np.transpose(N) # 初始随机城市放入禁忌表
  T=T.astype(int)
  for j in range(1,n):
    for i in range(m):
      visited=T[i,0:j] # 已访问的城市
      visited=visited.astype(int)
      J=np.zeros(n-j)
      J=J.astype(int) # 待访问的城市
      P=J # 待访问城市的选择概率分布
      P=P.astype(float)
      Jc=0
      for k in range(n):
        if all(a!=k for a in visited):
          J[Jc]=k
          Jc+=1
      #print(J)
      #########计算待选城市的概率分布########
      for k in range(len(J)):
        P[k] = ((tau[visited[-1],J[k]])**alpha) * ((eta[visited[-1],J[k]])**beta)
        # 赋值时一定要注意左右两边数据类型，整型和浮点型不能相互赋值
      #print(P)
      Pcum=(P/(np.sum(P,axis=0))).cumsum() # 此处累计概率最后一个元素和为1
      #print(Pcum)
      #########按概率选取下一个城市##########
      select=[i for i in range(len(Pcum)) if Pcum[i]>=np.random.rand()] # 获取列表中满足条件的下标
      #print(select)
      to_visited=J[select[0]]
      T[i,j]=to_visited
  if NC>=2:
    T[0,:]=R_best[NC-2,:]
  L=np.zeros((m,1)) # 长度
  for i in range(m):
    R=T[i,:]
    for j in range(n-1):
      L[i]=L[i]+D[R[j],R[j+1]]
    #print(L[i])
    L[i]=L[i]+D[R[0],R[n-1]]
  #print(L)
  pos=np.argmin(L,axis=0) # pos时列表类型
  L_best[NC-1,0]=L[pos[0],0]
  print(L_best[NC-1,0])
  R_best[NC-1,:]=T[pos[0],:]
  #print(R_best[NC-1,:])
  #########信息素更新##########
  delta_tau=np.zeros((n,n))
  for i in range(m):
    for j in range(n-1):
      delta_tau[T[i,j],T[i,j+1]]=delta_tau[T[i,j],T[i,j+1]]+Q/L[i]
    delta_tau[T[i,n-1],T[i,0]]=delta_tau[T[i,n-1],T[i,0]]+Q/L[i]
  tau=(1-rho)*tau+delta_tau
  T=np.zeros((m,n)) # 禁忌表清零
  for i in range(n-1):
    plt.plot([C[R_best[NC-1,i],0],C[R_best[NC-1,i+1],0]],[C[R_best[NC-1,i],1],C[R_best[NC-1,i+1],1]],'bo-')

  plt.plot([C[R_best[NC-1,n-1],0],C[R_best[NC-1,0],0]],[C[R_best[NC-1,n-1],1],C[R_best[NC-1,0],1]],'bo-')
  plt.title('shortest length:{}'.format(L_best[NC-1,0]))
  plt.pause(0.005)
  plt.cla()
  NC+=1

Pos = np.argmin(L_best, axis=0)
shortest_route = R_best[Pos[0], :]
shortest_length = L_best[Pos[0]]
plt.figure(2)
plt.plot(L_best)
plt.xlabel('iterations')
plt.ylabel('target function value')
plt.title('fitness evolution curve')
plt.show()