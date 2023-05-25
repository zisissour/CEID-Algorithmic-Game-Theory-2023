import numpy as np
from numpy import genfromtxt
from numpy.linalg import matrix_rank

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import linprog

import os

def checkForPNE(m,n,R,C):
 maxR=[]
 maxC=[]
 pne=[]
 max1=0
 max2=0
 for j in range(n):
     for i in range(m):
       if max1<R[i][j]:
        max1=R[i][j]
     maxR.append(max1)
      
 for i in range(m)  :
     for j in range(n):
         if max2<C[i][j]:
           max2=C[i][j]
     maxC.append(max2)

 for i in range(m):
     for j in range(n):
         if R[i][j]==maxR[j] and C[i][j]==maxC[i]:  
          return i+1,j+1  
          #pne.append((i,j))


 if not pne:
   print("No equilibria")
   return(0,0)





r=[[]]
c=[[]]
r=[[0,0,0],[0,0,0],[0,0,0]]
c=[[0,0,0],[0,0,0],[0,0,5]]

(i,j)=checkForPNE(1,1,r,c)
print(i,j)
