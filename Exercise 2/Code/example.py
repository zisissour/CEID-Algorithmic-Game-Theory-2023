import numpy as np


import random

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
     max1=0
      
 for i in range(m)  :
     for j in range(n):
         if max2<C[i][j]:
           max2=C[i][j]

     maxC.append(max2)
     max2=0

 for i in range(m):
     for j in range(n):
         if R[i][j]==maxR[j] and C[i][j]==maxC[i]:  
          pne.append((i+1,j+1))


 if not pne:
   return(0,0)
 
 else:
     return pne



r=[[2,2,0,0], [3,1,3,1]]
c=[[2,2,1,1], [0,1,0,1]]

a =checkForPNE(2,4,r,c)
print(a)

def computeApproximationGuarantees(m,n,R,C,x,y):
    
    epsAPPROX = 1
    epsWSNE = 1

    #Make sure everything is a numpy array
    R=np.array(R)
    C=np.array(C)
    x=np.array(x)
    y=np.array(y)

    Ry = np.matmul(R,y)
    xC = np.matmul(x.T,C)
        
    #Calculating epsAPPROX
    xRy = np.matmul(x.T,Ry)
    xCy = np.matmul(xC,y)

    eps1 = max(Ry) - xRy
    eps2 = max(xC) - xCy
    epsAPPROX = max(eps1, eps2)

    #Calculating epsWSNE
    
    Ry_i=[]
    for i in range(m):
       if x[i]>0:
          Ry_i.append(np.matmul(R[i],y))
    
    Cx_j = []
    for j in range(n):
       if y[j]>0:
          Cx_j.append(np.matmul(C.T[j],x))

    eps1 = max(Ry) - min(Ry_i)
    eps2 = max(xC) - min(Cx_j)

    epsWSNE = max(eps1, eps2)

    return(epsAPPROX,epsWSNE)


#x = [0.5, 0.5]
#y = [0,1,0,0]

#a = computeApproximationGuarantees(2,4,r,c,x,y)
#print(a)


def approxNEConstructionDMP(m,n,R,C):
    #Make sure that everything is an numpy array
    R = np.array(R)
    C = np.array(C)


    #Get random action for row player
    s = random.randint(0,m-1)

    #Get PBR for column player
    mx = max(C[s])
    t = random.choice([i for i, j in enumerate(C[s]) if j == mx]) #Choose randomly if >1 PBRs

    #Get PBR for row player
    mx = max(R.T[t])
    r = random.choice([i for i, j in enumerate(R.T[t]) if j == mx]) #Choose randomly if >1 PBRs

    #Initialize output
    x=np.zeros(m)
    y=np.zeros(n)

    y[t] = 1
    x[s] += 0.5
    x[r] += 0.5

    epsAPPROX,epsWSNE = computeApproximationGuarantees(m,n,R,C,x,y)



    return(x,y,epsAPPROX,epsWSNE)

a = approxNEConstructionDMP(2,4,r,c)

print(a[1])