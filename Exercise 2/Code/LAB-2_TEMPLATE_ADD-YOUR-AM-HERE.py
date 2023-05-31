import numpy as np
from numpy import genfromtxt
from numpy.linalg import matrix_rank

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import random
import os

############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    # pure colors...
    GREY      = '\033[90m'
    RED       = '\033[91m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    BLUE      = '\033[94m'
    PURPLE    = '\033[95m'
    CYAN      = '\033[96m'
    # color styles...
    HEADER      = '\033[95m\033[1m'
    MSG         = '\033[95m'
    QUESTION    = '\033[93m\033[3m'
    COMMENT     = '\033[96m'
    IMPLEMENTED = '\033[92m' + '[IMPLEMENTED] ' +'\033[96m'
    TODO        = '\033[94m' + '[TO DO] ' +'\033[96m'
    WARNING     = '\033[91m'
    ERROR       = '\033[91m\033[1m'
    ENDC        = '\033[0m'    # RECOVERS DEFAULT TEXT COLOR
    BOLD        = '\033[1m'
    ITALICS     = '\033[3m'
    UNDERLINE   = '\033[4m'

    def disable(self):
        self.HEADER     = ''
        self.OKBLUE     = ''
        self.OKGREEN    = ''
        self.WARNING    = ''
        self.FAIL       = ''
        self.ENDC       = ''

def screen_clear():
   # for mac and linux(here, os.name is 'posix')
   if os.name == 'posix':
      _ = os.system('clear')
   else:
      # for windows platfrom
      _ = os.system('cls')

### A. CONSTRUCTION OF RANDOM GAMES TO SOLVE ###

def generate_random_binary_array(N,K):
    print( bcolors.IMPLEMENTED + '''
    # ROUTINE:  generate_random_binary_array
    # PRE:      N = length of the 1xN binary array to be constructed
    #           K = number of ones within the 1xN binary array
    # POST:     A randomly constructed numpy array with K 1s and (N-K) zeros''' + bcolors.ENDC )

    if K <= 0: # construct an ALL-ZEROS array
        randomBinaryArray = np.zeros(N)
    
    elif K >= N:  # construct an ALL-ONES array
        randomBinaryArray = np.ones(N)

    else:
        randomBinaryArray = np.array([1] * K + [0] * (N-K))
        np.random.shuffle(randomBinaryArray)

    return(randomBinaryArray)

def generate_winlose_game_without_pne(m,n,G01,G10,earliestColFor01,earliestRowFor10):
   
    print( bcolors.IMPLEMENTED + '''
    # ROUTINE:   generate_random_binary_array
    # PRE:       (m,n) = the dimensions of the game to construct
    #            (G10,G01) = numbers of (1,0)-elements and (0,1) elements in the game
    # POST:      Construct a mxn win-lose game randomly, so that:
    #             * There are G10 (1,0)-elements and G01 (0,1)-elements.
    #             * (1,1)-elements are forbidden
    #             * Each row possesses at least one (0,1)-element
    #             * Each column possesses at least one (1,0)-element
    #             * (0,1)-elements lie in columns from earliestColFor01 to n
    #             * 10-elements lie in rows from earliestRowFor10 to n
    # ERROR HANDLING:
    #   [EXITCODE =  0] SUCCESSFUL CREATION OF RANDOM WIN-LOSE GAME 
    #   [EXITCODE = -1] WRONG PARAMETERS 
    #   [EXITCODE = -2] INSUFFICIENT 10-ELEMENTS OR 01-ELEMENTS 
    #   [EXITCODE = -3] TOO MANY 10-ELEMENTS OR 01-ELEMENTS 
    #   [EXITCODE = -4] NOT ENOUGH SPACE TO POSITION 10-ELEMENTS, GIVEN POSITIONS OF 01-ELEMENTS 
    #   [EXITCODE = -5] BAD LUCK, SOME COLUMN WITHIN 10-ELIGIBLE AREA IS ALREADY FILLED WITH 01-ELEMENTS''' + bcolors.ENDC )

    isIntegerFlag = True

    try:
        # try converting to integer
        int(m)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(n)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(G01)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(G10)
    except ValueError:
        isIntegerFlag = False

    try:
        # try converting to integer
        int(earliestColFor01)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(earliestRowFor10)
    except ValueError:
        isIntegerFlag = False

    if not isIntegerFlag or np.amin([m,n]) < 2 or np.amax([m,n]) > maxNumberOfActions or m > n or np.amin([earliestRowFor10,earliestColFor01]) < 0 or (earliestRowFor10 > m-1) or (earliestColFor01 > n-1):
        # WRONG INPUT PARAMETERS
        print( bcolors.ERROR + "ERROR MESSAGE GEN 1: wrong input parameters" + bcolors.ENDC )
        return( -1 , np.zeros([maxNumberOfActions,maxNumberOfActions]) , np.zeros([maxNumberOfActions,maxNumberOfActions]))

    # initialization of the two payoff matrices...
    R = np.zeros([m,n])
    C = np.zeros([m,n])    

    if (G10 < n or G01 < m):
        print( bcolors.ERROR + "ERROR MESSAGE GEN 2: NOT ENOUGH 10-elements and/or 01-elements: G10 =", G10," < n =",n,"? G01 = ",G01, "< m =",m,"?" + bcolors.ENDC )
        return( -2 , R , C )

    if G10 > (m-earliestRowFor10)*n or G01 > m*(n-earliestColFor01) or G01+G10 > m*n - earliestRowFor10*earliestColFor01:
        print( bcolors.ERROR + "ERROR MESSAGE GEN 3: TOO MANY 10-elements and/or 01-elements:" + bcolors.ENDC )
        print("\tG10 =", G10, "> (m-earliestRowFor10)*n =", (m-earliestRowFor10)*n,"?")
        print("\tG01 =", G01, "> m*(n-earliestColFor01) =", m*(n-earliestColFor01),"?")
        print("\tG01+G10 =", G01+G10, "> m*n - earliestRowFor10*earliestColFor01 =", m*n - earliestRowFor10*earliestColFor01,"?")
        return( -3 , R , C )

    # choose the random positions for 01-elements, within the eligible area of the bimatrix...
    # eligible area for 01-elements: rows = 0,...,m-1, columns = earliestColFor01,...,n-1 

    # STEP 1: choose m 01-elements, one per row, within the eligible area [0:m]x[earliestColFor01s:n] of the bimatrix.

    numEligibleCellsFor01 = m * ( n - earliestColFor01 ) # all cells in bimatrix are currently 00-elements

    ArrayForOne01PerRow = np.zeros(numEligibleCellsFor01)
    for i in range(m):
        random_j = np.random.randint(earliestColFor01,n)
        position = (n-earliestColFor01)*i + random_j - (earliestColFor01)
        ArrayForOne01PerRow[position] = 1

    # STEP 2: choose G01 – m 01-elements within the eligible area [0:m]x[earliestColFor01s:n] of the bimatrix
    # differently from those cells chosen in STEP 1.
    binaryArrayFor01s = generate_random_binary_array( numEligibleCellsFor01 - m , G01 - m ) 

    # Position ALL the 01-elements within the eligible area of the bimatrix...
    for i in range(m):
        for j in range(earliestColFor01,n):
            position = (n-earliestColFor01)*i + j - (earliestColFor01)
            if ArrayForOne01PerRow[position] == 1:
                # insert this enforced 10-element in binArrayFor01s

                if position <= 0: # checking cell (0,earliestColFor01)...
                    binaryArrayFor01sPrefix = np.array([])
                else:
                    binaryArrayFor01sPrefix = binaryArrayFor01s[0:position]

                if position >= numEligibleCellsFor01: #checking cell (m,n)...
                    binaryArrayFor01sSuffix = np.array([])
                else:
                    binaryArrayFor01sSuffix = binaryArrayFor01s[position:]

                binaryArrayFor01s = np.concatenate((binaryArrayFor01sPrefix,np.array([1]),binaryArrayFor01sSuffix),axis=None)

            # print("next position to check for 01-element:",position,"related to the cell [",i,j,"].")
            if binaryArrayFor01s[position] == 1:
                C[i,j] = 1

    # STEP 3: choose n 10-elements, one per column, within the eligible area [earliestRowFor10s:m]x[0:n] of the bimatrix. They should be different from those cells chosen in STEPS 1+2    

    numEligibleCellsFor10 = (m - earliestRowFor10) * n # all cells in bimatrix are currently 00-elements

    # Count only the (0,0)-elements within eligible area of the bimatrix for 10-elements...
    # eligible area for 10-elements: rows = earliestRowFor10,...,m-1, columns = 0,...,n-1 
    numFreeEligibleCellsFor10 = 0

    ArrayForOne10PerCol = np.zeros(numEligibleCellsFor10)

    # Count the non-01-elements within the eligible area of the bimatrix for 10-elements
    for i in range(earliestRowFor10,m):
        for j in range(0,n):
            if C[i,j] == 0:
                numFreeEligibleCellsFor10 += 1

    # print("Actual number for eligible cells for 10-elements: numEligibleCellsFor10 = ",numFreeEligibleCellsFor10)
    if numFreeEligibleCellsFor10 < G10:
        print(bcolors.ERROR + "ERROR MESSAGE GEN 4: Not enough space to position all the 10-elements within the selected block of the bimatrix and the random position of the 01-elements" + bcolors.ENDC)
        return( -4 , np.zeros( [m,n] ) , np.zeros( [m,n] ) )
      
    # choose the n random positions of 10-elements, one per column, in positions which are NOT already 
    # 01-elements, within the 10-eligible area of the bimatrix
    for j in range(n):
        if sum(C[earliestRowFor10:,j:j+1]) == n - earliestRowFor10:
            # the j-th row of the 10-eligible area in the bimatrix is already filled with 01-elements
            print(bcolors.ERROR + "ERROR MESSAGE 5: Bad luck, column",j,"of the bimatrix is already filled with 01-elements." + bcolors.ENDC )
            return( -5 , np.zeros( [m,n] ) , np.zeros( [m,n] ) )
         
        Flag_EmptyCellDiscovered = False
        while not Flag_EmptyCellDiscovered:
            random_i = np.random.randint(earliestRowFor10,m)
            if C[random_i,j] == 0:
                Flag_EmptyCellDiscovered = True
        position = n * (random_i - earliestRowFor10) + j
        ArrayForOne10PerCol[position] = 1

    # choose the remaining G10-n random positions for 10-elements, in positions which are NOT already
    # used by 01-elements or other (the necessary) 10-elements, within the eligible area of the bimatrix
    binaryArrayFor10s = generate_random_binary_array(numFreeEligibleCellsFor10-n,G10-n)
    # expand the binaryArrayFor10s to cover the entire eligible area for 10-elements, so that
    # all cells which are already 01-elements get 0-value and all cells with a necessary 10-element 
    # get 1-value.

    # print("INITIAL length of binaryArrayFor10s is",len(binaryArrayFor10s))
    for i in range(earliestRowFor10,m):
        for j in range(0,n):
            position = n*(i-earliestRowFor10) + j
            if C[i,j] == 1:
                # A 01-element was discovered. Insert a ZERO in binaryArrayFor10s, at POSITION, 
                # on behalf of cell (i,j)...

                # print("01-element discovered at position (",i,",",j,"). Inserting an additional ZERO at position ",position)
                    
                if position <= 0: # checking cell (earliestRowFor10,0)...
                    binaryArrayFor10sPrefix = np.array([])
                else:
                    binaryArrayFor10sPrefix = binaryArrayFor10s[0:position]

                if position >= len(binaryArrayFor10s): #checking cell (m,n)...
                    binaryArrayFor10sSuffix = np.array([])
                else:
                    binaryArrayFor10sSuffix = binaryArrayFor10s[position:]  

                binaryArrayFor10s = np.concatenate((binaryArrayFor10sPrefix,np.array([0]),binaryArrayFor10sSuffix),axis=None)

                # print("binaryArrayFor10s[position] =",binaryArrayFor10s[position])

            elif ArrayForOne10PerCol[position] == 1:
                # A necessary 10-element discovered. Insert a new ONE in binaryArrayFor10s, at POSITION, 
                # on behalf of cell (i,j)...
                # print("A necessary 10-element was discovered at position (",i,",",j,"). Inserting an additional ONE at position ",position)
                    
                if position <= 0: # checking cell (earliestRowFor10,0)...
                    binaryArrayFor10sPrefix = np.array([])
                else:
                    binaryArrayFor10sPrefix = binaryArrayFor10s[0:position]

                if position >= len(binaryArrayFor10s): #checking cell (m,n)...
                    binaryArrayFor10sSuffix = np.array([])
                else:
                    binaryArrayFor10sSuffix = binaryArrayFor10s[position:]  

                binaryArrayFor10s = np.concatenate((binaryArrayFor10sPrefix,np.array([1]),binaryArrayFor10sSuffix),axis=None)

                # print("binaryArrayFor10s[position] =",binaryArrayFor10s[position])

    # print("ACTUAL length of binaryArrayFor10s is",len(binaryArrayFor10s))

    # Insert the G10 10-elements in the appropriate positions of the bimatrix...
    for i in range(earliestRowFor10,m):
        for j in range(0,n):
            position = n*(i-earliestRowFor10) + j
            # print("next position to check for 10-element:",position,"related to the cell [",i,j,"], with C-value = ",C[i,j],"and binaryArrayFor10s-value = ",binaryArrayFor10s[position])
            if binaryArrayFor10s[position] == 1:
                R[i,j] = 1

    return(0,R,C)

### B. MANAGEMENT OF BIMATRICES ###

def drawLine(lineLength,lineCharacter):

    LINE          = '\t'
    consecutiveLineCharacters    = lineCharacter
    for i in range(lineLength):
            consecutiveLineCharacters    = consecutiveLineCharacters  + lineCharacter
    LINE          = '\t' + consecutiveLineCharacters
    return(LINE)

def drawBimatrix(m,n,R,C):

    print( bcolors.IMPLEMENTED + '''
    ROUTINE:    drawBimatrix
    PRE:        Dimensions and payoff matrices of a win-lose bimatrix game
    POST:       The bimatrix game, with RED for 10-elements, GREEN for 01-elements, and BLUE for 11-elements
    ''' + bcolors.ENDC )
    
    for i in range(m):
        #PRINTING ROW i...
        if i == 0:
            print(EQLINE)
        else:
            print(MINUSLINE)

        printRowString = ''

        for j in range(n):
            # PRINTING CELL (i,j)...
            if R[i,j] == 1:
                if C[i,j] == 1:
                    CellString = bcolors.CYAN + "("     
                else:
                    CellString = bcolors.RED + "("     
            elif C[i,j] == 1:
                CellString = bcolors.GREEN + "("     
            else:
                CellString = "("     
                                  
            CellString += str(int(R[i,j])) + "," + str(int(C[i,j])) + ")" + bcolors.ENDC 
            if printRowString == '':
                printRowString = '\t[ ' + CellString
            else:
                printRowString =  printRowString + ' | ' + CellString

        printRowString = printRowString + ' ]'
        print (printRowString)

    print ( EQLINE )

### ALGORITHMS FOR SOLVING BIMATRIX GAMES

# ALG0: Solver for ZERO-SUM games...


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

def solveZeroSumGame(m,n,A):
    print(bcolors.IMPLEMENTED + '''
    # ROUTINE: solveZeroSumGame
    # PRE:  An arbirary payoff matrix A, with real values (not necessarily in [0,1])
    # METHOD:
    #    Construct the LP describing the MAGNASARIAN-STONE formulation for the 0_SUM case: R = A, C = -A 
    #    [0SUMLP]  
    #    minmize          1*r           + 1*c +  np.zeros(m).reshape([1,m]@x + np.zeros(n).reshape([1,n]@y 
    #      s.t.
    #           -np.ones(m)*r + np.zeros(m)*c +            np.zeros([m,m])@x +                          R@y <= np.zeros(m), 
    #           np.zeros(n)*r -  np.ones(n)*c +                          C'x +            np.zeros([n,n])@y <= np.zeros(n),
    #                     0*r             0*c +  np.ones(m).reshape([1,m])@x + np.zeros(n).reshape([1,n])@y = 1,
    #                     0*r             0*c + np.zeros(m).reshape([1,m])@x +  np.ones(n).reshape([1,n])@y = 1,
    #                                                                   np.zeros(m) <= x,              np.zeros(n) <= y
    #
    # vector of unknowns is a (1+1+m+n)x1 array: chi = [ r, c, x^T , y^T ], 
    # where r is ROW's payoff and c is col's payoff, wrt the profile (x,y).
    #''' + bcolors.ENDC)
    
    c = np.block( [ np.ones(2), np.zeros(m+n) ] )
 
    Coefficients_a = np.block( [ (-1)*np.ones(m), np.zeros(n), np.array([0,0]) ] )                                                      # 1x(m+n+2) array...
    Coefficients_b = np.block( [ np.zeros(m), (-1)*np.ones(n), np.array([0,0]) ] )                                                      # 1x(m+n+2) array...
    Coefficients_x = (np.block( [np.zeros([m,m]), (-1)*A, np.ones(m).reshape([m,1]), np.zeros(m).reshape([m,1]) ] )).transpose()        # mx(m+n+2) array...
    Coefficients_y = (np.block( [A.transpose(), np.zeros([n,n]),np.zeros(n).reshape([n,1]), np.ones(n).reshape([n,1]) ] )).transpose()  # nx(m+n+2) array...

    SIGMA0 = ( np.block([ Coefficients_a.reshape([m+n+2,1]), Coefficients_b.reshape([m+n+2,1]), Coefficients_x, Coefficients_y ]) )

    SIGMA0_ub = SIGMA0[0:m+n,:]
    Constants_vector_ub = np.zeros(m+n)
    
    SIGMA0_eq = SIGMA0[m+n:m+n+2,:]
    Constants_vector_eq = np.ones(2)

    #variable bounds
    Var_bounds = [ (None, None) , (None,None) ]
    for i in range(m+n):
        Var_bounds.append( (0,None) )  # type: ignore

    zero_sum_res = linprog(     c, 
                                A_ub=SIGMA0_ub, 
                                b_ub=Constants_vector_ub, 
                                A_eq=SIGMA0_eq, 
                                b_eq=Constants_vector_eq, 
                                bounds=Var_bounds, 
                                method='highs', callback=None, options=None, x0=None )

    chi = zero_sum_res.x

    x = chi[2:m+2]
    y = chi[m+2:m+n+2]

    #rint("Success in solving 0SUMLP for (X,-X) ?\t", zero_sum_res.success)
    #print("Message of the solver for 0SUMLP ?\t",zero_sum_res.message)
    #print("0SUmLP's objective value (additive-wsne guarantee) \t=\t",zero_sum_res.fun)
    #print("NE point for (X,-X) is ( x=",x.reshape([1,m])," , y=",y.reshape([1,n])," ).")

    return(x,y)
 
def removeStrictlyDominatedStrategies(m,n,R,C):
    #Make sure everything is a numpy array
    R=np.array(R)
    C=np.array(C)

    maxR=0
    maxC=0

    for i in range(m):
        for j in range(n): 
            if maxR<R[i][j]:
                maxR=R[i][j]

        if maxR==0:
            R=np.delete(R,i,axis=0)
            C=np.delete(C,i,axis=0)    
            m=m-1
            m,n,R,C = removeStrictlyDominatedStrategies(m,n,R,C)
            break
        maxR=0

    for j in range(n):
        for i in range(m):
            if maxC<C[i][j]:
                maxC=C[i][j]

        if maxC==0:
            C=np.delete(C,j,axis=1)
            R=np.delete(R,j,axis=1)
            n=n-1
            m,n,R,C = removeStrictlyDominatedStrategies(m,n,R,C)
            break
        maxC=0
    
    return m,n,R,C

def interpretReducedStrategiesForOriginalGame(reduced_x,reduced_y,reduced_R,reduced_C,R,C):

    print(bcolors.TODO + '''
    ROUTINE:    interpretReducedStrategiesForOriginalGame
    PRE:        A profile of strategies (reduced_x,reduced_y) for the reduced 
                game (reduced_R,reduced_C), without (0,*)-rows or (*,0)-columns.
    POST:       The corresponding profile for the original game (R,C).
    ''' + bcolors.ENDC)

    x = reduced_x
    y = reduced_y

    return(x,y)

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

def approxNEConstructionFPPBR(m,n,R,C):
    T=1000

    #Make sure everything is an numpy array
    R=np.array(R)
    C=np.array(C)

    x=np.array([])
    y=np.array([])
    x_t = []
    y_t = []

    #First Move
    x=np.zeros(m)
    x[0]=1
    x_t.append(x)
    y=np.zeros(n)
    y[0]=1
    y_t.append(y)

    #T itterations
    for t in range(1,T):
        x=np.zeros(m) #Set new move to zeros 
        x[np.argmax(np.matmul(R,y_t[t-1]))] = 1 #Set one for the first PBR you find
        x_t.append(((t-1)*x_t[t-1] + x)/t) #Recalculate the strategy

        y=np.zeros(n)
        y[np.argmax(np.matmul(x_t[t-1].T,C))] = 1
        y_t.append(((t-1)*y_t[t-1] + y)/t)

    epsAPPROX, epsWSNE = computeApproximationGuarantees(m,n,R,C,x_t[T-1],y_t[T-1]) #Calculate ε-approxes 

    return(x_t[T-1],y_t[T-1],epsAPPROX,epsWSNE)

def approxNEConstructionFPUNI(m,n,R,C):

    T=1000

    #Make sure everything is a numpy array
    R=np.array(R)
    C=np.array(C)

    x=np.array([])
    y=np.array([])
    x_t = []
    y_t = []

    #First Move
    x=np.ones(m)/m
    x_t.append(x)

    y=np.ones(n)/n
    y_t.append(y)

    for t in range(1,T):
        x=np.zeros(m) #Set new move to zeros
        Ry = np.matmul(R,y_t[t-1]) #Calculate R after column's move

        #Uniformly distribute possibility to every PBR
        mx = max(Ry)
        mx_counter = 0
        for i, j in enumerate(Ry):
            if j == mx:
                x[i]=1
                mx_counter += 1
        x = x/mx_counter           

        x_t.append(((t-1)*x_t[t-1] + x)/t) #Recalculate the strategy

        y=np.zeros(n)
        xC = np.matmul(x_t[t-1].T,C)
        
        mx = max(xC)
        mx_counter = 0
        for i, j in enumerate(xC):
           if j == mx:
              y[i]=1
              mx_counter += 1
        y = y/mx_counter

        y_t.append(((t-1)*y_t[t-1] + y)/t)

    epsAPPROX, epsWSNE = computeApproximationGuarantees(m,n,R,C,x_t[T-1],y_t[T-1])

    return(x_t[T-1],y_t[T-1],epsAPPROX,epsWSNE)

def approxNEConstructionDEL(m,n,R,C):
    print(bcolors.TODO + '''
    ROUTINE: approxNEConstructionDEL
    PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
    POST:   A profile of strategies (x,y) produced by the DEL algorithm.''' + bcolors.ENDC)
    
    #... provide your code here, commenting all previous (unnecessary) prints ...

    # these steps are just to do something 

    x,y = solveZeroSumGame(m,n,R) # this example solves (R,-R)
    x = x.transpose()
    y = y.transpose()
    epsAPPROX,epsWSNE = computeApproximationGuarantees(m,n,R,C,x,y)

    # these steps are just to do something 

    return(x,y,epsAPPROX,epsWSNE)

### C. GET INPUT PARAMETERS ###
def determineGameDimensions():

    m = 0
    while m < 2 or m > maxNumberOfActions:
        RowActionsString = input(bcolors.QUESTION +'Determine the size 2 =< m =< ' + str(maxNumberOfActions) + ', for the mxn bimatrix game: ' + bcolors.ENDC)
        if RowActionsString.isdigit():
            m = int(RowActionsString)
            print(bcolors.MSG + "You provided the value m =" + str(m) + bcolors.ENDC)
            if m < 2 or m > maxNumberOfActions:
                print( bcolors.ERROR + 'ERROR INPUT 1: Only positive integers between 2 and ' + str(maxNumberOfActions) + ' are allowable values for m. Try again...' + bcolors.ENDC ) 
        else:
            m = 0
            print( bcolors.ERROR + 'ERROR INPUT 2: Only integer values between 2 and ' + str(maxNumberOfActions) + ' are allowable values for m. Try again...' + bcolors.ENDC ) 

    n = 0
    while n < 2 or n > maxNumberOfActions:
        ColActionsString = input(bcolors.QUESTION +'Determine the size 1 =< n =< ' + str(maxNumberOfActions) + ', for the mxn bimatrix game: ' + bcolors.ENDC)
        if ColActionsString.isdigit():
            n = int(ColActionsString)
            print(bcolors.MSG + "You provided the value n =" + str(n) + bcolors.ENDC)
            if n < 2 or n > maxNumberOfActions:
                print( bcolors.ERROR + 'ERROR INPUT 3: Only positive integers between 2 and ' + str(maxNumberOfActions) + ' are allowable values for m. Try again...' + bcolors.ENDC ) 
        else:
            n = 0
            print( bcolors.ERROR + 'ERROR INPUT 4: Only integer values between 2 and ' + str(maxNumberOfActions) + ' are allowable values for n. Try again...' + bcolors.ENDC ) 

    return( m, n )

def determineNumRandomGamesToSolve():

    numOfRandomGamesToSolve = 0
    while numOfRandomGamesToSolve < 1 or numOfRandomGamesToSolve > 10000:
        numOfRandomGamesToSolveString = input(bcolors.QUESTION +'Determine the number of random games to solve: ' + bcolors.ENDC)
        if numOfRandomGamesToSolveString.isdigit():
            numOfRandomGamesToSolve = int(numOfRandomGamesToSolveString)
            print( bcolors.MSG + "You requested to construct and solve " + str(numOfRandomGamesToSolve) + " random games to solve." + bcolors.ENDC )
            if n < 2 or m > maxNumberOfActions:
                print( bcolors.ERROR + 'ERROR INPUT 5: Only positive integers between 1 and ' + str(maxNumOfRandomGamesToSolve) + ' are allowable values for m. Try again...' + bcolors.ENDC ) 
        else:
            numOfRandomGamesToSolve = 0
            print( bcolors.ERROR + 'ERROR INPUT 6: Only integer values between 2 and ' + str(maxNumOfRandomGamesToSolve) + ' are allowable values for n. Try again...' + bcolors.ENDC ) 

    return( numOfRandomGamesToSolve )

def determineNumGoodCellsForPlayers(m,n):

    G10 = 0
    G01 = 0
    
    while G10 < 1 or G10 > m*n:
        G10String = input( bcolors.QUESTION + 'Determine the number of (1,0)-elements in the bimatrix: ' + bcolors.ENDC )
        if G10String.isdigit():
            G10 = int(G10String)
            print(bcolors.MSG + "You provided the value G10 =" + str(G10) + bcolors.ENDC)
            if G10 < 0 or G10 > m*n:
                print( bcolors.ERROR + 'ERROR INPUT 7: Only non-negative integers up to ' + str(m*n) + ' are allowable values for G10. Try again...' + bcolors.ENDC ) 
        else:
            G10 = 0
            print( bcolors.ERROR + 'ERROR INPUT 8: Only integer values up to ' + str(m*n) + ' are allowable values for G10. Try again...' + bcolors.ENDC ) 

    while G01 < 1 or G01 > m*n:
        G01String = input( bcolors.QUESTION + 'Determine the number of (0,1)-elements in the bimatrix: ' + bcolors.ENDC )
        if G01String.isdigit():
            G01 = int(G01String)
            print(bcolors.MSG + "You provided the value G01 =" + str(G01) + bcolors.ENDC)
            if G01 < 0 or G01 > m*n:
                print( bcolors.ERROR + 'ERROR INPUT 9: Only non-negative integers up to ' + str(m*n) + ' are allowable values for G01. Try again...' + bcolors.ENDC ) 
        else:
            G01 = 0
            print( bcolors.ERROR + 'ERROR INPUT 10: Only integer values up to ' + str(m*n) + ' are allowable values for G01. Try again...' + bcolors.ENDC ) 

    return( G10, G01 )

### D. PREAMBLE FOR LAB-2 ###

def print_LAB2_preamble():
    screen_clear()

    print(bcolors.HEADER + MINUSLINE + """
                        CEID-NE509 (2022-3) / LAB-2""")  
    print(MINUSLINE + """
        STUDENT NAME:           < provide your name here >
        STUDENT AM:             < provide your AM here >
        JOINT WORK WITH:        < provide your partner's name and AM here >""")
    print(MINUSLINE + bcolors.ENDC)

    input("Press ENTER to continue...")
    screen_clear()

    print(bcolors.HEADER + MINUSLINE + """
        LAB-2 OBJECTIVE: EXPERIMENTATION WITH WIN-LOSE BIMATRIX GAMES\n""" + MINUSLINE + """  
        1.      GENERATOR OF INSTANCES: Construct rando win-lose games 
        with given densities for non-(0,0)-elements, and without pure 
        Nash equilibria.                          (PROVIDED IN TEMPLATE)

        2.      BIMATRIX CLEANUP: Remove all STRICTLY DOMINATED actions 
        for the players, ie, all (0,*)-rows, and all (*,0)-columns from 
        the bimatrix.                                (TO BE IMPLEMENTED) 

        3.      Implementation of elementary algorithms for constructing
        strategy profiles that are then tested for their quality as 
        ApproxNE, or WSNE points.                    (TO BE IMPLEMENTED)

        4.      EXPERIMENTAL EVALUATION: Construct P random games, for
        some user-determined input parameter P, and solve each of them 
        with each of the elementary algorithms. Record the observed 
        approximation guarantees (both epsAPPROXNE and epsWSNE) for the 
        provided strategy profiles.                  (TO BE IMPLEMENTED)
        
        5.      VISUALIZATION OF RESULTS: Show the performances of the 
        algorithms (as approxNE or WSNE constructors), by constructin 
        the appropriate histograms (bucketing the observewd approximation 
        guarantees at one-decimal-point precision).  (TO BE IMPLEMENTED)
    """ + MINUSLINE + bcolors.ENDC )

    input("Press ENTER to continue...")

### E. EXPERIMENTS ###
def defaultExperiments():

    m,n = 10,10
    numOfRandomGamesToSolve = 100
    earliestColFor01 = 0
    earliestRowFor10 = 0

    G01=0
    G10=0

    num = int(input(bcolors.QUESTION+ 'Choose experiment: \n1)P1 \n2)P2 \n3)P3 \n4)P4\n' +bcolors.ENDC))

    
    if num == 1:
        G10 =20
        G01 =20
    
    elif num == 2:
        G10 =20
        G01 =50
    
    elif num == 3:
        G10 =20
        G01 =70

    elif num == 4:
        G10 =35
        G01 =35
    
    else:
        return

    DMPepsAPPROX =np.zeros(numOfRandomGamesToSolve)
    DMPepsWSNE =np.zeros(numOfRandomGamesToSolve)
    DELepsAPPROX =np.zeros(numOfRandomGamesToSolve)
    DELepsWSNE =np.zeros(numOfRandomGamesToSolve)
    FPPBRepsAPPROX =np.zeros(numOfRandomGamesToSolve)
    FPPBRepsWSNE =np.zeros(numOfRandomGamesToSolve)
    FPUNIepsAPPROX =np.zeros(numOfRandomGamesToSolve)
    FPUNIepsWSNE =np.zeros(numOfRandomGamesToSolve)
    

    minDMPepsAPPROX =0
    minDMPepsWSNE =0
    minDELepsAPPROX =0
    minDELepsWSNE =0
    minFPPBRepsAPPROX =0
    minFPPBRepsWSNE =0
    minFPUNIepsAPPROX =0
    minFPUNIepsWSNE =0
    
    DMPApproxNEWorstGame1ROW = []
    DMPApproxNEWorstGame1COL = []
    DMPApproxNEWorstGame2ROW = []
    DMPApproxNEWorstGame2COL = []
    
    DELApproxNEWorstGame1ROW = []
    DELApproxNEWorstGame1COL = []
    DELApproxNEWorstGame2ROW = []
    DELApproxNEWorstGame2COL = []
    
    FPPBRApproxNEWorstGame1ROW = []
    FPPBRApproxNEWorstGame1COL = []
    FPPBRApproxNEWorstGame2ROW = []
    FPPBRApproxNEWorstGame2COL = []

    FPUNIApproxNEWorstGame1ROW = []
    FPUNIApproxNEWorstGame1COL = []
    FPUNIApproxNEWorstGame2ROW = []
    FPUNIApproxNEWorstGame2COL = []

    DMPApproxNEHistogram = np.zeros(10)
    DMPWSNEHistogram = np.zeros(10)
    DELApproxNEHistogram = np.zeros(10)
    DELWSNEHistogram = np.zeros(10)
    FPPBRApproxNEHistogram = np.zeros(10)
    FPPBRWSNEHistogram = np.zeros(10)
    FPUNIApproxNEHistogram = np.zeros(10)
    FPUNIWSNEHistogram = np.zeros(10)

    for i in range(0,numOfRandomGamesToSolve):
        EXITCODE = -5
        numOfAttempts = 0

        # TRY GETTING A NEW RANDOM GAME
        # REPEAT UNTIL EXITCODE = 0, ie, a valid game was constructed.
        # NOTE: EXITCODE in {-1,-2,-3} indicates invalid parameters and exits the program)
        while EXITCODE < 0: 
            # EXIT CODE = -4 ==> No problem with parameters, only BAD LUCK, TOO MANY 01-elements within 10-eligible area
            # EXIT CODE = -5 ==> No problem with parameters, only BAD LUCK, ALL-01 column exists within 10-eligible area
            numOfAttempts += 1
            print("Attempt #" + str(numOfAttempts) + " to construct a random game...")
            EXITCODE,R,C = generate_winlose_game_without_pne(m,n,G01,G10,earliestColFor01,earliestRowFor10)

            if EXITCODE in [-1,-2,-3]:
                print(bcolors.ERROR + "ERROR MESSAGE MAIN 1: Invalid parameters were provided for the construction of the random game." + bcolors.ENDC)
                exit()

        reduced_m,reduced_n, reduced_R,reduced_C = removeStrictlyDominatedStrategies(m,n,R,C)

        ### EXECUTING DMP ALGORITHM...  
        x, y, DMPepsAPPROX[i], DMPepsWSNE[i] = approxNEConstructionDMP(reduced_m,reduced_n,reduced_R,reduced_C)
        ### EXECUTING DEL ALGORITHM...
        x, y, DELepsAPPROX[i], DELepsWSNE[i]= approxNEConstructionDEL(reduced_m,reduced_n,reduced_R,reduced_C)
        ### EXECUTING FICTITIOUS PLAY PBR ALGORITHM...
        x, y, FPPBRepsAPPROX[i], FPPBRepsWSNE[i] = approxNEConstructionFPPBR(reduced_m,reduced_n,reduced_R,reduced_C)
        ### EXECUTING FICTITIOUS PLAY UNIFORM ALGORITHM...
        x, y,  FPUNIepsAPPROX[i],  FPUNIepsWSNE[i] = approxNEConstructionFPUNI(reduced_m,reduced_n,reduced_R,reduced_C)

        
        if DMPepsAPPROX[i] > minDMPepsAPPROX:
            minDMPepsAPPROX = DMPepsAPPROX[i]
            DMPApproxNEWorstGame1ROW = R
            DMPApproxNEWorstGame1COL = C
        if DMPepsWSNE[i] > minDMPepsWSNE:
            minDMPepsWSNE = DMPepsWSNE[i]
            DMPApproxNEWorstGame2ROW = R
            DMPApproxNEWorstGame2COL = C
        if DELepsAPPROX[i] > minDELepsAPPROX:
            minDELepsAPPROX = DELepsAPPROX[i]
            DELApproxNEWorstGame1ROW = R
            DELApproxNEWorstGame1COL = C
        if DELepsWSNE[i] > minDELepsWSNE:
            minDELepsWSNE = DELepsWSNE[i]
            DELApproxNEWorstGame2ROW = R
            DELApproxNEWorstGame2COL = C
        if FPPBRepsAPPROX[i] > minFPPBRepsAPPROX:
            minFPPBRepsAPPROX = FPPBRepsAPPROX[i]
            FPPBRApproxNEWorstGame1ROW = R
            FPPBRApproxNEWorstGame1COL = C
        if FPPBRepsWSNE[i] > minFPPBRepsWSNE:
            minFPPBRepsWSNE = FPPBRepsWSNE[i]
            FPPBRApproxNEWorstGame2ROW = R
            FPPBRApproxNEWorstGame2COL = C
        if FPUNIepsAPPROX[i] > minFPUNIepsAPPROX:
            minFPUNIepsAPPROX = FPUNIepsAPPROX[i]
            FPUNIApproxNEWorstGame1ROW = R
            FPUNIApproxNEWorstGame1COL = C
        if FPUNIepsWSNE[i] > minFPUNIepsWSNE:
            minFPUNIepsWSNE = FPUNIepsWSNE[i]
            FPUNIApproxNEWorstGame2ROW = R
            FPPBRApproxNEWorstGame2COL = C

    
  
    counts, bins = np.histogram(DMPepsAPPROX, range=(0,1))
    DMPApproxNEHistogram = counts
    plt.title('DMPepsApproxNE')
    plt.xlabel('ε')
    plt.ylabel('Counts')
    plt.stairs(counts,bins)
    plt.savefig('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DMPApproxNEHistogram.jpg')
    plt.close()

    counts, bins = np.histogram(DELepsAPPROX, range=(0,1))
    DELApproxNEHistogram = counts
    plt.title('DELepsApproxNE')
    plt.xlabel('ε')
    plt.ylabel('Counts')
    plt.stairs(counts,bins)
    plt.savefig('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DELApproxNEHistogram.jpg')
    plt.close()

    counts, bins = np.histogram(FPPBRepsAPPROX, range=(0,1))
    FPPBRApproxNEHistogram = counts
    plt.title('FPPBRepsAPPROX')
    plt.xlabel('ε')
    plt.ylabel('Counts')
    plt.stairs(counts,bins)
    plt.savefig('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPPBRApproxNEHistogram.jpg')
    plt.close()

    counts, bins = np.histogram(FPUNIepsAPPROX, range=(0,1))
    FPUNIApproxNEHistogram = counts
    plt.title('FPUNIepsApproxNE')
    plt.xlabel('ε')
    plt.ylabel('Counts')
    plt.stairs(counts,bins)
    plt.savefig('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPUNIApproxNEHistogram.jpg')
    plt.close()

    counts, bins = np.histogram(DMPepsWSNE, range=(0,1))
    DMPWSDMPepsWSNEHistogram = counts
    plt.title('DMPepsWSNE')
    plt.xlabel('ε')
    plt.ylabel('Counts')
    plt.stairs(counts,bins)
    plt.savefig('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DMPepsWSNEHistogram.jpg')
    plt.close()

    counts, bins = np.histogram(DELepsWSNE, range=(0,1))
    DELWSDMPepsWSNEHistogram = counts
    plt.title('DELepsWSNE')
    plt.xlabel('ε')
    plt.ylabel('Counts')
    plt.stairs(counts,bins)
    plt.savefig('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DELepsWSNEHistogram.jpg')
    plt.close()

    counts, bins = np.histogram(FPPBRepsWSNE, range=(0,1))
    FPPBRWSDMPepsWSNEHistogram = counts
    plt.title('FPPBRepsWSNE')
    plt.xlabel('ε')
    plt.ylabel('Counts')
    plt.stairs(counts,bins)
    plt.savefig('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPPBRepsWSNEHistogram.jpg')
    plt.close()

    counts, bins = np.histogram(FPUNIepsWSNE, range=(0,1))
    FPUNIWSDMPepsWSNEHistogram = counts
    plt.title('FPUNIepsWSNE')
    plt.xlabel('ε')
    plt.ylabel('Counts')
    plt.stairs(counts,bins)
    plt.savefig('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPUNIepsWSNEHistogram.jpg')
    plt.close()

    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DMPApproxNEHistogram', DMPApproxNEHistogram, delimiter=',', fmt='%4d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DMPWSNEHistogram', DMPWSNEHistogram, delimiter=',', fmt='%4d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DMPApproxNEWorstGame1ROW', DMPApproxNEWorstGame1ROW, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DMPApproxNEWorstGame1COL', DMPApproxNEWorstGame1COL, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DMPApproxNEWorstGame2ROW', DMPApproxNEWorstGame2ROW, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DMPApproxNEWorstGame2COL', DMPApproxNEWorstGame2COL, delimiter=',', fmt='%1d')

    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DELApproxNEHistogram', DELApproxNEHistogram, delimiter=',', fmt='%4d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DELWSNEHistogram', DELWSNEHistogram, delimiter=',', fmt='%4d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DELApproxNEWorstGame1ROW', DELApproxNEWorstGame1ROW, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DELApproxNEWorstGame1COL', DELApproxNEWorstGame1COL, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DELApproxNEWorstGame2ROW', DELApproxNEWorstGame2ROW, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'DELApproxNEWorstGame2COL', DELApproxNEWorstGame2COL, delimiter=',', fmt='%1d')

    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPPBRApproxNEHistogram', FPPBRApproxNEHistogram, delimiter=',', fmt='%4d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPPBRWSNEHistogram', FPPBRWSNEHistogram, delimiter=',', fmt='%4d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPPBRApproxNEWorstGame1ROW', FPPBRApproxNEWorstGame1ROW, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPPBRApproxNEWorstGame1COL', FPPBRApproxNEWorstGame1COL, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPPBRApproxNEWorstGame2ROW', FPPBRApproxNEWorstGame2ROW, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPPBRApproxNEWorstGame2COL', FPPBRApproxNEWorstGame2COL, delimiter=',', fmt='%1d')

    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPUNIApproxNEHistogram', FPUNIApproxNEHistogram, delimiter=',', fmt='%4d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPUNIWSNEHistogram', FPUNIWSNEHistogram, delimiter=',', fmt='%4d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPUNIApproxNEWorstGame1ROW', FPUNIApproxNEWorstGame1ROW, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPUNIApproxNEWorstGame1COL', FPUNIApproxNEWorstGame1COL, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPUNIApproxNEWorstGame2ROW', FPUNIApproxNEWorstGame2ROW, delimiter=',', fmt='%1d')
    np.savetxt('./EXPERIMENTS/P'+str(num)+'/P'+str(num)+'FPUNIApproxNEWorstGame2COL', FPUNIApproxNEWorstGame2COL, delimiter=',', fmt='%1d')

### F. MENU ###
def menu():
    
    m=[]
    n=[]
    R=[]
    C=[]
    
    while True:

        num = input(bcolors.QUESTION+ 'What would you like to do? \n1)Load Bimatrix game \n2)Save new Bimatrix game \n3)Run experiments\n' + bcolors.ENDC)
        if int(num) == 1:
            try:
                filename = input(bcolors.QUESTION+ 'Type file name containing the R matrix: ' + bcolors.ENDC)
                R = np.genfromtxt(filename, delimiter=',', skip_header=0)
                filename = input(bcolors.QUESTION+ 'Type file name containing the C matrix: ' + bcolors.ENDC)
                C = np.genfromtxt(filename, delimiter=',', skip_header=0)
                m=len(R)
                n=len(R[0])

                break

            except:
                print(bcolors.ERROR+ 'File not found!' + bcolors.ENDC)

        elif int(num) == 2:
            try:
                m = int(input(bcolors.QUESTION+ 'Enter number of rows: ' + bcolors.ENDC))
                n = int(input(bcolors.QUESTION+ 'Enter number of columns: ' + bcolors.ENDC))
                R = np.zeros((m,n))
                C = R

                print(bcolors.GREEN + 'Creating R matrix: \n' + bcolors.ENDC)
                for i in range(m):
                    for j in range(n):
                        R[i][j] = input(bcolors.QUESTION+ '\nEnter value for row ' + str(i+1) + ', column '+ str(j+1) +': ' + bcolors.ENDC)
                
                filename = input(bcolors.QUESTION+ 'Enter file name: '+ bcolors.ENDC)
                np.savetxt(filename,R,delimiter=',', fmt='%1.4e')

                print(bcolors.GREEN + 'Creating C matrix: \n' + bcolors.ENDC)
                for i in range(m):
                    for j in range(n):
                        C[i][j] = input(bcolors.QUESTION+ '\nEnter value for row ' + str(i+1) + ', column '+ str(j+1) +': ' + bcolors.ENDC)

                filename = input(bcolors.QUESTION+ 'Enter file name: '+ bcolors.ENDC)
                np.savetxt(filename,C,delimiter=',', fmt='%1.4e')

            except:
                print(bcolors.ERROR + 'Wrong input' + bcolors.ENDC)
        
        elif int(num) == 3:
            num = int(input(bcolors.QUESTION+ 'What would you like to do? \n1)Run premade experiments \n2)Run custom experiment\n'+ bcolors.ENDC))
            if num == 1:
                defaultExperiments()
            elif num == 2:
                print('custom')
                #TODO
            else:
                return

        else:
            return

    num = int(input(bcolors.QUESTION+ 'What would you like to do? \n1)Calculate pure NE \n2)Calculate approximate NE\n' +bcolors.ENDC))
    
    if num == 1:
        drawBimatrix(m,n,R,C)
        ### SEEKING FOR PNE IN THE GAME (R,C)...

        (i,j)= checkForPNE(m,n,R,C)
        if (i,j) != (0,0):
            print( bcolors.MSG + "A pure NE (",i,",",j,") was discovered for (R,C)." + bcolors.ENDC )
            #exit()
        else:
            print( bcolors.MSG + "No pure NE exists for (R,C)." + bcolors.ENDC )
    
    elif num == 2:
        num = int(input(bcolors.QUESTION+ 'Select algorithm:\n 1)DMP \n2)DEL \n3)FP PBR \n4)FP UNIFORM\n' + bcolors.ENDC))
        
        drawBimatrix(m,n,R,C)

        if num == 1:
            ### EXECUTING DMP ALGORITHM...
            x, y, DMPepsAPPROX, DMPepsWSNE = approxNEConstructionDMP(m,n,R,C)
            #DMPx, DMPy = interpretReducedStrategiesForOriginalGame(x, y, R, C, reduced_R, reduced_C)
            print( bcolors.MSG + PLUSLINE) 
            print("\tConstructed solution for DMP:")
            print(MINUSLINE)
            print("\tDMPx =",x,"\n\tDMPy =",y)
            print("\tDMPepsAPPROX =",DMPepsAPPROX,".\tDMPepsWSNE =",DMPepsWSNE,"." + bcolors.ENDC)
            print( PLUSLINE + bcolors.ENDC )

        elif num == 2:
            ### EXECUTING DEL ALGORITHM...
            x, y, DELepsAPPROX, DELepsWSNE = approxNEConstructionDEL(m,m,R,C)
            #DELx, DELy = interpretReducedStrategiesForOriginalGame(x, y, R, C, reduced_R, reduced_C)
            print( bcolors.MSG + PLUSLINE )
            print("\tConstructed solution for DEL:")
            print(MINUSLINE)
            print("\tDELx =",x,"\n\tDELy =",y)
            print("\tDELepsAPPROX =",DELepsAPPROX,".\tDELepsWSNE =",DELepsWSNE,".")
            print( PLUSLINE + bcolors.ENDC )
        
        elif num == 3:
            ### EXECUTING FICTITIOUS PLAY PBR ALGORITHM...
            x, y, FPPBRepsAPPROX, FPPBRepsWSNE = approxNEConstructionFPPBR(m,n,R,C)
            #FPx, FPy = interpretReducedStrategiesForOriginalGame(x, y, R, C, reduced_R, reduced_C)
            print( bcolors.MSG + PLUSLINE )
            print("\tConstructed solution for FICTITIOUS PLAY PBR:")
            print(MINUSLINE)
            print("\tFPPBRx =",x,"\n\tFPPBRy =",y)
            print("\tFPPBRepsAPPROX =",FPPBRepsAPPROX,".\tFPBRPepsWSNE =",FPPBRepsWSNE,".")
            print( PLUSLINE + bcolors.ENDC )

        elif num == 4:
            ### EXECUTING FICTITIOUS PLAY UNIFORM ALGORITHM...
            x, y,  FPUNIepsAPPROX,  FPUNIepsWSNE = approxNEConstructionFPUNI(m,n,R,C)
            #FPx, FPy = interpretReducedStrategiesForOriginalGame(x, y, R, C, reduced_R, reduced_C)
            print( bcolors.MSG + PLUSLINE )
            print("\tConstructed solution for FICTITIOUS PLAY PBR:")
            print(MINUSLINE)
            print("\t FPUNIx =",x,"\n\t FPUNIy =",y)
            print("\t FPUNIepsAPPROX =", FPUNIepsAPPROX,".\tFPBRPepsWSNE =", FPUNIepsWSNE,".")
            print( PLUSLINE + bcolors.ENDC )
        
        else:
            return

    else:
        return
### MAIN PROGRAM FOR LAB-2 ###

LINELENGTH  = 80
EQLINE      = drawLine(LINELENGTH,'=')
MINUSLINE   = drawLine(LINELENGTH,'-')
PLUSLINE    = drawLine(LINELENGTH,'+')

print_LAB2_preamble()

screen_clear()

maxNumOfRandomGamesToSolve = 10000
maxNumberOfActions = 20

menu()




drawBimatrix(m,n,R,C)






print(bcolors.MSG + "Reduced bimatrix, after removal of strictly dominated actions:")
drawBimatrix(reduced_m,reduced_n,reduced_R,reduced_C)







