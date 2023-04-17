# 2-D NIM Game
import os
import random

############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    # pure colors...
    GREY      = '\033[90m'
    RED       = '\033[91m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    BLUE      = '\033[94m'
    BLUE      = '\033[94m'
    PURPLE    = '\033[95m'
    CYAN      = '\033[96m'
    # color styles...
    HEADER      = '\033[95m'
    QUESTION    = '\033[93m\033[3m'
    MSG         = '\033[96m'
    WARNING     = '\033[93m'
    ERROR       = '\033[91m'
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
   
def initializeBoard(N):
        board = ['']*(N*N+1)

        # this is the COUNTER of cells in the board already filled with R or G
        board[0] = 0
        
        # each EMPTY cell in the board contains its cardinal number 
        for i in range(N*N):
                if i < 9:
                        board[i+1] = ' ' + str(i+1)
                else:
                        board[i+1] = str(i+1)
        return board

def drawNimPalette(board,N):

        EQLINE          = '\t'
        MINUSLINE       = '\t'
        CONSECEQUALS    = ''
        CONSECMINUS     = ''
        for i in range(5):
                CONSECEQUALS    = CONSECEQUALS  + '='
                CONSECMINUS     = CONSECMINUS   + '-'

        for i in range(10):
                EQLINE          = EQLINE        + CONSECEQUALS
                MINUSLINE       = MINUSLINE     + CONSECMINUS

        for i in range(N):
                #PRINTING ROW i...
                if i == 0:
                        print(EQLINE)
                else:
                        print(MINUSLINE)

                printRowString = ''

                for j in range(N):
                        # PRINTING CELL (i,j)...
                        CellString = str(board[N*i+j+1])
                        if CellString == 'R': 
                                CellString = ' ' + bcolors.RED + CellString + bcolors.ENDC
                        
                        if CellString == 'G':
                                CellString = ' ' + bcolors.GREEN + CellString + bcolors.ENDC

                        if printRowString == '':
                                printRowString = '\t[ ' + CellString
                        else:
                                printRowString =  printRowString + ' | ' + CellString
                printRowString = printRowString + ' ]'
                print (printRowString)
        print ( EQLINE )
        print ( bcolors.PURPLE + '\t\t\tCOUNTER = [ ' + str(board[0]) + ' ]'  + bcolors.ENDC )
        print ( EQLINE )

def inputPlayerLetter():
        # The player chooses which label (letter) will fill the cells
        letter = ''
        while not(letter == 'G' or letter == 'R'):
                print ( bcolors.QUESTION + '[Q1] What letter do you choose to play? [ G(reen) | R(ed) ]' + bcolors.ENDC )
                letter = input().upper()
                # The first letter corresponds to the HUMAN and the second element corresponds to the COMPUTER
                if letter == 'G':
                        return ['G','R']
                else:
                        if letter == 'R':
                                return ['R','G']
                        else:
                                print (bcolors.ERROR + 'ERROR1: You provided an invalid choice. Please try again...' + bcolors.ENDC)

def whoGoesFirst():
        if random.randint(0,1) == 0:
                return 'computer'
        else:
                return 'player'

def howComputerPlays():
        
        while True:
                print ( bcolors.QUESTION + '[Q5] How will the computer play? [ R (randomly) | F (first Free) | C (copycat)]' + bcolors.ENDC )
                strategyLetter = input().upper()
        
                if strategyLetter == 'R':
                        return 'random'
                else: 
                        if strategyLetter == 'F':
                                return 'first free'
                        else:
                                if strategyLetter == 'C':
                                        return 'copycat'
                                else:
                                        print( bcolors.ERROR + 'ERROR 3: Incomprehensible strategy was provided. Try again...' + bcolors.ENDC )

def getBoardSize():

        BoardSize = 0
        while BoardSize < 1 or BoardSize > 10:
                GameSizeString = input('Determine the size 1 =< N =< 10, for the NxN board to play: ')
                if GameSizeString.isdigit():
                        BoardSize = int(GameSizeString)
                        if BoardSize < 1 or BoardSize > 10:
                                print( bcolors.ERROR + 'ERROR 4: Only positive integers between 1 and 10 are allowable values for N. Try again...' + bcolors.ENDC ) 
                else:
                        print( bcolors.ERROR + 'ERROR 5: Only positive integers between 1 and 10 are allowable values for N. Try again...' + bcolors.ENDC ) 
        return( BoardSize )

def startNewGame():
        # Function for starting a new game
        print(bcolors.QUESTION + '[Q0] Would you like to start a new game? (yes or no)' + bcolors.ENDC)
        return input().lower().startswith('y')

def continuePlayingGame():
        # Function for starting a new game
        print(bcolors.QUESTION + '[Q2] Would you like to continue playing this game? (yes or no)' + bcolors.ENDC)
        return input().lower().startswith('y')

def playAgain():
        # Function for replay (when the player wants to play again)
        print(bcolors.QUESTION + '[Q3] Would you like to continue playing this game? (yes or no)' + bcolors.ENDC)
        return input().lower().startswith('y')

def isBoardFull(board,N):
        return board[0] == N*N

def getRowAndColumn(move,N):
        moveRow         = 1 + (move - 1) // N
        moveColumn      = 1 + (move - 1) % N
        return(moveRow,moveColumn)




####### Student Functions Start Here ########
def play(board, N, player):
        finish = False
        moveCounter = 0 #Moves counter (up to 2 {three moves})
        move = [] #Array that holds the player's moves
        
        while not finish:
                move.append (input(bcolors.QUESTION+'Enter a number to make a move: '+bcolors.ENDC))

                if move[moveCounter].isnumeric():

                        move[moveCounter] = int(move[moveCounter])
                        #Check if move is on the board
                        if move[moveCounter] <= N*N and move[moveCounter] > 0:

                                #First move (player has to play this one)
                                if moveCounter == 0:
                                        screen_clear()
                                        check, finish = firstMoveChecks(board,N,move,player)
                                        drawNimPalette(board,N)
                                        if check:
                                                moveCounter += 1
                                                if not finish:
                                                        finish = not continuePlayingGame()
                                #Second move
                                elif moveCounter == 1:
                                        screen_clear()
                                        if secondMoveChecks(board,N,move,player):
                                                drawNimPalette(board,N)
                                                moveCounter += 1
                                                finish = not continuePlayingGame() 
                                        else:
                                                drawNimPalette(board,N)
                                                finish = True
                                #Third move (last move)
                                elif moveCounter == 2:
                                        screen_clear()
                                        thirdMoveChecks(board,N,move,player)
                                        drawNimPalette(board,N)
                                        finish = True
                        else:
                                print(bcolors.ERROR+"This is not a legitimate move"+bcolors.ENDC)
                else:
                        print(bcolors.ERROR+"Input must be an integer"+bcolors.ENDC) 


def getComputerMove_random (board, N, player):
   finish = False 
   
   moveCounter = 0 #Moves counter (up to 2 {three moves})
   move = [] #Array that holds the player's moves
   while not finish:
               emptyCells=getEmptyCells(board,N)
               if not emptyCells:
                         finish=True


               else :
                         move.append(random.choice(emptyCells)) 
                         numMove=random.randint(0,2)

                  #First move (player has to play this one)
                         screen_clear()
                         finish = firstMoveRandChecks(board,N,move,player)
                         drawNimPalette(board,N)
                         moveCounter += 1
                         if numMove==0 or finish==True:
                                   finish=True
                                   break
                        
                           #Second move
                         else:
                                screen_clear()
                                emptyCells=getEmptyCells(board,N)

                                for i in range(len(emptyCells)):
                                         row,column=getRowAndColumn(emptyCells[i],N)
                                         if row!=column :
                                            if isSequential(emptyCells[i],move[0],N):
                                                  move.append(emptyCells[i])
                                                  board[move[1]] = player
                                                  board[0] +=1
                                                  break
                                         else :
                                                i+=1
                                drawNimPalette(board,N)
                                moveCounter += 1
                                if numMove==1 or finish==True:
                                   finish=True
                                   break

                                elif numMove==2:
                                   #Third move (last move)
                                     screen_clear()
                                     emptyCells=getEmptyCells(board,N)
                                     for i in range(len(emptyCells)):
                                         row,column=getRowAndColumn(emptyCells[i],N)
                                         if row!=column :
                                            if isSequential2Cells(emptyCells[i],move[0],move[1], N):
                                                  board[emptyCells[i]] = player
                                                  board[0] +=1
                                                  break
                                         else :
                                                   i+=1
                                            
                                     screen_clear()
                                     
                                     drawNimPalette(board,N)
                                     finish = True



      
                              



def firstMoveRandChecks(board, N, move,player): #Checks if the first move is valid

        #Flags for checking and player movement ending
        endflag = False

      
         #Make move
        board[move[0]] = player
        board[0] +=1 
        

                #If cell is on the diagonal, player can't play again
        row,column = getRowAndColumn(move[0],N)
        if row == column:
          endflag = True
 
        return  endflag



def firstMoveChecks(board, N, move,player): #Checks if the first move is valid

        #Flags for checking and player movement ending
        checkflag, endflag = False, False

        #Check if cell is empty
        if board[move[0]] == 'G' or board[move[0]] == 'R':
                print(bcolors.ERROR+"This cell is not empty!"+bcolors.ENDC)

        else:

                #Make move
                board[move[0]] = player
                board[0] +=1 
                checkflag = True

                #If cell is on the diagonal, player can't play again
                row,column = getRowAndColumn(move[0],N)
                if row == column:
                        endflag = True

        

        return checkflag, endflag

def secondMoveRandChecks(board,N,move,player):
    finish=False
    avCells=getEmptyCells(board,N)

    for i in range (len(avCells)):
            if isSequential(move[0],avCells[i], N) :
                move[1]=avCells[i]
                break
    if not move[1]:
            finish=True
    else:
             board[move[1]] = player
             board[0] +=1

    return finish 









def secondMoveChecks(board, N, move,player): #Checks if the second move is valid

        checkflag = False

        if board[move[1]] == 'G' or board[move[1]] == 'R':
                print(bcolors.ERROR+"This cell is not empty!"+bcolors.ENDC)

        else: #Checks if cell is empty 

                row,column =getRowAndColumn(move[1], N)

                #Checks if the player is trying to play on the diagonal
                if row == column:
                        print(bcolors.ERROR+"You can't play on the diagonal!"+bcolors.ENDC)
        
                #If it's in squential cells let it play
                elif isSequential(move[0],move[1], N):
                        board[move[1]] = player
                        board[0] +=1
                        checkflag= True              

                else:
                        print(bcolors.ERROR+'The cells have to be squential'+bcolors.ENDC)               
        
        return checkflag

def thirdMoveRandChecks(board,N,move,player):
        finish=False
        avCellNum=getEmptyCellNum(board,N)
        for i in range(len(avCellNum)):
                 if isSequential2Cells(move[0],move[1],move[2], N):
                      board[move[2]] = player    
                      board[0] +=1
                 else:
                         finish=True
        return finish      
   



def thirdMoveChecks(board, N, move, player): #Checks if the third move is valid

        if board[move[2]] == 'G' or board[move[2]] == 'R':
                print(bcolors.ERROR+"This cell is not empty!"+bcolors.ENDC)

        else: #Checks if cell is empty 

                row,column =getRowAndColumn(move[2], N)

                #Checks if the player is trying to play on the diagonal
                if row == column:
                        print(bcolors.ERROR+"You can't play on the diagonal!"+bcolors.ENDC)
        
                #If it's in squential cells let it play
                elif isSequential2Cells(move[0],move[1],move[2], N):
                        board[move[2]] = player    
                        board[0] +=1      

                else:
                        print(bcolors.ERROR+'The cells have to be squential'+bcolors.ENDC)                       

def isSequential(prevMove, nextMove, N):#Checks if two moves are made in sequential cells

        #Get next and previous move's positions
        prevRow, prevColumn = getRowAndColumn(prevMove,N)
        nextRow, nextColumn = getRowAndColumn(nextMove,N)

        #Check if they are at the same row or column
        sameRow = prevRow == nextRow
        sameColumn = prevColumn == nextColumn

        #Check if they are next to each other
        if sameRow and (prevColumn == nextColumn-1 or prevColumn == nextColumn+1) :
                return True
        elif sameColumn and (prevRow == nextRow-1 or prevRow == nextRow+1) :
                return True
        else :
                return False

def isSequential2Cells(move1, move2, move3, N):
        flag = False

        #Get moves positions
        row1, col1 = getRowAndColumn(move1, N)
        row2, col2 = getRowAndColumn(move2, N)
        row3, col3 = getRowAndColumn(move3, N)
        
        #Check if the first two moves are on the same column or row
        sameRow = row1 == row2
        sameColumn = col1 == col2

        #Check if the next move is on the same column/row and sequential to the previous ones
        if sameRow and row2 == row3:
                check1 = col2 == col3-1
                check2 = col2 == col3+1
                check3 = col1 == col3-1
                check4 = col1 == col3+1
                if check1 or check2 or check3 or check4:
                        flag = True
        elif sameColumn and col2 == col3:
                check1 = row2 == row3-1
                check2 = row2 == row3+1
                check3 = row1 == row3-1
                check4 = row1 == row3+1
                if check1 or check2 or check3 or check4:
                        flag = True

        return flag

def getMove(row,column, N):
        move = N*(row-1) + column
        return move

def getEmptyCellNum(board,N):
        return N*N-board[0]

def getEmptyCells(board,N):
        cells =[]
        for i in range(1,N*N):
                if board[i] != 'R' and board[i] != 'G':
                        cells.append(i)
        
        return cells



######### MAIN PROGRAM BEGINS #########

screen_clear()

print(bcolors.HEADER + """
---------------------------------------------------------------------
                     CEID NE509 / LAB-1  
---------------------------------------------------------------------
STUDENT NAME:           < provide your name here >
STUDENT AM:             < provide your AM here >
JOINT WORK WITH:        < provide your partner's name and AM here >
---------------------------------------------------------------------
""" + bcolors.ENDC)

input("Press ENTER to continue...")
screen_clear()

print(bcolors.HEADER + """
---------------------------------------------------------------------
                     2-Dimensional NIM Game: RULES (I)
---------------------------------------------------------------------
    1.      A human PLAYER plays against the COMPUTER.
    2.      The starting position is an empty NxN board.
    3.      One player (the green) writes G, the other player 
            (the red) writes R, in empty cells.
""" + bcolors.ENDC ) 

input("Press ENTER to continue...")
screen_clear()

print(bcolors.HEADER + """
---------------------------------------------------------------------
                     2-Dimensional NIM Game: RULES (II) 
---------------------------------------------------------------------
    4.      The cells within the NxN board are indicated as 
            consecutive numbers, from 1 to N^2, starting from the 
            upper-left cell. E.g. for N=4, the starting position 
            and some intermediate position of the game would be 
            like those:
                    INITIAL POSITION        INTERMEDIATE POSITION
                    =====================   =====================
                    [  1 |  2 |  3 |  4 ]   [  1 |  2 |  3 |  4 ]
                    ---------------------   ---------------------
                    [  5 |  6 |  7 |  8 ]   [  5 |  R |  7 |  8 ]    
                    ---------------------   ---------------------
                    [  9 | 10 | 11 | 12 ]   [  9 |  R | 11 | 12 ] 
                    ---------------------   ---------------------
                    [ 13 | 14 | 15 | 16 ]   [  G |  G | 15 |  G ] 
                    =====================   =====================
                       COUNTER = [ 0 ]         COUNTER = [ 5 ]
                    =====================   =====================
""" + bcolors.ENDC )

input("Press ENTER to continue...")
screen_clear()

print(bcolors.HEADER + """
---------------------------------------------------------------------
                     2-Dimensional NIM Game: RULES (III) 
---------------------------------------------------------------------
    5.      In each round the current player's turn is to fill with 
            his/her own letter (G or R) at least one 1 and at most 
            3 CONSECUTIVE, currently empty cells of the board, all 
            of them lying in the SAME ROW, or in the SAME COLUMN 
            of the board. Alternatively, the player may choose ONLY
            ONE empty diagonal cell to play.
    6.      The player who fills the last cell in the board WINS.
    7.      ENJOY!!!
---------------------------------------------------------------------
""" + bcolors.ENDC)


maxNumMoves = 3

playNewGameFlag = True

while playNewGameFlag:

        if not startNewGame():
                break

        N = getBoardSize()

        nimBoard = initializeBoard(N)

        playerLetter, computerLetter = inputPlayerLetter()

        turn = whoGoesFirst()

        computerStrategy = howComputerPlays()

        print( bcolors.MSG + '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'    + bcolors.ENDC )
        print( bcolors.MSG + 'A new ' +str(N) +'x' + str(N) +' game is about to start. The ' + turn + ' makes the first move.' + bcolors.ENDC )
        print( bcolors.MSG + ' * The computer will play according to the ' + bcolors.HEADER + computerStrategy + bcolors.MSG +' strategy.' + bcolors.ENDC )
        print( bcolors.MSG + ' * The player will use the letter ' + playerLetter + ' and the computer will use the ' + computerLetter +'.' + bcolors.ENDC )
        print( bcolors.MSG + ' * The first move will be done by the ' + turn +'.' + bcolors.ENDC )
        print( bcolors.MSG + '---------------------------------------------------------------------'    + bcolors.ENDC )
        print( bcolors.MSG + 'Provide your own code here, to implement the workflow of the game:'       + bcolors.ENDC )
        print( bcolors.MSG + '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'    + bcolors.ENDC )
        drawNimPalette(nimBoard,N)
        
######### Student Main Program Code Starts Here ##########
        gameon = True

        while gameon:
                print(bcolors.GREEN+ turn + ' plays!'+ bcolors.ENDC)
                if turn == 'player':
                        play(nimBoard,N,playerLetter)
                        turn = 'computer'
                else: 
                       getComputerMove_random(nimBoard,N,computerLetter)
                      # play(nimBoard,N,computerLetter) #Wiil be replaced
                       turn = 'player'

                gameon = not isBoardFull(nimBoard, N)
        
        print(bcolors.RED+turn+' lost :('+bcolors.ENDC)
######### MAIN PROGRAM ENDS #########
