import numpy as np
import random

class FreecellEnvironment:
    def __init__(self, maxMove):
        self.maxMove = maxMove
        self.tableau = [[] for _ in range(8)]
        self.foundations = [[] for _ in range(4)]
        self.freeCells = [[] for _ in range(4)]
        self.numOfMove = 0
        cards = [
            26, 40, 28, 42, 41, 16, 30, 48, 43, 31,  6,
            15, 39, 36,  3,  0, 14, 50, 17,  2, 29,  5,
            27, 38, 21, 20, 47,  7, 19,  8, 13,  1, 46,
            22, 35, 10, 34, 51, 49, 33, 45, 24,  9, 23,
            12, 11, 37,  4, 18, 32, 44, 25
            ]

        stock = [
            201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 301, 302, 303, 304,
            305, 306, 307, 308, 309, 310, 311, 312, 313, 401, 402, 403, 404, 405, 406, 407, 408,
            409, 410, 411, 412, 413, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
            113,
        ]

        stock_copy = []
        for i in range(len(stock)):
            stock_copy.append(stock[cards[i]])
        
        for i in range(7):
            for j in range(8):
                if len(stock_copy)>0:
                    card = stock_copy.pop()
                    self.tableau[j].append(card)
        
        self.prevMove = None
   
    
    def get_movable(self):
        moveToFoundation = []
        cardAcceptToFoundation = []

        for j in range(len(self.foundations)):
            colFoundation = self.foundations[j]
            if len(colFoundation)==0:
                cardAcceptToFoundation.append(101)
                cardAcceptToFoundation.append(201)
                cardAcceptToFoundation.append(301)
                cardAcceptToFoundation.append(401)
                continue
            
            card = colFoundation[-1]
            cardAcceptToFoundation.append(card+1)
        

        cardAcceptToFoundation = list(dict.fromkeys(cardAcceptToFoundation))

        for i in range(len(self.tableau)):
            col = self.tableau[i]

            if len(col) == 0: continue

            card = col[-1]
            cardSuit = card // 100

            for j in range(len(self.foundations)):
                colFoundation = self.foundations[j]
                cardOnFoundation = 0 + cardSuit*100

                if len(colFoundation) !=0:
                    cardOnFoundation = colFoundation[-1]
                
                if cardOnFoundation+1==card:
                    moveToFoundation.append({
                        "from": 'T'+str(i),
                        "card": card,
                        "dest": 'F'+str(j),
                        "isMoveToFoundation": True
                    })
        
        movable = []
        self.find_movable_Tableau(self.tableau, movable, 'T')
        self.find_movable_Tableau(self.freeCells, movable, 'C')

        listMovable = moveToFoundation + movable
        random.shuffle(listMovable)
        return listMovable


    def find_movable_Tableau(self, tableau , movable, key):
        n = len(list(filter(lambda x: len(x)==0, self.freeCells)))
        m = len(list(filter(lambda x: len(x)==0, self.tableau)))

        for i in range(len(tableau)):
            col = tableau[i]

            if len(col) == 0: continue

            for x in range(len(col)-1, -1,-1):
                card = col[x]
                cardSuit = card //100
                cardNumber = card % 100
                stack = len(col) - x
                if x < len(col) -1:
                    lastCard = col[x+1]
                    lastCardSuit = lastCard // 100
                    lastCardNumber = lastCard % 100
                    if (cardSuit >=3) == (lastCardSuit<=3) or (cardNumber -1) != lastCardNumber : break
                
                if stack <= 2**(m-1) *(n+1):
                    for j in range(len(self.tableau)):
                        colDest = self.tableau[j]
                        if len(colDest) == 0:
                            movable.append({
                                "from": key+str(i),
                                "card": card,
                                "dest": 'T'+str(j),
                                "isMoveToFoundation": False
                            })
                            break
                
                if stack <= 2**m * (n+1):
                    for j in range(len(self.tableau)):
                        colDest = self.tableau[j]

                        if len(colDest)==0: continue
                        cardDest = colDest[-1]
                        cardSuitDest = cardDest //100
                        cardNumberDest = cardDest % 100

                        if (cardSuit >= 3) != (cardSuitDest >= 3):
                            if cardNumber +1 == cardNumberDest:
                                movable.append({
                                    "from": key+str(i),
                                    "card": card,
                                    "dest": 'T'+str(j),
                                    "isMoveToFoundation": False
                                })
                                break
                
                if stack == 1 and key !='C':
                    for j in range(len(self.freeCells)):
                        colDest = self.freeCells[j]
                        if len(colDest)==0:
                            movable.append({
                                    "from": key+str(i),
                                    "card": card,
                                    "dest": 'C'+str(j),
                                    "isMoveToFoundation": False
                                })
                            break

    def _move(self, action):
        self.numOfMove+=1

        if action["from"][0] == 'T':
            source = self.tableau
        else:
            source = self.freeCells
        
        if action["dest"][0] == 'T':
            dests = self.tableau
        elif action["dest"][0] == 'F':
            dests = self.foundations
        elif action["dest"][0] == 'C':
            dests = self.freeCells
        
        dest = dests[int(action["dest"][1])]
        _from = source[int(action["from"][1])]

        card = action["card"]
        idx = _from.index(card)
        dests[int(action["dest"][1])] = dest + _from[idx:]
        source[int(action["from"][1])] = _from[:idx]

    def printBoard(self):
        print("freeCells: ", self.freeCells)
        print("foundations: ", self.foundations)
        
        for i in range(len(self.tableau)):
            print(self.tableau[i])
        
    def reset(self):
        self.prevMove = None
        self.tableau = [[] for _ in range(8)]
        self.foundations = [[] for _ in range(4)]
        self.freeCells = [[] for _ in range(4)]
        self.numOfMove = 0

        cards = [
            26, 40, 28, 42, 41, 16, 30, 48, 43, 31,  6,
            15, 39, 36,  3,  0, 14, 50, 17,  2, 29,  5,
            27, 38, 21, 20, 47,  7, 19,  8, 13,  1, 46,
            22, 35, 10, 34, 51, 49, 33, 45, 24,  9, 23,
            12, 11, 37,  4, 18, 32, 44, 25
            ]

        stock = [
            201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 301, 302, 303, 304,
            305, 306, 307, 308, 309, 310, 311, 312, 313, 401, 402, 403, 404, 405, 406, 407, 408,
            409, 410, 411, 412, 413, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
            113,
        ]

        stock_copy = []
        for i in range(len(stock)):
            stock_copy.append(stock[cards[i]])
        
        for i in range(7):
            for j in range(8):
                if len(stock_copy)>0:
                    card = stock_copy.pop()
                    self.tableau[j].append(card)
        return self.getState()

    def check_win(self, action):
        if self.numOfMove >self.maxMove:
            return -5, True
        
        movable = self.get_movable()

        if len(movable) == 0:
            for i in range(len(self.foundations)):
                if len(self.foundations[i])<13:
                    return -5, True
                return 5, True
        else:
            isMoveToFoundation = action["isMoveToFoundation"]
            if isMoveToFoundation:
                return 1, False
        return 0, False
    
    def getState(self):
        temp = []
        for i in range(len(self.tableau)):
            temp += self.tableau[i]
        
        for i in range(len(self.foundations)):
            temp += self.foundations[i]
        
        for i in range(len(self.freeCells)):
            temp += self.freeCells[i]
        
        return np.array(temp)
    
    def step(self, action):
        if self.prevMove!=None and self.prevMove['from'] == action['dest'] and self.prevMove['dest'] == action['from'] and self.prevMove['card'] == action['card']:
            return self.getState(), -5, True, None
        
        self._move(action)
        
        self.prevMove = action

        reward, done = self.check_win(action)

        state = self.getState()
        return state, reward, done, None