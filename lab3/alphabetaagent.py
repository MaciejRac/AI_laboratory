import math
import copy

from exceptions import AgentException

class AlphaBetaAgent():
    def __init__(self, my_token,d=4):
        self.my_token = my_token
        self.d=d

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        a=-math.inf
        v=-math.inf
        y=None
        for i in connect4.possible_drops():
            copiedboard=copy.deepcopy(connect4)
            copiedboard.drop_token(i)
            newValue= self.alphabeta(copiedboard, 0, self.d - 1, a, math.inf)
            if newValue > v:
                v=newValue
                y=i
            a =max(a, v)
        return y
    
    def alphabeta(self,connect4, x, d, a, β)->int:
        if (connect4.game_over==True):
            if (connect4.wins==self.my_token):
                return 1    
            elif (connect4.wins!=None):
                return -1
            else:
                return 0
        elif (d==0):
            return self.evaluate(connect4) #h(s)
        elif (x==1):
            v=-math.inf
            for i in connect4.possible_drops():
                copiedboard=copy.deepcopy(connect4)
                copiedboard.drop_token(i)
                v= max(v, self.alphabeta(copiedboard, 0, d - 1, a, β))
                a = max(a, v)
                if (v >= β):
                    break # β cutoff
            return v
        else:
            v=math.inf
            for i in connect4.possible_drops():
                copiedboard=copy.deepcopy(connect4)
                copiedboard.drop_token(i)
                v = min(v, self.alphabeta(copiedboard, 1, d - 1, a, β))
                β =min(β, v)
                if (v <= a) :
                    break # α cutoff
        return v
    
    def evaluate(self, connect4):
        count_x, count_o = self.count_symbols(connect4)
        total_count=count_o+count_x
        if (self.my_token=='x'):
            normalized_difference = (count_x - count_o) / (total_count+1)
        else:
            normalized_difference = (count_o - count_x) / (total_count+1)
        #print(normalized_difference)
        return normalized_difference

    def count_symbols(self, connect4):
        count_x = 0
        count_o = 0
        for sequence in connect4.iter_fours():
            if sequence.count('x')>0 and sequence.count('o')==0:
                count_x+=1
            if sequence.count('o')>0 and sequence.count('x')==0:
                count_o+=1
            # for symbol in sequence:

            #     if symbol == 'x':
            #         count_x += 1
            #     elif symbol == 'o':
            #         count_o += 1
        return count_x, count_o