# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:15:53 2016

@author: QueensGambit
"""



class Straight:
    
    def __init__(self, dM, dC):
        self.dM = dM
        self.dC = dC
        self.description = "class for a straigh line"
        self.author = "QueensGambit"
        
    #checks if a given point is above or part of the straight
    # return True XOR False
    def pointAboveOrInStraight(self, dX, dY):
        dRes = dY - (self.dM * dX + self.dC)
        
        if dRes >= 0:
            return 0
        else:
            return 1
                
            
    def printStraight(self):
        print("f(x) = " + str(self.dM) + "X" + " + " + str(self.dC))

