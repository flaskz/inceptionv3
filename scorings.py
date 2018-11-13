# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:28:16 2018

@author: l.ikeda
"""

def accuracy(dic):
    acertos = 0
    erros = 0
    for key, values in dic.items():
        splitted = key.split('--')
        if splitted[0] == splitted[1]:
            acertos += values
        else:
            erros += values
    return acertos/(acertos+erros)

def precision(dic, value):
    tp = 0
    fp = 0
    for key, values in dic.items():
        splitted = key.split('--')
        if splitted[1] == value:
            if splitted[0] == value:
                tp += values
            else:
                fp += values
    if (tp+fp) == 0:
        return 0
    return tp/(tp+fp)

def recall(dic, value):
    tp = 0
    fn = 0
    for key, values in dic.items():
        splitted = key.split('--')
        if splitted[0] == value:
            if splitted[1] == value:
                tp += values
            else:
                fn += values
    if (tp+fn) == 0:
        return 0
    return tp/(tp+fn)    
    
def F1_score(recall, precision):
    if (recall+precision) == 0:
        return 0
    return 2*(recall*precision)/(recall+precision)
            
