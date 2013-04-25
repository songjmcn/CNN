#coding=gbk
'''
Created on 2013-4-25

@author: songjm
'''
import numpy as np
def sigmod(input):
    '激励函数'
    out=1.7159*np.tanh(2/3*input)
    return out
def dsigmoid(input):
    out=2/3/1.7159*(1.7159+input)*(1.7159-input)
    return out