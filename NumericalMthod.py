
import numpy as np
import microstructure

import torch
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import pandas as pd
from spyder_kernels.utils.iofuncs import load_dictionary
import scipy
import csv

import pickle
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F 

def bisect_root(f, a, b, n, sigma_0):
    '''
    Bisection method for finding the root of the function f within the interval [a, b].
    Inputs: f -- a function taking one argument
            a, b -- left and right edges of the interval
            n -- the number of bisections to do.
            sigma_0 -- target value
    Outputs: x_star -- the estimated solution of f(x) = sigma_0
             eps -- an upper bound on the error
    ''' 
    c = f(a) - sigma_0
    d = f(b) - sigma_0
 
    if c * d > 0.0:
        x_star = 1
        eps = None
        return x_star, eps  # Return None if no root can be found within the given interval

    for k in range(n):
        x_star = (a + b) / 2
        y = f(x_star) - sigma_0

        if y == 0.0:  # Solved the equation exactly
            eps = 0
            break  # Jumps out of the loop

        if c * y < 0:
            b = x_star
        else:
            a = x_star

        eps = (b - a) / 2

        if eps < 1e-5:
            break

    return x_star, eps

