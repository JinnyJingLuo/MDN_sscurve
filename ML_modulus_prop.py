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

class MDN(nn.Module):
    # load MDN structure
    def __init__(self, n_input, n_hidden, n_layers, n_gaussians):
        super(MDN, self).__init__()

        # Create a list to hold the layers
        layers = []

        # Input layer
        layers.append(nn.Linear(n_input, n_hidden))
        layers.append(nn.Tanh())

        # Additional hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())

        # Convert the list of layers into a Sequential container
        self.z_h = nn.Sequential(*layers)

        # Output layers for the MDN
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu
    
def construct_input(GS,strains,strainrates,ShearModulus,LatticeConstant,PossionRatio,Yieldpoint):
    if isinstance(GS, (float, int)):
        GS = [GS] # incase we only have one gs, we need to construct it 
        
    index = index = range(len(GS)) # get length of it
    df = pd.DataFrame({
        'grainsize': GS, # unit: m
        'strains': strains,
        'strainrates':strainrates,
        "Yieldpoint":Yieldpoint, # unit: MPa // noticed the yield point is resolved shear yield point
        "ShearModuluses": ShearModulus,
        "PoissonRatios": PossionRatio,
        "LatticeConsts": LatticeConstant,
        
    },index = index)# construct the input 
    return df


def normalizeInput(df,transformer_strain, transformer_gs, transformer_strainrate,scaler_YP,\
                   scaler_ShearModulus,scaler_LatticeConstant,scaler_PossionRatio):   
    df_after = df.copy()
    df_after['strains'] = transformer_strain.transform(np.array(df_after['strains']).reshape(-1, 1))
    df_after['strainrates']  = transformer_strainrate.transform(np.array(df_after['strainrates']).reshape(-1, 1))   
    df_after['grainsize'] = transformer_gs.transform(np.array(df_after['grainsize']).reshape(-1, 1)) 
    df_after['Yieldpoint'] = scaler_YP.transform(df_after[['Yieldpoint']])   
    df_after['ShearModuluses'] = scaler_ShearModulus.transform(df_after[['ShearModuluses']])   
    df_after['LatticeConsts'] = scaler_LatticeConstant.transform(df_after[['LatticeConsts']])   
    df_after['PoissonRatios'] = scaler_PossionRatio.transform(df_after[['PoissonRatios']])   
    return df_after
