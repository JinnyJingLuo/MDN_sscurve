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
    
def construct_input(GS,strains,strainrates,Material_Ni,Material_Al,Material_Cu,Yieldpoint):
    """
    construct the input with unit 
    'grainsize': grain size: m,
    'strains': strains,
    'strainrates':strainrates,
    "Yieldpoint": Resolved shear stress when yielding,
    "Material_Al": bool value if it is Al,
    "Material_Cu": bool value if it is Cu,
    "Material_Ni": bool value if it is Ni,
    Returns:
    Dataframe: with all input 
    """
    df = pd.DataFrame({
        'grainsize': GS,
        'strains': strains,
        'strainrates':strainrates,
        "Yieldpoint":Yieldpoint,
        "Material_Al": Material_Al,
        "Material_Cu": Material_Cu,
        "Material_Ni": Material_Ni,
    })
    return df

def normalizeInput(df,transformer_strain, transformer_gs, transformer_strainrate,scaler_YP):
    """
    Normalize the unit 
    df: unnormalized dataframe 
    transformer_strain: strain transformer
    transformer_gs: grain size transformer 
    transformer_strainrate: strainrate transformer
    scaler_YP: Yield point transformer 
    return: Datafram with normalized data 
    """
    df_after = df.copy()
    # hard copy df, otherwise df will be changed as well 

    df_after['strains'] = transformer_strain.transform(np.array(df_after['strains']).reshape(-1, 1))
    
    df_after['strainrates']  = transformer_strainrate.transform(np.array(df_after['strainrates']).reshape(-1, 1))
   
    df_after['grainsize'] = transformer_gs.transform(np.array(df_after['grainsize']).reshape(-1, 1)) 

    df_after['Yieldpoint'] = scaler_YP.transform(df_after[['Yieldpoint']])
    
    return df_after