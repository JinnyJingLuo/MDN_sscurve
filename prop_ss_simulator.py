# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:12:35 2024

@author: luoji
"""

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
# Define constants



class MDN(nn.Module):
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
    


def initial_dislocation_density(tau0,alp,mu,b,gs,bet):
    
    rho0 = []
    for kk in range(0,len(tau0)):
        A = bet[kk]/(gs[kk])
        B = alp[kk]*(b)
        tau_bar = tau0[kk]/mu
        if (tau_bar/B)**2 - 4*A/B > 0:
            rho0.append(0.25*(tau_bar/B + np.sqrt((tau_bar/B)**2 - 4*A/B))**2)
        else:
            rho0.append(0.25*(tau_bar/B)**2)
        
    return np.array(rho0)

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


def sigma_elastic(eps, rho_grain,E_modulus):
    sigma_ee = E_modulus * eps
    density = rho_grain
    return sigma_ee, density

def sigma_plastic(material_type, eps, schimd_factor_grain, gs, alp, bet, strainrate, materials, rho0, trained_model, \
                  transformer_strain, transformer_gs, transformer_strainrate,  scaler_density,scaler_YP,\
                  scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio):
    mu = materials["ShearModulus"]*1e9
    b = np.array(materials['LatticeConst'])/np.sqrt(2)*1e-10
   
    SFactor = np.max(schimd_factor_grain, axis=1)
    yieldpoints =density_to_stress(rho0, alp, mu, b, gs, bet) # Yield shear stress
    
    ShearModulus  = mu * np.ones_like(gs)
    LatticeConst  = np.array(materials['LatticeConst'])*1e-10 * np.ones_like(gs)
    PoissonRatio = materials["PoissonRatio"]*  np.ones_like(gs)

    strains = eps*np.ones_like(gs)
    
    strainrates = strainrate*np.ones_like(gs)


    df_test = construct_input(gs,strains,strainrates,ShearModulus,LatticeConst,PoissonRatio,yieldpoints)
    #(GS,strains,strainrates,ShearModulus,LatticeConstant,PossionRatio,Yieldpoint):
    df_test = normalizeInput(df=df_test,transformer_strain=transformer_strain, transformer_gs=transformer_gs,\
                             transformer_strainrate=transformer_strainrate, scaler_YP=scaler_YP,\
                             scaler_ShearModulus = scaler_ShearModulus, scaler_LatticeConstant = scaler_LatticeConst,\
                             scaler_PossionRatio = scaler_PoissonRatio)



    df_test = df_test.astype('float32')

    X = torch.from_numpy(df_test.values)
    x_test_variable = Variable(X)
    pi_den, sigma_den, mean_den = model(x_test_variable)
    
    mean_stress, dev_stress, mean_den, dev_den = distribution_stress(pi_den,sigma_den,mean_den,gs,b,alp,bet,mu, scaler_density)


    # sigma_ep = mu * bet.reshape(len(gs),1) /( gs.reshape(len(gs),1) * np.sqrt(density).reshape(len(gs),1)) + alp.reshape(len(gs),1)*mu*b * np.sqrt(density).reshape(len(gs),1)
    sigma_ep = np.array(mean_stress).reshape(len(gs),1) /SFactor.reshape(len(gs),1)
    sigma_ep_upper = np.array(np.array(mean_stress) + 3* np.array( dev_stress)).reshape(len(gs),1) /SFactor.reshape(len(gs),1)
    sigma_ep_lower = np.array(np.array(mean_stress)  - 3*  np.array( dev_stress)).reshape(len(gs),1) /SFactor.reshape(len(gs),1)
    upper_density = np.array(mean_den )+  3* np.array(dev_den)
    lower_density = np.array(mean_den )-  3* np.array(dev_den)
    # 

    # sigma_ep = np.where(sigma_ep > E_modulus * eps_cr.reshape(sigma_ep.shape), sigma_ep, E_modulus * eps_cr.reshape(sigma_ep.shape))
    return sigma_ep,  np.array(mean_den), sigma_ep_upper,upper_density, sigma_ep_lower,lower_density





def solve_strain(prediction_type, material_type, Schimd_factor, gs, alp, bet, strainrate, materials, rho0, trained_model,  \
            transformer_strain, transformer_gs, transformer_strainrate, scaler_density,  scaler_YP,weight,  sigma_z,scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio,
           ):
    mu = materials["ShearModulus"]*1e9
    b = np.array(materials['LatticeConst'])/np.sqrt(2)*1e-10
    E_modulus = 2*mu*(1+materials["PoissonRatio"])
    def stress(eps):
        #  strainrate,possion_ratio, LatticeConst,Yieldpoints, StackingFaultE, trained_model,  AtomicVolume, transformer_strain, transformer_gs, transformer_strainrate, scaler_LC, scaler_PR, scaler_SF, scaler_SM, scaler_V, scaler_YP, scaler_density
        [s_grain_average, ave_density, s_grain_p_upper,upper_density, s_grain_p_lower,lower_density]\
        = sigma_plastic(material_type, eps, Schimd_factor, gs, alp, bet, strainrate, materials, rho0, trained_model, \
                        transformer_strain, transformer_gs, transformer_strainrate,scaler_density,scaler_YP,\
                       scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio)
        s_grain_e = sigma_elastic(eps * np.ones_like(rho_dis), rho0,E_modulus)[0]
        # sigma_plastic(schimd_factor_grain, mu, gs, alphamub, bet, eps, strainrate,possion_ratio, LatticeConst,Yieldpoints, StackingFaultE, trained_model,  AtomicVolume, transformer_strain, transformer_gs, transformer_strainrate, scaler_LC, scaler_PR, scaler_SF, scaler_SM, scaler_V, scaler_YP, scaler_density ,sigma_z):
        
        if prediction_type == "average":
                s_grain_p = s_grain_average
                density = ave_density
        if prediction_type == "upper":
                s_grain_p = s_grain_p_upper
                density = upper_density
        if prediction_type == "lower":
                s_grain_p = s_grain_p_lower
                density = lower_density
     
        s_grain =  np.where((s_grain_e.reshape(np.shape(s_grain_e)) > s_grain_p.reshape(np.shape(s_grain_e))), s_grain_p.reshape(np.shape(s_grain_e)), s_grain_e.reshape(np.shape(s_grain_e)))
        # eps_p = eps - s_grain/np.max(Schimd_factor, axis=1).reshape(rho0.shape)/E_modulus
        s0 = np.sum(s_grain * weight) 
        return s0
    
    def stress_equation(eps):
        s0 = stress(eps)
        return s0-sigma_z

   
    eps_sss,_ = bisect_root(stress,0.0, 0.2, 5000,sigma_z)
   
    
    return eps_sss




def density_to_stress(dislocation_den,alp,mu,b,gs,bet):
    tau = mu *(alp.reshape(alp.shape)*b*np.sqrt(dislocation_den).reshape(alp.shape) + bet.reshape(alp.shape)/gs.reshape(alp.shape)/np.sqrt(dislocation_den).reshape(alp.shape))
    return tau


def bisect_root(f, a, b, n, sigma_0):
    # Bisection method for finding the root of the function f within the interval [a, b].
    # Inputs: f -- a function taking one argument
    #         a, b -- left and right edges of the interval
    #         n -- the number of bisections to do.
    #         sigma_0 -- target value
    # Outputs: x_star -- the estimated solution of f(x) = sigma_0
    #          eps -- an upper bound on the error

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

#%%==================Distribution of  stress ====================================
# Parameters for the distribution of A and the constants C1, C2
# mu, sigma = np.sum(scaler_density.inverse_transform(pi_data * mu_data,axis=1)[0]), np.sqrt(np.sum(pi_data**2 * sigma_data**2,axis=1))[0]  # Mean and standard deviation for A
def distribution_stress(pi_variable,sigma_variable,mu_variable,gs_list,b_v,alp_list,bet_list,shear_modulus,scaler_density):# unit is shear modulus
    pi_data = pi_variable.data.numpy()
    sigma_data = sigma_variable.data.numpy()
    mu_data = mu_variable.data.numpy()
    
    upper_bar =  scaler_density.inverse_transform((np.sum(pi_data * mu_data,axis=1) + np.sqrt(np.sum(pi_data**2 * sigma_data**2,axis=1))).reshape(-1,1))
    mu_list =  scaler_density.inverse_transform((np.sum(pi_data * mu_data,axis=1).reshape(-1,1)))
    sigma_list = upper_bar - mu_list
    mean_stress = []
    dev_stress = []
    mean_den = []
    dev_den = []
    for n in range(len(gs_list)):
        
        mu = mu_list[n][0]
        sigma = sigma_list[n][0]
        alp = alp_list[n]
        bet = bet_list[n]
        gs = gs_list[n]
        # gs = transformer_gs.inverse_transform(np.array(X_test['grainsize']).reshape(-1, 1))[5][0]  
        # C1, C2 = 0.57*b_v, 1.76e-3/gs   # Constants
        n_samples = 1000  # Number of samples
        
        # Generate samples from the distribution of A
        A_samples = np.random.normal(mu, sigma, n_samples)
        
        # Calculate B = exp(A)
        den_samples = np.exp(A_samples)
        mean_den.append(np.mean(den_samples))
        dev_den.append(np.std(den_samples))
        
        stress_samples =density_to_stress(den_samples,alp*np.ones_like(den_samples),shear_modulus*np.ones_like(den_samples),b_v*np.ones_like(den_samples),gs*np.ones_like(den_samples),bet*np.ones_like(den_samples))
        
        mean_stress.append(np.mean(stress_samples))
        dev_stress.append(np.std(stress_samples))
        
    return mean_stress, dev_stress, mean_den, dev_den


def strain_to_stress(material_type, Schimd_factor, gs, alp, bet, strainrate, materials, rho0, trained_model, \
 transformer_strain, transformer_gs, transformer_strainrate, scaler_YP,scaler_density, eps, scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio):
    mu = materials["ShearModulus"]*1e9
    
    E_modulus = 2*mu*(1+materials["PoissonRatio"])
       
    #  strainrate,possion_ratio, LatticeConst,Yieldpoints, StackingFaultE, trained_model,  AtomicVolume, transformer_strain, transformer_gs, transformer_strainrate, scaler_LC, scaler_PR, scaler_SF, scaler_SM, scaler_V, scaler_YP, scaler_density
    [s_grain_p, density, s_grain_p_upper,upper_density, s_grain_p_lower,lower_density]\
    = sigma_plastic(material_type, eps, Schimd_factor, gs, alp, bet, strainrate, materials, rho0, trained_model,  transformer_strain, transformer_gs, \
    transformer_strainrate, scaler_density, scaler_YP, scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio)
  
 # sigma_plastic(material_type, eps, schimd_factor_grain, gs, alp, bet, strainrate, materials, rho0, trained_model,  transformer_strain, transformer_gs, transformer_strainrate, scaler_YP, scaler_density):  
    s_grain_e = sigma_elastic(eps * np.ones_like(rho0), rho0,E_modulus)[0]
    # sigma_plastic(schimd_factor_grain, mu, gs, alphamub, bet, eps, strainrate,possion_ratio, LatticeConst,Yieldpoints, StackingFaultE, trained_model,  AtomicVolume, transformer_strain, transformer_gs, transformer_strainrate, scaler_LC, scaler_PR, scaler_SF, scaler_SM, scaler_V, scaler_YP, scaler_density ,sigma_z):

    s_grain = np.where(s_grain_p.reshape(np.shape(s_grain_e)) < s_grain_e, s_grain_p.reshape(np.shape(s_grain_e)), s_grain_e)   
    s_grain_upper = np.where(s_grain_p_upper.reshape(np.shape(s_grain_e)) < s_grain_e, s_grain_p_upper.reshape(np.shape(s_grain_e)), s_grain_e)
    s_grain_lower = np.where(s_grain_p_lower.reshape(np.shape(s_grain_e)) < s_grain_e, s_grain_p_lower.reshape(np.shape(s_grain_e)), s_grain_e)

    density = np.where(s_grain_p_upper.reshape(np.shape(s_grain_e)) < s_grain_e, density.reshape(np.shape(s_grain_e)), rho0)
    upper_density = np.where(s_grain_p_upper.reshape(np.shape(s_grain_e)) < s_grain_e, upper_density.reshape(np.shape(s_grain_e)), rho0)
    lower_density = np.where(s_grain_p_lower.reshape(np.shape(s_grain_e)) < s_grain_e, lower_density.reshape(np.shape(s_grain_e)), rho0)
        
    mu = materials["ShearModulus"]*1e9
    b = np.array(materials['LatticeConst'])/np.sqrt(2)*1e-10
    E_modulus = 2*mu*(1+materials["PoissonRatio"])
    yieldpoints =density_to_stress(rho0, alp, mu, b, gs, bet)/np.max(Schimd_factor, axis=1).reshape(rho0.shape)
    eps_cr = yieldpoints / E_modulus 
    
    return s_grain,s_grain_upper, s_grain_lower, density, upper_density, lower_density



    #----------------main ————————————# 
    # Main code starts here
    # Loop over the random seed
#%% 
'''
model learning
'''

filename = '../finalized_MDN_prop.sav'
with open(filename,'rb') as f:
    model, transformer_strain, transformer_gs, transformer_strainrate,  scaler_density,scaler_YP,scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio = pickle.load(f)
bet_rho =1.76e-3
alpha_rho = 0.57
path = 'samplesize97_withObserve_withExpyieldpoint3.spydata'


#%% 
'''
generate strain stress curve 

'''

filename = "material_properties.csv"
with open(filename, 'r') as csv_file:
    # Create a reader object
    reader = csv.DictReader(csv_file)   
    # Convert the reader object to a list of dictionaries
    data_materials = list(reader)
# Convert the list of dictionaries back to a dictionary of dictionaries
material_list = {row.pop('Material'): {k: float(v) for k, v in row.items()} for row in data_materials}

data = load_dictionary(path)
datas = data[0]["data"]


strain_rate = 1e-3
material_type = 'Ni'
materials = material_list[material_type]
file1 = open('MDN_Ni_test1.txt', "wb") 
data_exps = [[d['size'],d['sscurve']] for d in datas if ('size' in d and d['material'] == material_type)]
ss_save = {}
data_index = 0
bet_rho = 1.76e-3
alpha_rho = 0.57

for data_exp in data_exps[0:1]:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    data_index = data_index +1 
    ss_save_pergrain = []
    gs_ave = data_exp[0]*1e-6
    maxstrain = 0.2
    rho_cr = bet_rho/(alpha_rho *materials['LatticeConst']*1e-10/np.sqrt(2)*gs_ave)
    tau_cr = (alpha_rho *materials['LatticeConst']*1e-10/np.sqrt(2) * np.sqrt(rho_cr) + bet_rho /gs_ave / np.sqrt(rho_cr))*materials['ShearModulus']*2.7*1e9 
    size_n =20
    for log_disrho in [12]:
        y_average = []
        y_upper = []
        y_lower = []    
        for rand_seed in range(2020,2021):
             for maxstress in [100]:
                if max(data_exp[1][:, 0]) > 0.2:
                    maxload = maxstress #np.interp(0.2, data_exp[1][:, 0], data_exp[1][:, 1])
                else:
                    maxload = maxstress
                np.random.seed(rand_seed)
                random.seed(rand_seed)
                print([rand_seed,log_disrho],flush = True) 
                alp_detail,bet_detail,section_number, thickness_section, \
                    number_gs,weight_detail,Schimd_factor_detail,gs_detail,\
                        rho_detail, Area_detail \
                            = microstructure.microstructure(rand_seed,gs_ave,log_disrho)
                # load the trained_model from 'nn_learned_model_for_drho_dep.mat'
                sigma_z = 0  # applied force
                StepM, SectionM = np.meshgrid(np.arange(1, size_n + 1), np.arange(1, section_number + 1))
                StepL = StepM.flatten()
                SectionL = SectionM.flatten()
                step_section = np.stack((StepL, SectionL), axis=0)
                strain_average = np.zeros((size_n, section_number))
                strain_upper = np.zeros((size_n, section_number))
                strain_lower = np.zeros((size_n, section_number))
                stress = np.linspace(10, maxload*4e6, size_n)
                for kk in range(step_section.shape[1]):
                    print(data_index,"finished:",kk/step_section.shape[1])
                    section = step_section[1, kk]
                    step = step_section[0, kk]
                    gs_number = np.array(number_gs)[section-1]
                    gs = gs_detail[section-1]
                    weight = weight_detail[section-1]
                    Schimd_factor = Schimd_factor_detail[section-1]
                    rho_dis = rho_detail[section-1]
                    alp_section = alp_detail[section-1]
                    bet_section = bet_detail[section-1]    
                    sigma_z = stress[step-1]
                    
                    eps_sss_average= solve_strain("average", material_type, Schimd_factor, gs, alp_section, bet_section,strain_rate, materials, rho_dis, model, transformer_strain, \
                        transformer_gs, transformer_strainrate,  scaler_density,scaler_YP, weight, sigma_z,scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio)
                    eps_sss_upper = solve_strain("upper", material_type, Schimd_factor, gs, alp_section, bet_section,strain_rate, materials, rho_dis, model, transformer_strain, \
                         transformer_gs, transformer_strainrate, scaler_density,scaler_YP, weight, sigma_z,scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio)
                    eps_sss_lower = solve_strain("lower", material_type, Schimd_factor, gs, alp_section, bet_section,strain_rate, materials, rho_dis, model, transformer_strain, \
                         transformer_gs, transformer_strainrate, scaler_density, scaler_YP,weight, sigma_z,scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio)
    
                    # solve_strain(Schimd_factor, gs, alp, bet, strainrate, materials, rho0, trained_model,  transformer_strain, transformer_gs, transformer_strainrate, scaler_LC, scaler_PR, scaler_SF, scaler_SM, scaler_V, scaler_YP, scaler_density, rho_dis, weight, sigma_z):
                    
                    strain_average[step - 1, section - 1] = eps_sss_average
                    strain_upper[step - 1, section - 1] = eps_sss_upper
                    strain_lower[step - 1, section - 1] = eps_sss_lower
            
                weight_T = thickness_section / np.sum(thickness_section)
                strain_total = (strain_average@np.array(weight_T)).reshape(stress.shape)
                strain_total_upper = (strain_upper@np.array(weight_T)).reshape(stress.shape)
                strain_total_lower = (strain_lower@np.array(weight_T)).reshape(stress.shape)

            # strain_final, stress_final = continuity(np.array(strain_final), np.array(stress_final))
            # strain_final, stress_final = continuity(np.array(strain_final), np.array(stress_final))
            # strain_final, stress_final = continuity(np.array(strain_final), np.array(stress_final))
                indices = np.where( strain_total < maxstrain)[0]
                last_point_ave = np.interp(0.2,strain_total,stress)
                last_point_upper = np.interp(0.2,strain_total_upper,stress)
                last_point_lower = np.interp(0.2,strain_total_lower,stress)
                
               
                
                x_data = x_data = np.arange(0,0.18,0.001)
                y_average.append([np.interp(x_data, np.array(strain_total[indices]), np.array(stress[indices]) )] )     
                y_upper.append([np.interp(x_data,np.array(strain_total_upper[indices]), np.array(stress[indices]))])
                y_lower.append([np.interp(x_data, np.array(strain_total_lower[indices]), np.array(stress[indices]))])
                
            
                dict_temp = {'rand_seed':rand_seed,'logrho':log_disrho,
                             'sscurve_ave':np.array([strain_total[indices],stress[indices]/1e6]),
                             'sscurve_upper':np.array([strain_total_upper[indices],stress[indices]/1e6]),
                             'sscurve_lower':np.array([strain_total_lower[indices],stress[indices]/1e6])}

                ss_save_pergrain.append(dict_temp)
           
        y_average_ave = np.mean(np.squeeze(np.array(y_average),axis=1),axis = 0)
        y_upper_ave = np.mean(np.squeeze(np.array(y_upper),axis=1),axis = 0)
        y_lower_ave = np.mean(np.squeeze(np.array(y_lower),axis=1),axis = 0)       
        ax2.fill_between(x_data, y_upper_ave/1e6, y_lower_ave/1e6,alpha=0.2,label = r"$\rho_0$"+f"= {10**log_disrho:.2e}")
            # plt.scatter(strain_total, y_average/1e6)
                  
        ax2.plot(x_data,y_average_ave/1e6,lw=2.0,linestyle='--')
    ss_save[gs_ave] =  ss_save_pergrain
   
    
    ax2.scatter(data_exp[1][:,0],data_exp[1][:,1],label = 'experiment',color = 'black',alpha=0.5)
    ax2.set_xlim([0,0.14])
    ax2.set_ylim([0,400])
    ax2.set_xlabel('strain',fontsize = 20)
    ax2.set_ylabel('stress(MPa)',fontsize = 20)    
    ax2.legend(loc="lower right")
    ax2.set_title(f'grain size = {gs_ave/1e-6:.2f}um',fontsize = 20)   
    ax2.tick_params(axis='both', labelsize=15)


    fig2.savefig('MDN_Ni'+str(data_index) +'.png', dpi=300, bbox_inches='tight')  # Adjust filename and options

    pickle.dump(ss_save, file1)    
    file1.close

    # for gs in np.arange(5,200,20):
    


    