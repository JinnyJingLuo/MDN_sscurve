# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:12:35 2024

@author: luoji
"""

import numpy as np
import microstructure

import torch
import matplotlib.pyplot as plt



from spyder_kernels.utils.iofuncs import load_dictionary

import csv

import pickle
import random
from torch import nn

import torch.nn.functional as F

import PhysicalLaw_prop
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
    

    #----------------main ————————————# 
    # Main code starts here
    # Loop over the random seed
#%% 
'''
model learning
'''

filename = './finalized_MDN_prop.sav'
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
                    
                    eps_sss_average= PhysicalLaw_prop.solve_strain("average", material_type, Schimd_factor, gs, alp_section, bet_section,strain_rate, materials, rho_dis, model, transformer_strain, \
                        transformer_gs, transformer_strainrate,  scaler_density,scaler_YP, weight, sigma_z,scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio)
                    eps_sss_upper = PhysicalLaw_prop.solve_strain("upper", material_type, Schimd_factor, gs, alp_section, bet_section,strain_rate, materials, rho_dis, model, transformer_strain, \
                         transformer_gs, transformer_strainrate, scaler_density,scaler_YP, weight, sigma_z,scaler_ShearModulus, scaler_LatticeConst ,scaler_PoissonRatio)
                    eps_sss_lower = PhysicalLaw_prop.solve_strain("lower", material_type, Schimd_factor, gs, alp_section, bet_section,strain_rate, materials, rho_dis, model, transformer_strain, \
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
    


    