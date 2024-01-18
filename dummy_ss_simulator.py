import numpy as np
import microstructure

import matplotlib.pyplot as plt



from spyder_kernels.utils.iofuncs import load_dictionary

import csv

import pickle
import random

import PhysicalLaw
from ML_modulus import MDN

#----------------main ————————————# 
# Main code starts here
# Loop over the random seed

#%%
'''
CONST PREPARE FOR SIMULATION 
'''
ss_save = {}
data_index = 0
bet_rho = 1.76e-3
alpha_rho = 0.57
# how many point in one curve to calculate
size_n =20
# file save name
file1 = open('./MDN_Ni_test1.txt', "wb") 



#%% 
'''
MODEL LOADED
'''
list_save = ['network', 'transformer_strain', 'transformer_gs', 'transformer_strainrate', 'scaler_density','scaler_YP']
filename = './finalized_MDN2.sav'
with open(filename,'rb') as f:  # Python 3: open(..., 'rb')
   model, transformer_strain, transformer_gs, transformer_strainrate, scaler_density,scaler_YP= pickle.load(f)
bet_rho =1.76e-3
alpha_rho = 0.57



#%% 
'''
MATERIAL AND PHYSICAL LOADED 
'''

filename = "material_properties.csv"
with open(filename, 'r') as csv_file:  
    reader = csv.DictReader(csv_file)   
    data_materials = list(reader)
material_list = {row.pop('Material'): {k: float(v) for k, v in row.items()} for row in data_materials}

strain_rate = 1e-3
material_type = 'Ni'
maxstrain = 0.2
materials = material_list[material_type]



#%% 
'''
EXPERIMENTAL DATA LOADED 
'''
path = 'samplesize97_withObserve_withExpyieldpoint3.spydata'
data = load_dictionary(path)
datas = data[0]["data"]
data_exps = [[d['size'],d['sscurve']] for d in datas if ('size' in d and d['material'] == material_type)]



#%% 
for data_exp in data_exps[0:1]:
    
    '''
    prepare for simulation of one grain size 
    '''
    # figures plot 
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    # index record 
    data_index = data_index +1 
    # allocate a list to record all data that is necessary to record
    ss_save_persample = []
    # get grain size
    gs_ave = data_exp[0]*1e-6    
    
    for log_disrho in [12]:
        # For different initial dislocation density
        y_average = []
        y_upper = []
        y_lower = []    
        for rand_seed in range(2020,2021): # test three random case 
             for maxstress in [100]: # can be used for different maxstress
                maxload = maxstress
                np.random.seed(rand_seed)
                random.seed(rand_seed) # fixed the random seed such that we can get the same results 
                print([rand_seed,log_disrho,maxstress],flush = True) # print which sample we are calculating 
                # generate microstructure 
                alp_detail,bet_detail,section_number, thickness_section, \
                    number_gs,weight_detail,Schimd_factor_detail,gs_detail,\
                        rho_detail, Area_detail \
                            = microstructure.microstructure(rand_seed,gs_ave,log_disrho)
                # create a mesh for different sigma_z and section number
                StepM, SectionM = np.meshgrid(np.arange(1, size_n + 1), np.arange(1, section_number + 1))
                StepL = StepM.flatten()
                SectionL = SectionM.flatten()
                step_section = np.stack((StepL, SectionL), axis=0)
                strain_average = np.zeros((size_n, section_number))
                strain_upper = np.zeros((size_n, section_number))
                strain_lower = np.zeros((size_n, section_number)) # initialize the mesh
                stress = np.linspace(10, maxload*4e6, size_n) # stress level to calculate 
                for kk in range(step_section.shape[1]):
                    print(data_index,"finished:",kk/step_section.shape[1]) # print how much finished 
                    section = step_section[1, kk] # which section 
                    step = step_section[0, kk] # which stress level 
                    sigma_z = stress[step-1]
                    gs_number = np.array(number_gs)[section-1] # grain number details
                    gs = gs_detail[section-1] # grain size at that section 
                    weight = weight_detail[section-1] # weight at that section 
                    Schimd_factor = Schimd_factor_detail[section-1] # schmid factor at that section 
                    rho_dis = rho_detail[section-1]# intial dislocation density at that section 
                    alp_section = alp_detail[section-1] # alpha value : const now
                    bet_section = bet_detail[section-1] # beta value: constant now    
                    
                    # solve strain
                    eps_sss_average= PhysicalLaw.solve_strain("average", material_type, Schimd_factor, gs, alp_section, bet_section,strain_rate, materials, rho_dis, model, transformer_strain, \
                        transformer_gs, transformer_strainrate,  scaler_density,scaler_YP, weight, sigma_z)
                    eps_sss_upper = PhysicalLaw.solve_strain("upper", material_type, Schimd_factor, gs, alp_section, bet_section,strain_rate, materials, rho_dis, model, transformer_strain, \
                         transformer_gs, transformer_strainrate, scaler_density,scaler_YP, weight, sigma_z)
                    eps_sss_lower = PhysicalLaw.solve_strain("lower", material_type, Schimd_factor, gs, alp_section, bet_section,strain_rate, materials, rho_dis, model, transformer_strain, \
                         transformer_gs, transformer_strainrate, scaler_density, scaler_YP,weight, sigma_z)
    
                    # save to matrices
                    strain_average[step - 1, section - 1] = eps_sss_average
                    strain_upper[step - 1, section - 1] = eps_sss_upper
                    strain_lower[step - 1, section - 1] = eps_sss_lower
                # summation on thickness direction 
                weight_T = thickness_section / np.sum(thickness_section)
                strain_total = (strain_average@np.array(weight_T)).reshape(stress.shape)
                strain_total_upper = (strain_upper@np.array(weight_T)).reshape(stress.shape)
                strain_total_lower = (strain_lower@np.array(weight_T)).reshape(stress.shape)

                
                indices_ave = np.where( strain_total < maxstrain)[0]
                indices_upper = np.where( strain_total_upper < maxstrain)[0]
                indices_lower = np.where( strain_total_lower < maxstrain)[0]
                last_point_ave = np.interp(0.2,strain_total,stress)
                last_point_upper = np.interp(0.2,strain_total_upper,stress)
                last_point_lower = np.interp(0.2,strain_total_lower,stress)
                
               
                
                x_data = x_data = np.arange(0,0.18,0.001)
                y_average.append([np.interp(x_data, np.array(strain_total[indices_ave]), np.array(stress[indices_ave]) )] )     
                y_upper.append([np.interp(x_data,np.array(strain_total_upper[indices_upper]), np.array(stress[indices_upper]))])
                y_lower.append([np.interp(x_data, np.array(strain_total_lower[indices_lower]), np.array(stress[indices_lower]))])
                
            
                dict_temp = {'rand_seed':rand_seed,'logrho':log_disrho,
                             'sscurve_ave':np.array([strain_total[indices_ave],stress[indices_ave]/1e6]),
                             'sscurve_upper':np.array([strain_total_upper[indices_upper],stress[indices_upper]/1e6]),
                             'sscurve_lower':np.array([strain_total_lower[indices_lower],stress[indices_lower]/1e6])}

        # post process    
                ss_save_persample.append(dict_temp)
        y_average_ave = np.mean(np.squeeze(np.array(y_average),axis=1),axis = 0)
        y_upper_ave = np.mean(np.squeeze(np.array(y_upper),axis=1),axis = 0)
        y_lower_ave = np.mean(np.squeeze(np.array(y_lower),axis=1),axis = 0)             
        ax2.fill_between(x_data, y_upper_ave/1e6, y_lower_ave/1e6,alpha=0.2,label = r"$\rho_0$"+f"= {10**log_disrho:.2e}")            # plt.scatter(strain_total, y_average/1e6)
                  
        ax2.plot(x_data,y_average_ave/1e6,lw=2.0,linestyle='--')

    ax2.scatter(data_exp[1][:,0],data_exp[1][:,1],label = 'experiment',color = 'black',alpha=0.5)
    ax2.set_xlim([0,0.14])
    ax2.set_ylim([0,400])
    ax2.set_xlabel('strain',fontsize = 20)
    ax2.set_ylabel('stress(MPa)',fontsize = 20)    
    ax2.legend(loc="lower right")
    ax2.set_title(f'grain size = {gs_ave/1e-6:.2f}um',fontsize = 20)   
    ax2.tick_params(axis='both', labelsize=15)
    fig2.savefig('./MDN_Ni'+str(data_index) +'.png', dpi=300, bbox_inches='tight')  # Adjust filename and options
    
    # save data
    ss_save[gs_ave] =  ss_save_persample
pickle.dump(ss_save, file1)    
file1.close

    # for gs in np.arange(5,200,20):
    


    