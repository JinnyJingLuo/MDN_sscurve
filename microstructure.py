# -*- coding: utf-8 -*-
import numpy as np
import stat_microstructure
import scipy.io
import random

def microstructure(rand_seed,gs_ave,log_disrho):
   
    
    mat_file = scipy.io.loadmat('m_sch.mat')
    m_sch = mat_file['m_sch']
    
        
    number_section_max =20
    number_section_min = 10
    Cross_area =20* gs_ave ** 2
    
    #gs_ave = 10.0e-6
    # (b-a)^2/12 : uniform sqaure deviation
    gs_var = (0.5 * gs_ave) 
    gs_max = gs_ave +  np.sqrt(gs_var**2 *12) /2
    gs_min = gs_max - np.sqrt(gs_var**2 *12) 
    
    # parameters
    # bet = 1.76e-3


    

    
    # mean_gs = []
    # gs_theo = []
    #log_disrho = 1
    
    gs_detail = []
    number_gs = []
    grain_number = []
    go_index = []
    Area_detail = []
    rho_detail = []
    weight_detail = []
    Schimd_factor_detail = []
    alpha_detail = []
    bet_detail = []
    
    
    
    mat_file = scipy.io.loadmat('go_pdf.mat')
    go_pdf = mat_file['go_pdf']
    
    
    
    go_cdf = np.cumsum(go_pdf)
    cum_go_p_l = np.concatenate(([0], go_cdf[:-1]))
    cum_go_p_u = go_cdf
    
    
    
    section_number =  random.randint(number_section_min,number_section_max) # You can modify this value as needed
        
    thickness, pd, cdf_thickness, x = stat_microstructure.gs_lognormal_dist(section_number, gs_min, gs_max, gs_ave, gs_var, 10000)
    
        
    for section in range(1, section_number + 1):
        thickness_section = thickness[section - 1]
        number_gs_section = int(Cross_area / (thickness_section ** 2)) + 1
        grain_number.append(number_gs_section)
        go_index_section = np.zeros(grain_number[section - 1], dtype=int)
        alp_section = 0.57 + (np.random.rand(grain_number[section - 1]) - 0.5) * 0 * 0.57
        bet_section = 1.76e-3 + (np.random.rand(grain_number[section - 1]) - 0.5) * 0 * 1.76e-3
        Area_var = np.random.rand(grain_number[section - 1]) * 0.5+2
        Area_detail_section = Cross_area * (Area_var) / np.sum(Area_var)
        rho_ave = (1.0 * np.random.rand() + 0.5) * 10 ** float(log_disrho)
        rho_var = rho_ave
        rho_detail_section = (np.random.rand(grain_number[section - 1]) - 0.5) * rho_var + rho_ave
        gs_detail_section = (thickness_section * Area_detail_section) ** (1 / 3)
        rand_p2 = np.random.rand(grain_number[section - 1])
        for k in range(grain_number[section - 1]):
            index_of_go_k = np.min(np.where((cum_go_p_l <= rand_p2[k]) & (cum_go_p_u > rand_p2[k])))
            go_index_section[k] = index_of_go_k
        go_index.append(go_index_section)
        weight_detail.append(Area_detail_section / np.sum(Area_detail_section))
        Schimd_factor_detail.append(np.maximum(m_sch[go_index_section, :], 1e-6))
    
        number_gs.append(number_gs_section)
        Area_detail.append(Area_detail_section)
        rho_detail.append(rho_detail_section)
        gs_detail.append(gs_detail_section)
        alpha_detail.append(alp_section)
        bet_detail.append(bet_section)
    D_simulated = np.sum([np.sum(gs) for gs in gs_detail]) / np.sum(number_gs)
    # Rho_ave = rho_ave
    # print(thickness_section)
    import matplotlib.pyplot as plt
    gs_all = []
    [gs_all.extend(gs*10) for gs in gs_detail]
    # plt.hist(gs_all,10)
    # plt.xlabel('grain size(m)')
    # plt.ylabel('frequency')
    # plt.title(f'D_ave = {D_simulated*1e6:2f}')
    # plt.show()
    return alpha_detail,bet_detail, section_number, thickness, number_gs,weight_detail,Schimd_factor_detail,gs_detail,rho_detail, Area_detail