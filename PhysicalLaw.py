import numpy as np

import torch


from torch.autograd import Variable

import ML_modulus
import NumericalMthod

def initial_dislocation_density(tau0,alp,mu,b,gs,bet):
    """
    Parameters:
    tau0 (float):Resolved yield shear stress:Pa
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    mu (float): Shear modulus:Pa
    b (float): Burger's vector: m
    gs (float): grain size: m

    Returns:
    float: dislocation density 
    """
    rho0 = []
    for kk in range(0,len(tau0)):
        # equation becomes:  (sqrt(\rho)^2) - tau0/mu/(alpha *b) *sqrt(rho)  + beta / d/(alpha *b) == 0
        A = bet[kk]/(gs[kk])
        B = alp[kk]*(b)
        tau_bar = tau0[kk]/mu
        if (tau_bar/B)**2 - 4*A/B > 0:
            rho0.append(0.25*(tau_bar/B + np.sqrt((tau_bar/B)**2 - 4*A/B))**2)
        else:
        # cannot solve the equation, use the minimum value for the initial dislocation density 
            rho0.append(0.25*(tau_bar/B)**2)
        
    return np.array(rho0)

def sigma_elastic(eps, rho_grain,E_modulus):
    """
    calculate elastic stress
    Parameters:
    eps (float): strain level 
    rho_grain (float): initial dislocation density 
    E_modulus (float): Young's modulus 

    Returns:
    float, float : elastic stress, initial dislocation density
    """
    sigma_ee = E_modulus * eps
    density = rho_grain
    return sigma_ee, density

def sigma_plastic(material_type, eps, schimd_factor_grain, gs, alp, bet, strainrate, materials, rho0, trained_model,  transformer_strain, transformer_gs, transformer_strainrate,  scaler_density,scaler_YP):
    """
    calculate plastic stress at one section 
    Parameters:
    material_type: string, can be "Ni", "Al" or "Cu"
    eps(float): strain level;
    schimd_factor_grain: a [n * 12] array, with the schmid factor on all 12 systems
    gs (array ): grain size: m
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    strainrate: strainrate
    materials: dictionary for this material
    rho0: initial dislocation density 
    trained_model: trained ML model 
    transformer_strain: strain transformer
    transformer_gs: grain size transformer 
    transformer_strainrate: strainrate transformer
    scaler_YP: Yield point transformer 

    Returns:
    float: plastic stress and predicted density 
    """
    mu = materials["ShearModulus"]*1e9
    # dictionary unit: ShearModulus: GPa, lattice constant: A; 
    b = np.array(materials['LatticeConst'])/np.sqrt(2)*1e-10
    # convert lattice constant to bugers vector 
    SFactor = np.max(schimd_factor_grain, axis=1)  
    # get the maxiumum Schmid factor to determined which system start slip
    yieldpoints =density_to_stress(rho0, alp, mu, b, gs, bet)
    # calculate resolved shear stress as yield 
    if material_type == 'Ni':
       Material_Ni = np.ones_like(gs)
       Material_Al = 0 * np.ones_like(gs)
       Material_Cu = 0*  np.ones_like(gs)
    elif material_type == 'Cu':
       Material_Ni = 0 * np.ones_like(gs)
       Material_Al = 0 * np.ones_like(gs)
       Material_Cu = np.ones_like(gs)
    else:
        Material_Ni = 0 * np.ones_like(gs)
        Material_Al =  np.ones_like(gs)
        Material_Cu = 0 * np.ones_like(gs)
    # all grain share the same properties, strain level, strainrate at one section        
    strains = eps*np.ones_like(gs) 
    strainrates = strainrate*np.ones_like(gs)
    # construct the input
    df_test =  ML_modulus.construct_input(gs,strains,strainrates,Material_Ni,  Material_Al ,  Material_Cu,yieldpoints)
    # normalize the input 
    df_test =  ML_modulus.normalizeInput(df=df_test,transformer_strain=transformer_strain, transformer_gs=transformer_gs, transformer_strainrate=transformer_strainrate, scaler_YP=scaler_YP)
    # set all data type to float and get all data in df as the input matrix
    df_test = df_test.astype('float32')
    X = torch.from_numpy(df_test.values)
    # construct the input
    x_test_variable = Variable(X)
    # get result predicted
    pi_den, sigma_den, mean_den = trained_model(x_test_variable)
    # with the normalized result to calculate stress and density statistical parameters
    mean_stress, dev_stress, mean_den, dev_den = distribution_stress(pi_den,sigma_den,mean_den,gs,b,alp,bet,mu, scaler_density)
    # change shear stress back to normal stress
    sigma_ep = np.array(mean_stress).reshape(len(gs),1) /SFactor.reshape(len(gs),1)
    sigma_ep_upper = np.array(np.array(mean_stress) + 3* np.array( dev_stress)).reshape(len(gs),1) /SFactor.reshape(len(gs),1)
    sigma_ep_lower = np.array(np.array(mean_stress)  - 3*  np.array( dev_stress)).reshape(len(gs),1) /SFactor.reshape(len(gs),1)
    upper_density = np.array(mean_den )+  3* np.array(dev_den)
    lower_density = np.array(mean_den )-  3* np.array(dev_den)
    # 3 can be changed, here we use 3Sigma principle

    # sigma_ep = np.where(sigma_ep > E_modulus * eps_cr.reshape(sigma_ep.shape), sigma_ep, E_modulus * eps_cr.reshape(sigma_ep.shape))
    return sigma_ep,  np.array(mean_den), sigma_ep_upper,upper_density, sigma_ep_lower,lower_density


def solve_strain(prediction_type, material_type, Schimd_factor, gs, alp, bet, strainrate, materials, rho0, trained_model,  transformer_strain, transformer_gs, transformer_strainrate, scaler_density,  scaler_YP,weight, sigma_z):
    """
    calculate strain at a stress level 
    Parameters:
    prediction_type: can be a string "average", "upper" or "lower"
    material_type: string, can be "Ni", "Al" or "Cu"
    sigma_z (float): stress level Pa;
    schimd_factor_grain: a [n * 12] array, with the schmid factor on all 12 systems
    gs (array ): grain size: m
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    strainrate: strainrate
    materials: dictionary for this material
    rho0: initial dislocation density 
    trained_model: trained ML model 
    transformer_strain: strain transformer
    transformer_gs: grain size transformer 
    transformer_strainrate: strainrate transformer
    weight: weight of grains on one section
    scaler_YP: Yield point transformer 

    Returns:
    float: plastic stress and predicted density 
    """

    # get material properties
    mu = materials["ShearModulus"]*1e9
    b = np.array(materials['LatticeConst'])/np.sqrt(2)*1e-10
    E_modulus = 2*mu*(1+materials["PoissonRatio"])

    # define a one-variable function that output the stress given a strain
    def stress(eps):
        #plastic stress
        [s_grain_average, ave_density, s_grain_p_upper,upper_density, s_grain_p_lower,lower_density]= sigma_plastic(material_type, eps, Schimd_factor, gs, alp, bet, strainrate, materials, rho0, trained_model,  transformer_strain, transformer_gs, transformer_strainrate,scaler_density,scaler_YP)
        #elastic stress
        s_grain_e = sigma_elastic(eps * np.ones_like(rho0), rho0,E_modulus)[0]
        if prediction_type == "average":
                s_grain_p = s_grain_average
                density = ave_density
        if prediction_type == "upper":
                s_grain_p = s_grain_p_upper
                density = upper_density
        if prediction_type == "lower":
                s_grain_p = s_grain_p_lower
                density = lower_density
        # a grain prefer to choose the lower stress value
        s_grain =  np.where((s_grain_e.reshape(np.shape(s_grain_e)) > s_grain_p.reshape(np.shape(s_grain_e))), s_grain_p.reshape(np.shape(s_grain_e)), s_grain_e.reshape(np.shape(s_grain_e)))
        # final stress is the weight of these grains 
        s0 = np.sum(s_grain * weight) 
        return s0
    # define a one-variable function where 0 point is the solution
    def stress_equation(eps):
        s0 = stress(eps)
        return s0-sigma_z

    # find the solution through bisection method 
    eps_sss,_ = NumericalMthod.bisect_root(stress,0.0, 0.2, 5000,sigma_z)

    return eps_sss


def density_to_stress(dislocation_den,alp,mu,b,gs,bet):
    """
    convert density to strain 
    Parameters:
    dislocation_den (float): dislocation density:1/m^2
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    mu (float): Shear modulus:Pa
    b (float): Burger's vector: m
    gs (float): grain size: m

    Returns:
    float: resolved shear stress(Pa)
    """
    tau = mu *(alp.reshape(alp.shape)*b*np.sqrt(dislocation_den).reshape(alp.shape) + bet.reshape(alp.shape)/gs.reshape(alp.shape)/np.sqrt(dislocation_den).reshape(alp.shape))
    return tau

def sample_from_mdn(means, std_devs, mix_coeffs, num_samples=1):
    # Choose components for each sample
    components = np.random.choice(len(mix_coeffs), size=num_samples, p=mix_coeffs)

    # Generate samples for each component
    samples = np.random.normal(means[components], std_devs[components])

    return samples
        
def distribution_stress(pi_variable,sigma_variable,mu_variable,gs_list,b_v,alp_list,bet_list,shear_modulus,scaler_density):# unit is shear modulus
    '''
    Distribution of  stress 
    # Parameters for the distribution of A and the constants C1, C2
    pi_variable,sigma_variable,mu_variable: model output
    gs_list: grain size list (m)
    b_v: Burgers vector list
    alp_list:coeff for equation between density and stress
    bet_list: coeff for equation between density and stress
    shear_modulus: Shear modulus 
    scaler_density: change normalized density to density
    '''  
    pi_data = pi_variable.data.numpy()
    sigma_data = sigma_variable.data.numpy()
    mu_data = mu_variable.data.numpy()
    
    # calculate the mean and variance for dislocation density with inverse_transform
    # CAUTION: IT SHOULD BE NOTICED THAT MDN IS THE SUMMATION OF DISTRIBUTION FUNCTION, NOT THE SUMMATION OF VARIABLES THAT SATISFY DIFFERENT DISTRIBUTION!!!

    mean_stress = []
    dev_stress = []
    mean_den = []
    dev_den = []
    
    for n in range(len(gs_list)):
        
        alp = alp_list[n]
        bet = bet_list[n]
        gs = gs_list[n]
        n_samples = 1000  # Number of samples
        
        # for k in range(len(pi_data[0])): # how  many different distribution
        mu = mu_data[n]
        sigma = sigma_data[n]
        pi = pi_data[n]
        
        normalized_logrho = sample_from_mdn(mu, sigma, pi, n_samples)
        logden_samples = scaler_density.inverse_transform(normalized_logrho.reshape(-1,1))
        den_samples = np.exp(logden_samples)
        # get statistical value fpr stress
        stress_samples =density_to_stress(den_samples,alp*np.ones_like(den_samples),shear_modulus*np.ones_like(den_samples),b_v*np.ones_like(den_samples),
        gs*np.ones_like(den_samples),bet*np.ones_like(den_samples))
        
        mean_stress.append(np.mean(stress_samples))
        dev_stress.append(np.std(stress_samples))
        
    return mean_stress, dev_stress, mean_den, dev_den

