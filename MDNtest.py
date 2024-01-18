# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:18:03 2024

@author: luoji
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:32:13 2023

@author: luoji
"""


from spyder_kernels.utils.iofuncs import load_dictionary
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
import random
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
from torch.autograd import Variable # storing data while learning
import pickle
import torch.nn.functional as F

#%% pre training process 
randseed = 1995
np.random.seed(randseed)
random.seed(randseed)
torch.manual_seed(randseed)
AveSchmidFactor = 2.2 #the average value from mtex gives Schmid factor 2.2 
oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians
path = '../samplesize97_withObserve_withExpyieldpoint3.spydata'
test_size = 0.33
# scaling law: 
def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI


#%% Physics 
def initial_dislocation_density(tau0,alp,mu,b,gs,bet):    
    '''
    Parameters
    ----------
    tau0 (float):Resolved yield shear stress:Pa
    alp (float): Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    bet (float): Coefficient beta in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)). 
    mu (float): Shear modulus:Pa
    b (float): Burger's vector: m
    gs (float): grain size: m

    Returns
    -------
    rho0 intial dislocation density 
        array of float 
    found : bool
        if we found the solution
    '''
    rho0 = []
    found = []
    for kk in range(0,len(tau0)):
        C0 = bet/(gs[kk])
        C2 = alp*(b[kk])
        C1 = - tau0[kk]/mu[kk]
        rho0_kk = np.roots([C2,C1,C0])# find roots for equation between density and stress 
        if np.isreal(rho0_kk[0]):           
            rho0.append(max(rho0_kk)**2)
            found.append(1)
        else:
            temp =- 1/(2*C2)*(C1)
            rho0.append(temp**2)
            found.append(0)      
    return np.array(rho0),found


def dislocation_density(tau_list,alp,mu,b,tau0,gs,bet):
    '''
    Convert resolved shear stress to density     

    Parameters
    ----------
    tau_list : float array
        (Resolved yield shear stress:Pa
    alp :  float 
        Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).
    mu : float 
        Shear modulus
    b : float 
        Burgers vector : m
    tau0 : float array 
        initial dislocation density 1/m^2
    gs : float 
        grain size  grain size 
    bet : float 
        Coefficient alpha in euqation \tau = alpha * \mu b *sqrt(\rho) + beta \mu / (d sqrt(\rho)).

    Returns
    -------
    float array 
        list of density 

    '''
    rho0, found =  initial_dislocation_density(tau0,alp,mu,b,gs,bet)
    tau_cr = alp*mu*b*np.sqrt(rho0) + bet/(gs)/np.sqrt(rho0)*mu # minimum stress for yielding 
    dislocation_den = []
    for kk in range(0,len(tau_list)):
        if np.array(tau_list)[kk] < tau_cr[kk] :
            dislocation_den.append(rho0[kk]) # if small than rho_cr, it is rho_0
        else:
            A = bet/(gs[kk])
            B = alp*(b[kk])
            tau_bar = tau_list[kk]/mu[kk]
            dislocation_den.append(0.25*(tau_bar/B + np.sqrt((tau_bar/B)**2 - 4*A/B))**2) # otherwise it is the solution 
    return np.array(dislocation_den)   


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
    tau = mu *(alp*b*np.sqrt(dislocation_den) + bet/gs/np.sqrt(dislocation_den))
    return tau



#%%data processing 
def loadingdata(dataname):    
    ''' 
    Parameters
    ----------
    dataname : string 
        the path of loading  spydata file .
    ------
    df : datafram 
        the initial dataset without any processing 
    '''
    data = load_dictionary(dataname)
    datas = data[0]["data"]
    # these are raw data from experiments
    filename = "material_properties.csv"
    with open(filename, 'r') as csv_file:    
        reader = csv.DictReader(csv_file)     
        data = list(reader)
    # get material list
    material_list = {row.pop('Material'): {k: float(v) for k, v in row.items()} for row in data}
   #assign all list 
    materials = []
    stresses = []
    strains = []
    GS = []
    samplesizes = []
    strainrates = []
    processingmethods = []
    ShearModuluses = []
    PoissonRatios = []
    LatticeConsts = []
    AtomicVolumes = []
    StackingFaultEs = []
    Yieldpoint = []
    YE_observe = []
    k = 0
    while k<len(datas):        
        sscurve = datas[k]
        # -----------assign the first data point of  ss curve in training set -------------#
        strains.append(float((sscurve["YE_observe"]))) # assign observed yield  strain        
        if sscurve["YS_exp"] == 'nan':                
            stresses.append(float(sscurve["YS_observe"])) # if we cannot find the exeprimental data, use observed one
        else:
            stresses.append(float((sscurve["YS_exp"])))    # otherwise use reported one 
        #material type 
        materials.append(sscurve["material"])
        # grain size m 
        GS.append(sscurve["size"])
        # size sample 3 element array 
        samplesize = sscurve["samplesize"]
        while len(sscurve["samplesize"]) < 3:
            samplesize.append(np.nan) # if not find it, assign nan valyue 
        samplesizes.append(sscurve["samplesize"])
        # processing method
        processingmethods.append(sscurve["Processingmethod"])
        #strain rate
        strainrates.append(float(sscurve["StrainRate"] or 1e-4))
        if sscurve["YS_exp"] == 'nan':                
            Yieldpoint.append(float(sscurve["YS_observe"]))
        else:
            Yieldpoint.append(float((sscurve["YS_exp"])))
        ShearModuluses.append(material_list[sscurve["material"]]['ShearModulus'])
        PoissonRatios.append(material_list[sscurve["material"]]['PoissonRatio'])
        LatticeConsts.append(material_list[sscurve["material"]]['LatticeConst'])
        AtomicVolumes.append(material_list[sscurve["material"]]['AtomicVolume'])
        StackingFaultEs.append(material_list[sscurve["material"]]['StackingFaultE'])
        YE_observe.append(float((sscurve["YE_observe"])))
        # -----------assign the first data point of  ss curve in training set -------------#
       
        
       # ----------- add other data point into training set   -----------#
        for strain, stress in sscurve["sscurve"]:
            strains.append(strain)
            stresses.append(stress)
            materials.append(sscurve["material"])
            GS.append(sscurve["size"])
            samplesize = sscurve["samplesize"]
            while len(sscurve["samplesize"]) < 3:
                samplesize.append(np.nan)
            samplesizes.append(sscurve["samplesize"])
            
            processingmethods.append(sscurve["Processingmethod"])
            strainrates.append(float(sscurve["StrainRate"] or 1e-4))
            if sscurve["YS_exp"] == 'nan':                
                Yieldpoint.append(float(sscurve["YS_observe"]))
            else:
                Yieldpoint.append(float((sscurve["YS_exp"])))
            ShearModuluses.append(material_list[sscurve["material"]]['ShearModulus'])
            PoissonRatios.append(material_list[sscurve["material"]]['PoissonRatio'])
            LatticeConsts.append(material_list[sscurve["material"]]['LatticeConst'])
            AtomicVolumes.append(material_list[sscurve["material"]]['AtomicVolume'])
            StackingFaultEs.append(material_list[sscurve["material"]]['StackingFaultE'])
            YE_observe.append(float((sscurve["YE_observe"])))
        k += 1
            
    # how we deal with nan value in sample size
    samplesizes_np = np.array(samplesizes)
    samplesizes_np[np.isnan(samplesizes_np[:,0]),0] = 2 # mm 
    samplesizes_np[np.isnan(samplesizes_np[:,1]),1] = 2
    samplesizes_np[np.isnan(samplesizes_np[:,2]),2] = 5*samplesizes_np[np.isnan(samplesizes_np[:,2]),1]
    df = pd.DataFrame({
        'grainsize': GS,#unit um
        'strains': strains,
        'strainrates':strainrates,
        'Material':materials,
        'ShearModuluses':ShearModuluses,#unit:GPa
        'PoissonRatios': PoissonRatios,
        'LatticeConsts': LatticeConsts,#unit:A
        'AtomicVolumes': AtomicVolumes,#unit:A^3
        'StackingFaultEs': StackingFaultEs,#unit: mJ
        'stresses':stresses,#unit MPa
        "Yieldpoint":Yieldpoint,#unit MPa
        "YE_observe":YE_observe
    }) # assign the initial dataset without any processing 
    return df

def normalizedata(df,test_size):
    '''
    Parameters
    ----------
    df : dataframe
        raw data without processing 
    test_size : percentage that is used as test sample 
        DESCRIPTION.

    Returns
    -------
    X_train : training set input 
    X_test : test set input
    y_train : training set output 
    y_test : test set output
    transformer_strain : normalizer for strain 
    transformer_gs : normalizer for grain size 
    transformer_strainrate : normalizer for strain rate 
    scaler_density : normalizer for density AFTER LN FUNCTION
    scaler_YP : normalizer for yield points 
    '''
    features = df.drop('stresses', axis=1) # get the remaining except stress as features 
    target = df['stresses'] # target is the stress 
    
    
    ShearModulus = np.array(features['ShearModuluses'])*1e9  # change the shear modulus unit to be Pa
    features['ShearModuluses'] = ShearModulus
    features['LatticeConsts'] = np.array(features['LatticeConsts'])*1e-10 # change to unit m 
    grainsize = np.array(features['grainsize'])*1e-6 # change unit to m
    features['grainsize'] = grainsize
    tau_list = pd.Series(np.array(target)*1e6/AveSchmidFactor)# change normal stress to shear stress
   
    Yieldpoint = np.array(features['Yieldpoint'])*1e6/AveSchmidFactor # change normal stress to shear stress 
    features['Yieldpoint'] = Yieldpoint 
    
    
    alp = 0.57
    bet = 0.00176
    
    BurgersVector = np.array(features['LatticeConsts'])/np.sqrt(2)# change unit to m

    # change shear stress to density 
    density_train = dislocation_density(tau_list ,alp,ShearModulus,BurgersVector,Yieldpoint,grainsize, bet)
    # set the density as the output df type 
    density_train = pd.DataFrame(density_train)
    #-----------uncomment to plot density distribution-----------#
    plt.figure()
    plt.hist(np.log1p(density_train),bins=20)
    plt.xlabel(r'lg($\rho$)')
    plt.show()
    #-----------uncomment to plot density distribution-----------#
    
    # get initial dislocation density 
    density_y , found= initial_dislocation_density(Yieldpoint,0.57,ShearModulus,BurgersVector,grainsize,0.00176)
   
    # the following part gives YP0 and den0 which is smaller than YP and den_y
    # we do not use this it as the feature since it does not change so much 
    #-----------uncomment to plot the difference -----------#
    # E_modulus = ShearModulus * 2*(1+features["PoissonRatios"])  # get Youngs Modulus 
    # eps_cr =  features['YE_observe'] #yield strain 
    # A_para= 1
    # rho0 = rho_y 
    # density_increase = A_para*(eps_cr)/BurgersVector/grainsize
    # density0 = np.where( density_y -  density_increase > 0, density_y -  density_increase, density_y )
    # YP0 = density_to_stress(density0, alp, ShearModulus, BurgersVector, grainsize , bet)
    # plt.figure()
    # plt.xlabel(r'lg($\rho_0$)')
    # plt.show()
    # color = grainsize
    # color_normalized = (color - np.min(color)) /( 200e-6 - 1e-6)
    # plt.figure()
    # plt.scatter(features['strains'], np.log1p( density_train), c = color_normalized,cmap='viridis',alpha = 0.2)
    # plt.xlabel('strain')
    # plt.ylabel(r'lg($\rho$)')
    # plt.xlim([0,0.25])
    # plt.colorbar(label='Grain size (1 to 200 um )')
    # plt.show() 
    # plt.scatter(grainsize/1e-6, np.log1p(density0),alpha = 0.1, color='r',label = 'After')
    # plt.scatter(grainsize/1e-6, np.log1p(density_y),alpha = 0.1, color='b',label = 'Before')
    # plt.xlabel('grain size(um)')
    # plt.ylabel(r'lg($\rho_0$)')
    # for kk in range(len(density0)):
    #     plt.plot([grainsize[kk]/1e-6, grainsize[kk]/1e-6], [np.log1p(density0[kk]), np.log1p(density_y[kk])],color='black')
    # plt.xlim([0,100])
    # plt.legend()
    # plt.show()
    # plt.scatter(grainsize/1e-6, np.log1p(density_y),alpha = 0.1, color='b')
    # plt.xlabel('grain size(um)')
    # plt.ylabel(r'lg($\rho_0$)')
    # plt.xlim([25,100])
    # plt.show()
    #-----------uncomment to plot the difference -----------#
    
    # drop some features we do not use/ but we can use latter 
    drop_column = ['ShearModuluses','LatticeConsts','PoissonRatios','AtomicVolumes','StackingFaultEs','YE_observe']
    features = features.drop(columns=drop_column, axis = 1)
    # get onehotencode such for material type
    features = pd.get_dummies(features, columns=['Material'])
    # split the training and test set by the percentage given 
    X_train, X_test, density_train, density_test = train_test_split(features,density_train, test_size=test_size, random_state=2024)
    # No test for MDN 
    X_train = X_train.copy()
    X_test = X_test.copy()# hard copy 
    
    # until now, we have the features: strain, strain rate, grain size, RESOLVED SHEAR STRESS AT YIELD, material type 
    # we also have the output : density
   
    # normalization
    transformer_strain = QuantileTransformer(output_distribution='uniform',n_quantiles=20)
    strain_quantile_transformed = transformer_strain.fit_transform(np.array(X_train['strains']).reshape(-1, 1))
    X_train['strains'] = strain_quantile_transformed 
    X_test['strains'] = transformer_strain.transform(np.array(X_test['strains']).reshape(-1, 1)) 

    transformer_strainrate = QuantileTransformer(output_distribution='uniform',n_quantiles=20)
    strainrate_quantile_transformed = transformer_strainrate.fit_transform(np.array(X_train['strainrates']).reshape(-1, 1)) 
    X_train['strainrates'] = strainrate_quantile_transformed 
    X_test['strainrates'] = transformer_strainrate.transform(np.array(X_test['strainrates']).reshape(-1, 1)) 

    transformer_gs = QuantileTransformer(output_distribution='uniform',n_quantiles=10)
    gs_quantile_transformed = transformer_gs.fit_transform(np.array(X_train['grainsize']).reshape(-1, 1))
    X_train.loc[:,'grainsize'] = gs_quantile_transformed 
    X_test['grainsize'] = transformer_gs.transform(np.array(X_test['grainsize']).reshape(-1, 1)) 
    
    scaler_YP =  MinMaxScaler() #QuantileTransformer(output_distribution='normal',n_quantiles=n_quantiles)
    scaled_feature = scaler_YP.fit_transform(X_train[['Yieldpoint']])
    X_train['Yieldpoint'] = scaled_feature
    X_test['Yieldpoint'] = scaler_YP.transform(X_test[['Yieldpoint']])
    
    
    data = np.array(density_train).reshape(-1, 1)
    data_log = np.log1p(data) 
    scaler_density = MinMaxScaler() #QuantileTransformer(output_distribution='normal',n_quantiles=n_quantiles)
    y_train = scaler_density.fit_transform(data_log)
    y_train = pd.DataFrame(y_train)    # change the data type to dataframe for output 
    testdata_log = np.log1p((np.array(density_test).reshape(-1, 1)))
    y_test = scaler_density.transform(testdata_log)
    y_test = pd.DataFrame(y_test)  
    

    
    return X_train, X_test, y_train, y_test, transformer_strain, transformer_gs, transformer_strainrate,scaler_density, scaler_YP

    
#%% ML part 
    
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


# define loss function of MDN model
def mdn_loss_fn(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result + 1e-6)
    # result = (result >  0) * result
    # if torch.mean(result) < 0:
    #      print(torch.exp(-result))
    return torch.mean(result)


# early stopping mechanism 
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):# how many steps not improve
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



    
def train_mdn(x_variable,y_variable,x_val,y_val,network, optimizer,scheduler):
    val_losses = []
    # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=20, min_delta=0.0001)
    for epoch in range(10000):# during this epoches 
        pi_variable, sigma_variable, mu_variable = network(x_variable)# output the network result 
        loss = mdn_loss_fn(pi_variable, sigma_variable, mu_variable, y_variable) #calculate the lost 
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() # backfoward and update 
        pi_val, sigma_val, mu_val = network(x_val) # try on validate data 
        val_loss = mdn_loss_fn(pi_val, sigma_val, mu_val, y_val)
        val_losses.append(val_loss.detach().numpy())
        early_stopping(val_loss)# to check if early stop needed 
        if early_stopping.early_stop:
            break
    #---------------uncomment to see the performance ------------#
    # plt.plot(np.arange(0,epoch+1,1),np.array(val_losses))
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss function")
    # plt.title("Training process of loss function")
    # plt.show()
    #---------------uncomment to see the performance ------------#


#%%--------------------Main function -------------------------------


data = loadingdata(path)


# get training set and test set separated 
X_train, X_test, y_train, y_test, transformer_strain, transformer_gs, transformer_strainrate,  scaler_density,scaler_YP = normalizedata(data,test_size)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2025)

n_hidden = 10
n_gaussians = 2
n_layers = 4
lr = 0.0003990130683249108
n_input = 7
n_output = 1
network = MDN(n_input, n_hidden, n_layers, n_gaussians)
optimizer = torch.optim.Adam(network.parameters(),lr,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)   

# change datatype to MDN satisfied type(torch)     
n_samples = len(X_train)
y_tensor = torch.from_numpy(y_train.to_numpy().reshape(n_samples, n_output)).float()
x_tensor = torch.from_numpy(X_train.to_numpy().reshape(n_samples, n_input)).float()
x_variable = Variable(x_tensor)
y_variable = Variable(y_tensor, requires_grad=False)

n_samples = len(X_val)
y_tensor = torch.from_numpy(y_val.to_numpy().reshape(n_samples, n_output)).float()
x_tensor = torch.from_numpy(X_val.to_numpy().reshape(n_samples, n_input)).float()
xx_val = Variable(x_tensor)
yy_val = Variable(y_tensor, requires_grad=False)

# training process  
train_mdn(x_variable, y_variable,xx_val,yy_val,network, optimizer,scheduler)


# show the performance 
n_samples = len(X_test)
x_tensor = torch.from_numpy(X_test.to_numpy().reshape(n_samples, n_input)).float()
x_test_variable = Variable(x_tensor)
pi_variable, sigma_variable, mu_variable = network(x_test_variable)
test_loss =  network(x_test_variable)


#%%-----------------------Visualization pf  the result -----------------------------
pi_data = pi_variable.data.numpy()
sigma_data = sigma_variable.data.numpy()
mu_data = mu_variable.data.numpy()

#--------------uncomment it to see the learned individual distribution  --------------
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,8))
# ax1.plot(x_test_data, pi_data)
# ax1.set_title('$\Pi$')
# ax2.plot(x_test_data, sigma_data)
# ax2.set_title('$\sigma$')
# ax3.plot(x_test_data, mu_data)
# ax3.set_title('$\mu$')
# plt.xlim([-15,15])
# plt.show()

#%%------------ performance of trained model ----------------------------------
space_n = 100
plt.figure(figsize=(8, 8), facecolor='white')
print("Here")
plt.plot(np.linspace(1e10,1e14,space_n),np.linspace(1e10,1e14,space_n))    
den_true =np.exp( scaler_density.inverse_transform( y_test.to_numpy().reshape(-1,1)))
den_pred = np.exp( scaler_density.inverse_transform(  np.sum(pi_data * mu_data,axis=1).reshape(-1,1)))
upper_bar =  np.exp( scaler_density.inverse_transform((np.sum(pi_data * mu_data,axis=1) - np.sqrt(np.sum(pi_data**2 * sigma_data**2,axis=1))).reshape(-1,1)))
upper_bar = upper_bar - den_pred
lower_bar =  np.exp( scaler_density.inverse_transform((np.sum(pi_data * mu_data,axis=1) + np.sqrt(np.sum(pi_data**2 * sigma_data**2,axis=1))).reshape(-1,1)))
lower_bar = den_pred - lower_bar
asymmetric_error = np.vstack([lower_bar.T, upper_bar.T])
# plt.fill_between(den_true, upper_bar,lower_bar)
# plt.plot(den_true,den_pred)
plt.errorbar(den_true, den_pred, yerr=asymmetric_error, fmt='o', capsize=5, ecolor='red', linestyle=' ', color='blue', label='Data with Error')
plt.xscale('log')
plt.yscale('log')

plt.xlim([5e10,5e14])
plt.ylim([5e10,5e14])
# plt.ylim([0,1])
plt.ylabel("Prediction",fontsize = 15)
plt.xlabel("Experiment",fontsize = 15)
plt.legend()
plt.title("Prediction vs Experiments on Test data",fontsize = 15)
plt.show()

# #%%
plt.figure(figsize=(8, 8), facecolor='white')

# plt.rc('axes', labelsize=20) #fontsize of the x and y labels
# plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
# plt.rc('ytick', labelsize=20) #fontsize of the y tick labels


plt.plot(np.linspace(0,1,space_n),np.linspace(0,1,space_n))    


plt.errorbar( y_test.to_numpy(),  np.sum(pi_data * mu_data,axis=1), np.sqrt(np.sum(pi_data**2 * sigma_data**2,axis=1)), fmt='o', capsize=5, ecolor='red', linestyle=' ', color='blue', label='Data with Error')


plt.ylabel("Prediction",fontsize = 15)
plt.xlabel("Experiment",fontsize = 15)
plt.legend(fontsize = 15)
plt.title("Prediction vs Experiments on Test data(After normalization)",fontsize = 15)
plt.show()


#%%
list_save = [network, transformer_strain, transformer_gs, transformer_strainrate,  scaler_density,scaler_YP]
filename = 'finalized_MDN2.sav'
with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(list_save, f)
# =============================================================================

space_n = 100
strain = np.linspace(1e-6,1e-3,space_n)


#%% =============================================================================
for mu_k, sigma_k in zip(mu_data.T, sigma_data.T):
    plt.figure(figsize=(8, 8), facecolor='white')
    average_density_afterlog1 = scaler_density.inverse_transform( mu_k.reshape(-1,1))
    density = np.exp(average_density_afterlog1)
    plt.scatter(y_test.to_numpy(), density)
    average_density_afterlog1 = scaler_density.inverse_transform( (mu_k+sigma_k).reshape(-1,1))
    density_upper = np.exp(average_density_afterlog1)
    average_density_afterlog1 = scaler_density.inverse_transform( (mu_k-sigma_k).reshape(-1,1))
    density_lower = np.exp(average_density_afterlog1)
    plt.scatter(y_test.to_numpy(), density_upper,alpha=0.1)
    plt.scatter(y_test.to_numpy(), density_lower,alpha=0.1)
    # plt.scatter(mdn_x_data, mdn_y_data, marker='.', lw=0, alpha=0.2, c='black')
    # plt.xlim([0,1])
    # plt.ylim([0,1])
    plt.show()
    
#%%==================Distribution of  stress ====================================
# Parameters for the distribution of A and the constants C1, C2
upper_bar =  scaler_density.inverse_transform((np.sum(pi_data * mu_data,axis=1) + np.sqrt(np.sum(pi_data**2 * sigma_data**2,axis=1))).reshape(-1,1))
mu_list =  scaler_density.inverse_transform((np.sum(pi_data * mu_data,axis=1).reshape(-1,1)))

mean_C = []
dev_C = []
for n in  [0]:
    sigma_list = upper_bar - mu_list
    mu = mu_list[5][0]
    sigma = sigma_list[5][0]
    gs = transformer_gs.inverse_transform(np.array(X_test['grainsize']).reshape(-1, 1))[5][0]
    b_v = 2.5e-10
    C1, C2 = 0.57*b_v, 1.76e-3/gs   # Constants
    n_samples = (n+1)*1000 # Number of samples
    
    # Generate samples from the distribution of A
    A_samples = np.random.normal(mu, sigma, n_samples)
    plt.hist(A_samples, bins=20)
    plt.title(r'Histogram of lg($\rho0$)')
    plt.xlabel('normalized density')
    plt.ylabel('Probability Density')
    plt.show()
    # Calculate B = exp(A)
    B_samples = np.exp(A_samples)
    
    plt.hist(B_samples, bins=20)
    plt.title('Histogram of density')
    plt.xlabel('density')
    plt.ylabel('Probability Density')
    plt.show()
    # Calculate C = C1/sqrt(B) + C2*sqrt(B)
    C_samples = (C1 * np.sqrt(B_samples) + C2 / np.sqrt(B_samples))*76e9
    
    # Plotting the distribution of C
    plt.hist(C_samples, bins=20)
    plt.title('Histogram of stress')
    plt.xlabel('Stress(Pa)')
    plt.ylabel('Probability Density')
    plt.show()
    
    mean_C.append(np.mean(C_samples))
    dev_C.append(np.std(C_samples))
    # You can also calculate and print descriptive statistics if needed
    # print(f"Mean of C: {np.mean(C_samples)}")
    # print(f"Standard Deviation of C: {np.std(C_samples)}")
    
#%% =============================================================================   
