# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:33:51 2023

@author: luoji
"""

import numpy as np
from scipy.stats import lognorm
from scipy.stats import uniform
from scipy.stats import norm
def gs_lognormal_dist(thickness_number, lower_bound, upper_bound, mu, std, N_size):
    # Parameters
    sigma = np.sqrt(np.log(1 + (std/mu)**2))
    scale = mu / np.sqrt(1 + (std/mu)**2)
    

    dx = (-lower_bound + upper_bound) / (N_size + 1)

    # Log-normal distribution
    dist = lognorm(s=sigma, scale=scale)

    # Truncate the distribution
    samples = dist.rvs(100000)  # Large number of samples
    # samples = samples[(samples > lower_bound) & (samples < upper_bound)]

    E2pd = dist.std()**2 + dist.mean()**2

    # Generate x values
    x = np.arange(-upper_bound, upper_bound, dx)

    # PDF values
    pdf = dist.pdf(x) * x**2 / E2pd
    cdf = np.cumsum(pdf) * dx

    # Get the thickness size
    # thickness_number = 1000
    p = np.random.rand(thickness_number, 1)
    location_upper = np.array([x[min(np.sum(cdf < val) + 1, len(cdf)-1)] for val in p])
    thickness_size = np.maximum(location_upper - np.random.rand(thickness_number, 1).reshape(location_upper.shape) * dx, 0)

    return thickness_size, dist, cdf, x

def gs_norm_dist(thickness_number, lower_bound, upper_bound, mu, std, N_size):
    # Parameters
    # sigma = np.sqrt(np.log(1 + (std/mu)**2))
    # scale = mu / np.sqrt(1 + (std/mu)**2)
    

    dx = (-lower_bound + upper_bound) / (N_size + 1)

    # # Log-normal distribution
    dist = norm(loc = mu, scale = std); 

    # Truncate the distribution
    samples = dist.rvs(100000)  # Large number of samples
    samples = samples[(samples > lower_bound) & (samples < upper_bound)]

    E2pd = dist.std()**2 + dist.mean()**2

    # Generate x values
    x = np.arange(lower_bound, upper_bound, dx)

    # PDF values
    pdf = dist.pdf(x) * x**2 / E2pd
    cdf = np.cumsum(pdf) * dx

    # Get the thickness size
    # thickness_number = 1000
    p = np.random.rand(thickness_number, 1)
    location_upper = np.array([x[min(np.sum(cdf < val) + 1, len(cdf)-1)] for val in p])
    thickness_size = np.maximum(location_upper - np.random.rand(thickness_number, 1).reshape(location_upper.shape) * dx, 0)

    return thickness_size, dist, cdf, x


def gs_uniform_dist(thickness_number, lower_bound, upper_bound, mu, std, N_size):
    # Parameters
    # sigma = np.sqrt(np.log(1 + (std/mu)**2))
    # scale = mu / np.sqrt(1 + (std/mu)**2)
    

    dx = (-lower_bound + upper_bound) / (N_size + 1)

    # # Log-normal distribution
    dist = uniform(loc = lower_bound,scale = upper_bound -lower_bound)

    # Truncate the distribution
    samples = dist.rvs(100000)  # Large number of samples
    

    E2pd = dist.std()**2 + dist.mean()**2

    # Generate x values
    x = np.arange(lower_bound, upper_bound, dx)

    # PDF values
    pdf = dist.pdf(x) * x**2 / E2pd
    cdf = np.cumsum(pdf) * dx

    # Get the thickness size
    # thickness_number = 1000
    p = np.random.rand(thickness_number, 1)
    location_upper = np.array([x[min(np.sum(cdf < val) + 1, len(cdf)-1)] for val in p])
    thickness_size = np.maximum(location_upper - np.random.rand(thickness_number, 1).reshape(location_upper.shape) * dx, 0)

    return thickness_size, dist, cdf, x

