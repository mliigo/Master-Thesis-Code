"""
Multi-Level TVP-DFM Simulation

This file contains functions for simulating a multi-level dynamic factor model 
with time-varying loadings and stochastic volatility as used in Del Negro and 
Otrok (2008). The main function is "sim", which uses previously defined functions 
to simulate the observables, factors and parameters, and uses them to compute the 
variance decompositions. It returns a dictionary with all these simulated values.

Functions:
----------
* sim_rw         :  simulates random walk without drift
* sim_arcoeffs   :  simulates stationary AR coefficients
* sim_ar_sv      :  simulates AR process with stochastic volatility
* comp_var_arsv  :  computes variances of AR process with stochastic volatility
* sim            :  simulates the model (observables, factors, parameters & 
                    variance decompositions)

Imports:
--------
* pandas
* numpy

References:
-----------
Del Negro, M. & Otrok, C. (2008). Dynamic Factor Models with Time-Varying 
Parameters: Measuring Changes in International Business Cycles (Staff Report 
No. 326). Federal Reserve Bank of New York. 
DOI: https://dx.doi.org/10.2139/ssrn.1136163
"""

# import packages
import pandas as pd
import numpy as np


## Functions to Simulate Processes --------------------------------------------

# Random walk without drift
def sim_rw(mu, std, T, RNG, burn=0):
    '''
    Simulates a driftless random walk of length T, with normal innovations with
    mean mu and standard deviation std.

    mu   : mean of iid normal innovations
    std  : standard deviation of iid normal innovations
    T    : number of time periods
    RNG  : numpy random generator
    burn : number of burn-in periods (default=100000)
    '''
    # allocate space and set initial values
    y = np.empty(T+burn)
    y[0] = 0

    # iteratively simulate process
    for t in range(1,T+burn):
        y[t] = y[t-1] + RNG.normal(mu, std)
    
    # return simulated process
    return y[-T:]

# Stationary AR coefficients (positive decreasing mean)
def sim_arcoeffs(L, mu_x, std, RNG):
    '''
    Simulate AR coefficients from univariate normal distributions.
    For l in (1,...,L), a coefficient is drawn from normal((L-l)*mu_x, std).

    L    : number of lags in AR process
    mu_x : multiplier in mean of normal 
    std  : standard deviation of normal distribution
    RNG  : numpy random generator
    '''
    # define empty array for storing
    phi = np.zeros(L)

    # draw each coefficient from univariate normal
    for l in range(L):
        phi[l] = RNG.normal((L-l)*mu_x, std)

    # get coefficients of corresponding lag polynomial
    coeffs = np.append(-np.flip(phi), 1)

    # check if roots lie outside unit circle and redraw if not
    while not all(i >= 1.001 for i in abs(np.roots(coeffs))):
        phi = np.zeros(L)
        for l in range(L):
            phi[l] = RNG.normal((L-l)*mu_x, std)
        coeffs = np.append(-np.flip(phi), 1)

    # return stationary AR coefficients
    return phi

# AR process with stochastic volatility
def sim_ar_sv(phi, mu, std, mu_rw, std_rw, T, RNG, burn=100000):
    '''
    Simulate AR process of time length T with stochastic volatility.
    Assumption: No stochastic volatility before beginning of sample period.

    phi    :  array of AR coefficients
    mu     :  mean of normal innovations
    std    :  standard deviation of normal innovations
    mu_rw  :  mean of iid normal innovations to RW
    std_rw :  standard deviation of iid normal innovations to RW
    T      :  number of time periods
    RNG    : numpy random generator
    burn   :  number of burn-in periods (default=100000)
    '''
    # get number of lags
    p = len(phi)

    # allocate space and set initial values
    y = np.empty(T+burn)
    y[0:p] = 0

    # simulate random walk (starting in t=1, 0 for t<=0)
    rw = sim_rw(mu_rw, std_rw, T, RNG=RNG, burn=1)

    # reverse order of AR coefficients
    phi_flip = np.flip(phi)

    # iteratively simulate process without stoch vola (up to end of burn)
    for t in range(p,burn):
        y[t] = np.sum(phi_flip*y[t-p:t]) + RNG.normal(mu, std)
    
    # iteratively simulate process with stoch vola (for final sample)
    for t in range(burn,burn+T):
        y[t] = np.sum(phi_flip*y[t-p:t]) + np.exp(rw[t-burn])*RNG.normal(mu, std)
    
    # return simulated process and rw process in volatilities
    return y[-T:], rw

# Compute variances of AR process with stochastic volatility
def comp_var_arsv(phi, s2, h):
    '''
    Compute path of variances of AR process with stochastic volatility.
    Needed for computation of Variance Decompositions. Assumes that no
    stochastic volatility at t <= 0.

    phi  :  array of AR coefficients
    s2   :  non time-varying variance part (sigma^2)
    h    :  RW process of SV
    '''
    # get hyperparameters
    T = len(h)
    P = len(phi)

    # compute time-varying variance part (e^h)**2
    eh2 = np.exp(h)**2

    # create companion matrix
    Phi = np.zeros((P,P))
    Phi[0,:] = phi
    if P>1:
        Phi[1:,:P-1] = np.identity(P-1)
    
    # compute variance of u_tilde at time 0
    var_u_tilde_0 = np.zeros((P,P))
    var_u_tilde_0[0,0] = s2

    # compute variance of process before SV (t<=0) and create storage
    var_y_tilde = np.zeros((T+1,P,P))
    var_y_tilde[0,:,:] = np.dot(np.linalg.inv(np.identity(P**2)-np.kron(Phi,Phi)),
                             var_u_tilde_0.flatten()).reshape((P,P), order='F')
    
    # create storage for variance of process with SV (t>0) and iteratively compute it
    var_y = np.zeros(T)
    for t in range(T):
        var_u_tilde_t = np.zeros((P,P))
        var_u_tilde_t[0,0] = eh2[t]*s2
        var_y_tilde[t+1,:,:] = (np.dot(np.dot(Phi, var_y_tilde[t,:,:]), Phi.transpose())+
                                  var_u_tilde_t)
        var_y[t] = var_y_tilde[t+1,0,0]
    
    # return variance process
    return var_y


## Function to Simulate Model -------------------------------------------------

# Simulate Multi-Level TVP-DFM
def sim(T, K, N_k, Q, P, a_mean, a_std, b0_mean_w, b0_std_w, b0_mean_k, 
        b0_std_k, s2, s2_eta_w, s2_eta_k, s2_zeta, s2_zeta_w, s2_zeta_K, 
        s2_w, s2_K, RNG, print_stat=0):
    '''
    Function to simulate data for Del Negro & Otrok (2008) Model with 
    1 world factor and K group factors. Returns a dictionary with the
    simulated observables, factors, parameters and variance decompositions.

    T          : number of time periods
    K          : number of group factors
    N_k        : number of observed series per group factor
    Q          : number of lags in factor series
    P          : number of lags in error series
    a_mean     : mean of Normal for generating intercepts
    a_std      : std. of Normal for generating intercepts
    b0_mean_w  : mean of Normal for generating initial world loadings (t=0)
    b0_std_w   : std. of Normal for generating initial world loadings (t=0)
    b0_mean_k  : mean of Normal for generating initial group loadings (t=0)
    b0_std_k   : std. of Normal for generating initial group loadings (t=0)
    s2         : list of non time-varying components of variances of error processes
    s2_eta_w   : list of variances of innovations to world loadings
    s2_eta_k   : list of variances of innovations to group loadings
    s2_zeta    : list of variances of innovations to SV of errors
    s2_zeta_w  : variance of innovations to SV of world factor
    s2_zeta_K  : list of variances of innovations to SV of group factors
    s2_w       : non time-varying component of variance of world factor
    s2_K       : list of non time-varying components of variances of group factor
    seed       : seed used to set numpy random generator (RNG)
    '''
    # get total number of series
    N = K*N_k

    # make lists of variances if none provided
    if len([s2])==1:
        s2 = [s2 for i in range(N)]
    if len([s2_K])==1:
        s2_K = [s2_K for k in range(K)]
    if len([s2_eta_w])==1:
        s2_eta_w = [s2_eta_w for i in range(N)]
    if len([s2_eta_k])==1:
        s2_eta_k = [s2_eta_k for i in range(N)]
    if len([s2_zeta])==1:
        s2_zeta = [s2_zeta for i in range(N)]
    if len([s2_zeta_K])==1:
        s2_zeta_K = [s2_zeta_K for k in range(K)]
    
    # specify group factor indicators (which series loads on which factor)
    # first N_k on first, next N_k on second, ...
    select_k = np.zeros(N).astype(int)
    for k in range(K):
        select_k[k*N_k:(k*N_k)+N_k] = k
    
    # specify sign restriction indicators 
    # (first pos. on world & first in each group pos. on group)
    sign_w = np.zeros(N).astype(int)
    sign_w[0] = 1
    sign_k = np.zeros(N).astype(int)
    for k in range(K):
        sign_k[k*N_k] = 1

    # create empty arrays to store results
    phi_w = np.zeros(Q)      # world factor AR coefficients
    phi_K = np.zeros((Q,K))  # group factor AR coefficients
    phi = np.zeros((P,N))    # error AR coefficients
    a = np.zeros(N)          # intercepts
    b_w = np.zeros((T,N))    # RW loadings on world factor
    b_k = np.zeros((T,N))    # RW loadings on group factor
    h_w = np.zeros(T)        # RW in SV of world factor process
    h_K = np.zeros((T,K))    # RW in SV of group factor processes
    h = np.zeros((T,N))      # RW in SV of error processes
    f_w = np.zeros(T)        # world factor process
    f_K = np.zeros((T,K))    # group factor processes
    eps = np.zeros((T,N))    # error processes
    y = np.zeros((T,N))      # simulated observed series

    # simulate world factor AR coefficients
    phi_w[:] = sim_arcoeffs(Q, 0.2, 0.1, RNG=RNG)

    # simulate world factor process and world factor SV
    f_w[:], h_w[:] = sim_ar_sv(phi_w, 0, np.sqrt(s2_w), 0, np.sqrt(s2_zeta_w), 
                               T, RNG=RNG)
    
    # loop over groups
    for k in range(K):

        # simulate group k factor AR coefficients
        phi_K[:,k] = sim_arcoeffs(Q, 0.2, 0.1, RNG=RNG)

        # simulate group k factor process and group k factor SV
        f_K[:,k], h_K[:,k] = sim_ar_sv(phi_K[:,k], 0, np.sqrt(s2_K[k]), 0, 
                                       np.sqrt(s2_zeta_K[k]), T, RNG=RNG)
    
    # loop over observed series
    for i in range(N):

        # generate error AR coefficients
        phi[:,i] = sim_arcoeffs(P, 0.3, 0.1, RNG=RNG)

        # generate error process and error SV
        eps[:,i], h[:,i] = sim_ar_sv(phi[:,i], 0, np.sqrt(s2[i]), 0, 
                                     np.sqrt(s2_zeta[i]), T, RNG=RNG)

        # generate intercepts
        a[i] = RNG.normal(a_mean, a_std)

        # generate world loadings
        b_w[:,i] = RNG.normal(b0_mean_w, b0_std_w) + sim_rw(0, np.sqrt(s2_eta_w[i]), 
                                                            T, RNG=RNG, burn=1)
        if (sign_w[i]==1) and not all(b_w[:,i]>0.5):
            wrong_sign = 1
            while wrong_sign:
                b_w[:,i] = RNG.normal(b0_mean_w, b0_std_w) + sim_rw(0, np.sqrt(s2_eta_w[i]), 
                                                                    T, RNG=RNG, burn=1)
                if all(b_w[:,i]>0.5):
                    wrong_sign = 0
            
        # generate group loadings
        b_k[:,i] = RNG.normal(b0_mean_k, b0_std_k) + sim_rw(0, np.sqrt(s2_eta_k[i]), 
                                                            T, RNG=RNG, burn=1)
        if (sign_k[i]==1) and not all(b_k[:,i]>0.5):
            wrong_sign = 1
            while wrong_sign:
                b_k[:,i] = RNG.normal(b0_mean_k, b0_std_k) + sim_rw(0, np.sqrt(s2_eta_k[i]), 
                                                                    T, RNG=RNG, burn=1)
                if all(b_k[:,i]>0.5):
                    wrong_sign = 0

        # compute observed (simulated) series
        for t in range(T):
            y[t,i] = a[i] + b_w[t,i]*f_w[t] + b_k[t,i]*f_K[t,select_k[i]] + eps[t,i]
    
    # compute variances of factors and idiosyncratic processes
    var_f_w = comp_var_arsv(phi_w, s2_w, h_w)
    var_f_K = np.zeros((T,K))
    for k in range(K):
        var_f_K[:,k] = comp_var_arsv(phi_K[:,k], s2_K[k], h_K[:,k])
    var_I = np.zeros((T,N))
    for i in range(N):
        var_I[:,i] = comp_var_arsv(phi[:,i], s2[i], h[:,i])
    
    # compute sum of variances by series
    sumvar = np.zeros((T,N))
    for i in range(N):
        sumvar[:,i] = (b_w[:,i]**2)*var_f_w[:] + (b_k[:,i]**2)*var_f_K[:,select_k[i]] + var_I[:,i]

    # compute variances attributable to the different components
    vd_w = np.zeros((T,N))
    vd_K = np.zeros((T,N))
    vd_I = np.zeros((T,N))
    for i in range(N):
        vd_w[:,i] = (b_w[:,i]**2)*var_f_w[:] / sumvar[:,i]
        vd_K[:,i] = (b_k[:,i]**2)*var_f_K[:,select_k[i]] / sumvar[:,i]
        vd_I[:,i] = var_I[:,i] / sumvar[:,i]

    # create dictionary with results
    results = {'f_w': f_w, 'f_K': f_K, 'y': y, 'phi_w': phi_w, 'phi_K': phi_K, 'phi': phi, 
               'a': a, 'b_w': b_w, 'b_k': b_k, 's2_w': s2_w, 's2_K': s2_K, 's2': s2, 
               's2_eta_w': s2_eta_w, 's2_eta_k': s2_eta_k, 's2_zeta_w': s2_zeta_w, 
               's2_zeta_K': s2_zeta_K, 's2_zeta': s2_zeta, 'h_w': h_w, 'h_K': h_K, 'h': h, 
               'vd_w': vd_w, 'vd_K': vd_K, 'vd_I': vd_I, 'select_k': select_k, 'sign_w': sign_w, 
               'sign_k': sign_k}

    # return results
    return results

