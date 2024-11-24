"""
Multi-Level TVP-DFM Gibbs Estimation

This file contains functions for estimating a multi-level dynamic factor model 
with time-varying loadings and stochastic volatility as used in Del Negro and 
Otrok (2008). The main function is "gibbs", which uses the defined conditional
posterior samplers to generate draws from the model's joint posterior, and 
uses those to compute the time-varying variance decompositions.

Functions:
----------
* Sigma_comp       : computes part of the variance of the first p observations 
                     of an AR(p) process
* Psi_comp         : computes part of the density kernel for sampling the 
                     AR parameters of series i
* Psi_0_comp       : computes part of the density kernel for sampling the 
                     AR parameters of a factor
* S_i_comp         : computes part of the variance of the first p observations 
                     of series i
* var_arsv_comp    : computes path of variance of AR process with stochastic 
                     volatility from t=1 onwards
* a_s2_i_sampler   : draws constants and non time-varying variance components
* phi_i_sampler    : draws AR parameters in idiosyncratic processes
* phi_0_sampler    : draws AR parameters in factor processes
* s2_eta_i_sampler : draws innovation variances of loading processes
* s2_zeta_sampler  : draws innovation variances of stochastic volatility 
                     processes of series or factors
* f_w_sampler      : draws world factor
* f_k_sampler      : draws group k factor
* b_i_sampler      : draws world or group loadings
* h_i_sampler      : draws stochastic volatility components of series i
* h_0_sampler      : draws stochastic volatility components of factors
* gibbs            : performs the Gibbs sampling procedure using the above
                     sampling functions and returns a dictionary of Gibbs draws

Imports:
--------
* pandas
* numpy
* datetime

References:
-----------
Del Negro, M. & Otrok, C. (2008). Dynamic Factor Models with Time-Varying 
Parameters: Measuring Changes in International Business Cycles (Staff Report 
No. 326). Federal Reserve Bank of New York. 
DOI: https://dx.doi.org/10.2139/ssrn.1136163

Kim, S., Shephard, N. & Chib, S. (1998). Stochastic Volatility: Likelihood 
Inference and Comparison with ARCH Models. Review of Economic Studies, 65(3), 
361–393. DOI: https://doi.org/10.1111/1467-937X.00050

Omori, Y., Chib, S., Shephard, N. \& Nakajima, J. (2007). Stochastic Volatility 
with Leverage: Fast and Efficient Likelihood Inference. Journal of Econometrics,
140(2), 425–449. DOI: https://doi.org/10.1016/j.jeconom.2006.07.008

Otrok, C. & Whiteman, C. H. (1998). Bayesian Leading Indicators: Measuring and 
Predicting Economic Conditions in Iowa. International Economic Review, 39(4), 
997–1014. DOI: https://doi.org/10.2307/2527349
"""

# import packages
import pandas as pd
import numpy as np
from datetime import datetime


## Functions used in Samplers -------------------------------------------------

# Compute Sigma_i 
def Sigma_comp(phi_i):
    '''
    Compute Sigma_i part of covariance matrix of first p observations of AR(p) 
    process. See Otrok & Whiteman (1998) p.1001.

    phi_i : vector of AR(p) paremeters in error process of series i (px1)
    '''
    # define lag order p
    p = len(phi_i)

    # create companion matrix 
    Phi_i = np.zeros((p,p))
    Phi_i[0] = phi_i
    if p>1:
        Phi_i[1:,:p-1] = np.identity(p-1)
    
    # define e = (1,0,...,0)'(1,0,...,0)
    e = np.zeros((p,p))
    e[0,0] = 1

    # compute and return Sigma
    return np.dot(np.linalg.pinv(np.identity(p**2)-np.kron(Phi_i, Phi_i)), 
                  e.flatten()).reshape((p,p), order='F')

# Compute Psi_i 
def Psi_comp(y_i, f_w, f_k, a_i, b_w_i, b_k_i, phi_i, eh_i, s2_i, p):
    '''
    Compute Psi(phi_i) part of density kernel for sampling AR parameters phi_i 
    of error series i. See Del Negro & Otrok (2008) pp.29-31.

    y_i       : vector of observations on series i (Tx1)
    f_w       : vector of world factor values (Tx1)
    f_k       : vector of group factor values (Tx1)
    a_i       : intercept of series i
    b_w_i     : vector of world loadings of series i (Tx1)
    b_k_i     : vector of group loadings of series i (Tx1)
    phi_i     : vector of AR(p) paremeters for error process of series i (px1)
    eh_i      : vector of SV part of error process of series i (Tx1)
    s2_i      : non time-varying component of innovation variance of series i
    p         : order of AR(p) error process
    '''
    # comupte S_i
    S_i = S_i_comp(phi_i=phi_i, eh_i=eh_i, p=p)

    # compute corresponding inverted Cholesky factor
    Q_i_inv = np.linalg.pinv(np.linalg.cholesky(S_i))

    # define y_i1_tilde and x_i1_tilde
    y_i1_tilde = y_i[:p] - b_w_i[:p]*f_w[:p] - b_k_i[:p]*f_k[:p]
    x_i1_tilde = np.ones(p)

    # compute y_i1_tilde_star 
    y_i1_tilde_star = np.dot(Q_i_inv, y_i1_tilde)

    # compute part of Psi_i (part in exponential)
    err_i = y_i1_tilde_star - a_i*x_i1_tilde
    Psi_i_part = -(1/(2*s2_i)) * np.dot(np.dot(err_i.transpose(), np.linalg.pinv(S_i)), err_i)
    
    # compute and return Psi_i
    return np.linalg.det(S_i)**(-0.5)*np.exp(Psi_i_part)

# Compute Psi_0
def Psi_0_comp(f, phi_0, s2_0, q):
    '''
    Compute Psi(phi_0) part of density kernel for sampling AR parameters phi_0
    of factor processes. See Del Negro & Otrok (2008) p.31.

    f      : vector of factor values (Tx1)
    phi_0  : vector of AR(q) paremeters in factor process (qx1)
    s2_0   : non time-varying component of factor innovation variance
    q      : order of AR(q) factor process
    '''
    # compute Sigma_0
    Sigma_0 = Sigma_comp(phi_0)

    # compute part of Psi_0 (part in exponential)
    Psi_0_part = -(1/(2*s2_0)) * np.dot(np.dot(f[:q].transpose(), np.linalg.pinv(Sigma_0)), f[:q])
    
    # compute and return Psi_0
    return np.linalg.det(Sigma_0)**(-0.5)*np.exp(Psi_0_part)

# Compute S_i 
def S_i_comp(phi_i, eh_i, p):
    '''
    Compute S_i, part of variance of first p observations of series i.

    phi_i  : vector of AR(p) paremeters in error process of series i (px1)
    eh_i   : vector of SV part of error process of series i (Tx1)
    p      : order of AR(p) error process
    '''
    # create companion matrix 
    Phi_i = np.zeros((p,p))
    Phi_i[0] = phi_i
    if p>1:
        Phi_i[1:,:p-1] = np.identity(p-1)
    
    # define e_1 = [1 0 ... 0]'
    e_1 = np.zeros(p)
    e_1[0] = 1

    # compute Z_i matrix
    Z_i = np.zeros((p,p))
    for j in range(p):
        Z_i[:,j] = eh_i[p-1-j] * np.dot(np.linalg.matrix_power(Phi_i,j), e_1)
    
    # compute Sigma_i
    Sigma_i = Sigma_comp(phi_i)

    # comupte S_i
    Phi_i_p = np.linalg.matrix_power(Phi_i,p)
    S_i = np.dot(np.dot(Phi_i_p, Sigma_i), Phi_i_p.transpose()) + np.dot(Z_i,Z_i.transpose())

    # return S_i
    return S_i

# Compute variances of AR process with stochastic volatility 
def var_arsv_comp(phi, s2, h):
    '''
    Compute path of variances of AR process with stochastic volatility.
    Needed for computation of variance decompositions. Assumes that no
    stochastic volatility is present at t <= 0.

    phi  :  array of AR coefficients 
    s2   :  non time-varying variance part
    h    :  random walk process of stochastic volatility components (Tx1)
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

    # compute variance of process before SV (t<=0)
    var_y_tilde = np.zeros((T+1,P,P))
    var_y_tilde[0,:,:] = np.dot(np.linalg.inv(np.identity(P**2)-np.kron(Phi,Phi)),
                             var_u_tilde_0.flatten()).reshape((P,P), order='F')
    
    # iteratively compute variance of process with SV (t>0)
    var_y = np.zeros(T)
    for t in range(T):
        var_u_tilde_t = np.zeros((P,P))
        var_u_tilde_t[0,0] = eh2[t]*s2
        var_y_tilde[t+1,:,:] = (np.dot(np.dot(Phi, var_y_tilde[t,:,:]), 
                                       Phi.transpose())+var_u_tilde_t)
        var_y[t] = var_y_tilde[t+1,0,0]
    
    # return variance process
    return var_y


## Conditional Distribution Samplers ------------------------------------------

# Sample a_i (intercepts) and s2_i (non time-varying variance components)
def a_s2_i_sampler(y_i, f_w, f_k, b_w_i, b_k_i, phi_i, eh_i, s2_i, a_bar_i, 
                   A_bar_i, nu_bar_i, delta2_bar_i,T, p, RNG):
    '''
    Function for sampling intercepts a_i and non time-varying variance components
    s^2_i. See Del Negro & Otrok (2008) pp. 29-30. The Inverse-Gamma prior on s^2_i
    is parametrized as a Scaled-Inverse-Chi^2 distribution.

    y_i          : vector of observations on series i (Tx1)
    f_w          : vector of world factor values (Tx1)
    f_k          : vector of group factor values (Tx1)
    b_w_i        : vector of world loadings of series i (Tx1)
    b_k_i        : vector of group loadings of series i (Tx1)
    phi_i        : vector of AR(p) paremeters in error process of series i (px1)
    eh_i         : vector of SV part of error process of series i (Tx1)
    s2_i         : non time-varying component of innovation variance of series i
    a_bar_i      : mean of Normal prior on a_i
    A_bar_i      : precision of Normal prior on a_i
    nu_bar_i     : degrees of freedom of Inverse-Gamma prior on s^2_i
    delta2_bar_i : scale parameter of Inverse-Gamma prior on s^2_i
    T            : number of time periods
    p            : order of AR(p) error process
    RNG          : numpy random number generator
    '''
    # comupte S_i
    S_i = S_i_comp(phi_i=phi_i, eh_i=eh_i, p=p)

    # compute corresponding inverted Cholesky factor
    Q_i_inv = np.linalg.pinv(np.linalg.cholesky(S_i))

    # define y_i1_tilde
    y_i1_tilde = np.flip(y_i[:p] - b_w_i[:p]*f_w[:p] - b_k_i[:p]*f_k[:p])

    # compute y_i1_tilde_star and x_i1_tilde_star
    y_i1_tilde_star = np.dot(Q_i_inv, y_i1_tilde)
    x_i1_tilde_star = np.dot(Q_i_inv, np.ones(p))

    # define y_i2_tilde_star
    y_i2_tilde_star = np.zeros(T-p)
    y_i_esc = (y_i - b_w_i*f_w - b_k_i*f_k)/eh_i
    for t in range(T-p):
        y_i2_tilde_star[t] = y_i_esc[t+p] - np.sum(np.flip(phi_i)*y_i_esc[t:t+p])

    # define x_i2_tilde_star
    x_i2_tilde_star = (1 - np.sum(phi_i))/eh_i[p:]

    # define y_i_tilde_star and x_i_tilde_star
    y_i_tilde_star = np.hstack((y_i1_tilde_star, y_i2_tilde_star))
    x_i_tilde_star = np.hstack((x_i1_tilde_star, x_i2_tilde_star))

    # compute A_i_inv (variance of normal cond. posterior of a_i)
    A_i_inv = 1/(A_bar_i + (1/s2_i)*np.dot(x_i_tilde_star.transpose(),x_i_tilde_star))

    # compute mean of normal cond. posterior of a_i
    mean_a_i = A_i_inv * (A_bar_i*a_bar_i + (1/s2_i)*np.dot(x_i_tilde_star.transpose(),y_i_tilde_star))

    # sample new a_i
    new_a_i = RNG.normal(mean_a_i, np.sqrt(A_i_inv))

    # compute parameters of IG cond. posterior of s^2_i
    e_i = y_i_tilde_star - new_a_i*x_i_tilde_star
    nu = nu_bar_i + T
    delta2 = (nu_bar_i*delta2_bar_i + np.dot(e_i.transpose(), e_i))/nu

    # sample new s2_i
    new_s2_i = (delta2*nu)/RNG.chisquare(nu)

    # return new a_i and new s2_i
    return new_a_i, new_s2_i

# Sample phi_i (AR coefficients in idiosyncratic processes) 
def phi_i_sampler(y_i, f_w, f_k, a_i, b_w_i, b_k_i, phi_i, eh_i, s2_i, 
                phi_bar_i, V_bar_i, T, p, RNG):
    '''
    Function for sampling phi_i, the vector of AR(p) paremeters in the
    error process of series i. See Del Negro & Otrok (2008) pp. 30-31.

    y_i       : vector of observations on series i (Tx1)
    f_w       : vector of world factor values (Tx1)
    f_k       : vector of group factor values (Tx1)
    a_i       : intercept of series i
    b_w_i     : vector of world loadings of series i (Tx1)
    b_k_i     : vector of group loadings of series i (Tx1)
    phi_i     : vector of AR(p) paremeters in error process of series i 
                (previous draw)
    eh_i      : vector of SV part of error process of series i (Tx1)
    s2_i      : non time-varying component of innovation variance of series i
    phi_bar_i : mean vector of Normal prior on phi_i (px1)
    V_bar_i   : precision matrix of Normal prior on phi_i (pxp)
    T         : number of time periods
    p         : order of AR(p) error process
    RNG       : numpy random number generator
    '''
    # compute e_i_t
    e_i_t = y_i - a_i - b_w_i*f_w - b_k_i*f_k

    # compute e_i
    e_i = e_i_t[p:] / eh_i[p:]

    # compute E_i
    E_i = np.zeros((T-p,p))
    for j in range(p):
        E_i[:,j] = e_i_t[p-(j+1):-(j+1)] / eh_i[p:]

    # compute variance of normal part of cond. posterior of phi_i
    V_i_inv = np.linalg.pinv(V_bar_i + (1/s2_i)*np.dot(E_i.transpose(),E_i))

    # compute mean of normal part of cond. posterior of phi_i
    mean_phi_i = np.dot(V_i_inv, (np.dot(V_bar_i,phi_bar_i) + 
                                  (1/s2_i)*np.dot(E_i.transpose(),e_i)))

    # set indicator for stationarity and counter for tries
    stationary = 0
    iter = 0

    # draw phi_i candidate until stationary or max. iter reached (50)
    while not stationary:

        # draw phi_i candidate
        phi_i_new = mean_phi_i + np.dot(np.linalg.cholesky(V_i_inv), 
                                        RNG.normal(size=p))

        # get roots of coefficients of corresponding lag polynomial
        roots = np.roots(np.append(-np.flip(phi_i_new), 1))

        # check for stationarity
        if all(i >= 1.001 for i in abs(roots)):
            # consider candidate if condition met
            stationary = 1
        else:
            iter = iter + 1
            if iter > 50:
                # take previous value if stationarity not met after 50 tries
                phi_i_new = phi_i
                stationary = 1

    # if no new draw was generated return previous phi_i
    if all(phi_i_new == phi_i):
        return phi_i

    # if new draw was generated decide if to keep it
    else:

        # compute previous and new Psi_i
        Psi_i_old = Psi_comp(y_i=y_i, f_w=f_w, f_k=f_k, a_i=a_i, b_w_i=b_w_i, 
                             b_k_i=b_k_i, phi_i=phi_i, eh_i=eh_i, s2_i=s2_i, 
                             p=p)
        Psi_i_new = Psi_comp(y_i=y_i, f_w=f_w, f_k=f_k, a_i=a_i, b_w_i=b_w_i, 
                             b_k_i=b_k_i, phi_i=phi_i_new, eh_i=eh_i, s2_i=s2_i, 
                             p=p)

        # accept or reject new draw
        if Psi_i_old == 0:
            accept = 1
        else:
            accept = RNG.uniform() < min(Psi_i_new/Psi_i_old, 1)
    
        # return new or old draw of phi_i
        return phi_i_new * accept + phi_i * (1-accept)

# Sample phi_0 (AR coefficients in factor processes) 
def phi_0_sampler(f, phi_0, eh_0, s2_0, phi_bar_0, V_bar_0, T, q, RNG):
    '''
    Function for sampling phi_0, the vector of AR(q) paremeters for the
    factor process. See Del Negro & Otrok (2008) pp. 30-31.

    f         : vector of factor values (Tx1)
    phi_0     : vector of AR(q) paremeters in factor process (qx1) (previous draw)
    eh_0      : vector of SV part of factor process (Tx1)
    s2_0      : non time-varying component of factor innovation variance
    phi_bar_0 : mean vector of Normal prior on phi_0 (qx1)
    V_bar_0   : precision matrix of Normal prior on phi_0 (qxq)
    T         : number of time periods
    q         : order of AR(q) factor process
    RNG       : numpy random number generator
    '''
    # compute e_0
    e_0 = f[q:] / eh_0[q:]

    # compute E_0
    E_0 = np.zeros((T-q,q))
    for j in range(q):
        E_0[:,j] = f[q-(j+1):-(j+1)] / eh_0[q:]
    
    # compute variance of normal part of cond. posterior of phi_0
    V_0_inv = np.linalg.pinv(V_bar_0 + (1/s2_0)*np.dot(E_0.transpose(),E_0))

    # compute mean of normal part of cond. posterior of phi_0
    mean_phi_0 = np.dot(V_0_inv, (np.dot(V_bar_0,phi_bar_0) + 
                                  (1/s2_0)*np.dot(E_0.transpose(),e_0)))

    # set indicator for stationarity and counter for tries
    stationary = 0
    iter = 0

    # draw phi_0 candidate until stationary or max. iter reached (50)
    while not stationary:

        # draw phi_i candidate
        phi_0_new = mean_phi_0 + np.dot(np.linalg.cholesky(V_0_inv), 
                                        RNG.normal(size=q))

        # get roots of coefficients of corresponding lag polynomial
        roots = np.roots(np.append(-np.flip(phi_0_new), 1))

        # check for stationarity
        if all(i >= 1.001 for i in abs(roots)):
            # consider candidate if condition met
            stationary = 1
        else:
            iter = iter + 1
            if iter > 50:
                # take previous value if stationarity not met after 50 tries
                phi_0_new = phi_0
                stationary = 1

    # if no new draw was generated return previous phi_0
    if all(phi_0_new == phi_0):
        return phi_0
    
    # if new draw was generated decide if to keep it
    else:

        # compute previous and new Psi_0
        Psi_0_old = Psi_0_comp(f=f, phi_0=phi_0, s2_0=s2_0, q=q)
        Psi_0_new = Psi_0_comp(f=f, phi_0=phi_0_new, s2_0=s2_0, q=q)

        # accept or reject new draw
        if Psi_0_old == 0:
            accept = 1
        else:
            accept = RNG.uniform() < min(Psi_0_new/Psi_0_old, 1)
    
        # return new or old draw of phi_0
        return phi_0_new * accept + phi_0 * (1-accept)

# Sample s2_eta_i (innovation variance of loading processes)
def s2_eta_i_sampler(b_i, nu_eta_bar_i, delta2_eta_bar_i, T, RNG):
    '''
    Function for sampling s^2_eta_i, the variance of the innovations
    to the law of motions of the loadings. See Del Negro & Otrok (2008) p.11.
    The Inverse-Gamma prior on s^2_eta_i is parametrized as a 
    Scaled-Inverse-Chi^2 distribution.

    b_i               : vector of loadings of series i (Tx1)
    nu_eta_bar_i      : degrees of freedom of Inverse-Gamma prior on s^2_eta_i
    delta2_eta_bar_i  : scale parameter of Inverse-Gamma prior on s^2_eta_i
    T                 : number of time periods
    RNG               : numpy random number generator
    '''
    # compute posterior parameters
    nu_eta = nu_eta_bar_i + T
    delta2_eta = (nu_eta_bar_i*delta2_eta_bar_i + 
                  np.sum((b_i[1:]-b_i[:-1])**2))/nu_eta

    # draw and return new s2_eta_i
    return (delta2_eta*nu_eta)/RNG.chisquare(nu_eta)

# Sample s2_zeta_i and s2_zeta_0 (innovation variance of SV processes)
def s2_zeta_sampler(h_i, nu_zeta_bar, delta2_zeta_bar, T, RNG):
    '''
    Function for sampling s^2_zeta_i and s^2_zeta_0, the variance of the 
    innovations to the law of motions of the stochastic volatilities. 
    See Del Negro & Otrok (2008) pp.11-12. The Inverse-Gamma priors are 
    parametrized as Scaled-Inverse-Chi^2 distributions.

    h_i              : vector of SV part of error process of series i or factor
    nu_zeta_bar      : degrees of freedom of Inverse-Gamma prior on s^2_zeta_i 
                       (and on s^2_zeta_0)
    delta2_zeta_bar  : scale parameter of Inverse-Gamma prior on s^2_eta_i
                       (and on s^2_zeta_0)
    T                : number of time periods
    RNG              : numpy random number generator
    '''
    # compute posterior parameters
    nu_zeta = nu_zeta_bar + T
    delta2_zeta = (nu_zeta_bar*delta2_zeta_bar + 
                   (h_i[0]**2 + np.sum((h_i[1:]-h_i[:-1])**2)))/nu_zeta

    # draw and return new s2_zeta
    return (delta2_zeta*nu_zeta)/RNG.chisquare(nu_zeta)

# Sample f_w (world factor)
def f_w_sampler(y, f_K, select_k, a, b_w, b_k, phi_w, phi, eh_w, eh, 
                s2_w, s2, T, n, p, q, RNG):
    '''
    Function for sampling world factor using Carter & Kohn Algorithm. 
    See Del Negro & Otrok (2008) pp.12-13. Assumes that q-1 = p, where
    q is order of AR factor process and p is order of AR process of
    idiosyncratic components.

    y         : matrix of observed series (Txn)
    f_K       : matrix of group factors (TxK)
    select_k  : list of group indicators (1xn)
    a         : vector of intercepts (nx1)
    b_w       : matrix of world loadings (Txn)
    b_k       : matrix of group loadings (Txn)
    phi_w     : vector of AR(q) paremeters in world factor process (qx1)
    phi       : matrix of AR(p) parameters in error processes (pxn)
    eh_w      : vector of SV part of world factor process (Tx1)
    eh        : matrix of SV part of error processes (Txn)
    s2_w      : non time-varying component of world factor innovation variance
    s2        : vector of non time-varying components of innovation variances (nx1)
    T         : number of time periods
    n         : number of observed series
    p         : order of AR(p) error process
    q         : order of AR(q) factor process
    RNG       : numpy random number generator
    '''
    # create companion matrix 
    Phi_0 = np.zeros((q,q))
    Phi_0[0] = phi_w
    if q>1:
        Phi_0[1:,:q-1] = np.identity(q-1)
    
    # compute y_tilde_star
    y_x = np.zeros((T,n))
    for i in range(n):
        y_x[:,i] = y[:,i] - b_k[:,i]*f_K[:,select_k[i]]
    y_tilde_star = np.zeros((n, T-p))
    for t in range(p,T):
        for i in range(n):
            y_tilde_star[i,t-p] = y_x[t,i] - np.sum(np.flip(phi[:,i]) * 
                                                    y_x[t-p:t,i])
    
    # compute a_tilde_star
    a_tilde_star = np.zeros(n)
    for i in range(n):
        a_tilde_star[i] = a[i] - np.sum(phi[:,i] * a[i])
    
    ## START of copmuting initial conditions for Kalman Filter ---
        
    # unconditional mean and variance of f_tilde_t
    Q_0 = np.zeros((q,q))
    Q_0[0,0] = s2_w
    f_tilde_00 = np.zeros(q)
    s_tilde_00 = np.dot(np.linalg.pinv(np.identity(q**2)-np.kron(Phi_0, Phi_0)), 
                        Q_0.flatten()).reshape((q,q), order='F')
    
    # I_y
    I_y = np.zeros((p,n,n))
    for j in range(p):
        I_y[j,:,:] = np.identity(n)
    I_y = np.vstack(I_y)
    
    # B_bar_t for t = 1,...,p
    B_bar = np.zeros((p,n,q))
    for j in range(p):
        B_bar[j,:,0] = b_w[j,:]
    
    # B_y
    B_y = np.zeros((p,n,q))
    for j in range(p):
        B_y[j,:,:] = np.dot(B_bar[-(j+1),:,:], np.linalg.matrix_power(Phi_0, p-j))
    B_y = np.vstack(B_y)
    
    # U_y
    U_y = np.zeros((p,p,n,q))
    for i in range(p):
        for j in range(p-i):
            U_y[j,j+i,:,:] = np.dot(B_bar[-(j+1),:,:], 
                                    np.linalg.matrix_power(Phi_0, i))
    U_y = U_y.swapaxes(1,2).reshape((p*n,p*q))
    
    # U_f
    U_f = np.zeros((p,q,q))
    for j in range(p):
        U_f[j,:,:] = np.linalg.matrix_power(Phi_0, j)
    U_f = np.hstack(U_f)

    # s2_i*S_i for all i = 1,...,n
    S = np.zeros((n,p,p))
    for i in range(n):
        S[i,:,:] = s2[i] * S_i_comp(phi_i=phi[:,i], eh_i=eh[:,i], p=p)
    
    # Sigma_e_p1
    Sigma_e_p1 = np.zeros((p,p,n,n))
    for k in range(p):
        for l in range(p):
            Sigma_e_p1[k,l,:,:] = np.diag(S[:,k,l])
    Sigma_e_p1 = Sigma_e_p1.swapaxes(1,2).reshape((p*n,p*n))
    
    # Sigma_0
    Sigma_0 = np.zeros((p*q, p*q))
    for j in range(p):
        Sigma_0[j*q,j*q] = s2_w * (eh_w[p-(j+1)]**2)
    
    # y_tilde_p1
    y_tilde_p1 = np.zeros((p,n))
    for j in range(p):
        y_tilde_p1[j,:] = y_x[p-(j+1),:]
    y_tilde_p1 = y_tilde_p1.flatten()  

    # conditional mean and variance of f_tilde_p (computed in parts)
    Phi_0_p = np.linalg.matrix_power(Phi_0, p)
    fs1 = (np.dot(np.dot(Phi_0_p, s_tilde_00), B_y.transpose()) + 
           np.dot(np.dot(U_f, Sigma_0), U_y.transpose()))
    fs2 = (np.dot(np.dot(B_y, s_tilde_00), B_y.transpose()) + 
           np.dot(np.dot(U_y, Sigma_0), U_y.transpose()))
    fs3 = np.dot(fs1, np.linalg.pinv(fs2 + Sigma_e_p1))
    f1 =  np.dot(Phi_0_p, f_tilde_00)
    f2 = y_tilde_p1 - np.dot(I_y, a) - np.dot(B_y, f_tilde_00)
    ss1 = np.dot(np.dot(Phi_0_p, s_tilde_00), Phi_0_p.transpose())
    ss2 = np.dot(np.dot(U_f, Sigma_0), U_f.transpose())
    f_tilde_pp = f1 + np.dot(fs3, f2)                       
    s_tilde_pp = ss1 + ss2 - np.dot(fs3, fs1.transpose())   
    
    ## END of computing initial conditions for Kalman Filter ---

    # empty arrays to store results of Kalman Filter
    f_tilde_tt_store = np.zeros((T-p+1, q))
    s_tilde_tt_store = np.zeros((T-p+1, q, q))

    # assign initial conditions to storage
    f_tilde_tt_store[0,:] = f_tilde_pp
    s_tilde_tt_store[0,:,:] = s_tilde_pp
    
    ## START of Kalman Filter ---

    # start loop from t=p+1 to T (note: Python indexes from 0)
    for t in range(p, T):

        # compute variances of innovations
        Q_t = np.zeros((q,q))
        Q_t[0,0] = s2_w * (eh_w[t]**2)
        R_t = np.diag(s2 * (eh[t,:]**2))

        # compute B_star_t
        B_star_t = np.zeros((n, p+1))
        B_star_t[:,0] = b_w[t,:]
        for j in range(1,p+1):
            B_star_t[:,j] = -b_w[t-j,:] * phi[j-1,:]

        # forecast f_tilde_t
        f_tilde_tt1 = np.dot(Phi_0, f_tilde_tt_store[t-p,:])
        s_tilde_tt1 = np.dot(np.dot(Phi_0, s_tilde_tt_store[t-p,:,:]), 
                             Phi_0.transpose()) + Q_t

        # forecast y_tilde_star_t
        y_tilde_star_tt1 = a_tilde_star + np.dot(B_star_t,f_tilde_tt1)
        g_tt1 = np.dot(np.dot(B_star_t,s_tilde_tt1), B_star_t.transpose()) + R_t

        # update forecast of f_tilde_t
        K_t = np.dot(np.dot(s_tilde_tt1,B_star_t.transpose()), 
                     np.linalg.pinv(g_tt1))
        f_tilde_tt = f_tilde_tt1 + np.dot(K_t, (y_tilde_star[:,t-p] - 
                                                y_tilde_star_tt1))
        s_tilde_tt = s_tilde_tt1 - np.dot(np.dot(K_t,B_star_t), s_tilde_tt1)

        # store results
        f_tilde_tt_store[t-p+1,:] = f_tilde_tt
        s_tilde_tt_store[t-p+1,:,:] = s_tilde_tt

    ## END of Kalman Filter ---
    
    # draw and store f_T from N(f_tilde_TT(1), s_tilde_TT(1,1))
    f_store = np.zeros(T)
    if s_tilde_tt_store[-1,0,0]>0:
        f_store[-1] = (f_tilde_tt_store[-1,0] + 
                       np.sqrt(s_tilde_tt_store[-1,0,0]) * RNG.normal())
    else:
        f_store[-1] = f_tilde_tt_store[-1,0] + 1e-6 * RNG.normal()

    # move backwards in time and draw remaining f_t
    for t in range(2, T-p+1):
        
        # get mean and variance of N(f_tilde_ttpl1, s_tilde_ttpl1)
        g_star_t = np.dot(np.dot(Phi_0[0,:], s_tilde_tt_store[-t,:,:]),
                          Phi_0[0,:].transpose()) + s2_w * (eh_w[-t+1]**2)
        K_star_t = np.dot(s_tilde_tt_store[-t,:,:], Phi_0[0,:].transpose())/g_star_t
        f_tilde_ttpl1 = f_tilde_tt_store[-t,:] + np.dot(K_star_t, (f_store[-t+1] -
                                                                   np.dot(Phi_0[0,:], f_tilde_tt_store[-t,:])))
        s_tilde_ttpl1 = s_tilde_tt_store[-t,:,:] - np.dot(np.outer(K_star_t,Phi_0[0,:]),
                                                          s_tilde_tt_store[-t,:,:])

        # draw and store f_t
        if s_tilde_ttpl1[0,0]>0:
            f_store[-t] = f_tilde_ttpl1[0] + np.sqrt(s_tilde_ttpl1[0,0]) * RNG.normal()
        else:
            f_store[-t] = f_tilde_ttpl1[0] + 1e-6 * RNG.normal()

        # draw first q in one go (f_1,...,f_q)
        if t==(T-p):
            try:
                f_store[:q] = np.flip(f_tilde_ttpl1 + np.dot(np.linalg.cholesky(s_tilde_ttpl1), 
                                                             RNG.normal(size=q)))
            except:
                shift = np.identity(q)*(-min(np.linalg.eigvals(s_tilde_ttpl1))+1e-6)
                f_store[:q] = np.flip(f_tilde_ttpl1 + np.dot(np.linalg.cholesky(s_tilde_ttpl1+shift),
                                                             RNG.normal(size=q)))

    # return new f
    return f_store

# Sample f_k (group factor)
def f_k_sampler(y, f_w, a, b_w, b_k, phi_k, phi, eh_k, eh, s2_k, s2, 
                T, p, q, RNG):
    '''
    Function for sampling group k factor using Carter & Kohn Algorithm. 
    See Del Negro & Otrok (2008) pp.12-13. Assumes that q-1 = p, where
    q is order of AR factor process and p is order of AR process of
    idiosyncratic components.

    y         : matrix of observed series that belong to group k (Txn_k)
    f_w       : vector of world factor (Tx1)
    a         : vector of intercepts (n_kx1)
    b_w       : matrix of world loadings (Txn_k)
    b_k       : matrix of group loadings (Txn_k)
    phi_k     : vector of AR(q) paremeters in group k factor process (qx1)
    phi       : matrix of AR(p) parameters in error processes (pxn_k)
    eh_k      : vector of SV part of group k factor process (Tx1)
    eh        : matrix of SV part of error processes (Txn_k)
    s2_k      : non time-varying component of group k factor innovation variance
    s2        : vector of non time-varying components of innovation variances (n_kx1)
    T         : number of time periods
    p         : order of AR(p) error process
    q         : order of AR(q) factor process
    RNG       : numpy random number generator
    '''
    # get number of observed series in group k
    n_k = y.shape[1]

    # create companion matrix 
    Phi_0 = np.zeros((q,q))
    Phi_0[0] = phi_k
    if q>1:
        Phi_0[1:,:q-1] = np.identity(q-1)
    
    # compute y_tilde_star
    y_x = np.zeros((T,n_k))
    for i in range(n_k):
        y_x[:,i] = y[:,i] - b_w[:,i]*f_w
    y_tilde_star = np.zeros((n_k, T-p))
    for t in range(p,T):
        for i in range(n_k):
            y_tilde_star[i,t-p] = y_x[t,i] - np.sum(np.flip(phi[:,i]) * 
                                                    y_x[t-p:t,i])
    
    # compute a_tilde_star
    a_tilde_star = np.zeros(n_k)
    for i in range(n_k):
        a_tilde_star[i] = a[i] - np.sum(phi[:,i] * a[i])
    
    ## START of computing initial conditions for Kalman Filter ---
        
    # unconditional mean and variance of f_tilde_t
    Q_0 = np.zeros((q,q))
    Q_0[0,0] = s2_k
    f_tilde_00 = np.zeros(q)
    s_tilde_00 = np.dot(np.linalg.pinv(np.identity(q**2)-np.kron(Phi_0, Phi_0)), 
                        Q_0.flatten()).reshape((q,q), order='F')
    
    # I_y
    I_y = np.zeros((p,n_k,n_k))
    for j in range(p):
        I_y[j,:,:] = np.identity(n_k)
    I_y = np.vstack(I_y)
    
    # B_bar_t for t = 1,...,p
    B_bar = np.zeros((p,n_k,q))
    for j in range(p):
        B_bar[j,:,0] = b_k[j,:]
    
    # B_y
    B_y = np.zeros((p,n_k,q))
    for j in range(p):
        B_y[j,:,:] = np.dot(B_bar[-(j+1),:,:], 
                            np.linalg.matrix_power(Phi_0, p-j))
    B_y = np.vstack(B_y)
    
    # U_y
    U_y = np.zeros((p,p,n_k,q))
    for i in range(p):
        for j in range(p-i):
            U_y[j,j+i,:,:] = np.dot(B_bar[-(j+1),:,:], 
                                    np.linalg.matrix_power(Phi_0, i))
    U_y = U_y.swapaxes(1,2).reshape((p*n_k,p*q))
    
    # U_f
    U_f = np.zeros((p,q,q))
    for j in range(p):
        U_f[j,:,:] = np.linalg.matrix_power(Phi_0, j)
    U_f = np.hstack(U_f)

    # s2_i*S_i for all i in group k
    S = np.zeros((n_k,p,p))
    for i in range(n_k):
        S[i,:,:] = s2[i] * S_i_comp(phi_i=phi[:,i], eh_i=eh[:,i], p=p)
    
    # Sigma_e_p1
    Sigma_e_p1 = np.zeros((p,p,n_k,n_k))
    for k in range(p):
        for l in range(p):
            Sigma_e_p1[k,l,:,:] = np.diag(S[:,k,l])
    Sigma_e_p1 = Sigma_e_p1.swapaxes(1,2).reshape((p*n_k,p*n_k))
    
    # Sigma_0
    Sigma_0 = np.zeros((p*q, p*q))
    for j in range(p):
        Sigma_0[j*q,j*q] = s2_k * (eh_k[p-(j+1)]**2)
    
    # y_tilde_p1
    y_tilde_p1 = np.zeros((p,n_k))
    for j in range(p):
        y_tilde_p1[j,:] = y_x[p-(j+1),:]
    y_tilde_p1 = y_tilde_p1.flatten()  

    # conditional mean and variance of f_tilde_p (computed in parts)
    Phi_0_p = np.linalg.matrix_power(Phi_0, p)
    fs1 = (np.dot(np.dot(Phi_0_p, s_tilde_00), B_y.transpose()) + 
           np.dot(np.dot(U_f, Sigma_0), U_y.transpose()))
    fs2 = (np.dot(np.dot(B_y, s_tilde_00), B_y.transpose()) + 
           np.dot(np.dot(U_y, Sigma_0), U_y.transpose()))
    fs3 = np.dot(fs1, np.linalg.pinv(fs2 + Sigma_e_p1))
    f1 =  np.dot(Phi_0_p, f_tilde_00)
    f2 = y_tilde_p1 - np.dot(I_y, a) - np.dot(B_y, f_tilde_00)
    ss1 = np.dot(np.dot(Phi_0_p, s_tilde_00), Phi_0_p.transpose())
    ss2 = np.dot(np.dot(U_f, Sigma_0), U_f.transpose())
    f_tilde_pp = f1 + np.dot(fs3, f2)                       
    s_tilde_pp = ss1 + ss2 - np.dot(fs3, fs1.transpose())  

    ## END of computing initial conditions for Kalman Filter ---

    # empty arrays to store results of Kalman Filter
    f_tilde_tt_store = np.zeros((T-p+1, q))
    s_tilde_tt_store = np.zeros((T-p+1, q, q))

    # assign initial conditions to storage
    f_tilde_tt_store[0,:] = f_tilde_pp
    s_tilde_tt_store[0,:,:] = s_tilde_pp
    
    ## START of Kalman Filter ---

    # start loop from t=p+1 to T (note: Python indexes from 0)
    for t in range(p, T):

        # compute variances of innovations
        Q_t = np.zeros((q,q))
        Q_t[0,0] = s2_k * (eh_k[t]**2)
        R_t = np.diag(s2 * (eh[t,:]**2))

        # compute B_star_t
        B_star_t = np.zeros((n_k, p+1))
        B_star_t[:,0] = b_k[t,:]
        for j in range(1,p+1):
            B_star_t[:,j] = -b_k[t-j,:] * phi[j-1,:]

        # forecast f_tilde_t
        f_tilde_tt1 = np.dot(Phi_0, f_tilde_tt_store[t-p,:])
        s_tilde_tt1 = np.dot(np.dot(Phi_0, s_tilde_tt_store[t-p,:,:]), 
                             Phi_0.transpose()) + Q_t

        # forecast y_tilde_star_t
        y_tilde_star_tt1 = a_tilde_star + np.dot(B_star_t,f_tilde_tt1)
        g_tt1 = np.dot(np.dot(B_star_t,s_tilde_tt1), B_star_t.transpose()) + R_t

        # update forecast of f_tilde_t
        K_t = np.dot(np.dot(s_tilde_tt1,B_star_t.transpose()), np.linalg.pinv(g_tt1))
        f_tilde_tt = f_tilde_tt1 + np.dot(K_t, (y_tilde_star[:,t-p] - 
                                                y_tilde_star_tt1))
        s_tilde_tt = s_tilde_tt1 - np.dot(np.dot(K_t,B_star_t), s_tilde_tt1)

        # store results
        f_tilde_tt_store[t-p+1,:] = f_tilde_tt
        s_tilde_tt_store[t-p+1,:,:] = s_tilde_tt

    ## END of Kalman Filter ---
    
    # draw and store f_T from N(f_tilde_TT(1), s_tilde_TT(1,1))
    f_store = np.zeros(T)
    if s_tilde_tt_store[-1,0,0]>0:
        f_store[-1] = (f_tilde_tt_store[-1,0] + 
                       np.sqrt(s_tilde_tt_store[-1,0,0]) * RNG.normal())
    else:
        f_store[-1] = f_tilde_tt_store[-1,0] + 1e-6 * RNG.normal()

    # move backwards in time and draw remaining f_t
    for t in range(2, T-p+1):
        
        # get mean and variance of N(f_tilde_ttpl1, s_tilde_ttpl1)
        g_star_t = np.dot(np.dot(Phi_0[0,:], s_tilde_tt_store[-t,:,:]),
                          Phi_0[0,:].transpose()) + s2_k * (eh_k[-t+1]**2)
        K_star_t = np.dot(s_tilde_tt_store[-t,:,:], Phi_0[0,:].transpose())/g_star_t
        f_tilde_ttpl1 = f_tilde_tt_store[-t,:] + np.dot(K_star_t, (f_store[-t+1] -
                                                                   np.dot(Phi_0[0,:], f_tilde_tt_store[-t,:])))
        s_tilde_ttpl1 = s_tilde_tt_store[-t,:,:] - np.dot(np.outer(K_star_t,Phi_0[0,:]),
                                                          s_tilde_tt_store[-t,:,:])

        # draw and store f_t
        if s_tilde_ttpl1[0,0]>0:
            f_store[-t] = f_tilde_ttpl1[0] + np.sqrt(s_tilde_ttpl1[0,0]) * RNG.normal()
        else:
            f_store[-t] = f_tilde_ttpl1[0] + 1e-6 * RNG.normal()

        # draw first q in one go (f_1,...,f_q)
        if t==(T-p):
            try:
                f_store[:q] = np.flip(f_tilde_ttpl1 + np.dot(np.linalg.cholesky(s_tilde_ttpl1),
                                                         RNG.normal(size=q)))
            except:
                shift = np.identity(q)*(-min(np.linalg.eigvals(s_tilde_ttpl1))+1e-6)
                f_store[:q] = np.flip(f_tilde_ttpl1 + np.dot(np.linalg.cholesky(s_tilde_ttpl1+shift),
                                                         RNG.normal(size=q)))

    # return new f
    return f_store

# Sample b_w_i and b_k_i (world and group factor loadings)
def b_i_sampler(y_i, f, f_x, a_i, b_x_i, phi_i, eh_i, s2_i, s2_eta_i, 
              b_bar_i, B_bar_i, T, p, RNG):
    '''
    Function for sampling b_w_i or b_k_i, the random walk processes of world or
    group loadings of series i, using Carter & Kohn Algorithm. See Del Negro & 
    Otrok (2008) pp.13-14 and 33-34.

    y_i       : vector of observations on series i (Tx1)
    f         : vector of factor values (Tx1)
                (loadings on this factor are being estimated)
    f_x       : vector of factor values (other factor) (Tx1)
    a_i       : intercept of series i
    b_x_i     : vector of loadings on other factor, f_x (Tx1)
    phi_i     : vector of AR(p) paremeters in error process of series i (px1)
    eh_i      : vector of SV part of error process of series i (Tx1)
    s2_i      : non time-varying component of variance of error process of series i
    s2_eta_i  : variance of innovations to law of motions of loadings
    b_bar_i   : mean of Normal prior on b_i_0
    B_bar_i   : precision of Normal prior on b_i_0
    T         : number of time periods
    p         : order of AR(p) error process
    RNG       : numpy random number generator
    '''
    # compute y_star_i (t = p+1, ..., T)
    y_x_i = y_i - b_x_i*f_x
    y_star_i = np.zeros(T-p)
    for t in range(p,T):
        y_star_i[t-p] = y_x_i[t] - np.sum(np.flip(phi_i) * y_x_i[t-p:t])
    
    # compute a_star_i
    a_star_i = a_i - np.sum(phi_i * a_i)

    ## START of computing initial conditions for Kalman Filter ---

    # unconditional mean and variance of b_i_0 (from prior)
    b_bar_i_0 = b_bar_i
    s_bar_i_0 = 1/B_bar_i

    # needed parameters
    I_y = np.ones(p)
    B_y = np.flip(f[:p])
    U_y = np.zeros((p,p))
    for j in range(p):
        U_y[j,j:] = np.repeat(f[p-(j+1)], p-j)
    B_b = np.ones(p+1)
    U_b = np.zeros((p+1, p))
    for j in range(p):
        U_b[:j+1,j] = np.ones(j+1)

    # compute S_i
    S_i = S_i_comp(phi_i=phi_i, eh_i=eh_i, p=p)

    # conditional mean and variance of b_tilde_i_p (computed in parts)
    b1 = (np.outer(B_b*s_bar_i_0, B_y.transpose()) + 
          s2_eta_i * np.dot(U_b, U_y.transpose()))
    b2 = (np.outer(B_y*s_bar_i_0, B_y.transpose()) + 
          s2_eta_i * np.dot(U_y, U_y.transpose())) + s2_i*S_i
    b3 = np.flip(y_x_i[:p]) - I_y*a_i - B_y*b_bar_i_0
    b4 = np.dot(b1, np.linalg.pinv(b2))
    s1 = (np.outer(B_b*s_bar_i_0, B_b.transpose()) + 
          s2_eta_i * np.dot(U_b, U_b.transpose()))
    b_tilde_i_pp = B_b*b_bar_i_0 + np.dot(b4, b3) 
    s_tilde_i_pp = s1 - np.dot(b4, b1.transpose())

    ## END of computing initial conditions for Kalman Filter ---

    # create Xi matrix
    Xi = np.zeros((p+1,p+1))
    Xi[0,0] = 1
    Xi[1:,:p] = np.identity(p)

    # compute variance of eta_tilde_i
    Q = np.zeros((p+1,p+1))
    Q[0,0] = s2_eta_i

    # compute variance of sigma_it * u_it for t=1,...,T
    R = s2_i * (eh_i**2)

    # compute w_star_i_t for t=p+1,...,T
    w_star_i = np.zeros((T-p,p+1))
    w_star_i[:,0] = f[p:]
    for j in range(p):
        w_star_i[:,j+1] = -phi_i[j] * f[p-(j+1):-(j+1)]

    # empty arrays to store results of Kalman Filter
    b_tilde_i_tt_store = np.zeros((T-p+1, p+1))
    s_tilde_i_tt_store = np.zeros((T-p+1, p+1, p+1))

    # assign initial conditions to storage
    b_tilde_i_tt_store[0,:] = b_tilde_i_pp
    s_tilde_i_tt_store[0,:,:] = s_tilde_i_pp

    ## START of Kalman Filter ---

    # start loop from t=p+1 to T (note: Python indexes from 0)
    for t in range(p, T):
        
        # forecast b_tilde_i_t
        b_tilde_i_tt1 = np.dot(Xi, b_tilde_i_tt_store[t-p,:])
        s_tilde_i_tt1 = np.dot(np.dot(Xi, s_tilde_i_tt_store[t-p,:,:]), 
                               Xi.transpose()) + Q

        # forecast y_star_i_t
        y_star_i_tt1 = a_star_i + np.dot(w_star_i[t-p,:], b_tilde_i_tt1)
        g_i_tt1 = np.dot(np.dot(w_star_i[t-p,:], s_tilde_i_tt1), 
                         w_star_i[t-p,:].transpose()) + R[t]
        
        # update forecast of b_tilde_i_t
        K_i_t = np.dot(s_tilde_i_tt1, w_star_i[t-p,:].transpose()) * 1/g_i_tt1
        b_tilde_i_tt = b_tilde_i_tt1 + K_i_t * (y_star_i[t-p]-y_star_i_tt1)
        s_tilde_i_tt = s_tilde_i_tt1 - np.dot(np.outer(K_i_t, w_star_i[t-p,:]), 
                                              s_tilde_i_tt1)

        # store results
        b_tilde_i_tt_store[t-p+1,:] = b_tilde_i_tt
        s_tilde_i_tt_store[t-p+1,:,:] = s_tilde_i_tt

    ## END of Kalman Filter ---
    
    # draw and store b_i_T from N(b_tilde_i_TT(1), s_tilde_i_TT(1,1))
    b_i_store = np.zeros(T)
    if s_tilde_i_tt_store[-1,0,0]>0:
        b_i_store[-1] = (b_tilde_i_tt_store[-1,0] + 
                         np.sqrt(s_tilde_i_tt_store[-1,0,0]) * RNG.normal())
    else:
        b_i_store[-1] = b_tilde_i_tt_store[-1,0] + 1e-6 * RNG.normal()

    # move backwards in time and draw remaining b_tilde_i_t
    for t in range(2, T-p+1):
        
        # get mean and var of N(b_tilde_i_ttpl1, s_tilde_i_ttpl1)
        g_star_i_t = np.dot(np.dot(Xi[0,:], s_tilde_i_tt_store[-t,:,:]),
                            Xi[0,:].transpose()) + s2_eta_i
        K_star_i_t = np.dot(s_tilde_i_tt_store[-t,:,:], Xi[0,:].transpose())/g_star_i_t
        b_tilde_i_ttpl1 = b_tilde_i_tt_store[-t,:] + np.dot(K_star_i_t, (b_i_store[-t+1] -
                                                                         np.dot(Xi[0,:], b_tilde_i_tt_store[-t,:])))
        s_tilde_i_ttpl1 = s_tilde_i_tt_store[-t,:,:] - np.dot(np.outer(K_star_i_t, Xi[0,:]),
                                                              s_tilde_i_tt_store[-t,:,:])

        # draw and store b_i_t
        if s_tilde_i_ttpl1[0,0]>0:
            b_i_store[-t] = b_tilde_i_ttpl1[0] + np.sqrt(s_tilde_i_ttpl1[0,0]) * RNG.normal()
        else:
            b_i_store[-t] = b_tilde_i_ttpl1[0] + 1e-6 * RNG.normal()

        # draw first p+1 in one go (b_i_1,...,b_i_{p+1})
        if t==(T-p):
            try:
                b_i_store[:p+1] = np.flip(b_tilde_i_ttpl1 + np.dot(np.linalg.cholesky(s_tilde_i_ttpl1),
                                                                   RNG.normal(size=p+1)))
            except:
                shift = np.identity(p+1)*(-min(np.linalg.eigvals(s_tilde_i_ttpl1))+1e-6)
                b_i_store[:p+1] = np.flip(b_tilde_i_ttpl1 + np.dot(np.linalg.cholesky(s_tilde_i_ttpl1+shift),
                                                                   RNG.normal(size=p+1)))

    # return new b_i
    return b_i_store

# Sample h_i (SV process of series)
def h_i_sampler(y_i, f_w, f_k, a_i, b_w_i, b_k_i, phi_i, eh_i, s2_i, 
               s2_zeta_i, q_K, m_K, v2_K, T, p, RNG):
    '''
    Function for sampling h_i, the random walk process in the stochastic 
    volatility component of series i. See Del Negro & Otrok (2008) pp.14 
    and 34-35. Follows approach of Kim, Shephard and Chib (1998) with
    updated mixture parameters of Omori et al. (2007).

    y_i       : vector of observations on series i (Tx1)
    f_w       : vector of world factor values (Tx1)
    f_k       : vector of group factor values (Tx1)
    a_i       : intercept of series i
    b_w_i     : vector of world loadings of series i (Tx1)
    b_k_i     : vector of group loadings of series i (Tx1)
    phi_i     : vector of AR(p) paremeters in error process of series i (px1)
    eh_i      : vector of SV part of error process of series i (Tx1)
    s2_i      : non time-varying component of innovation variance of series i
    s2_zeta_i : variance of innovations to stochastic volatilities of series i
    q_K       : probability parameters of normal mixture distribution
    m_K       : mean parameters of normal mixture distribution
    v2_K      : variance parameters of normal mixture distribution
    T         : number of time periods
    p         : order of AR(p) error process
    RNG       : numpy random number generator
    '''
    # compute e_it for t = 1, ..., T
    e_i = y_i - a_i - b_w_i*f_w - b_k_i*f_k

    # draw e_it for t = 1-p, ..., 0
    u_i = RNG.normal(size=p)
    if p==1:
        if phi_i == 0:
            e_i = np.insert(e_i, 0, 0)
        else:
            e_ij = (e_i[0] - np.sqrt(s2_i)*eh_i[0]*u_i) / phi_i
            e_i = np.insert(e_i, 0, e_ij)
    else:
        for j in range(p):
            if phi_i[p-1] == 0:
                e_i = np.insert(e_i, 0, 0)
            else:
                e_ij = (e_i[p-1] - np.sum(np.flip(phi_i[:p-1])*e_i[:p-1]) - 
                        np.sqrt(s2_i)*eh_i[p-1-j]*u_i[j]) / phi_i[p-1]
                e_i = np.insert(e_i, 0, e_ij)
    
    # compute z_i_t for t = 1, ..., T
    z_i = np.zeros(T)
    for t in range(T):
        z_i[t] = e_i[t+p] - np.sum(np.flip(phi_i)*e_i[t:t+p])

    # compute z_star_i_t for t = 1, ..., T
    z_star_i = np.log(z_i**2 + 0.0001)

    # compute u_star_i_t for t = 1, ..., T
    u_star_i = z_star_i - 2*np.log(eh_i) - np.log(s2_i)

    # compute s_it for t = 1, ..., T
    s_i = np.zeros(T).astype(int)
    for t in range(T):
        perc_it = np.zeros(10)
        for k in range(10):
            perc_it[k] = q_K[k]/np.sqrt(v2_K[k]) * np.exp(-1/(2*v2_K[k])*
                                                          ((u_star_i[t]-
                                                            m_K[k]+1.2704)**2))
        perc_it2 = perc_it/np.sum(perc_it)
        s_i[t] = RNG.choice(range(10), 1, p=perc_it2)
    
    # empty arrays to store results of Kalman Filter (initial conditions zero)
    h_i_tt_store = np.zeros(T+1)
    s_i_tt_store = np.zeros(T+1)

    ## START of Kalman Filter ---
    for t in range(1,T+1):
        
        # forecast h_i_t
        h_i_tt1 = h_i_tt_store[t-1]
        s_i_tt1 = s_i_tt_store[t-1] + s2_zeta_i

        # forecast z_star_i_t
        z_star_i_tt1 = 2*h_i_tt1 + np.log(s2_i) + m_K[s_i[t-1]] - 1.2704
        g_i_tt1 = 4*s_i_tt1 + v2_K[s_i[t-1]]

        # update forecast of h_i_t
        K_i_t = s_i_tt1 * 2 * (1/g_i_tt1)
        h_i_tt_store[t] = h_i_tt1 + K_i_t * (z_star_i[t-1] - z_star_i_tt1)
        s_i_tt_store[t] = s_i_tt1 - K_i_t * 2 * s_i_tt1
    ## END of Kalman Filter ---
    
    # empty array to store final draws of h_i_t
    h_i = np.zeros(T)

    # draw and store h_i_T from N(h_i_TT, s_i_TT)
    h_i[-1] = RNG.normal(h_i_tt_store[-1], np.sqrt(s_i_tt_store[-1]))

    # move backwards in time and draw remaining h_i_t
    for t in range(2, T+1):

        # get mean and var of N(h_i_ttpl1, s_i_ttpl1)
        K_star_i_t = s_i_tt_store[-t] / (s_i_tt_store[-t] + s2_zeta_i)
        h_i_ttpl1 = h_i_tt_store[-t] + K_star_i_t * (h_i[-t+1]-h_i_tt_store[-t])
        s_i_ttpl1 = s_i_tt_store[-t] - K_star_i_t * s_i_tt_store[-t]

        # draw and store h_i_t
        h_i[-t] = RNG.normal(h_i_ttpl1, np.sqrt(s_i_ttpl1))

    # return h_i
    return h_i

# Sample h_0 (SV process of factors) 
def h_0_sampler(f, phi_0, eh_0, s2_0, s2_zeta_0, q_K, m_K, v2_K, T, 
                q, RNG):
    '''
    Function for sampling h_0, the random walk process in the stochastic 
    volatility component of a factor. See Del Negro & Otrok (2008) pp.14 
    and 34-35. Follows approach of Kim, Shephard and Chib (1998) with
    updated mixture parameters of Omori et al. (2007).

    f         : vector of factor values (Tx1)
    phi_0     : vector of AR(q) paremeters for factor process (qx1)
    eh_0      : vector of SV part of factor process (Tx1)
    s2_0      : non time-varying component of factor innovation variance
    s2_zeta_0 : variance of innovations to stochastic volatilities of factor
    q_K       : probability parameters of normal mixture distribution
    m_K       : mean parameters of normal mixture distribution
    v2_K      : variance parameters of normal mixture distribution
    T         : number of time periods
    q         : order of AR(q) factor process
    RNG       : numpy random number generator
    '''
    # draw f_t for t = 1-q, ..., 0
    u_i = RNG.normal(size=q)
    if q==1:
        if phi_0 == 0:
            f = np.insert(f, 0, 0)
        else:
            f_j = (f[0] - np.sqrt(s2_0)*eh_0[0]*u_i) / phi_0
            f = np.insert(f, 0, f_j)
    else:
        for j in range(q):
            if phi_0[q-1] == 0:
                f = np.insert(f, 0, 0)
            else:
                f_j = (f[q-1] - np.sum(np.flip(phi_0[:q-1])*f[:q-1]) - 
                       np.sqrt(s2_0)*eh_0[q-1-j]*u_i[j]) / phi_0[q-1]
                f = np.insert(f, 0, f_j)
    
    # compute z_0_t for t = 1, ..., T
    z_0 = np.zeros(T)
    for t in range(T):
        z_0[t] = f[t+q] - np.sum(np.flip(phi_0)*f[t:t+q])

    # compute z_star_0_t for t = 1, ..., T
    z_star_0 = np.log(z_0**2 + 0.0001)

    # compute u_star_0_t for t = 1, ..., T
    u_star_0 = z_star_0 - 2*np.log(eh_0) - np.log(s2_0)

    # compute s_0_t for t = 1, ..., T
    s_0 = np.zeros(T).astype(int)
    for t in range(T):
        perc_0t = np.zeros(10)
        for k in range(10):
            perc_0t[k] = q_K[k]/np.sqrt(v2_K[k]) * np.exp(-1/(2*v2_K[k])*
                                                          ((u_star_0[t]-
                                                            m_K[k]+1.2704)**2))
        perc_0t = perc_0t/np.sum(perc_0t)
        s_0[t] = RNG.choice(range(10), 1, p=perc_0t)
    
    # empty arrays to store results of Kalman Filter (initial conditions zero)
    h_0_tt_store = np.zeros(T+1)
    s_0_tt_store = np.zeros(T+1)

    ## START of Kalman Filter ---
    for t in range(1,T+1):
        
        # forecast h_0_t
        h_0_tt1 = h_0_tt_store[t-1]
        s_0_tt1 = s_0_tt_store[t-1] + s2_zeta_0

        # forecast z_star_0_t
        z_star_0_tt1 = 2*h_0_tt1 + np.log(s2_0) + m_K[s_0[t-1]] - 1.2704
        g_0_tt1 = 4*s_0_tt1 + v2_K[s_0[t-1]]

        # update forecast of h_0_t
        K_0_t = s_0_tt1 * 2 * (1/g_0_tt1)
        h_0_tt_store[t] = h_0_tt1 + K_0_t * (z_star_0[t-1] - z_star_0_tt1)
        s_0_tt_store[t] = s_0_tt1 - K_0_t * 2 * s_0_tt1
    ## END of Kalman Filter ---
    
    # empty array to store final draws of h_0_t
    h_0 = np.zeros(T)

    # draw and store h_0_T from N(h_0_TT, s_0_TT)
    h_0[-1] = RNG.normal(h_0_tt_store[-1], np.sqrt(s_0_tt_store[-1]))

    # move backwards in time and draw remaining h_0_t
    for t in range(2, T+1):

        # get mean and var of N(h_0_ttpl1, s_0_ttpl1)
        K_star_0_t = s_0_tt_store[-t] / (s_0_tt_store[-t] + s2_zeta_0)
        h_0_ttpl1 = h_0_tt_store[-t] + K_star_0_t * (h_0[-t+1]-h_0_tt_store[-t])
        s_0_ttpl1 = s_0_tt_store[-t] - K_star_0_t * s_0_tt_store[-t]

        # draw and store h_0_t
        h_0[-t] = RNG.normal(h_0_ttpl1, np.sqrt(s_0_ttpl1))

    # return h_0
    return h_0


## Gibbs Sampling Procedure ---------------------------------------------------

# Function to perform gibbs sampling
def gibbs(N_runs, y, K, select_k, s2_0, sign_w, sign_k, a_bar_i, A_bar_i, 
          b_bar_i, B_bar_i, phi_bar_i, V_bar_i, phi_bar_0, V_bar_0, nu_bar_i, 
          delta2_bar_i, nu_eta_bar_i, delta2_eta_bar_i, nu_zeta_bar, 
          delta2_zeta_bar, RNG, print_progress=1, print_time=1):
    '''
    Function to perform Gibbs Sampling for Dynamic Factor Model with 1 world
    factor, K group factors, time-varying loadings and stochastic volatility in 
    the factor and idiosyncratic processes. Factors and idiosyncratic errors 
    follow AR(q) and AR(p) processes, respectively. See Del Negro & Otrok (2008).
    Returns a dictionary containing the Gibbs draws of the factors and model 
    parameters, as well as the resulting variance decompositions.


    Gibbs Sampling Hyperparameters
    ------------------------------
    N_runs            : number of gibbs sampling iterations

    Model Specifications
    --------------------
    y                 : matrix of observed series (Txn)
    K                 : number of groups
    select_k          : array of group indicators (which series loads on which 
                        group, 0,1,2...) (1xn)

    Scale Normalization
    -------------------
    s2_0              : normalized world & group factor innovation variance 
                        (non time-varying component)

    Sign Restrictions
    -----------------
    sign_w            : array of sign indicators (which series loads positively 
                        on world factor, one entry is 1 and rest 0) (1xn)
    sign_k            : array of sign indicators (which series loads positively 
                        on group factor, one entry per group is 1, rest 0) (1xn)

    Prior Parameter Values
    ----------------------
    a_bar_i           : mean of Normal prior on a_i
    A_bar_i           : precision of Normal prior on a_i
    b_bar_i           : mean of Normal prior on b_i_0
    B_bar_i           : precision of Normal prior on b_i_0
    phi_bar_i         : mean vector of Normal prior on phi_i (px1)
    V_bar_i           : precision matrix of Normal prior on phi_i (pxp)
    phi_bar_0         : mean vector of Normal prior on phi_0 (qx1)
    V_bar_0           : precision matrix of Normal prior on phi_0 (qxq)
    nu_bar_i          : degrees of freedom of Inverse-Gamma prior on s^2_i
    delta2_bar_i      : scale parameter of Inverse-Gamma prior on s^2_i
    nu_eta_bar_i      : degrees of freedom of Inverse-Gamma prior on s^2_eta_i
    delta2_eta_bar_i  : scale parameter of Inverse-Gamma prior on s^2_eta_i
    nu_zeta_bar       : degrees of freedom of Inverse-Gamma prior on s^2_zeta_i
                        (and s^2_zeta_0)
    delta2_zeta_bar   : scale parameter of Inverse-Gamma prior on s^2_zeta_i
                        (and s^2_zeta_0)

    Settings
    --------
    RNG               : numpy random number generator
    print_progress    : print gibbs iteration progress if true (default=True)
    print_time        : print verification that completed and time needed if 
                        true (default=True)
    '''
    # get starting time
    start_time = datetime.now().replace(microsecond=0)

    # if y is dataframe get series names and turn into array
    if isinstance(y, pd.core.frame.DataFrame):
        df = 1
        names = list(y.columns)
        y = np.array(y)
    else:
        df = 0
    
    # change indicators to numpy arrays if they are lists
    if isinstance(select_k, list):
        select_k = np.array(select_k)
    if isinstance(sign_w, list):
        sign_w = np.array(sign_w)
    if isinstance(sign_k, list):
        sign_k = np.array(sign_k)

    # get number of observed series
    n = y.shape[1]

    # get number of time periods
    T = y.shape[0]

    # get order of AR(p) processes (series)
    p = len(phi_bar_i)

    # get order of AR(q) processes (factors)
    q = len(phi_bar_0)

    # create objects to store estimates
    a = np.zeros((N_runs+1,n))          # intercepts (series)
    b_w = np.zeros((N_runs+1,T,n))      # world loadings (series)
    b_k = np.zeros((N_runs+1,T,n))      # group loadings (series)
    s2 = np.zeros((N_runs+1,n))         # non time-varying components of variance (series)
    s2_eta_w = np.zeros((N_runs+1,n))   # variance of innovations to world loadings (series)
    s2_eta_k = np.zeros((N_runs+1,n))   # variance of innovations to group loadings (series)
    s2_zeta = np.zeros((N_runs+1,n))    # variance of innovations to stochastic volatilities (series)
    s2_zeta_w = np.zeros(N_runs+1)      # variance of innovations to stochastic volatilities (world factor)
    s2_zeta_K = np.zeros((N_runs+1,K))  # variance of innovations to stochastic volatilities (group factors)
    h = np.zeros((N_runs+1,T,n))        # stochastic volatility components (series)
    h_w = np.zeros((N_runs+1,T))        # stochastic volatility components (world factor)
    h_K = np.zeros((N_runs+1,T,K))      # stochastic volatility components (group factors)
    phi = np.zeros((N_runs+1,p,n))      # AR(p) parameters (series)
    phi_w = np.zeros((N_runs+1,q))      # AR(q) parameters (world factor)
    phi_K = np.zeros((N_runs+1,q,K))    # AR(q) parameters (group factors)
    f_w = np.zeros((N_runs+1,T))        # world factor
    f_K = np.zeros((N_runs+1,T,K))      # group factors
    vd_w = np.zeros((N_runs+1,T,n))     # variance decomposition w.r.t. world component
    vd_K = np.zeros((N_runs+1,T,n))     # variance decomposition w.r.t. group component
    vd_I = np.zeros((N_runs+1,T,n))     # variance decomposition w.r.t. idiosyncratic component

    # set initial values (zero for all h, s2_eta and s2_zeta)
    # (mean across series for factors)
    for i in range(n):
        s2[0,i] = 0.5
    a[0,:] = np.repeat(a_bar_i, n)
    for i in range(n):
        b_w[0,:,i] = np.repeat(b_bar_i, T)
        b_k[0,:,i] = np.repeat(b_bar_i, T)
        phi[0,:,i] = phi_bar_i
    phi_w[0,:] = phi_bar_0
    for k in range(K):
        phi_K[0,:,k] = phi_bar_0
        f_K[0,:,k] = np.mean(y[:,[x==k for x in select_k]],1)
    f_w[0,:] = np.mean(y,1)

    # define parameter values of normal mixture (needed in h samplers)
    # (given by Omori et al. (2007))
    q_K = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 
                    0.12047, 0.05591, 0.01575, 0.00115])
    m_K = np.array([3.19717, 2.61784, 2.00544, 1.29306, 0.41867, -0.70238, 
                    -2.19748, -4.28206, -7.41344, -13.37960])
    v2_K = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 
                     1.57469, 2.54498, 4.16591, 7.33342])

    # define dictionary with indicators by group (used when sampling f_k)
    # (to know which series belong to group k)
    k_ind = dict.fromkeys(range(K))
    for k in range(K):
        k_ind[k] = [x==k for x in select_k]
    
    # get index of series that loads positively on world factor 
    pos_w = list(sign_w).index(1)
    # get indices of series that load positively on group factor
    pos_k = list(np.where(sign_k==1)[0])

    # gibbs sampling algorithm
    for r in range(1,N_runs+1):

        ## Block Ia: drawing non time-varying parameters for series ---
        for i in range(n):
            # a_i and s2_i
            a[r,i], s2[r,i] = a_s2_i_sampler(y_i=y[:,i], f_w=f_w[r-1,:], f_k=f_K[r-1,:,select_k[i]],
                                             b_w_i=b_w[r-1,:,i], b_k_i=b_k[r-1,:,i], phi_i=phi[r-1,:,i], 
                                             eh_i=np.exp(h[r-1,:,i]), s2_i=s2[r-1,i], a_bar_i=a_bar_i, 
                                             A_bar_i=A_bar_i, nu_bar_i=nu_bar_i, delta2_bar_i=delta2_bar_i, 
                                             T=T, p=p, RNG=RNG)
            # phi_i
            phi[r,:,i] = phi_i_sampler(y_i=y[:,i], f_w=f_w[r-1,:], f_k=f_K[r-1,:,select_k[i]], a_i=a[r,i], 
                                       b_w_i=b_w[r-1,:,i], b_k_i=b_k[r-1,:,i], phi_i=phi[r-1,:,i], 
                                       eh_i=np.exp(h[r-1,:,i]), s2_i=s2[r,i], phi_bar_i=phi_bar_i, 
                                       V_bar_i=V_bar_i, T=T, p=p, RNG=RNG)
            # s2_eta_w_i
            s2_eta_w[r,i] = s2_eta_i_sampler(b_i=b_w[r-1,:,i], nu_eta_bar_i=nu_eta_bar_i, 
                                             delta2_eta_bar_i=delta2_eta_bar_i, T=T, RNG=RNG)
            # s2_eta_k_i
            s2_eta_k[r,i] = s2_eta_i_sampler(b_i=b_k[r-1,:,i], nu_eta_bar_i=nu_eta_bar_i, 
                                             delta2_eta_bar_i=delta2_eta_bar_i, T=T, RNG=RNG)
            # s2_zeta_i
            s2_zeta[r,i] = s2_zeta_sampler(h_i=h[r-1,:,i], nu_zeta_bar=nu_zeta_bar, 
                                           delta2_zeta_bar=delta2_zeta_bar, T=T, RNG=RNG)
        
        ## Block Ib: drawing non time-varying parameters for factors ---
        # phi_w
        phi_w[r,:] = phi_0_sampler(f=f_w[r-1,:], phi_0=phi_w[r-1,:], eh_0=np.exp(h_w[r-1,:]), s2_0=s2_0, 
                                   phi_bar_0=phi_bar_0, V_bar_0=V_bar_0, T=T, q=q, RNG=RNG)
        # s_zeta_w
        s2_zeta_w[r] = s2_zeta_sampler(h_i=h_w[r-1,:], nu_zeta_bar=nu_zeta_bar, 
                                       delta2_zeta_bar=delta2_zeta_bar, T=T, RNG=RNG)
        # loop over groups
        for k in range(K):
            # phi_k
            phi_K[r,:,k] = phi_0_sampler(f=f_K[r-1,:,k], phi_0=phi_K[r-1,:,k], eh_0=np.exp(h_K[r-1,:,k]), 
                                         s2_0=s2_0, phi_bar_0=phi_bar_0, V_bar_0=V_bar_0, T=T, q=q, RNG=RNG)
            # s_zeta_k
            s2_zeta_K[r,k] = s2_zeta_sampler(h_i=h_K[r-1,:,k], nu_zeta_bar=nu_zeta_bar, 
                                             delta2_zeta_bar=delta2_zeta_bar, T=T, RNG=RNG)

        ## Block II: drawing factors ---
        # f_w
        f_w[r,:] = f_w_sampler(y=y, f_K=f_K[r-1,:,:], select_k=select_k, a=a[r,:], b_w=b_w[r-1,:,:], 
                               b_k=b_k[r-1,:,:], phi_w=phi_w[r,:], phi=phi[r,:,:], eh_w=np.exp(h_w[r-1,:]), 
                               eh=np.exp(h[r-1,:,:]), s2_w=s2_0, s2=s2[r,:], T=T, n=n, p=p, q=q, RNG=RNG)
        # loop over groups
        for k in range(K):
            # f_k
            f_K[r,:,k] = f_k_sampler(y=y[:,k_ind[k]], f_w=f_w[r,:], a=a[r,k_ind[k]], b_w=b_w[r-1,:,k_ind[k]].transpose(), 
                                     b_k=b_k[r-1,:,k_ind[k]].transpose(), phi_k=phi_K[r,:,k], phi=phi[r,:,k_ind[k]].transpose(),
                                     eh_k=np.exp(h_K[r-1,:,k]), eh=np.exp(h[r-1,:,k_ind[k]].transpose()), s2_k=s2_0,
                                     s2=s2[r,k_ind[k]], T=T, p=p, q=q, RNG=RNG)

        ## Block III: drawing loadings ---

        # draw world loadings of series with sign restriction on world factor
        b_w[r,:,pos_w] = b_i_sampler(y_i=y[:,pos_w], f=f_w[r,:], f_x=f_K[r,:,select_k[pos_w]], a_i=a[r,pos_w], 
                                        b_x_i=b_k[r-1,:,pos_w], phi_i=phi[r,:,pos_w], eh_i=np.exp(h[r-1,:,pos_w]), 
                                        s2_i=s2[r,pos_w], s2_eta_i=s2_eta_w[r,pos_w], b_bar_i=b_bar_i, B_bar_i=B_bar_i, 
                                        T=T, p=p, RNG=RNG)
        # redraw loadings if not all positive
        if not all(b_w[r,:,pos_w]>0):
            wrong_sign = 1
            tries = 1
            while wrong_sign and tries<=50:
                b_w[r,:,pos_w] = b_i_sampler(y_i=y[:,pos_w], f=f_w[r,:], f_x=f_K[r,:,select_k[pos_w]], a_i=a[r,pos_w], 
                                                b_x_i=b_k[r-1,:,pos_w], phi_i=phi[r,:,pos_w], eh_i=np.exp(h[r-1,:,pos_w]), 
                                                s2_i=s2[r,pos_w], s2_eta_i=s2_eta_w[r,pos_w], b_bar_i=b_bar_i, B_bar_i=B_bar_i, 
                                                T=T, p=p, RNG=RNG)
                tries = tries + 1
                if all(b_w[r,:,pos_w]>0):
                    wrong_sign = 0
            # switch signs if all negative after max tries or take previous if mixed (pos/neg)
            if not all(b_w[r,:,pos_w]>0):
                if all(b_w[r,:,pos_w]<0):
                    b_w[r,:,pos_w] = -b_w[r,:,pos_w]
                    f_w[r,:] = -f_w[r,:]
                else:
                    b_w[r,:,pos_w] = b_w[r-1,:,pos_w]
        
        # draw world loadings of remaining series
        for i in [i for i in range(n) if i != pos_w]:
            b_w[r,:,i] = b_i_sampler(y_i=y[:,i], f=f_w[r,:], f_x=f_K[r,:,select_k[i]], a_i=a[r,i], 
                                        b_x_i=b_k[r-1,:,i], phi_i=phi[r,:,i], eh_i=np.exp(h[r-1,:,i]), 
                                        s2_i=s2[r,i], s2_eta_i=s2_eta_w[r,i], b_bar_i=b_bar_i, B_bar_i=B_bar_i, 
                                        T=T, p=p, RNG=RNG)
        
        # draw group loadings of series with sign restriction on group factors
        for i in pos_k:
            b_k[r,:,i] = b_i_sampler(y_i=y[:,i], f=f_K[r,:,select_k[i]], f_x=f_w[r,:], a_i=a[r,i], 
                                        b_x_i=b_w[r,:,i], phi_i=phi[r,:,i], eh_i=np.exp(h[r-1,:,i]), 
                                        s2_i=s2[r,i], s2_eta_i=s2_eta_k[r,i], b_bar_i=b_bar_i, B_bar_i=B_bar_i, 
                                        T=T, p=p, RNG=RNG)
            # redraw loadings if not all positive
            if not all(b_k[r,:,i]>0):
                wrong_sign = 1
                tries = 1
                while wrong_sign and tries<=50:
                    b_k[r,:,i] = b_i_sampler(y_i=y[:,i], f=f_K[r,:,select_k[i]], f_x=f_w[r,:], a_i=a[r,i], 
                                                b_x_i=b_w[r,:,i], phi_i=phi[r,:,i], eh_i=np.exp(h[r-1,:,i]), 
                                                s2_i=s2[r,i], s2_eta_i=s2_eta_k[r,i], b_bar_i=b_bar_i, B_bar_i=B_bar_i, 
                                                T=T, p=p, RNG=RNG)
                    tries = tries + 1
                    if all(b_k[r,:,i]>0):
                        wrong_sign = 0
                # switch signs if all negative after max tries or take previous if mixed (pos/neg)
                if not all(b_k[r,:,i]>0):
                    if all(b_k[r,:,i]<0):
                        b_k[r,:,i] = -b_k[r,:,i]
                        f_K[r,:,select_k[i]] = -f_K[r,:,select_k[i]]
                    else:
                        b_k[r,:,i] = b_k[r-1,:,i]

        # draw group loadings of remaining series
        for i in [i for i in range(n) if i not in pos_k]:
            b_k[r,:,i] = b_i_sampler(y_i=y[:,i], f=f_K[r,:,select_k[i]], f_x=f_w[r,:], a_i=a[r,i], 
                                        b_x_i=b_w[r,:,i], phi_i=phi[r,:,i], eh_i=np.exp(h[r-1,:,i]), 
                                        s2_i=s2[r,i], s2_eta_i=s2_eta_k[r,i], b_bar_i=b_bar_i, B_bar_i=B_bar_i, 
                                        T=T, p=p, RNG=RNG)
        
        ## Block IV: drawing stochastic volatility components ---
        for i in range(n):
            # h_i
            h[r,:,i] = h_i_sampler(y_i=y[:,i], f_w=f_w[r,:], f_k=f_K[r,:,select_k[i]], a_i=a[r,i], b_w_i=b_w[r,:,i], 
                                 b_k_i=b_k[r,:,i], phi_i=phi[r,:,i], eh_i=np.exp(h[r-1,:,i]), s2_i=s2[r,i], 
                                 s2_zeta_i=s2_zeta[r,i], q_K=q_K, m_K=m_K, v2_K=v2_K, T=T, p=p, RNG=RNG)
        # h_w
        h_w[r,:] = h_0_sampler(f=f_w[r,:], phi_0=phi_w[r,:], eh_0=np.exp(h_w[r-1,:]), s2_0=s2_0, 
                               s2_zeta_0=s2_zeta_w[r], q_K=q_K, m_K=m_K, v2_K=v2_K, T=T, q=q, RNG=RNG)
        # loop over groups
        for k in range(K):
            # h_k
            h_K[r,:,k] = h_0_sampler(f=f_K[r,:,k], phi_0=phi_K[r,:,k], eh_0=np.exp(h_K[r-1,:,k]), s2_0=s2_0, 
                                     s2_zeta_0=s2_zeta_K[r,k], q_K=q_K, m_K=m_K, v2_K=v2_K, T=T, q=q, RNG=RNG)

        # compute variances of factors and idiosyncratic processes
        var_f_w_r = var_arsv_comp(phi_w[r,:], s2_0, h_w[r,:])
        var_f_K_r = np.zeros((T,K))
        for k in range(K):
            var_f_K_r[:,k] = var_arsv_comp(phi_K[r,:,k], s2_0, h_K[r,:,k])
        var_I_r = np.zeros((T,n))
        for i in range(n):
            var_I_r[:,i] = var_arsv_comp(phi[r,:,i], s2[r,i], h[r,:,i])
    
        # compute sum of variances by series
        sumvar_r = np.zeros((T,n))
        for i in range(n):
            sumvar_r[:,i] = (b_w[r,:,i]**2)*var_f_w_r[:] + (b_k[r,:,i]**2)*var_f_K_r[:,select_k[i]] + var_I_r[:,i]

        # compute variances attributable to the different components and store them
        for i in range(n):
            vd_w[r,:,i] = (b_w[r,:,i]**2)*var_f_w_r[:] / sumvar_r[:,i]
            vd_K[r,:,i] = (b_k[r,:,i]**2)*var_f_K_r[:,select_k[i]] / sumvar_r[:,i]
            vd_I[r,:,i] = var_I_r[:,i] / sumvar_r[:,i]

        # print progess of algorithm
        if print_progress:
            print('Gibbs iteration: {0}/{1}'.format(r,N_runs))
    
    # get total time needed
    total_time = datetime.now().replace(microsecond=0)-start_time

    # print verification that gibbs sampling is complete and time needed 
    if print_time:
        print('GIBBS SAMPLING COMPLETE. Execution time:', total_time)

    # create dictionary of estimation history (include columnnames if y was initially a dataframe)
    if df:
        results = {'f_w': f_w, 'f_K': f_K, 'a': a, 'b_w': b_w, 'b_k': b_k, 's2': s2, 's2_eta_w': s2_eta_w,
                   's2_eta_k': s2_eta_k, 's2_zeta': s2_zeta, 's2_zeta_w': s2_zeta_w, 's2_zeta_K': s2_zeta_K, 
                   's2_0': s2_0, 'phi': phi, 'phi_w': phi_w, 'phi_K': phi_K, 'h': h, 'h_w': h_w, 
                   'h_K': h_K, 'vd_w': vd_w, 'vd_K': vd_K, 'vd_I': vd_I, 'time': total_time.total_seconds(),
                   'select_k': select_k, 'sign_w': sign_w, 'sign_k':sign_k, 'names': names}
    else:
        results = {'f_w': f_w, 'f_K': f_K, 'a': a, 'b_w': b_w, 'b_k': b_k, 's2': s2, 's2_eta_w': s2_eta_w,
                   's2_eta_k': s2_eta_k, 's2_zeta': s2_zeta, 's2_zeta_w': s2_zeta_w, 's2_zeta_K': s2_zeta_K, 
                   's2_0': s2_0, 'phi': phi, 'phi_w': phi_w, 'phi_K': phi_K, 'h': h, 'h_w': h_w, 
                   'h_K': h_K, 'vd_w': vd_w, 'vd_K': vd_K, 'vd_I': vd_I, 'time': total_time.total_seconds(),
                   'select_k': select_k, 'sign_w': sign_w, 'sign_k':sign_k}

    # return estimation results
    return results

