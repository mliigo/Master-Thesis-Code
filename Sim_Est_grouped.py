"""
Code for Grouping Model Simulation and Estimation

This file defines functions used in the simulation study implemented in the file
"Simulation_Study.py". Due to the functioning of multiprocessing, they need to 
be defined in a seperate file.

Functions:
----------
* get_res : takes the Gibbs draws from estimating a simulated model and returns
            point estimates and percentiles for factors, parameters and 
            variance decompositions based on the posterior draws
* sim_est : simulates the model, performs estimation based on the simulated
            observables and saves the simulated model as well as the estimation 
            results obtained with "get_res"

Imports:
--------
* ML_TVP_DFM_simulate
* ML_TVP_DFM_estimate
* numpy
* datetime
"""

# import simulation and estimation files
import ML_TVP_DFM_simulate as SIM
import ML_TVP_DFM_estimate as EST

# import remaining packages
import numpy as np
from datetime import datetime


# function to get needed results from estimation output
def get_res(est, burn):
    '''
    Takes the Gibbs draws from estimating a simulated model and returns 
    a dictionary of point estimates and percentiles for factors, parameters
    and variance decompositions based on the posterior draws.

    est  : dictionary with all Gibbs draws returned from running 
           ML_TVP_DFM_estimate.gibbs
    burn : number of burn-in iterations, draws after this are treated as
           posterior draws
    '''
    # get hyperparameters
    T = len(est['f_w'][0,:])
    n = len(est['a'][0,:])
    K = len(est['f_K'][0,0,:])
    p = len(est['phi'][0,:,0])
    q = len(est['phi_w'][0,:])

    # define empty dictionary to store results
    res = dict.fromkeys(['a', 's2', 's2_eta_w', 's2_eta_k', 's2_zeta', 
                         's2_zeta_w', 's2_zeta_K', 'phi', 'phi_w', 'phi_K', 
                         'b_w', 'b_k', 'h', 'h_w', 'h_K', 'f_w', 'f_K',
                         'vd_w', 'vd_K', 'vd_I', 'time'])
    
    # fill in time needed for estimation
    res['time'] = est['time']
    
    # fill in results of non-tv parameters of series
    for key in ['a', 's2', 's2_eta_w', 's2_eta_k', 's2_zeta']:
        # add subdictionary
        res[key] = dict([(j, np.zeros(n)) for j in ['mean','median','std']])
        # loop over series and fill subdictionary
        for i in range(n):
            hist = est[key][burn+1:,i]
            res[key]['mean'][i] = np.mean(hist)
            res[key]['median'][i] = np.median(hist)
            res[key]['std'][i] = np.std(hist)
    
    # fill in results of AR coefficients of series
    res['phi'] = dict([(j, np.zeros((p,n))) for j in ['mean','median','std']])
    # loop over lags and fill subdictionary
    for l in range(p):
        hist = est['phi'][burn+1:,l,:]
        res['phi']['mean'][l,:] = np.mean(hist,0)
        res['phi']['median'][l,:] = np.median(hist,0)
        res['phi']['std'][l,:] = np.std(hist,0)
    
    # fill in results of non-tv parameters of world factor
    res['s2_zeta_w'] = dict([(j, np.zeros(1)) for j in ['mean','median','std']])
    # fill subdictionary
    hist = est['s2_zeta_w'][burn+1:]
    res['s2_zeta_w']['mean'][0] = np.mean(hist)
    res['s2_zeta_w']['median'][0] = np.median(hist)
    res['s2_zeta_w']['std'][0] = np.std(hist)
    
    # fill in results of AR coefficients of world factor
    res['phi_w'] = dict([(j, np.zeros(q)) for j in ['mean','median','std']])
    # loop over lags and fill subdictionary
    for l in range(q):
        hist = est['phi_w'][burn+1:,l]
        res['phi_w']['mean'][l] = np.mean(hist)
        res['phi_w']['median'][l] = np.median(hist)
        res['phi_w']['std'][l] = np.std(hist)

    # fill in results of non-tv parameters of group factors
    res['s2_zeta_K'] = dict([(j, np.zeros(K)) for j in ['mean','median','std']])
    # loop over groups and fill subdictionary
    for k in range(K):
        hist = est['s2_zeta_K'][burn+1:,k]
        res['s2_zeta_K']['mean'][k] = np.mean(hist)
        res['s2_zeta_K']['median'][k] = np.median(hist)
        res['s2_zeta_K']['std'][k] = np.std(hist)

    # fill in results of AR coefficients of group factors
    res['phi_K'] = dict([(j, np.zeros((q,K))) for j in ['mean','median','std']])
    # loop over lags and fill subdictionary
    for l in range(q):
        hist = est['phi_K'][burn+1:,l,:]
        res['phi_K']['mean'][l,:] = np.mean(hist,0)
        res['phi_K']['median'][l,:] = np.median(hist,0)
        res['phi_K']['std'][l,:] = np.std(hist,0)
    
    # fill in results of tv parameters of series
    for key in ['b_w', 'b_k', 'h', 'vd_w', 'vd_K', 'vd_I']:
        # add subdictionary
        res[key] = dict([(j, np.zeros((T,n))) for j in ['mean','median','std','2.5','5','10',
                                                        '16','84','90','95','97.5']])
        # loop over series and fill subdictionary
        for i in range(n):
            hist = est[key][burn+1:,:,i]
            res[key]['mean'][:,i] = np.mean(hist,0)
            res[key]['median'][:,i] = np.median(hist,0)
            res[key]['std'][:,i] = np.std(hist,0)
            res[key]['2.5'][:,i] = np.percentile(hist,2.5,0)
            res[key]['5'][:,i] = np.percentile(hist,5,0)
            res[key]['10'][:,i] = np.percentile(hist,10,0)
            res[key]['16'][:,i] = np.percentile(hist,16,0)
            res[key]['84'][:,i] = np.percentile(hist,84,0)
            res[key]['90'][:,i] = np.percentile(hist,90,0)
            res[key]['95'][:,i] = np.percentile(hist,95,0)
            res[key]['97.5'][:,i] = np.percentile(hist,97.5,0)
    
    # fill in h_w and f_w
    for key in ['h_w', 'f_w']:
        # add subdictionary
        res[key] = dict([(j, np.zeros(T)) for j in ['mean','median','std','2.5','5','10',
                                                    '16','84','90','95','97.5']])
        # fill subdictionary
        hist = est[key][burn+1:,:]
        res[key]['mean'][:] = np.mean(hist,0)
        res[key]['median'][:] = np.median(hist,0)
        res[key]['std'][:] = np.std(hist,0)
        res[key]['2.5'][:] = np.percentile(hist,2.5,0)
        res[key]['5'][:] = np.percentile(hist,5,0)
        res[key]['10'][:] = np.percentile(hist,10,0)
        res[key]['16'][:] = np.percentile(hist,16,0)
        res[key]['84'][:] = np.percentile(hist,84,0)
        res[key]['90'][:] = np.percentile(hist,90,0)
        res[key]['95'][:] = np.percentile(hist,95,0)
        res[key]['97.5'][:] = np.percentile(hist,97.5,0)
    
    # fill in h_K and f_K
    for key in ['h_K', 'f_K']:
        # add subdictionary
        res[key] = dict([(j, np.zeros((T,K))) for j in ['mean','median','std','2.5','5','10',
                                                        '16','84','90','95','97.5']])
        # loop over groups and fill subdictionary
        for k in range(K):
            hist = est[key][burn+1:,:,k]
            res[key]['mean'][:,k] = np.mean(hist,0)
            res[key]['median'][:,k] = np.median(hist,0)
            res[key]['std'][:,k] = np.std(hist,0)
            res[key]['2.5'][:,k] = np.percentile(hist,2.5,0)
            res[key]['5'][:,k] = np.percentile(hist,5,0)
            res[key]['10'][:,k] = np.percentile(hist,10,0)
            res[key]['16'][:,k] = np.percentile(hist,16,0)
            res[key]['84'][:,k] = np.percentile(hist,84,0)
            res[key]['90'][:,k] = np.percentile(hist,90,0)
            res[key]['95'][:,k] = np.percentile(hist,95,0)
            res[key]['97.5'][:,k] = np.percentile(hist,97.5,0)
    
    # return dictionary
    return res

# function for simulation, estimation and results extraction
def sim_est(T, K, N_k, Q, P, N_runs, burnin, RNG, a_mean, a_std, b0_mean_w, 
            b0_std_w, b0_mean_k, b0_std_k, s2, s2_eta_w, s2_eta_k, s2_zeta, 
            s2_zeta_w, s2_zeta_K, s2_w, s2_K, s2_0, a_bar_i, A_bar_i, b_bar_i, 
            B_bar_i, phi_bar_i, V_bar_i, phi_bar_0, V_bar_0, nu_bar_i, 
            delta2_bar_i, nu_eta_bar_i, delta2_eta_bar_i, nu_zeta_bar, 
            delta2_zeta_bar, storage, i):
    '''
    Simulates a model using ML_TVP_DFM_simulate.sim, performs estimation based
    on the simulated observables using ML_TVP_DFM_estimate.gibbs and saves the 
    simulated model as well as the estimation results obtained with "get_res".
    Parameters are defined in ML_TVP_DFM_simulate.sim and 
    ML_TVP_DFM_estimate.gibbs.
    '''
    # get starting time
    start_time = datetime.now().replace(microsecond=0)

    try:
        # simulate model
        data = SIM.sim(T, K, N_k, Q, P, a_mean, a_std, b0_mean_w, b0_std_w, 
                       b0_mean_k, b0_std_k, s2, s2_eta_w, s2_eta_k, s2_zeta, 
                       s2_zeta_w, s2_zeta_K, s2_w, s2_K, RNG, print_stat=0)
    
        # estimate model
        print('process {0} estimation started. Time:'.format(i), 
              datetime.now().time().replace(microsecond=0))
        trace = EST.gibbs(N_runs=N_runs, y=data['y'], K=K, select_k=data['select_k'],
                          s2_0=s2_0, sign_w=data['sign_w'], sign_k=data['sign_k'], 
                          a_bar_i=a_bar_i, A_bar_i=A_bar_i, b_bar_i=b_bar_i, 
                          B_bar_i=B_bar_i, phi_bar_i=phi_bar_i, V_bar_i=V_bar_i, 
                          phi_bar_0=phi_bar_0, V_bar_0=V_bar_0, nu_bar_i=nu_bar_i, 
                          delta2_bar_i=delta2_bar_i, nu_eta_bar_i=nu_eta_bar_i, 
                          delta2_eta_bar_i=delta2_eta_bar_i, nu_zeta_bar=nu_zeta_bar, 
                          delta2_zeta_bar=delta2_zeta_bar, RNG=RNG, 
                          print_progress=0, print_time=0)

        # print confirmation that completed and elapsed time
        print('process {0} complete. Execution time: '.format(i), 
              datetime.now().replace(microsecond=0)-start_time)

        # get results
        res = get_res(trace, burnin)

        # store in dictionary
        storage[i] = {'status': 'success', 'data': data, 'res': res}
    
    except:

        # print that it failed and elapsed time
        print('process {0} failed. Execution time: '.format(i), 
              datetime.now().replace(microsecond=0)-start_time)

        # specify that iter has failed
        storage[i] = {'status': 'failed'}

