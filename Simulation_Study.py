"""
Simulation Study to Evaluate Performance of Model and Estimation Procedure
of Del Negro and Otrok (2008).

This file implements the simulation study. It simulates several models and uses
the simulated observables for estimating the factors, parameters and variance
decompositions. This is done in parallel using multiprocessing. The results are
saved as pickled dictionaries in the Sim_Saves folder. Four regimes are considered,
in which the degree of time-variation in the loadings and stochastic volatilities
is either set to a low or high value, respectively.

Note: Make sure that a folder called "Sim_Saves" is present in the working directory.
      The simulated and estimated values are saved as dictionaries (one for each
      regime) in this folder and then used in Sim_Evaluation.py.

Functions:
----------
* Multi_Sim_Est : simultaneously simulates and estimates several models and
                  saves the simulated models and estimation results for
                  subsequent evaluation

Imports:
--------
* Sim_Est_grouped
* pickle
* numpy
* multiprocessing
* datetime

References:
-----------
Del Negro, M. & Otrok, C. (2008). Dynamic Factor Models with Time-Varying 
Parameters: Measuring Changes in International Business Cycles (Staff Report 
No. 326). Federal Reserve Bank of New York. 
DOI: https://dx.doi.org/10.2139/ssrn.1136163
"""

# import file to perform parallelized simulation
import Sim_Est_grouped as sim_est
import pickle

# import remaining packages
import numpy as np
import multiprocessing as mp
from datetime import datetime


## Specify Hyperparameters ------------------------------------------------------------------

T = 100     # number time periods
K = 2       # number of groups
N_k = 5     # number of series per group
q = 3       # order of AR(q) factor processes
p = q - 1   # order of AR(p) error processes


## Specify Parameters for Simulation --------------------------------------------------------

a_mean = 0        # mean of Normal for generating intercepts
a_std = 1         # std. of Normal for generating intercepts
b0_mean_w = 1     # mean of Normal for generating initial world loadings (t=0)
b0_std_w = 1      # std. of Normal for generating initial world loadings (t=0)
b0_mean_k = 1     # mean of Normal for generating initial group loadings (t=0)
b0_std_k = 1      # std. of Normal for generating initial group loadings (t=0)
s2 = 1            # non time-varying components of variances of error processes
s2_0 = 1          # non time-varying components of variances of world and group factors


## Specify Prior Parameters -----------------------------------------------------------------

a_bar_i = 0                                # mean of N prior on a_i
A_bar_i = 0.1                              # precision of N prior on a_i
b_bar_i = 0                                # mean of N prior on b_i_0
B_bar_i = 0.1                              # precision of N prior on b_i_0
nu_bar_i = 100                             # dof of IG prior on s2_i
delta2_bar_i = 1                           # scale of IG prior on s2_i

nu_eta_bar_i = 10                          # dof of IG prior on s2_eta_i
delta2_eta_bar_i = 0.4**2                  # scale of IG prior on s2_eta_i
nu_zeta_bar = 100                          # dof of IG prior on s2_zeta_i (& s2_zeta_0)
delta2_zeta_bar = 0.1**2                   # scale of IG prior on s2_zeta_i (& s2_zeta_0)

phi_bar_i = np.zeros(p)                    # mean vector of N prior on phi_i (px1)
V_bar_i = np.diag([1 for l in range(p)])   # precision matrix of N prior on phi_i (pxp)
phi_bar_0 = np.zeros(q)                    # mean vector of N prior on phi_0 (qx1)
V_bar_0 = np.diag([1 for l in range(q)])   # precision matrix of N prior on phi_0 (qxq)


## Specify Estimation Hyperparameters -------------------------------------------------------

N_runs = 35000     # number of gibbs sampling iterations
burnin = 30000     # number of burn-in iterations


## Define Function for Iterative Multiprocess Simulation & Estimation -----------------------

def Multi_Sim_Est(ID, N_sims, N_cores, s2_eta_all, s2_zeta_all, seed):
    '''
    Simultaneously simulates and estimates several models and saves the simulated
    models and estimation results for subsequent evaluation. N_cores number of
    models are simulated and estimated at the same time until N_sims number of
    models have been successfuly simulated and estimated. The simulated and
    estimated values are saved as pickled dictionaries in the folder Sim_Saves.
    Make sure that such a folder is present in the working directory. Uses the 
    sim_est function defined in Sim_Est_grouped.py.

    ID          : identifier attached to the beginning of the saved dictionary's
                  name (ID_storage_N_sims)
    N_sims      : total number of models to simulate and estimate
    N_cores     : number of models to simulate and estimate in parallel
    s2_eta_all  : true value of the innovation variances to the loading processes
    s2_zeta_all : true value of the innovation variances to the stochastic
                  volatility processes
    seed        : seed to use for random number generation
    '''
    # get starting time of whole simulation
    start_time_sim = datetime.now().replace(microsecond=0)

    # create storage to save results across processes
    manager = mp.Manager()
    storage = manager.dict()

    # define identifiers for progression
    finished = 0   # number of succesfully simulated and estimated models
    failed = 0     # number of failed simulated and estimated models
    batch = 1      # number of batch of N_cores models

    # run simulation and estimation until N_sims models have been succesfully
    # simulated and estimated
    while finished<N_sims:

        # print starting of new batch and current time
        print('=== STARTING BATCH {0}.  TIME: '.format(batch),
              datetime.now().replace(microsecond=0))
        
        # get starting time of this batch
        start_time_batch = datetime.now().replace(microsecond=0)

        # define empty list to store processes
        processes = []

        # set up and start N_cores number of processes
        for i in range((batch-1)*N_cores+1, (batch-1)*N_cores+N_cores+1):
            
            # define random number generator, i is constructed to ensure 
            # replicability when using different N_cores
            rng = np.random.default_rng(seed+i)

            # create process and tell it to use sim_est function
            # (automatically save results in storage)
            proc = mp.Process(target=sim_est.sim_est, args=(T, K, N_k, q, p, 
                        N_runs, burnin, rng, a_mean, a_std, b0_mean_w, b0_std_w, 
                        b0_mean_k, b0_std_k, s2, s2_eta_all, s2_eta_all, 
                        s2_zeta_all, s2_zeta_all, s2_zeta_all, s2_0, s2_0, s2_0, 
                        a_bar_i, A_bar_i, b_bar_i, B_bar_i, phi_bar_i, V_bar_i, 
                        phi_bar_0, V_bar_0, nu_bar_i, delta2_bar_i, nu_eta_bar_i, 
                        delta2_eta_bar_i, nu_zeta_bar, delta2_zeta_bar, storage, i))
            
            # start and append process to list
            proc.start()
            processes.append(proc)

        # wait until all processes are done
        for proc in processes:
            proc.join()

        # check which processes were successful and which not, update identifiers
        # accordingly and delete failed processes from storage
        for j in range((batch-1)*N_cores+1,(batch-1)*N_cores+N_cores+1):
            if storage[j]['status'] == 'success':
                finished = finished + 1
            else:
                failed = failed + 1
                del storage[j]
        
        # print completion of batch, needed time and number of successful and
        # failed processes
        print('=== BATCH {0} COMPLETE.  EXECUTION TIME: '.format(batch),
              datetime.now().replace(microsecond=0)-start_time_batch)
        print('=== Current Number of Succesfully Finished Runs: {0}'.format(finished))
        print('=== Current Number of Failed Runs              : {0}'.format(failed),'\n')

        # update batch identifier
        batch = batch + 1
    
    # if more than N_sims number of models were successsfuly simulated and
    # estimated, then delete the excess
    if len(storage.keys())>N_sims:
        for j in list(storage.keys())[N_sims:]:
            del storage[j]

    # save the results as dictionary in Sim_Saves folder
    with open('Sim_Saves/{0}_storage_{1}.pkl'.format(ID, N_sims), 'wb') as x:
        pickle.dump(storage.copy(), x)

    # print completion of simulation and needed time
    print('SIMULATION ID: {0} COMPLETE. TOTAL EXECUTION TIME: '.format(ID), 
          datetime.now().replace(microsecond=0)-start_time_sim)


## Simulate, Estimate & Save ----------------------------------------------------------------

# To run the simulation study for a given time-variation regime (LL, LH, HL, HH) remove the
# hashtags for the respective regime and run it (done in this way since multiprocessing 
# reruns the main file and the hashtags prevent it from starting too many processes).
# Results are saved as pickled dictionaries in the Sim_Saves folder (make sure that a folder
# with this name is present in the working directory). The number of cores can be adjusted 
# with N_cores.

#if __name__=="__main__":
#    Multi_Sim_Est(ID="LL", N_sims=200, N_cores=20, s2_eta_all=0.3**2, s2_zeta_all=0.05**2, 
#                  seed=2345)

#if __name__=="__main__":
#    Multi_Sim_Est(ID="LH", N_sims=200, N_cores=20, s2_eta_all=0.3**2, s2_zeta_all=0.15**2, 
#                  seed=23456)

#if __name__=="__main__":
#    Multi_Sim_Est(ID="HL", N_sims=200, N_cores=20, s2_eta_all=0.5**2, s2_zeta_all=0.05**2, 
#                  seed=234567)

#if __name__=="__main__":
#    Multi_Sim_Est(ID="HH", N_sims=200, N_cores=20, s2_eta_all=0.5**2, s2_zeta_all=0.15**2, 
#                  seed=2345678)

