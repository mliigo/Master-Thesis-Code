"""
Application to International Inflation Dynamics

This file applies the multi-level dynamic factor model with time-varying loadings
and stochastic volatility of Del Negro and Otrok (2008) to decompose quarterly
inflation rates of 71 countries into world, group and idiosyncratic components.
Countries are split into advanced economies (AEs), emerging market economies 
(EMEs) and low-income developing countries (LIDCs). The considered time period 
is 1970-Q2 – 2023-Q3. Data is taken from the World Bank's global inflation 
database (Ha et al., 2023).

Posterior draws are saved as a python dictionary using the pickle package.
They are then evaluated in the file "App_Evaluation.py".

Imports:
--------
* ML_TVP_DFM_estimate
* pandas
* numpy
* pickle

References:
-----------
Del Negro, M. & Otrok, C. (2008). Dynamic Factor Models with Time-Varying 
Parameters: Measuring Changes in International Business Cycles (Staff Report 
No. 326). Federal Reserve Bank of New York. 
DOI: https://dx.doi.org/10.2139/ssrn.1136163

Ha, J., Kose, M. A. & Ohnsorge, F. (2023). One-stop Source: A Global Database 
of Inflation. Journal of International Money and Finance, 137 (October), 102896.
DOI: https://doi.org/10.1016/j.jimonfin.2023.102896
"""

# import packages
import pandas as pd
import numpy as np
import pickle

# import functions for estimation
import ML_TVP_DFM_estimate as EST


## Import and Clean Data -------------------------------------------------------

# import data (quarterly headline CPI)
data_q = pd.read_excel('Inflation_Data_WorldBank.xlsx', sheet_name='hcpi_q')

# remove columns that are not needed
data_q = data_q.drop(columns=['Country Code', 'IMF Country Code', 
                              'Indicator Type', 'Series Name', 'Data source', 
                              'Note', 'Unnamed: 223', '20231.1', '20232.1', 
                              '20233.1', '20234.1', 'Unnamed: 228',
                              'Unnamed: 229', 'Unnamed: 230'])

# transpose data (dates as index, countries as column names)
data_q = data_q.transpose()
data_q.columns = data_q.iloc[0]
data_q = data_q.drop('Country')
data_q = data_q.drop(data_q.columns[-1],axis=1)

# change index to datetime 
data_q.index = pd.date_range(start='1970', end='2024', freq='Q')

# select dates 1970Q1 - 2023Q3 (dropping last quarter to gain 10 countries)
DATA = data_q.loc['1970-01-31':'2023-09-30']



## Create Country Groups -------------------------------------------------------

# Advanced Economies (28)
aes = ['Australia', 'Austria', 'Belgium', 'Canada', 'Cyprus', 'Denmark', 
       'Finland', 'France', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Israel', 
       'Italy', 'Japan', 'Korea, Rep.', 'Luxembourg', 'Malta', 'Netherlands', 
       'New Zealand', 'Norway', 'Portugal', 'Singapore', 'Spain', 'Sweden', 
       'Switzerland', 'United Kingdom', 'United States']

# Emerging Market and Middle Income Economies (30)
emes = ['Argentina', 'Bahamas', 'Bolivia', 'Chile', 'Colombia', 
        'Dominican Republic', 'Ecuador', 'Egypt, Arab Rep.', 'El Salvador', 
        'Fiji', 'Gabon', 'Guatemala', 'India', 'Indonesia', 'Jamaica', 'Malaysia', 
        'Mauritius', 'Mexico', 'Morocco', 'Pakistan', 'Panama', 'Paraguay', 
        'Peru', 'Philippines', 'Samoa', 'South Africa', 'Sri Lanka', 
        'Trinidad and Tobago', 'Türkiye', 'Uruguay']

# Low Income Developing Countries (13)
lics = ['Burkina Faso', 'Burundi', 'Cameroon', "CÃ´te d'Ivoire", 'Gambia, The', 
        'Ghana', 'Haiti', 'Honduras', 'Nepal', 'Niger', 'Nigeria', 'Senegal', 
        'Tanzania, United Rep.']


## Select Countries and Compute Percentage Changes -----------------------------

# select available countries without missing data
DATA_clean = DATA[aes+emes+lics]

# compute percentage changes (and remove first row)
DATA_perc = DATA_clean.pct_change().loc['1970-06-30':'2023-09-30']


## Estimation ------------------------------------------------------------------

# specify number of groups
K = 3

# specify group compositions
select_k = np.zeros(len(DATA_clean.columns)).astype(int)
select_k[np.in1d(DATA_clean.columns, aes)] = 0
select_k[np.in1d(DATA_clean.columns, emes)] = 1
select_k[np.in1d(DATA_clean.columns, lics)] = 2

# specify sign restriction on world factor (United States)
sign_w = np.zeros(len(DATA_clean.columns)).astype(int)
sign_w[DATA_clean.columns=='United States'] = 1

# specify sign restriction on group factors
sign_k = np.zeros(len(DATA_clean.columns)).astype(int)
sign_k[DATA_clean.columns=='United States'] = 1     # AEs
sign_k[DATA_clean.columns=='India'] = 1             # EMDEs
sign_k[DATA_clean.columns=='Nigeria'] = 1           # LICs

# specify AR lags
q = 3       # order of AR(q) factor processes
p = q-1     # order of AR(p) error processes

# specify prior parameters
a_bar_i = 0                                # mean of N prior on a_i
A_bar_i = 0.1                              # precision of N prior on a_i
b_bar_i = 0                                # mean of N prior on b_i_0
B_bar_i = 0.1                              # precision of N prior on b_i_0
nu_bar_i = 100                             # df of IG prior on s2_i
delta2_bar_i = 0.001                       # scale of IG prior on s2_i
nu_eta_bar_i = 10                          # df of IG prior on s2_eta_i
delta2_eta_bar_i = 0.25**2                 # scale of IG prior on s2_eta_i
nu_zeta_bar = 100                          # df of IG prior on s2_zeta_i (& s2_zeta_0)
delta2_zeta_bar = 0.05**2                  # scale of IG prior on s2_zeta_i (& s2_zeta_0)
phi_bar_i = np.zeros(p)                    # mean vector of N prior on phi_i (px1)
V_bar_i = np.diag([1 for l in range(p)])   # precision matrix of N prior on phi_i (pxp)
phi_bar_0 = np.zeros(q)                    # mean vector of N prior on phi_0 (qx1)
V_bar_0 = np.diag([1 for l in range(q)])   # precision matrix of N prior on phi_0 (qxq)

# fix non time-varying components of variances of world and group factors
s2_0 = 0.001  

## Gibbs Sampling Procedure ----------------------------------------------------

# specify Gibbs sampling hyperparameters
N_runs = 60000      # number of Gibbs sampling iterations
burnin = 50000      # number of burn-in iterations

# set seed
RNG = np.random.default_rng(1999)

# estimate and get trace
trace = EST.gibbs(N_runs, DATA_perc, K, select_k, s2_0, sign_w, sign_k, a_bar_i, 
                  A_bar_i, b_bar_i, B_bar_i, phi_bar_i, V_bar_i, phi_bar_0, 
                  V_bar_0, nu_bar_i, delta2_bar_i, nu_eta_bar_i, delta2_eta_bar_i, 
                  nu_zeta_bar, delta2_zeta_bar, RNG, print_progress=1, print_time=1)

# save only draws after burn-in (last 10'000 iterations)
#posterior_draws = {'f_w':       trace['f_w'][burnin:N_runs+1,:], 
#                   'f_K':       trace['f_K'][burnin:N_runs+1,:,:], 
#                   'a':         trace['a'][burnin:N_runs+1,:], 
#                   'b_w':       trace['b_w'][burnin:N_runs+1,:,:], 
#                   'b_k':       trace['b_k'][burnin:N_runs+1,:,:], 
#                   's2':        trace['s2'][burnin:N_runs+1,:], 
#                   's2_eta_w':  trace['s2_eta_w'][burnin:N_runs+1,:],
#                   's2_eta_k':  trace['s2_eta_k'][burnin:N_runs+1,:], 
#                   's2_zeta':   trace['s2_zeta'][burnin:N_runs+1,:], 
#                   's2_zeta_w': trace['s2_zeta_w'][burnin:N_runs+1], 
#                   's2_zeta_K': trace['s2_zeta_K'][burnin:N_runs+1,:], 
#                   's2_0':      trace['s2_0'], 
#                   'phi':       trace['phi'][burnin:N_runs+1,:,:], 
#                   'phi_w':     trace['phi_w'][burnin:N_runs+1,:], 
#                   'phi_K':     trace['phi_K'][burnin:N_runs+1,:,:], 
#                   'h':         trace['h'][burnin:N_runs+1,:,:], 
#                   'h_w':       trace['h_w'][burnin:N_runs+1,:], 
#                   'h_K':       trace['h_K'][burnin:N_runs+1,:,:], 
#                   'vd_w':      trace['vd_w'][burnin:N_runs+1,:,:], 
#                   'vd_K':      trace['vd_K'][burnin:N_runs+1,:,:], 
#                   'vd_I':      trace['vd_I'][burnin:N_runs+1,:,:], 
#                   'time':      trace['time'],
#                   'select_k':  trace['select_k'], 
#                   'sign_w':    trace['sign_w'], 
#                   'sign_k':    trace['sign_k'], 
#                   'names':     trace['names']}

#with open('Application_Trace_last_10k.pkl', 'wb') as x:
#    pickle.dump(posterior_draws.copy(), x)