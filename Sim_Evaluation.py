"""
Evaluation of the Simulation Study

This file evaluates the simulation study results by computing and plotting
correlations between simulated (true) values and estimated values of factors,
time-varying parameters and variance decompositions. It also computes and
plots mean posterior confidence interval widths of the factors, time-varying 
parameters and variance decompositions. Correlations are seen as a measure of
accuracy of the estimates, while posterior confidence interval widths measure 
the uncertainty associated with these estimates.

Note: Functions are used to generate and directly save plots as PNG files.
      (often relatively large in size and hence not directly displayable)

Imports:
--------
* numpy
* pandas
* matplotlib.pyplot
* sns
* datetime
"""

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# change plot font globally to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# specify colors and linetypes to use across all plots
lcols = {'LL': 'black', 'LH': 'black', 'HL': 'darkgray', 'HH': 'darkgray'}
ltypes = {'LL': '-', 'LH': '--', 'HL': '-', 'HH': '--'}

# import simulation results
LL = pd.read_pickle('Sim_Saves/LL_storage_200.pkl')
LH = pd.read_pickle('Sim_Saves/LH_storage_200.pkl')
HL = pd.read_pickle('Sim_Saves/HL_storage_200.pkl')
HH = pd.read_pickle('Sim_Saves/HH_storage_200.pkl')

# get hyperparameters of simulation study
N = len(LL.keys())                                   # number samples per regime
n = len(LL[list(LL.keys())[0]]['data']['y'][0,:])    # number of series i
K = len(LL[list(LL.keys())[0]]['data']['f_K'][0,:])  # number of groups k


## Compute Correlations --------------------------------------------------------

# empty dictionary for storage
corrs = {key: dict.fromkeys(['f_w','f_K','b_w','b_k','h','h_w','h_K',
                             'vd_w','vd_K','vd_I']) 
         for key in ['LL', 'LH', 'HL', 'HH']}

# add empty arrays of appropriate dimensions to dictionary
for key in ['LL', 'LH', 'HL', 'HH']:
    for param in ['f_w','h_w']:
        corrs[key][param] = np.zeros(N)
    for param in ['f_K','h_K']:
        corrs[key][param] = np.zeros((N,K))
    for param in ['b_w','b_k','h','vd_w','vd_K','vd_I']:
        corrs[key][param] = np.zeros((N,n))

# loop to fill dictionary
for j in range(N):

    # get simulated and estimated values for iteration j in all regimes
    LL_j = LL[list(LL.keys())[j]]
    LH_j = LH[list(LH.keys())[j]]
    HL_j = HL[list(HL.keys())[j]]
    HH_j = HH[list(HH.keys())[j]]

    for param in ['f_w','h_w']:
        corrs['LL'][param][j] = np.corrcoef(LL_j['data'][param], 
                                            LL_j['res'][param]['mean'])[0,1]
        corrs['LH'][param][j] = np.corrcoef(LH_j['data'][param], 
                                            LH_j['res'][param]['mean'])[0,1]
        corrs['HL'][param][j] = np.corrcoef(HL_j['data'][param], 
                                            HL_j['res'][param]['mean'])[0,1]
        corrs['HH'][param][j] = np.corrcoef(HH_j['data'][param], 
                                            HH_j['res'][param]['mean'])[0,1]

    for param in ['f_K','h_K']:
        for k in range(K):
            corrs['LL'][param][j,k] = np.corrcoef(LL_j['data'][param][:,k], 
                                                  LL_j['res'][param]['mean'][:,k])[0,1]
            corrs['LH'][param][j,k] = np.corrcoef(LH_j['data'][param][:,k], 
                                                  LH_j['res'][param]['mean'][:,k])[0,1]
            corrs['HL'][param][j,k] = np.corrcoef(HL_j['data'][param][:,k], 
                                                  HL_j['res'][param]['mean'][:,k])[0,1]
            corrs['HH'][param][j,k] = np.corrcoef(HH_j['data'][param][:,k], 
                                                  HH_j['res'][param]['mean'][:,k])[0,1]

    for param in ['b_w','b_k','h','vd_w','vd_K','vd_I']:
        for i in range(n):
            corrs['LL'][param][j,i] = np.corrcoef(LL_j['data'][param][:,i], 
                                                  LL_j['res'][param]['mean'][:,i])[0,1]
            corrs['LH'][param][j,i] = np.corrcoef(LH_j['data'][param][:,i], 
                                                  LH_j['res'][param]['mean'][:,i])[0,1]
            corrs['HL'][param][j,i] = np.corrcoef(HL_j['data'][param][:,i], 
                                                  HL_j['res'][param]['mean'][:,i])[0,1]
            corrs['HH'][param][j,i] = np.corrcoef(HH_j['data'][param][:,i], 
                                                  HH_j['res'][param]['mean'][:,i])[0,1]

# compute mean correlations
mean_corrs = pd.DataFrame(columns=['LL', 'LH', 'HL', 'HH', 'ALL'], 
                          index=['f_w','f_K','b_w','b_k','h','h_w',
                                 'h_K','vd_w','vd_K','vd_I'])
for param in mean_corrs.index:
    for j in mean_corrs.columns[:-1]:
        mean_corrs.loc[param,j] = np.mean(corrs[j][param])
    mean_corrs.loc[param,'ALL'] = np.mean(np.concatenate((corrs['LL'][param], 
                                                          corrs['LH'][param], 
                                                          corrs['HL'][param], 
                                                          corrs['HH'][param])))

# compute median correlations
median_corrs = pd.DataFrame(columns=['LL', 'LH', 'HL', 'HH', 'ALL'], 
                            index=['f_w','f_K','b_w','b_k','h','h_w',
                                   'h_K','vd_w','vd_K','vd_I'])
for param in median_corrs.index:
    for j in median_corrs.columns[:-1]:
        median_corrs.loc[param,j] = np.median(corrs[j][param])
    median_corrs.loc[param,'ALL'] = np.median(np.concatenate((corrs['LL'][param], 
                                                              corrs['LH'][param], 
                                                              corrs['HL'][param], 
                                                              corrs['HH'][param])))

# inspect means and medians
mean_corrs.astype(float).round(2).transpose()
median_corrs.astype(float).round(2).transpose()


## Plot Correlations -----------------------------------------------------------

# 2x2 plot of correlations for factors and loadings
def plot_corrs_fb():
    fig, ax = plt.subplots(2,2, figsize=(6.3,4.8), dpi=600)
    for j in ['LL', 'LH', 'HL', 'HH']:
        x_fw = np.sort(corrs[j]['f_w'])
        ax[0,0].plot(x_fw, np.arange(1,len(x_fw)+1)/float(len(x_fw)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[0,0].set_title('World Factor', fontsize=9)
        x_fk = np.sort(corrs[j]['f_K'].flatten())
        ax[0,1].plot(x_fk, np.arange(1,len(x_fk)+1)/float(len(x_fk)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[0,1].set_title('Group Factors', fontsize=9)
        x_bw = np.sort(corrs[j]['b_w'].flatten())
        ax[1,0].plot(x_bw, np.arange(1,len(x_bw)+1)/float(len(x_bw)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[1,0].set_title('World Loadings', fontsize=9)
        x_bk = np.sort(corrs[j]['b_k'].flatten())
        ax[1,1].plot(x_bk, np.arange(1,len(x_bk)+1)/float(len(x_bk)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[1,1].set_title('Group Loadings', fontsize=9)
    for j in range(2):
        for i in range(2):
            ax[j,i].set_xlim(-1.1, 1.1)
            ax[j,i].spines[['right', 'top']].set_visible(False)
            ax[j,i].grid(alpha=0.25)
            ax[j,i].tick_params(axis='both', labelsize=9)
    plt.legend(['LL', 'LH', 'HL', 'HH'], loc='upper center',
               bbox_to_anchor=(-0.1, -0.15), ncol=4, fontsize=9,
               frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.savefig('CORRS_F_B.png', bbox_inches='tight')
plot_corrs_fb()

# 1x3 plot of correlations for stochastic volatilities of factors and series
def plot_corrs_sv():
    fig, ax = plt.subplots(1,3, figsize=(6.3,3), dpi=600)
    for j in ['LL', 'LH', 'HL', 'HH']:
        x_hw = np.sort(corrs[j]['h_w'])
        ax[0].plot(x_hw, np.arange(1,len(x_hw)+1)/float(len(x_hw)), 
                    color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[0].set_title('SV of World Factor', fontsize=9)
        x_hk = np.sort(corrs[j]['h_K'].flatten())
        ax[1].plot(x_hk, np.arange(1,len(x_hk)+1)/float(len(x_hk)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[1].set_title('SV of Group Factors', fontsize=9)
        x_hi = np.sort(corrs[j]['h'].flatten())
        ax[2].plot(x_hi, np.arange(1,len(x_hi)+1)/float(len(x_hi)), 
                   color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[2].set_title('SV of Idiosyncratic Components', fontsize=9)
    for j in range(3):
        ax[j].set_xlim(-1.1, 1.1)
        ax[j].spines[['right', 'top']].set_visible(False)
        ax[j].grid(alpha=0.25)
        ax[j].tick_params(axis='both', labelsize=9)
    ax[1].legend(['LL', 'LH', 'HL', 'HH'], loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9,
               frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig('CORRS_H.png', bbox_inches='tight')
plot_corrs_sv()

# 1x3 plot of correlations for variance decompositions
def plot_corrs_vd():
    fig, ax = plt.subplots(1,3, figsize=(6.3,3), dpi=600)
    for j in ['LL', 'LH', 'HL', 'HH']:
        x_hw = np.sort(corrs[j]['vd_w'].flatten())
        ax[0].plot(x_hw, np.arange(1,len(x_hw)+1)/float(len(x_hw)), 
                    color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[0].set_title('World', fontsize=9)
        x_hk = np.sort(corrs[j]['vd_K'].flatten())
        ax[1].plot(x_hk, np.arange(1,len(x_hk)+1)/float(len(x_hk)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[1].set_title('Group', fontsize=9)
        x_hi = np.sort(corrs[j]['vd_I'].flatten())
        ax[2].plot(x_hi, np.arange(1,len(x_hi)+1)/float(len(x_hi)), 
                   color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[2].set_title('Idiosyncratic', fontsize=9)
    for j in range(3):
        ax[j].set_xlim(-1.1, 1.1)
        ax[j].spines[['right', 'top']].set_visible(False)
        ax[j].grid(alpha=0.25)
        ax[j].tick_params(axis='both', labelsize=9)
    ax[1].legend(['LL', 'LH', 'HL', 'HH'], loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9,
               frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig('CORRS_VDS.png', bbox_inches='tight')
plot_corrs_vd()


## Compute Posterior Confidence Interval Widths --------------------------------

# empty dictionaries for storage
ci_68 = {key: dict.fromkeys(['f_w','f_K','b_w','b_k','h','h_w','h_K',
                             'vd_w','vd_K','vd_I']) 
         for key in ['LL', 'LH', 'HL', 'HH']}
ci_90 = {key: dict.fromkeys(['f_w','f_K','b_w','b_k','h','h_w','h_K',
                             'vd_w','vd_K','vd_I']) 
         for key in ['LL', 'LH', 'HL', 'HH']}

# add empty arrays of appropriate dimensions to dictionary
for key in ['LL', 'LH', 'HL', 'HH']:
    for param in ['f_w','h_w']:
        ci_68[key][param] = np.zeros(N)
        ci_90[key][param] = np.zeros(N)
    for param in ['f_K','h_K']:
        ci_68[key][param] = np.zeros((N,K))
        ci_90[key][param] = np.zeros((N,K))
    for param in ['b_w','b_k','h','vd_w','vd_K','vd_I']:
        ci_68[key][param] = np.zeros((N,n))
        ci_90[key][param] = np.zeros((N,n))

# loop to fill dictionary
for j in range(N):

    # get simulated and estimated values for iteration j in all regimes
    LL_j = LL[list(LL.keys())[j]]
    LH_j = LH[list(LH.keys())[j]]
    HL_j = HL[list(HL.keys())[j]]
    HH_j = HH[list(HH.keys())[j]]

    for param in ['f_w','h_w']:
        ci_68['LL'][param][j] = np.mean((LL_j['res'][param]['84'] - 
                                         LL_j['res'][param]['16']))
        ci_68['LH'][param][j] = np.mean((LH_j['res'][param]['84'] - 
                                         LH_j['res'][param]['16']))
        ci_68['HL'][param][j] = np.mean((HL_j['res'][param]['84'] - 
                                         HL_j['res'][param]['16']))
        ci_68['HH'][param][j] = np.mean((HH_j['res'][param]['84'] - 
                                         HH_j['res'][param]['16']))
        ci_90['LL'][param][j] = np.mean((LL_j['res'][param]['95'] - 
                                         LL_j['res'][param]['5']))
        ci_90['LH'][param][j] = np.mean((LH_j['res'][param]['95'] - 
                                         LH_j['res'][param]['5']))
        ci_90['HL'][param][j] = np.mean((HL_j['res'][param]['95'] - 
                                         HL_j['res'][param]['5']))
        ci_90['HH'][param][j] = np.mean((HH_j['res'][param]['95'] - 
                                         HH_j['res'][param]['5']))

    for param in ['f_K','h_K']:
        for k in range(K):
            ci_68['LL'][param][j,k] = np.mean((LL_j['res'][param]['84'][:,k] - 
                                               LL_j['res'][param]['16'][:,k]))
            ci_68['LH'][param][j,k] = np.mean((LH_j['res'][param]['84'][:,k] - 
                                               LH_j['res'][param]['16'][:,k]))
            ci_68['HL'][param][j,k] = np.mean((HL_j['res'][param]['84'][:,k] - 
                                               HL_j['res'][param]['16'][:,k]))
            ci_68['HH'][param][j,k] = np.mean((HH_j['res'][param]['84'][:,k] - 
                                               HH_j['res'][param]['16'][:,k]))
            ci_90['LL'][param][j,k] = np.mean((LL_j['res'][param]['95'][:,k] - 
                                               LL_j['res'][param]['5'][:,k]))
            ci_90['LH'][param][j,k] = np.mean((LH_j['res'][param]['95'][:,k] - 
                                               LH_j['res'][param]['5'][:,k]))
            ci_90['HL'][param][j,k] = np.mean((HL_j['res'][param]['95'][:,k] - 
                                               HL_j['res'][param]['5'][:,k]))
            ci_90['HH'][param][j,k] = np.mean((HH_j['res'][param]['95'][:,k] - 
                                               HH_j['res'][param]['5'][:,k]))

    for param in ['b_w','b_k','h','vd_w','vd_K','vd_I']:
        for i in range(n):
            ci_68['LL'][param][j,i] = np.mean((LL_j['res'][param]['84'][:,i] - 
                                               LL_j['res'][param]['16'][:,i]))
            ci_68['LH'][param][j,i] = np.mean((LH_j['res'][param]['84'][:,i] - 
                                               LH_j['res'][param]['16'][:,i]))
            ci_68['HL'][param][j,i] = np.mean((HL_j['res'][param]['84'][:,i] - 
                                               HL_j['res'][param]['16'][:,i]))
            ci_68['HH'][param][j,i] = np.mean((HH_j['res'][param]['84'][:,i] - 
                                               HH_j['res'][param]['16'][:,i]))
            ci_90['LL'][param][j,i] = np.mean((LL_j['res'][param]['95'][:,i] - 
                                               LL_j['res'][param]['5'][:,i]))
            ci_90['LH'][param][j,i] = np.mean((LH_j['res'][param]['95'][:,i] - 
                                               LH_j['res'][param]['5'][:,i]))
            ci_90['HL'][param][j,i] = np.mean((HL_j['res'][param]['95'][:,i] - 
                                               HL_j['res'][param]['5'][:,i]))
            ci_90['HH'][param][j,i] = np.mean((HH_j['res'][param]['95'][:,i] - 
                                               HH_j['res'][param]['5'][:,i]))

# compute mean PCI widths
mean_ci_68 = pd.DataFrame(columns=['LL', 'LH', 'HL', 'HH', 'ALL'], 
                          index=['f_w','f_K','b_w','b_k','h','h_w',
                                 'h_K','vd_w','vd_K','vd_I'])
mean_ci_90 = pd.DataFrame(columns=['LL', 'LH', 'HL', 'HH', 'ALL'], 
                          index=['f_w','f_K','b_w','b_k','h','h_w',
                                 'h_K','vd_w','vd_K','vd_I'])
for param in mean_ci_90.index:
    for j in mean_ci_90.columns[:-1]:
        mean_ci_68.loc[param,j] = np.mean(ci_68[j][param])
        mean_ci_90.loc[param,j] = np.mean(ci_90[j][param])
    mean_ci_68.loc[param,'ALL'] = np.mean(np.concatenate((ci_68['LL'][param], 
                                                          ci_68['LH'][param], 
                                                          ci_68['HL'][param], 
                                                          ci_68['HH'][param])))
    mean_ci_90.loc[param,'ALL'] = np.mean(np.concatenate((ci_90['LL'][param], 
                                                          ci_90['LH'][param], 
                                                          ci_90['HL'][param], 
                                                          ci_90['HH'][param])))

# compute median PCI widths
median_ci_68 = pd.DataFrame(columns=['LL', 'LH', 'HL', 'HH', 'ALL'], 
                            index=['f_w','f_K','b_w','b_k','h','h_w',
                                   'h_K','vd_w','vd_K','vd_I'])
median_ci_90 = pd.DataFrame(columns=['LL', 'LH', 'HL', 'HH', 'ALL'], 
                            index=['f_w','f_K','b_w','b_k','h','h_w',
                                   'h_K','vd_w','vd_K','vd_I'])
for param in median_ci_90.index:
    for j in median_ci_90.columns[:-1]:
        median_ci_68.loc[param,j] = np.median(ci_68[j][param])
        median_ci_90.loc[param,j] = np.median(ci_90[j][param])
    median_ci_68.loc[param,'ALL'] = np.median(np.concatenate((ci_68['LL'][param], 
                                                              ci_68['LH'][param], 
                                                              ci_68['HL'][param], 
                                                              ci_68['HH'][param])))
    median_ci_90.loc[param,'ALL'] = np.median(np.concatenate((ci_90['LL'][param], 
                                                              ci_90['LH'][param], 
                                                              ci_90['HL'][param], 
                                                              ci_90['HH'][param])))

# inspect means and medians
mean_ci_68.astype(float).round(2).transpose()
median_ci_68.astype(float).round(2).transpose()
mean_ci_90.astype(float).round(2).transpose()
median_ci_90.astype(float).round(2).transpose()


## Plot Posterior Confidence Interval Widths -----------------------------------

# 2x2 plot of mean PCI90 widths for factors and loadings
def plot_ci90_fb():
    fig, ax = plt.subplots(2,2, figsize=(6.3,4.8), dpi=600)
    for j in ['LL', 'LH', 'HL', 'HH']:
        x_fw = np.sort(ci_90[j]['f_w'])
        ax[0,0].plot(x_fw, np.arange(1,len(x_fw)+1)/float(len(x_fw)), 
                    color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[0,0].set_title('World Factor', fontsize=9)
        x_fk = np.sort(ci_90[j]['f_K'].flatten())
        ax[0,1].plot(x_fk, np.arange(1,len(x_fk)+1)/float(len(x_fk)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[0,1].set_title('Group Factors', fontsize=9)
        x_bw = np.sort(ci_90[j]['b_w'].flatten())
        ax[1,0].plot(x_bw, np.arange(1,len(x_bw)+1)/float(len(x_bw)), 
                   color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[1,0].set_title('World Loadings', fontsize=9)
        x_bk = np.sort(ci_90[j]['b_k'].flatten())
        ax[1,1].plot(x_bk, np.arange(1,len(x_bk)+1)/float(len(x_bk)), 
                   color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[1,1].set_title('Group Loadings', fontsize=9)
    for j in range(2):
        for i in range(2):
            ax[j,i].set_xlim(-0.1, 8.1)
            ax[j,i].spines[['right', 'top']].set_visible(False)
            ax[j,i].grid(alpha=0.25)
            ax[j,i].tick_params(axis='both', labelsize=9)
    plt.legend(['LL', 'LH', 'HL', 'HH'], loc='upper center',
               bbox_to_anchor=(-0.1, -0.15), ncol=4, fontsize=9,
               frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.savefig('CI90_F_B.png', bbox_inches='tight')
plot_ci90_fb()

# 1x3 plot of mean PCI90 widths for stochastic volatilities of factors and series
def plot_ci90_sv():
    fig, ax = plt.subplots(1,3, figsize=(6.3,3), dpi=600)
    for j in ['LL', 'LH', 'HL', 'HH']:
        x_hw = np.sort(ci_90[j]['h_w'])
        ax[0].plot(x_hw, np.arange(1,len(x_hw)+1)/float(len(x_hw)), 
                    color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[0].set_title('SV of World Factor', fontsize=9)
        x_hk = np.sort(ci_90[j]['h_K'].flatten())
        ax[1].plot(x_hk, np.arange(1,len(x_hk)+1)/float(len(x_hk)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[1].set_title('SV of Group Factors', fontsize=9)
        x_hi = np.sort(ci_90[j]['h'].flatten())
        ax[2].plot(x_hi, np.arange(1,len(x_hi)+1)/float(len(x_hi)), 
                   color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[2].set_title('SV of Idiosyncratic Components', fontsize=9)
    for j in range(3):
        ax[j].set_xlim(0.6, 1.8)
        ax[j].set_xticks([0.75, 1.25, 1.75])
        ax[j].spines[['right', 'top']].set_visible(False)
        ax[j].grid(alpha=0.25)
        ax[j].tick_params(axis='both', labelsize=9)
    ax[1].legend(['LL', 'LH', 'HL', 'HH'], loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9,
               frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig('CI90_H.png', bbox_inches='tight')
plot_ci90_sv()

# 1x3 plot of mean PCI90 widths for variance decompositions
def plot_ci90_vd():
    fig, ax = plt.subplots(1,3, figsize=(6.3,3), dpi=600)
    for j in ['LL', 'LH', 'HL', 'HH']:
        x_hw = np.sort(ci_90[j]['vd_w'].flatten())
        ax[0].plot(x_hw, np.arange(1,len(x_hw)+1)/float(len(x_hw)), 
                    color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[0].set_title('World', fontsize=9)
        x_hk = np.sort(ci_90[j]['vd_K'].flatten())
        ax[1].plot(x_hk, np.arange(1,len(x_hk)+1)/float(len(x_hk)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[1].set_title('Group', fontsize=9)
        x_hi = np.sort(ci_90[j]['vd_I'].flatten())
        ax[2].plot(x_hi, np.arange(1,len(x_hi)+1)/float(len(x_hi)), 
                   color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[2].set_title('Idiosyncratic', fontsize=9)
    for j in range(3):
        ax[j].set_xlim(-0.05, 1.05)
        ax[j].spines[['right', 'top']].set_visible(False)
        ax[j].grid(alpha=0.25)
        ax[j].tick_params(axis='both', labelsize=9)
    ax[1].legend(['LL', 'LH', 'HL', 'HH'], loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9,
               frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig('CI90_VDS.png', bbox_inches='tight')
plot_ci90_vd()


## Compute Mean Absolute Errors ------------------------------------------------

# empty dictionary for storage
mae = {key: dict.fromkeys(['f_w','f_K','b_w','b_k','h','h_w','h_K',
                           'vd_w','vd_K','vd_I']) 
        for key in ['LL', 'LH', 'HL', 'HH']}

# add empty arrays of appropriate dimensions to dictionary
for key in ['LL', 'LH', 'HL', 'HH']:
    for param in ['f_w','h_w']:
        mae[key][param] = np.zeros(N)
    for param in ['f_K','h_K']:
        mae[key][param] = np.zeros((N,K))
    for param in ['b_w','b_k','h','vd_w','vd_K','vd_I']:
        mae[key][param] = np.zeros((N,n))

# loop to fill dictionary
for j in range(N):

    # get simulated and estimated values for iteration j in all regimes
    LL_j = LL[list(LL.keys())[j]]
    LH_j = LH[list(LH.keys())[j]]
    HL_j = HL[list(HL.keys())[j]]
    HH_j = HH[list(HH.keys())[j]]

    for param in ['f_w','h_w']:
        mae['LL'][param][j] = np.mean(abs(LL_j['res'][param]['mean'] - 
                                          LL_j['data'][param]))
        mae['LH'][param][j] = np.mean(abs(LH_j['res'][param]['mean'] - 
                                          LH_j['data'][param]))
        mae['HL'][param][j] = np.mean(abs(HL_j['res'][param]['mean'] - 
                                          HL_j['data'][param]))
        mae['HH'][param][j] = np.mean(abs(HH_j['res'][param]['mean'] - 
                                          HH_j['data'][param]))

    for param in ['f_K','h_K']:
        for k in range(K):
            mae['LL'][param][j,k] = np.mean(abs(LL_j['res'][param]['mean'][:,k] - 
                                                LL_j['data'][param][:,k]))
            mae['LH'][param][j,k] = np.mean(abs(LH_j['res'][param]['mean'][:,k] - 
                                                LH_j['data'][param][:,k]))
            mae['HL'][param][j,k] = np.mean(abs(HL_j['res'][param]['mean'][:,k] - 
                                                HL_j['data'][param][:,k]))
            mae['HH'][param][j,k] = np.mean(abs(HH_j['res'][param]['mean'][:,k] - 
                                                HH_j['data'][param][:,k]))

    for param in ['b_w','b_k','h','vd_w','vd_K','vd_I']:
        for i in range(n):
            mae['LL'][param][j,i] = np.mean(abs(LL_j['res'][param]['mean'][:,i] - 
                                                LL_j['data'][param][:,i]))
            mae['LH'][param][j,i] = np.mean(abs(LH_j['res'][param]['mean'][:,i] - 
                                                LH_j['data'][param][:,i]))
            mae['HL'][param][j,i] = np.mean(abs(HL_j['res'][param]['mean'][:,i] - 
                                                HL_j['data'][param][:,i]))
            mae['HH'][param][j,i] = np.mean(abs(HH_j['res'][param]['mean'][:,i] - 
                                                HH_j['data'][param][:,i]))

# compute mean correlations
mean_mae = pd.DataFrame(columns=['LL', 'LH', 'HL', 'HH', 'ALL'], 
                          index=['f_w','f_K','b_w','b_k','h','h_w',
                                 'h_K','vd_w','vd_K','vd_I'])
for param in mean_mae.index:
    for j in mean_mae.columns[:-1]:
        mean_mae.loc[param,j] = np.mean(mae[j][param])
    mean_mae.loc[param,'ALL'] = np.mean(np.concatenate((mae['LL'][param], 
                                                        mae['LH'][param], 
                                                        mae['HL'][param], 
                                                        mae['HH'][param])))

# compute median correlations
median_mae = pd.DataFrame(columns=['LL', 'LH', 'HL', 'HH', 'ALL'], 
                            index=['f_w','f_K','b_w','b_k','h','h_w',
                                   'h_K','vd_w','vd_K','vd_I'])
for param in median_mae.index:
    for j in median_mae.columns[:-1]:
        median_mae.loc[param,j] = np.median(mae[j][param])
    median_mae.loc[param,'ALL'] = np.median(np.concatenate((mae['LL'][param], 
                                                            mae['LH'][param], 
                                                            mae['HL'][param], 
                                                            mae['HH'][param])))

# inspect means and medians
mean_mae.astype(float).round(2).transpose()
median_mae.astype(float).round(2).transpose()


## Plot Mean Absolute Errors ---------------------------------------------------

# 1x3 plot of mean MAEs for variance decompositions
def plot_maes_vd():
    fig, ax = plt.subplots(1,3, figsize=(6.3,3), dpi=600)
    for j in ['LL', 'LH', 'HL', 'HH']:
        x_hw = np.sort(mae[j]['vd_w'].flatten())
        ax[0].plot(x_hw, np.arange(1,len(x_hw)+1)/float(len(x_hw)), 
                    color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[0].set_title('World', fontsize=9)
        x_hk = np.sort(mae[j]['vd_K'].flatten())
        ax[1].plot(x_hk, np.arange(1,len(x_hk)+1)/float(len(x_hk)), 
                     color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[1].set_title('Group', fontsize=9)
        x_hi = np.sort(mae[j]['vd_I'].flatten())
        ax[2].plot(x_hi, np.arange(1,len(x_hi)+1)/float(len(x_hi)), 
                   color=lcols[j], ls=ltypes[j], linewidth=1)
        ax[2].set_title('Idiosyncratic', fontsize=9)
    for j in range(3):
        ax[j].set_xlim(-0.05, 0.55)
        ax[j].spines[['right', 'top']].set_visible(False)
        ax[j].grid(alpha=0.25)
        ax[j].tick_params(axis='both', labelsize=9)
    ax[1].legend(['LL', 'LH', 'HL', 'HH'], loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9,
               frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig('MAE_VDS.png', bbox_inches='tight')
plot_maes_vd()


## Summarize Computation Times -------------------------------------------------

# dictionary for storage
time_needed = {key: np.zeros(N) for key in ['LL', 'LH', 'HL', 'HH']}

# loop to fill dictionary
for j in range(N):

    # get time needed for iteration j in all regimes
    time_needed['LL'][j] = LL[list(LL.keys())[j]]['res']['time']
    time_needed['LH'][j] = LH[list(LH.keys())[j]]['res']['time']
    time_needed['HL'][j] = HL[list(HL.keys())[j]]['res']['time']
    time_needed['HH'][j] = HH[list(HH.keys())[j]]['res']['time']

# empty dataframe to store summary statistics of estimation times
time_summary = pd.DataFrame(columns = ['LL', 'LH', 'HL', 'HH'],
                            index = ['mean', 'median', 'std', 'min', 
                                     '10', '25', '50', '75', '90', 'max'])

# loop to fill df
for j in time_summary.columns:

    time_summary.loc['mean',j] = str(datetime.timedelta(seconds=round(np.mean(time_needed[j]))))
    time_summary.loc['median',j] = str(datetime.timedelta(seconds=round(np.median(time_needed[j]))))
    time_summary.loc['std',j] = str(datetime.timedelta(seconds=round(np.std(time_needed[j]))))
    time_summary.loc['min',j] = str(datetime.timedelta(seconds=round(np.min(time_needed[j]))))
    time_summary.loc['10',j] = str(datetime.timedelta(seconds=round(np.percentile(time_needed[j], 10))))
    time_summary.loc['25',j] = str(datetime.timedelta(seconds=round(np.percentile(time_needed[j], 25))))
    time_summary.loc['50',j] = str(datetime.timedelta(seconds=round(np.percentile(time_needed[j], 50))))
    time_summary.loc['75',j] = str(datetime.timedelta(seconds=round(np.percentile(time_needed[j], 75))))
    time_summary.loc['90',j] = str(datetime.timedelta(seconds=round(np.percentile(time_needed[j], 90))))
    time_summary.loc['max',j] = str(datetime.timedelta(seconds=round(np.max(time_needed[j]))))

# inspect summary statistics
time_summary
