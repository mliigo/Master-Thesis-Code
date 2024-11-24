"""
Evaluation of the Application Results

This file evaluates the results obtained in the file "App_Estimation.py".
It plots the mean factors, loadings, stochastic volatilities and variance 
decompositions. It also creates tables of summary statistics of variance
decompositions across countries by group.

Note: Functions are used to generate and directly save plots as PNG files.
      (often relatively large in size and hence not directly displayable)

Imports:
--------
* numpy
* pandas
* matplotlib.pyplot
* matplotlib.lines
"""

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# change plot font globally to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


## Import Estimation Results, Adjust Names and Define Groups -------------------

# import estimation data (posterior draws)
est_data = pd.read_pickle('Application_Trace_last_10k.pkl')

# get hyperparams
n = len(est_data['a'][0,:])      # number of series i
K = len(est_data['f_K'][0,0,:])  # number of groups k
T = len(est_data['f_K'][0,:,0])  # number of time periods t

# define time span to use on x-axis
time_span = pd.date_range(start='1970-06-30', end='2023-09-30', freq='Q')

# change some country names to shorter versions (for nicer plots)
name_changes = {'South Korea': 'Korea, Rep.', 'Egypt': 'Egypt, Arab Rep.', 
                'Ivory Coast': "CÃ´te d'Ivoire", 'Gambia': 'Gambia, The', 
                'Tanzania': 'Tanzania, United Rep.'}
rev_subs = { v:k for k, v in name_changes.items()}
est_data['names'] = [rev_subs.get(item, item)  for item in est_data['names']]

# Advanced Economies (28)
aes = ['Australia', 'Austria', 'Belgium', 'Canada', 'Cyprus', 'Denmark', 
       'Finland', 'France', 'Germany', 'Greece', 'Iceland', 'Ireland', 
       'Israel', 'Italy', 'Japan', 'South Korea', 'Luxembourg', 'Malta', 
       'Netherlands', 'New Zealand', 'Norway', 'Portugal', 'Singapore',
       'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'United States']

# Emerging Market and Middle Income Economies (30)
emes = ['Argentina', 'Bahamas', 'Bolivia', 'Chile', 'Colombia', 
        'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Fiji', 
        'Gabon', 'Guatemala', 'India', 'Indonesia', 'Jamaica', 'Malaysia', 
        'Mauritius', 'Mexico', 'Morocco', 'Pakistan', 'Panama', 'Paraguay', 
        'Peru', 'Philippines', 'Samoa', 'South Africa', 'Sri Lanka', 
        'Trinidad and Tobago', 'Türkiye', 'Uruguay']

# Low Income Developing Countries (13)
lics = ['Burkina Faso', 'Burundi', 'Cameroon', 'Ivory Coast', 'Gambia', 
        'Ghana', 'Haiti', 'Honduras', 'Nepal', 'Niger', 'Nigeria', 'Senegal', 
        'Tanzania']


## Plot Factors ----------------------------------------------------------------

# 2x2 plot with mean factors
def mean_fac_plotter():
    fig, ax = plt.subplots(2,2,figsize=(6.3,4.5), dpi=600)
    ax[0,0].plot(time_span, np.mean(est_data['f_w'][:,:],0), color='black', 
                 linewidth=1)
    ax[0,0].set_title('World Factor', fontsize=10)
    ax[0,1].plot(time_span, np.mean(est_data['f_K'][:,:,0],0), color='black', 
                 linewidth=1)
    ax[0,1].set_title('AE Factor', fontsize=10)
    ax[1,0].plot(time_span, np.mean(est_data['f_K'][:,:,1],0), color='black', 
                 linewidth=1)
    ax[1,0].set_title('EME Factor', fontsize=10)
    ax[1,1].plot(time_span, np.mean(est_data['f_K'][:,:,2],0), color='black', 
                 linewidth=1)
    ax[1,1].set_title('LIDC Factor', fontsize=10)
    for j in range(2):
        for i in range(2):
            ax[i,j].axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax[i,j].tick_params(axis='both', labelsize=8)
            ax[i,j].spines[['right', 'top']].set_visible(False)
            ax[i,j].grid(alpha=0.25)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.savefig('FACS_MEAN.png', bbox_inches='tight')
    plt.close()
mean_fac_plotter()


## Plot Loadings ---------------------------------------------------------------

# compute mean world and group loadings of each country
bw_means = np.zeros((T,n))
bk_means = np.zeros((T,n))
for i in range(n):
    bw_means[:,i] = np.mean(est_data['b_w'][:,:,i],0)
    bk_means[:,i] = np.mean(est_data['b_k'][:,:,i],0)

# compute fraction of positive mean world and group loadings by group
bw_pos = np.zeros((T,3))
bk_pos = np.zeros((T,3))
bw_pos[:,0] = np.round(np.sum(bw_means[:,:len(aes)]>0, 1)/
                       len(aes), 2)
bw_pos[:,1] = np.round(np.sum(bw_means[:,len(aes):len(aes)+len(emes)]>0, 1)/
                       len(emes), 2)
bw_pos[:,2] = np.round(np.sum(bw_means[:,len(aes)+len(emes):]>0, 1)/
                       len(lics), 2)
bk_pos[:,0] = np.round(np.sum(bk_means[:,:len(aes)]>0, 1)/
                       len(aes), 2)
bk_pos[:,1] = np.round(np.sum(bk_means[:,len(aes):len(aes)+len(emes)]>0, 1)/
                       len(emes), 2)
bk_pos[:,2] = np.round(np.sum(bk_means[:,len(aes)+len(emes):]>0, 1)/
                       len(lics), 2)

# 2x1 plot with fraction of positive mean world and group loadings by group
def bwk_pos_plotter():
    fig, ax = plt.subplots(2,1,figsize=(6.3,4.1), dpi=600, sharey=True)
    ax[0].plot(time_span, bw_pos[:,0], color='black', linewidth=1)
    ax[0].plot(time_span, bw_pos[:,1], color='black', linewidth=1, ls='--')
    ax[0].plot(time_span, bw_pos[:,2], color='black', linewidth=1, ls=':')
    ax[0].set_title('World Loadings', fontsize=10)
    ax[1].plot(time_span, bk_pos[:,0], color='black', linewidth=1)
    ax[1].plot(time_span, bk_pos[:,1], color='black', linewidth=1, ls='--')
    ax[1].plot(time_span, bk_pos[:,2], color='black', linewidth=1, ls=':')
    ax[1].set_title('Group Loadings', fontsize=10)
    for j in range(2):
        ax[j].tick_params(axis='both', labelsize=8)
        ax[j].spines[['right', 'top']].set_visible(False)
        ax[j].grid(alpha=0.25)
    ax[1].legend(['AEs', 'EMEs', 'LIDCs'], loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8,
               frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.savefig('B_WK_POS.png', bbox_inches='tight')
    plt.close()
bwk_pos_plotter()

# 7x4 plot of mean world loadings of all AEs (with 68% PCI)
def bw_i_AE_plotter():
    fig, ax = plt.subplots(7,4, figsize=(6.3,7.6), dpi=600, sharex=True)
    series = 0
    for i in range(7):
        for j in range(4):
            ax[i,j].plot(time_span, np.mean(est_data['b_w'][:,:,series],0), 
                         color='black')
            ax[i,j].fill_between(time_span,
                                np.percentile(est_data['b_w'][:,:,series],84,0),
                                np.percentile(est_data['b_w'][:,:,series],16,0),
                                alpha=0.25, color='gray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    plt.tight_layout()
    plt.savefig('BW_i_AE.png', bbox_inches='tight')
    plt.close()
bw_i_AE_plotter()

# 7x4 plot of mean group loadings of all AEs (with 68% PCI)
def bk_i_AE_plotter():
    fig, ax = plt.subplots(7,4, figsize=(6.3,7.6), dpi=600, sharex=True)
    series = 0
    for i in range(7):
        for j in range(4):
            ax[i,j].plot(time_span, np.mean(est_data['b_k'][:,:,series],0), 
                         color='black')
            ax[i,j].fill_between(time_span,
                                np.percentile(est_data['b_k'][:,:,series],84,0),
                                np.percentile(est_data['b_k'][:,:,series],16,0),
                                alpha=0.25, color='gray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    plt.tight_layout()
    plt.savefig('BK_i_AE.png', bbox_inches='tight')
    plt.close()
bk_i_AE_plotter()

# 8x4 plot of mean world loadings of all EMEs (with 68% PCI)
def bw_i_EME_plotter():
    fig, ax = plt.subplots(8,4, figsize=(6.3,8.8), dpi=600, sharex=True)
    series = 28
    for i in range(7):
        for j in range(4):
            ax[i,j].plot(time_span, np.mean(est_data['b_w'][:,:,series],0), 
                         color='black')
            ax[i,j].fill_between(time_span,
                                np.percentile(est_data['b_w'][:,:,series],84,0),
                                np.percentile(est_data['b_w'][:,:,series],16,0),
                                alpha=0.25, color='gray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    for j in range(2):
        ax[7,j].plot(time_span, np.mean(est_data['b_w'][:,:,series],0), 
                     color='black')
        ax[7,j].fill_between(time_span,
                            np.percentile(est_data['b_w'][:,:,series],84,0),
                            np.percentile(est_data['b_w'][:,:,series],16,0),
                            alpha=0.25, color='gray', edgecolor='none')
        ax[7,j].set_title(est_data['names'][series], fontsize=10)
        ax[7,j].set_xlim(time_span[0], time_span[-1])
        ax[7,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
        ax[7,j].set_xticklabels(['1970','1995','2020'])
        ax[7,j].tick_params(axis='both', labelsize=8)
        series = series + 1
    fig.delaxes(ax[7,2])
    fig.delaxes(ax[7,3])
    plt.tight_layout()
    plt.savefig('BW_i_EME.png', bbox_inches='tight')
    plt.close()
bw_i_EME_plotter()

# 8x4 plot of mean group loadings of all EMEs (with 68% PCI)
def bk_i_EME_plotter():
    fig, ax = plt.subplots(8,4, figsize=(6.3,8.8), dpi=600, sharex=True)
    series = 28
    for i in range(7):
        for j in range(4):
            ax[i,j].plot(time_span, np.mean(est_data['b_k'][:,:,series],0), 
                         color='black')
            ax[i,j].fill_between(time_span,
                                np.percentile(est_data['b_k'][:,:,series],84,0),
                                np.percentile(est_data['b_k'][:,:,series],16,0),
                                alpha=0.25, color='gray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    for j in range(2):
        ax[7,j].plot(time_span, np.mean(est_data['b_k'][:,:,series],0), 
                     color='black')
        ax[7,j].fill_between(time_span,
                            np.percentile(est_data['b_k'][:,:,series],84,0),
                            np.percentile(est_data['b_k'][:,:,series],16,0),
                            alpha=0.25, color='gray', edgecolor='none')
        ax[7,j].set_title(est_data['names'][series], fontsize=10)
        ax[7,j].set_xlim(time_span[0], time_span[-1])
        ax[7,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
        ax[7,j].set_xticklabels(['1970','1995','2020'])
        ax[7,j].tick_params(axis='both', labelsize=8)
        series = series + 1
    fig.delaxes(ax[7,2])
    fig.delaxes(ax[7,3])
    plt.tight_layout()
    plt.savefig('BK_i_EME.png', bbox_inches='tight')
    plt.close()
bk_i_EME_plotter()

# 4x4 plot of mean world loadings of all LIDCs (with 68% PCI)
def bw_i_LIDC_plotter():
    fig, ax = plt.subplots(4,4, figsize=(6.3,4.4), dpi=600, sharex=True)
    series = 58
    for i in range(3):
        for j in range(4):
            ax[i,j].plot(time_span, np.mean(est_data['b_w'][:,:,series],0), 
                         color='black')
            ax[i,j].fill_between(time_span,
                                np.percentile(est_data['b_w'][:,:,series],84,0),
                                np.percentile(est_data['b_w'][:,:,series],16,0),
                                alpha=0.25, color='gray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    for j in range(1):
        ax[3,j].plot(time_span, np.mean(est_data['b_w'][:,:,series],0), 
                     color='black')
        ax[3,j].fill_between(time_span,
                            np.percentile(est_data['b_w'][:,:,series],84,0),
                            np.percentile(est_data['b_w'][:,:,series],16,0),
                            alpha=0.25, color='gray', edgecolor='none')
        ax[3,j].set_title(est_data['names'][series], fontsize=10)
        ax[3,j].set_xlim(time_span[0], time_span[-1])
        ax[3,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
        ax[3,j].set_xticklabels(['1970','1995','2020'])
        ax[3,j].tick_params(axis='both', labelsize=8)
        series = series + 1
    fig.delaxes(ax[3,1])
    fig.delaxes(ax[3,2])
    fig.delaxes(ax[3,3])
    plt.tight_layout()
    plt.savefig('BW_i_LIDC.png', bbox_inches='tight')
    plt.close()
bw_i_LIDC_plotter()

# 4x4 plot of mean group loadings of all LIDCs (with 68% PCI)
def bk_i_LIDC_plotter():
    fig, ax = plt.subplots(4,4, figsize=(6.3,4.4), dpi=600, sharex=True)
    series = 58
    for i in range(3):
        for j in range(4):
            ax[i,j].plot(time_span, np.mean(est_data['b_k'][:,:,series],0), 
                         color='black')
            ax[i,j].fill_between(time_span,
                                np.percentile(est_data['b_k'][:,:,series],84,0),
                                np.percentile(est_data['b_k'][:,:,series],16,0),
                                alpha=0.25, color='gray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    for j in range(1):
        ax[3,j].plot(time_span, np.mean(est_data['b_k'][:,:,series],0), 
                     color='black')
        ax[3,j].fill_between(time_span,
                            np.percentile(est_data['b_k'][:,:,series],84,0),
                            np.percentile(est_data['b_k'][:,:,series],16,0),
                            alpha=0.25, color='gray', edgecolor='none')
        ax[3,j].set_title(est_data['names'][series], fontsize=10)
        ax[3,j].set_xlim(time_span[0], time_span[-1])
        ax[3,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
        ax[3,j].set_xticklabels(['1970','1995','2020'])
        ax[3,j].tick_params(axis='both', labelsize=8)
        series = series + 1
    fig.delaxes(ax[3,1])
    fig.delaxes(ax[3,2])
    fig.delaxes(ax[3,3])
    plt.tight_layout()
    plt.savefig('BK_i_LIDC.png', bbox_inches='tight')
    plt.close()
bk_i_LIDC_plotter()


### Plot Stochastic Volatilities -----------------------------------------------

# 2x2 plot of mean innovation std of factors (with 68% PCI)
def seh_f_plotter():
    f_names = ['World Factor', 'AE Factor', 'EME Factor', 'LIDC Factor']
    fig, ax = plt.subplots(2,2, figsize=(6.3,3.3), dpi=600, sharex=True, sharey=True)
    ax[0,0].plot(time_span, np.mean(np.sqrt(est_data['s2_0'])*np.exp(est_data['h_w'][:,:]),0), 
                 color='black')
    ax[0,0].fill_between(time_span,
                         np.percentile(np.sqrt(est_data['s2_0'])*np.exp(est_data['h_w'][:,:]),84,0),
                         np.percentile(np.sqrt(est_data['s2_0'])*np.exp(est_data['h_w'][:,:]),16,0),
                         alpha=0.25, color='gray', edgecolor='none')
    ax[0,0].set_xlim(time_span[0], time_span[-1])
    ax[0,0].set_title(f_names[0], fontsize=10)
    ax[0,0].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
    ax[0,0].set_xticklabels(['1970','1995','2020'])
    ax[0,0].tick_params(axis='both', labelsize=8)
    ax[0,1].plot(time_span, np.mean(np.sqrt(est_data['s2_0'])*np.exp(est_data['h_K'][:,:,0]),0), 
                 color='black')
    ax[0,1].fill_between(time_span,
                         np.percentile(np.sqrt(est_data['s2_0'])*np.exp(est_data['h_K'][:,:,0]),84,0),
                         np.percentile(np.sqrt(est_data['s2_0'])*np.exp(est_data['h_K'][:,:,0]),16,0),
                         alpha=0.25, color='gray', edgecolor='none')
    ax[0,1].set_xlim(time_span[0], time_span[-1])
    ax[0,1].set_title(f_names[1], fontsize=10)
    ax[0,1].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
    ax[0,1].set_xticklabels(['1970','1995','2020'])
    ax[0,1].tick_params(axis='both', labelsize=8)
    for j in range(2):
        ax[1,j].plot(time_span, np.mean(np.sqrt(est_data['s2_0'])*np.exp(est_data['h_K'][:,:,j+1]),0), 
                     color='black')
        ax[1,j].fill_between(time_span,
                             np.percentile(np.sqrt(est_data['s2_0'])*np.exp(est_data['h_K'][:,:,j+1]),84,0),
                             np.percentile(np.sqrt(est_data['s2_0'])*np.exp(est_data['h_K'][:,:,j+1]),16,0),
                             alpha=0.25, color='gray', edgecolor='none')
        ax[1,j].set_xlim(time_span[0], time_span[-1])
        ax[1,j].set_title(f_names[j+2], fontsize=10)
        ax[1,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
        ax[1,j].set_xticklabels(['1970','1995','2020'])
        ax[1,j].tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.savefig('SEH_FACS.png', bbox_inches='tight')
    plt.close()
seh_f_plotter()

# 7x4 plot of mean idio innovation std of all AEs (with 68% PCI)
def seh_i_AE_plotter():
    fig, ax = plt.subplots(7,4, figsize=(6.3,7.6), dpi=600, sharex=True)
    series = 0
    for i in range(7):
        for j in range(4):
            y = np.sqrt(est_data['s2'][:,series])[:,None]*np.exp(est_data['h'][:,:,series])
            ax[i,j].plot(time_span, np.mean(y,0), color='black')
            ax[i,j].fill_between(time_span,
                                 np.percentile(y,84,0),
                                 np.percentile(y,16,0),
                                 alpha=0.25, color='gray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    plt.tight_layout()
    plt.savefig('SEH_i_AE.png', bbox_inches='tight')
    plt.close()
seh_i_AE_plotter()

# 8x4 plot of mean idio innovation std of all EMEs (with 68% PCI)
def seh_i_EME_plotter():
    fig, ax = plt.subplots(8,4, figsize=(6.3,8.8), dpi=600, sharex=True)
    series = 28
    for i in range(7):
        for j in range(4):
            y = np.sqrt(est_data['s2'][:,series])[:,None]*np.exp(est_data['h'][:,:,series])
            ax[i,j].plot(time_span, np.mean(y,0), color='black')
            ax[i,j].fill_between(time_span,
                                 np.percentile(y,84,0),
                                 np.percentile(y,16,0),
                                 alpha=0.25, color='gray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    for j in range(2):
        y = np.sqrt(est_data['s2'][:,series])[:,None]*np.exp(est_data['h'][:,:,series])
        ax[7,j].plot(time_span, np.mean(y,0), color='black')
        ax[7,j].fill_between(time_span,
                             np.percentile(y,84,0),
                             np.percentile(y,16,0),
                             alpha=0.25, color='gray', edgecolor='none')
        ax[7,j].set_title(est_data['names'][series], fontsize=10)
        ax[7,j].set_xlim(time_span[0], time_span[-1])
        ax[7,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
        ax[7,j].set_xticklabels(['1970','1995','2020'])
        ax[7,j].tick_params(axis='both', labelsize=8)
        series = series + 1
    fig.delaxes(ax[7,2])
    fig.delaxes(ax[7,3])
    plt.tight_layout()
    plt.savefig('SEH_i_EME.png', bbox_inches='tight')
    plt.close()
seh_i_EME_plotter()

# 4x4 plot of mean idio innovation std of all LIDCs (with 68% PCI)
def seh_i_LIDC_plotter():
    fig, ax = plt.subplots(4,4, figsize=(6.3,4.4), dpi=600, sharex=True)
    series = 58
    for i in range(3):
        for j in range(4):
            y = np.sqrt(est_data['s2'][:,series])[:,None]*np.exp(est_data['h'][:,:,series])
            ax[i,j].plot(time_span, np.mean(y,0), color='black')
            ax[i,j].fill_between(time_span,
                                 np.percentile(y,84,0),
                                 np.percentile(y,16,0),
                                 alpha=0.25, color='gray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    for j in range(1):
        y = np.sqrt(est_data['s2'][:,series])[:,None]*np.exp(est_data['h'][:,:,series])
        ax[3,j].plot(time_span, np.mean(y,0), color='black')
        ax[3,j].fill_between(time_span,
                             np.percentile(y,84,0),
                             np.percentile(y,16,0),
                             alpha=0.25, color='gray', edgecolor='none')
        ax[3,j].set_title(est_data['names'][series], fontsize=10)
        ax[3,j].set_xlim(time_span[0], time_span[-1])
        ax[3,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
        ax[3,j].set_xticklabels(['1970','1995','2020'])
        ax[3,j].tick_params(axis='both', labelsize=8)
        series = series + 1
    fig.delaxes(ax[3,1])
    fig.delaxes(ax[3,2])
    fig.delaxes(ax[3,3])
    plt.tight_layout()
    plt.savefig('SEH_i_LIDC.png', bbox_inches='tight')
    plt.close()
seh_i_LIDC_plotter()


## Plot Variance Decompositions ------------------------------------------------

# 1x3 plot of mean VDs by group
def VD_g_plotter():
    x = [0,28,58]
    y = [28,58,71]
    names = ['AEs', 'EMEs', 'LIDCs']
    fig, ax = plt.subplots(1,3,figsize=(6.3,2.1), dpi=600, sharey=True)
    for j in range(3):
        ax[j].fill_between(time_span,
                        np.zeros(T),
                        np.mean(np.mean(est_data['vd_w'][:,:,x[j]:y[j]],0),1),
                        alpha=1, color='black', edgecolor='none')
        ax[j].fill_between(time_span,
                        np.mean(np.mean(est_data['vd_w'][:,:,x[j]:y[j]],0),1),
                        (np.mean(np.mean(est_data['vd_w'][:,:,x[j]:y[j]],0),1)+
                         np.mean(np.mean(est_data['vd_K'][:,:,x[j]:y[j]],0),1)),
                        alpha=1, color='gray', edgecolor='none')
        ax[j].fill_between(time_span,
                        (np.mean(np.mean(est_data['vd_w'][:,:,x[j]:y[j]],0),1)+
                         np.mean(np.mean(est_data['vd_K'][:,:,x[j]:y[j]],0),1)),
                        (np.mean(np.mean(est_data['vd_w'][:,:,x[j]:y[j]],0),1)+
                         np.mean(np.mean(est_data['vd_K'][:,:,x[j]:y[j]],0),1)+
                         np.mean(np.mean(est_data['vd_I'][:,:,x[j]:y[j]],0),1)),
                        alpha=1, color='lightgray', edgecolor='none')
        ax[j].set_title(names[j], fontsize=10)
        ax[j].set_ylim(0,1)
        ax[j].set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax[j].set_xlim(time_span[0], time_span[-1])
        ax[j].set_xticks(['1975-06-30','1990-06-30','2005-06-30','2020-06-30'])
        ax[j].set_xticklabels(['1975','1990','2005','2020'])
        ax[j].tick_params(axis='both', labelsize=8)
    custom_lines = [Line2D([0], [0], color='black', lw=4),
                    Line2D([0], [0], color='gray', lw=4),
                    Line2D([0], [0], color='lightgray', lw=4)]
    fig.legend(custom_lines, ['World Component', 'Group Component', 'Idiosyncratic Component'], 
                 loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=8, 
                 frameon=False)
    plt.tight_layout()
    plt.savefig('VD_g.png', bbox_inches='tight')
    plt.close()
VD_g_plotter()

# 4x3 plot of 10th, 50th and 90th percentile of VDs by group
def VD_g_4x3_plotter():
    x = [0,0,28,58]
    y = [71,28,58,71]
    names_i = ['All', 'AEs', 'EMEs', 'LIDCs']
    names_j = ['World', 'Group', 'Idiosyncratic']
    comp = ['vd_w', 'vd_K', 'vd_I']
    fig, ax = plt.subplots(4,3,figsize=(6.3,7.5), dpi=600, sharex=True, sharey=True)
    for i in range(4):
        for j in range(3):
            ax[i,j].plot(time_span, np.mean(np.mean(est_data[comp[j]][:,:,x[i]:y[i]],0),1),
                         color='black')
            ax[i,j].fill_between(time_span,
                                 np.percentile(np.mean(est_data[comp[j]][:,:,x[i]:y[i]],0),10,1),
                                 np.percentile(np.mean(est_data[comp[j]][:,:,x[i]:y[i]],0),90,1),
                                 alpha=0.5, color='gray', edgecolor='none')
            ax[i,j].set_title('{0} {1}'.format(names_i[i], names_j[j]), fontsize=10)
            ax[i,j].set_ylim(0,1)
            ax[i,j].set_yticks([0,0.25,0.5,0.75,1])
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_xticks(['1975-06-30','1990-06-30','2005-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1975','1990','2005','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.savefig('VD_g_4x3.png', bbox_inches='tight')
    plt.close()
VD_g_4x3_plotter()

# 7x4 plot of mean VDs of all AEs (with 68% PCI)
def vd_i_AE_plotter():
    fig, ax = plt.subplots(7,4, figsize=(6.3,7.6), dpi=600, sharex=True, sharey=True)
    series = 0
    for i in range(7):
        for j in range(4):
            ax[i,j].fill_between(time_span, np.zeros(T),
                                 np.mean(est_data['vd_w'][:,:,series],0),
                                 alpha=1, color='black', edgecolor='none')
            ax[i,j].fill_between(time_span,
                                 np.mean(est_data['vd_w'][:,:,series],0),
                                (np.mean(est_data['vd_w'][:,:,series],0)+
                                 np.mean(est_data['vd_K'][:,:,series],0)),
                                 alpha=1, color='gray', edgecolor='none')
            ax[i,j].fill_between(time_span,
                                (np.mean(est_data['vd_w'][:,:,series],0)+
                                 np.mean(est_data['vd_K'][:,:,series],0)),
                                (np.mean(est_data['vd_w'][:,:,series],0)+
                                 np.mean(est_data['vd_K'][:,:,series],0)+
                                 np.mean(est_data['vd_I'][:,:,series],0)),
                                alpha=1, color='lightgray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].tick_params(axis='both', labelsize=8)
            ax[i,j].set_ylim(0,1)
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            series = series + 1
    plt.tight_layout()
    plt.savefig('VD_i_AE.png', bbox_inches='tight')
    plt.close()
vd_i_AE_plotter()

# 8x4 plot of mean VDs of all EMEs (with 68% PCI)
def vd_i_EME_plotter():
    fig, ax = plt.subplots(8,4, figsize=(6.3,8.8), dpi=600, sharex=True, sharey=True)
    series = 28
    for i in range(7):
        for j in range(4):
            ax[i,j].fill_between(time_span, np.zeros(T),
                                 np.mean(est_data['vd_w'][:,:,series],0),
                                 alpha=1, color='black', edgecolor='none')
            ax[i,j].fill_between(time_span,
                                 np.mean(est_data['vd_w'][:,:,series],0),
                                (np.mean(est_data['vd_w'][:,:,series],0)+
                                 np.mean(est_data['vd_K'][:,:,series],0)),
                                 alpha=1, color='gray', edgecolor='none')
            ax[i,j].fill_between(time_span,
                                (np.mean(est_data['vd_w'][:,:,series],0)+
                                 np.mean(est_data['vd_K'][:,:,series],0)),
                                (np.mean(est_data['vd_w'][:,:,series],0)+
                                 np.mean(est_data['vd_K'][:,:,series],0)+
                                 np.mean(est_data['vd_I'][:,:,series],0)),
                                alpha=1, color='lightgray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_ylim(0,1)
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    for j in range(2):
        ax[7,j].fill_between(time_span, np.zeros(T),
                            np.mean(est_data['vd_w'][:,:,series],0),
                            alpha=1, color='black', edgecolor='none')
        ax[7,j].fill_between(time_span,
                            np.mean(est_data['vd_w'][:,:,series],0),
                            (np.mean(est_data['vd_w'][:,:,series],0)+
                             np.mean(est_data['vd_K'][:,:,series],0)),
                             alpha=1, color='gray', edgecolor='none')
        ax[7,j].fill_between(time_span,
                            (np.mean(est_data['vd_w'][:,:,series],0)+
                             np.mean(est_data['vd_K'][:,:,series],0)),
                            (np.mean(est_data['vd_w'][:,:,series],0)+
                             np.mean(est_data['vd_K'][:,:,series],0)+
                             np.mean(est_data['vd_I'][:,:,series],0)),
                             alpha=1, color='lightgray', edgecolor='none')
        ax[7,j].set_title(est_data['names'][series], fontsize=10)
        ax[7,j].set_xlim(time_span[0], time_span[-1])
        ax[7,j].set_ylim(0,1)
        ax[7,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
        ax[7,j].set_xticklabels(['1970','1995','2020'])
        ax[7,j].tick_params(axis='both', labelsize=8)
        series = series + 1
    fig.delaxes(ax[7,2])
    fig.delaxes(ax[7,3])
    plt.tight_layout()
    plt.savefig('VD_i_EME.png', bbox_inches='tight')
    plt.close()
vd_i_EME_plotter()

# 4x4 plot of mean VDs of all LIDCs (with 68% PCI)
def vd_i_LIDC_plotter():
    fig, ax = plt.subplots(4,4, figsize=(6.3,4.4), dpi=600, sharex=True, sharey=True)
    series = 58
    for i in range(3):
        for j in range(4):
            ax[i,j].fill_between(time_span, np.zeros(T),
                                 np.mean(est_data['vd_w'][:,:,series],0),
                                 alpha=1, color='black', edgecolor='none')
            ax[i,j].fill_between(time_span,
                                 np.mean(est_data['vd_w'][:,:,series],0),
                                (np.mean(est_data['vd_w'][:,:,series],0)+
                                 np.mean(est_data['vd_K'][:,:,series],0)),
                                 alpha=1, color='gray', edgecolor='none')
            ax[i,j].fill_between(time_span,
                                (np.mean(est_data['vd_w'][:,:,series],0)+
                                 np.mean(est_data['vd_K'][:,:,series],0)),
                                (np.mean(est_data['vd_w'][:,:,series],0)+
                                 np.mean(est_data['vd_K'][:,:,series],0)+
                                 np.mean(est_data['vd_I'][:,:,series],0)),
                                alpha=1, color='lightgray', edgecolor='none')
            ax[i,j].set_title(est_data['names'][series], fontsize=10)
            ax[i,j].set_xlim(time_span[0], time_span[-1])
            ax[i,j].set_ylim(0,1)
            ax[i,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
            ax[i,j].set_xticklabels(['1970','1995','2020'])
            ax[i,j].tick_params(axis='both', labelsize=8)
            series = series + 1
    for j in range(1):
        ax[3,j].fill_between(time_span, np.zeros(T),
                            np.mean(est_data['vd_w'][:,:,series],0),
                            alpha=1, color='black', edgecolor='none')
        ax[3,j].fill_between(time_span,
                            np.mean(est_data['vd_w'][:,:,series],0),
                            (np.mean(est_data['vd_w'][:,:,series],0)+
                             np.mean(est_data['vd_K'][:,:,series],0)),
                             alpha=1, color='gray', edgecolor='none')
        ax[3,j].fill_between(time_span,
                            (np.mean(est_data['vd_w'][:,:,series],0)+
                             np.mean(est_data['vd_K'][:,:,series],0)),
                            (np.mean(est_data['vd_w'][:,:,series],0)+
                             np.mean(est_data['vd_K'][:,:,series],0)+
                             np.mean(est_data['vd_I'][:,:,series],0)),
                             alpha=1, color='lightgray', edgecolor='none')
        ax[3,j].set_title(est_data['names'][series], fontsize=10)
        ax[3,j].set_xlim(time_span[0], time_span[-1])
        ax[3,j].set_ylim(0,1)
        ax[3,j].set_xticks(['1970-06-30','1995-06-30','2020-06-30'])
        ax[3,j].set_xticklabels(['1970','1995','2020'])
        ax[3,j].tick_params(axis='both', labelsize=8)
        series = series + 1
    fig.delaxes(ax[3,1])
    fig.delaxes(ax[3,2])
    fig.delaxes(ax[3,3])
    plt.tight_layout()
    plt.savefig('VD_i_LIDC.png', bbox_inches='tight')
    plt.close()
vd_i_LIDC_plotter()


### Tables of Variance Decompositions by Group----------------------------------

# define time stamps
time_indices = [20, 80, 140, 200]  # 1975-Q2, 1990-Q2, 2005-Q2, 2020-Q2

# empty dataframes to store results
vd_medians = pd.DataFrame(columns=time_indices, 
                          index=['All_w', 'All_k', 'AE_w', 'AE_k', 'EME_w', 
                                 'EME_k', 'LIDC_w', 'LIDC_k'])
vd_means = pd.DataFrame(columns=time_indices, 
                        index=['All_w', 'All_k', 'AE_w', 'AE_k', 'EME_w', 
                               'EME_k', 'LIDC_w', 'LIDC_k'])
vd_10 = pd.DataFrame(columns=time_indices, 
                     index=['All_w', 'All_k', 'AE_w', 'AE_k', 'EME_w', 
                            'EME_k', 'LIDC_w', 'LIDC_k'])
vd_90 = pd.DataFrame(columns=time_indices,
                     index=['All_w', 'All_k', 'AE_w', 'AE_k', 'EME_w', 
                            'EME_k', 'LIDC_w', 'LIDC_k'])

# fill dataframes
for t in vd_medians.columns:
    # medians
    vd_medians.loc['All_w',t] = np.median(np.mean(est_data['vd_w'][:,t,:],0))
    vd_medians.loc['All_k',t] = np.median(np.mean(est_data['vd_K'][:,t,:],0))
    vd_medians.loc['AE_w',t] = np.median(np.mean(est_data['vd_w'][:,t,:28],0))
    vd_medians.loc['AE_k',t] = np.median(np.mean(est_data['vd_K'][:,t,:28],0))
    vd_medians.loc['EME_w',t] = np.median(np.mean(est_data['vd_w'][:,t,28:58],0))
    vd_medians.loc['EME_k',t] = np.median(np.mean(est_data['vd_K'][:,t,28:58],0))
    vd_medians.loc['LIDC_w',t] = np.median(np.mean(est_data['vd_w'][:,t,58:],0))
    vd_medians.loc['LIDC_k',t] = np.median(np.mean(est_data['vd_K'][:,t,58:],0))
    # means
    vd_means.loc['All_w',t] = np.mean(np.mean(est_data['vd_w'][:,t,:],0))
    vd_means.loc['All_k',t] = np.mean(np.mean(est_data['vd_K'][:,t,:],0))
    vd_means.loc['AE_w',t] = np.mean(np.mean(est_data['vd_w'][:,t,:28],0))
    vd_means.loc['AE_k',t] = np.mean(np.mean(est_data['vd_K'][:,t,:28],0))
    vd_means.loc['EME_w',t] = np.mean(np.mean(est_data['vd_w'][:,t,28:58],0))
    vd_means.loc['EME_k',t] = np.mean(np.mean(est_data['vd_K'][:,t,28:58],0))
    vd_means.loc['LIDC_w',t] = np.mean(np.mean(est_data['vd_w'][:,t,58:],0))
    vd_means.loc['LIDC_k',t] = np.mean(np.mean(est_data['vd_K'][:,t,58:],0))
    # 10th
    vd_10.loc['All_w',t] = np.percentile(np.mean(est_data['vd_w'][:,t,:],0), 10)
    vd_10.loc['All_k',t] = np.percentile(np.mean(est_data['vd_K'][:,t,:],0), 10)
    vd_10.loc['AE_w',t] = np.percentile(np.mean(est_data['vd_w'][:,t,:28],0), 10)
    vd_10.loc['AE_k',t] = np.percentile(np.mean(est_data['vd_K'][:,t,:28],0), 10)
    vd_10.loc['EME_w',t] = np.percentile(np.mean(est_data['vd_w'][:,t,28:58],0), 10)
    vd_10.loc['EME_k',t] = np.percentile(np.mean(est_data['vd_K'][:,t,28:58],0), 10)
    vd_10.loc['LIDC_w',t] = np.percentile(np.mean(est_data['vd_w'][:,t,58:],0), 10)
    vd_10.loc['LIDC_k',t] = np.percentile(np.mean(est_data['vd_K'][:,t,58:],0), 10)
    # 90th
    vd_90.loc['All_w',t] = np.percentile(np.mean(est_data['vd_w'][:,t,:],0), 90)
    vd_90.loc['All_k',t] = np.percentile(np.mean(est_data['vd_K'][:,t,:],0), 90)
    vd_90.loc['AE_w',t] = np.percentile(np.mean(est_data['vd_w'][:,t,:28],0), 90)
    vd_90.loc['AE_k',t] = np.percentile(np.mean(est_data['vd_K'][:,t,:28],0), 90)
    vd_90.loc['EME_w',t] = np.percentile(np.mean(est_data['vd_w'][:,t,28:58],0), 90)
    vd_90.loc['EME_k',t] = np.percentile(np.mean(est_data['vd_K'][:,t,28:58],0), 90)
    vd_90.loc['LIDC_w',t] = np.percentile(np.mean(est_data['vd_w'][:,t,58:],0), 90)
    vd_90.loc['LIDC_k',t] = np.percentile(np.mean(est_data['vd_K'][:,t,58:],0), 90)

# print rounded values
(vd_medians*100).astype(float).round(1)
(vd_means*100).astype(float).round(1)
(vd_10*100).astype(float).round(1)
(vd_90*100).astype(float).round(1)

