"""
Inflation Data Analysis

This file analyses the dataset used for the application in the file 
"App_Estimation.py". It computes summary statistics and creates plots for 
quarterly inflation rates of 71 countries spanning the period 1970-Q2 – 2023-Q3.
Data is taken from the World Bank's global inflation database (Ha et al., 2023).

Imports:
--------
* pandas
* numpy
* matplotlib.pyplot
* matplotlib.ticker

References:
-----------
Ha, J., Kose, M. A. & Ohnsorge, F. (2023). One-stop Source: A Global Database 
of Inflation. Journal of International Money and Finance, 137 (October), 102896.
DOI: https://doi.org/10.1016/j.jimonfin.2023.102896
"""

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# change plot font globally to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


## Import and Clean Data -------------------------------------------------------

# import data (quarterly headline CPI)
data = pd.read_excel('Inflation_Data_WorldBank.xlsx', sheet_name='hcpi_q')

# remove columns that are not needed
data = data.drop(columns=['Country Code', 'IMF Country Code', 'Indicator Type', 
                              'Series Name', 'Data source', 'Note', 
                              'Unnamed: 223', '20231.1', '20232.1', '20233.1', 
                              '20234.1', 'Unnamed: 228', 'Unnamed: 229', 
                              'Unnamed: 230'])

# transpose data (dates as index, countries as column names)
data = data.transpose()
data.columns = data.iloc[0]
data = data.drop('Country')
data = data.drop(data.columns[-1],axis=1)

# change index to datetime 
data.index = pd.date_range(start='1970', end='2024', freq='Q')

# select dates 1970Q1 - 2023Q3
data_701_233 = data.loc['1970-01-31':'2023-09-30']



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

# select available countries
data_clean = data_701_233[aes+emes+lics]

# compute percentage changes (and remove first row)
data_pc = data_clean.pct_change().loc['1970-06-30':'2023-09-30']


## Summary Statistics and Median Evolution -------------------------------------

# dataframe with mean, median and standard devation of inflation by country
sum_stats = pd.DataFrame(columns=['Mean','Median','Std'], index=data_pc.columns)
for i in data_pc.columns:
    sum_stats.loc[i,'Mean'] = data_pc[i].mean()
    sum_stats.loc[i,'Median'] = data_pc[i].median()
    sum_stats.loc[i,'Std'] = data_pc[i].std()
print(sum_stats.to_string())

# average summary statistics by group
sum_stats.mean()
sum_stats.loc[aes,:].mean()
sum_stats.loc[emes,:].mean()
sum_stats.loc[lics,:].mean()

# plot median inflation rate and IQR by group in one figure 
# (saves as INFL_MED.png)
def med_iqr_plotter():
    fig, ax = plt.subplots(3,1,figsize=(6.3,5.2), dpi=600)
    ax[0].plot(data_pc.index, np.median(data_pc[aes], 1), color='black', linewidth=1)
    ax[0].fill_between(data_pc.index,
                np.percentile(data_pc[aes], 25, 1),
                np.percentile(data_pc[aes], 75, 1), 
                alpha=0.25, color='gray', edgecolor='none')
    ax[0].set_title('Advanced Economies', fontsize=10)
    ax[1].plot(data_pc.index, np.median(data_pc[emes], 1), color='black', linewidth=1)
    ax[1].fill_between(data_pc.index,
                np.percentile(data_pc[emes], 25, 1),
                np.percentile(data_pc[emes], 75, 1), 
                alpha=0.25, color='gray', edgecolor='none')
    ax[1].set_title('Emerging Market Economies', fontsize=10)
    ax[2].plot(data_pc.index, np.median(data_pc[lics], 1), color='black', linewidth=1)
    ax[2].fill_between(data_pc.index,
                np.percentile(data_pc[lics], 25, 1),
                np.percentile(data_pc[lics], 75, 1), 
                alpha=0.25, color='gray', edgecolor='none')
    ax[2].set_title('Low-Income Develpoing Economies', fontsize=10)
    for j in range(3):
        ax[j].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax[j].tick_params(axis='both', labelsize=8)
        ax[j].spines[['right', 'top']].set_visible(False)
        ax[j].grid(alpha=0.25)
        ax[j].set_ylim(-0.025, 0.125)
        ax[j].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45)
    plt.savefig('INFL_MED.png', bbox_inches='tight')

med_iqr_plotter()

