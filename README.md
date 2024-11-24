# Master Thesis Code

## Overview

This repository contains the Python code used for my Master Thesis titled: 

> Dynamic Factor Model with Time-Varying Parameters: 
> Simulation Study & Application to International Inflation Dynamics

It contains functions for estimating a multi-level Dynamic Factor Model with time-varying parameters and stochastic volatility as used in Del Negro & Otrok (2008).
The estimation is Bayesian and uses a Gibbs sampler. The performance of the model and estimation procedure is assessed under different model parametrizations and 
degrees of time-variation using a Monte Carlo simulation study. The model is applied to study international inflation dynamics by decomposing quarterly inflation 
rates of 71 countries into world, group and idiosyncratic components. Countries are split into advanced economies (AEs), emerging market economies (EMEs) and 
low-income developing countries (LIDCs). The considered time period is 1970-Q2 – 2023-Q3. Data is taken from the World Bank's global inflation database (Ha et al., 2023).

## Files

The files can be classified into three groups: Estimation, Simulation Study and Application. The following contains brief descriptions of the files. More detailed explanations of the files and the functions therein are given in the respective file.

* Estimation

    * ML_TVP_DFM_estimate.py : Functions for estimating the model. Its main function is "gibbs", which uses the defined conditional posterior samplers to generate draws from 
    the model's joint posterior.

* Simulation Study

    * ML_TVP_DFM_simulate.py : Functions for simulating the model. Its main function is "sim", which simulates the observables, factors and parameters, and uses them to compute the variance decompositions.
    * Sim_Est_grouped.py     : Groups simulation, estimation and result extraction into one function to be used in Simulation_Study.py.
    * Simulation_Study.py    : Implements the Monte Carlo simulation study. It simulates and estimates several models in parallel. Four regimes are considered, in which the degree of time-variation in the 
    loadings and stochastic volatilities is either set to a low or high value, respectively. The simulated and estimated values are saved in the Sim_Saves folder.
    * Sim_Evaluation.py      : Uses the simulated and estimated factors, parameters and variance decompositions to evaluate the performance of the model and estimation procedure.

* Application

    * Data_Analysis.py       : Plots median quarterly inflation rates by development-group and computes summary statistics. Uses the dataset "Inflation_Data_WorldBank.xlsx".
    * App_Estimation.py      : Estimates the model based on quarterly inflation rates of 71 countries from 3 different groups spanning the period 1970-Q2 – 2023-Q3. The posterior draws are saved as "Application_Trace_last_10k.pkl".
    * App_Evaluation.py      : Evaluates the estimation results to study the evolution of international inflation dynamics. It analyzes how the importance of global and group factors has evolved for different countries and groups.

## References

Del Negro, M. & Otrok, C. (2008). Dynamic Factor Models with Time-Varying 
Parameters: Measuring Changes in International Business Cycles (Staff Report 
No. 326). Federal Reserve Bank of New York. 
DOI: https://dx.doi.org/10.2139/ssrn.1136163

Ha, J., Kose, M. A. & Ohnsorge, F. (2023). One-stop Source: A Global Database 
of Inflation. Journal of International Money and Finance, 137 (October), 102896.
DOI: https://doi.org/10.1016/j.jimonfin.2023.102896

Data was taken from: https://www.worldbank.org/en/research/brief/inflation-database