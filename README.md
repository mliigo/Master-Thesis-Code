# Master Thesis Code

## Overview

This repository contains the Python code used for my Master Thesis titled: 

> Dynamic Factor Model with Time-Varying Parameters: 
> Simulation Study & Application to International Inflation Dynamics

It contains functions for estimating a multi-level Dynamic Factor Model with time-varying loadings and stochastic volatility as used in Del Negro & Otrok (2008).
The estimation is Bayesian and uses a Gibbs sampler. The performance of the model and estimation procedure is assessed under different model parametrizations and 
degrees of time-variation using a Monte Carlo simulation study. The model is applied to study international inflation dynamics by decomposing quarterly inflation 
rates of 71 countries into world, group and idiosyncratic components. Countries are split into advanced economies, emerging market economies and low-income 
developing countries. The considered time period is 1970-Q2 – 2023-Q3. Data is taken from the World Bank's global inflation database (Ha et al., 2023).

## Files

The files can be classified into three groups: Estimation, Simulation Study and Application. The following contains brief descriptions of the files. More detailed explanations of the files and the functions therein are given in the respective file.

* __Estimation__

    * _ML_TVP_DFM_estimate.py_ : Functions for estimating the model. Its main function is "gibbs", which uses the defined conditional posterior samplers to generate draws from the model's joint posterior.

* __Simulation Study__

    * _ML_TVP_DFM_simulate.py_ : Functions for simulating the model. Its main function is "sim", which simulates the observables, factors and parameters, and uses them to compute the variance decompositions.
    * _Sim_Est_grouped.py_     : Groups simulation, estimation and result extraction into one function to be used in _Simulation_Study.py_.
    * _Simulation_Study.py_    : Implements the Monte Carlo simulation study. It simulates and estimates several models in parallel. Four regimes are considered, in which the degree of time-variation in the loadings and stochastic volatilities is either set to a low or high value, respectively. It saves the simulated and estimated values as dictionaries in a folder called "Sim_Saves" to be used in _Sim_Evaluation.py_. 
    * _Sim_Evaluation.py_      : Uses the simulated and estimated factors, parameters and variance decompositions from the dictionaries in the "Sim_Saves" folder to evaluate the performance of the model and estimation procedure.

* __Application__

    * _Data_Analysis.py_       : Plots median quarterly inflation rates by development-group and computes summary statistics. Uses the dataset "Inflation_Data_WorldBank.xlsx".
    * _App_Estimation.py_      : Estimates the model based on quarterly inflation rates of 71 countries from 3 groups spanning the period 1970-Q2 – 2023-Q3. It saves the posterior draws as a dictionary named "Application_Trace_last_10k.pkl" to be used in _App_Evaluation.py_.
    * _App_Evaluation.py_      : Evaluates the estimation results from the dictionary "Application_Trace_last_10k.pkl" to study the evolution of international inflation dynamics. It analyzes how the importance of global and group factors has evolved for different countries and groups.

## References

Del Negro, M. & Otrok, C. (2008). Dynamic Factor Models with Time-Varying 
Parameters: Measuring Changes in International Business Cycles (Staff Report 
No. 326). Federal Reserve Bank of New York. 
DOI: https://dx.doi.org/10.2139/ssrn.1136163

Ha, J., Kose, M. A. & Ohnsorge, F. (2023). One-stop Source: A Global Database 
of Inflation. Journal of International Money and Finance, 137 (October), 102896.
DOI: https://doi.org/10.1016/j.jimonfin.2023.102896

Data was taken from: https://www.worldbank.org/en/research/brief/inflation-database
