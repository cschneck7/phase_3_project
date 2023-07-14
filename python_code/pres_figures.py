import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


'''
Used to create and save presentation figures
'''

def run_all_figs(X, y):
    '''
    Runs and saves all figures in py file
    
    Inputs: X = Full feature database
            y = target variable
    '''
    heat_map_doctor_recc_h1n1(X, y)
    heat_map_opinion_h1n1_vacc_effective(X, y)
    heat_map_opinion_h1n1_risk(X, y)
    
    

def heat_map_doctor_recc_h1n1(X, y):
    '''
    Creates heatmap for h1n1_vaccination vs. doctor_recc_h1n1
    
    Inputs: X = Full feature database
            y = target variable
    '''
#     Changes from binary to string format
    y = y.map({0: 'No', 1: 'Yes'})
    x = X[X.doctor_recc_h1n1.notnull()].doctor_recc_h1n1.map({0.0: 'No', 1.0: 'Yes'})
    
#     creates figure
    fig, ax = plt.subplots(figsize=(3,3), dpi=200);
    
#     creates heatmap
    sns.heatmap(data=pd.crosstab(y, x), 
                annot=True, 
                fmt="", 
                cbar=False, 
                ax=ax, 
                cmap=sns.light_palette("#808080"),
                linewidths=3,
                annot_kws={"fontsize":12});
    
#     Sets label titles
    ax.set_ylabel('H1N1 Vaccinated?')
    ax.set_xlabel('Recommended by Doctor?')
    
#     Saves figure
    fig.savefig('images/figures/doctor_recc_h1n1_heat_map.png', 
                dpi=200, 
                bbox_inches="tight",
                pad_inches=0)

    
def heat_map_opinion_h1n1_vacc_effective(X, y):
    '''
    Creates heatmap for h1n1_vaccination vs. opinion_h1n1_vacc_effective
    
    Inputs: X = Full feature database
            y = target variable
    '''
#     Changes from binary to string format
    y = y.map({0: 'No', 1: 'Yes'})
    
#     changes float values to int
    x = X[X.opinion_h1n1_vacc_effective.notnull()].opinion_h1n1_vacc_effective.astype(int)
    
#     creates figure
    fig, ax = plt.subplots(figsize=(7.5,3), dpi=100);
    
#     creates heatmap
    sns.heatmap(data=pd.crosstab(y, x), 
                annot=True, 
                fmt="", 
                cbar=False, 
                ax=ax, 
                cmap=sns.light_palette("seagreen"),
                linewidths=3,
                annot_kws={"fontsize":12});
    
    #     Sets label titles
    ax.set_ylabel('H1N1 Vaccinated?')
    ax.set_xlabel('Opinion of H1N1 Vaccine Effectiveness?')
    
#     Saves figure
    fig.savefig('images/figures/opinion_h1n1_vacc_effective_heat_map.png', 
                dpi=100, 
                bbox_inches="tight",
                pad_inches=0)
    

def heat_map_opinion_h1n1_risk(X, y):
    '''
    Creates heatmap for h1n1_vaccination vs. opinion_h1n1_risk
    
    Inputs: X = Full feature database
            y = target variable
    '''
#     Changes from binary to string format
    y = y.map({0: 'No', 1: 'Yes'})
    
#     changes float values to int
    x = X[X.opinion_h1n1_risk.notnull()].opinion_h1n1_risk.astype(int)
    
#     creates figure
    fig, ax = plt.subplots(figsize=(7.5,3), dpi=100);
    
#     creates heatmap
    sns.heatmap(data=pd.crosstab(y, x), 
                annot=True, 
                fmt="", 
                cbar=False, 
                ax=ax, 
                cmap=sns.light_palette("#338BA8"),
                linewidths=3,
                annot_kws={"fontsize":12});
      
    #     Sets label titles
    ax.set_ylabel('H1N1 Vaccinated?')
    ax.set_xlabel('Opinion of H1N1 Risk')
    
#     Saves figure
    fig.savefig('images/figures/opinion_h1n1_risk_heat_map.png', 
                dpi=100, 
                bbox_inches="tight",
                pad_inches=0)