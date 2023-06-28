import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
import sys
import pickle
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (LabelEncoder, FunctionTransformer,
                                   OneHotEncoder, OrdinalEncoder,
                                   MinMaxScaler)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, cross_validate)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import (BaseEstimator, TransformerMixin)
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_score,
                             make_scorer)
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from itertools import combinations
from xgboost import XGBClassifier


def vacc_rates(x, y):
    '''
    Calculates and creates dataframe with feature group rates and total counts
    
    input: x = feature
           y = target variable
    
    output: DataFrame
    '''
#     creates dataframe containing feature, target variable, and columns of ones for count
    df = pd.concat([x, y, pd.DataFrame(np.ones(len(y)))], axis=1)
    
#     Gets count of each groups unvac and vac
    counts = df.groupby(by=[x.name, y.name]).count().unstack()
    
#     calculates groups unvac percentage and count
    group_perc =  (counts[0][0]/(counts[0][0]+counts[0][1]))
    group_count = counts[0][0]+counts[0][1]
    
#     creates dataframe with calculated group percentage and counts
    group_df = pd.concat([group_perc, group_count], axis=1)
    
#     Renames columns
    group_df.rename(columns={0: 'group_perc', 1: 'group_count'}, inplace=True)
    
#     returns dataframe
    return group_df



def ordered_chi2(features, target='features', ascending=True, alpha=None):
    '''
    Returnes the ordered chi2 P-values between each feature and the target variable or variables
    
    Inputs: features = Dataframe of features
            target = pd.Series as target or default = 'features' which results in 
            finding relationships inside features DataFrame
            ascending = Determines order, default value = True
            alpha = Chi2 Pvalue threshold for which features get returned.
                        Returns all features with a Pvalue<=alpha,
                        default = None so all features are returned
            
    Output: Dataframe containing ordered P-values
    '''
    
    df = pd.DataFrame(columns=['var1', 'var2', 'Pvalue'])
                 
    if isinstance(target, pd.Series):
        for col in features.columns:
            temp_dict={}
            temp_dict['var1'] = target.name
            temp_dict['var2'] = col
            temp_dict['Pvalue'] = stats.chi2_contingency(pd.crosstab(target, features[col]))[1]
            df = df.append(temp_dict, ignore_index=True)
    
    elif target == 'features':
        combs = combinations(features.columns, 2)
        for comb in combs:
            temp_dict={}
            temp_dict['var1'] = comb[0]
            temp_dict['var2'] = comb[1]
            temp_dict['Pvalue'] = stats.chi2_contingency(pd.crosstab(features[comb[0]], features[comb[1]]))[1]
            df = df.append(temp_dict, ignore_index=True)
        
    else:
        sys.exit('''Incorrect input for parameter target.
        Parameter only accepts types pd.DataFrame, pd.Series, or left to default value.''')        
    
    if alpha == None:
        return df.sort_values(by='Pvalue', ascending=ascending)
    else:
        return df[df.Pvalue <= alpha].sort_values(by='Pvalue', ascending=ascending)
    


def missing_row_entries(df):
    '''
    Prints a DataFrame where:
        index = num of nan entries in a row
        frequency = number of rows with amount of nans defined by index
        cum_sum = cumulative sum starting with rows with most missing entries
        
    inputs: df = DataFrame with missing entries
    output: DataFrame, with nan per row information
    '''
    
    # Creates a list with number of missing values in each row
    nan_per_row = []
    for i in range(df.shape[0]):
        nan_per_row.append(df.iloc[i,:].isna().sum())
       
    # creates dataseries of missing values
    nan_per_row_ds = pd.Series(nan_per_row)
    nan_per_row_ds.rename('frequency', inplace=True)
    
    # gets frequeny of nan amounts in rows
    nan_row_counts = nan_per_row_ds.value_counts()
    
#     Orders nan_row_counts descending by most nans down
    ordered_missing_nan = nan_row_counts[nan_row_counts.keys().sort_values(ascending=False)]
    
#     Creates dataseries with cumulative sum
    nan_cum_sum = np.cumsum(ordered_missing_nan)   
    nan_cum_sum.rename('cum_sum', inplace=True)
    
#     Creates DataFrame with rows per missing values amount, and cum sum
    nan_df = pd.concat([ordered_missing_nan, nan_cum_sum], axis=1)
    nan_df.rename_axis('num_of_nans_in_row', inplace=True)
    
    return nan_df




def drop_rows_by_nans(df, y, nan_threshold=None):
    '''
    Drop rows by quantity of nan values with nan_threshold as cut-off point
    
    Inputs: 
            df = DataFrame to be altered, should contain features and target concatenated together
            y = target variable column name
            nan_threshold = cut-off point to drop rows, default = None which results in a value
                            of half or a little larger than number of features
    
    Outputs: 
            feature_df = feature DataFrame
            y = Target Variable
    '''
    
#     Checks value of nan_threshold
    if nan_threshold == None:
#         Sets to half of or rounded up from half of feature columns
        nan_threshold = math.ceil((len(df.columns)-1)/2)
    
#     Finds nans contained in each row 
    nan_per_row = []
    for i in range(df.shape[0]):
        nan_per_row.append(df.iloc[i,:].isna().sum())
    
#     Creates temporary column in df for number of nan values
    df['nans'] = nan_per_row
#     Creates dataframe of rows under nan threshold
    df = df[df.nans < nan_threshold]
#     Creates target variable
    y = df.h1n1_vaccine
#     returns tuple (feature_df, target)
    return (df.drop(['nans', 'h1n1_vaccine'], axis=1), y)




def rankings(data):
    '''
    Takes in GridSearchCV df column and returns scores mean
    and ranking
    '''
#     Finds columns of interest
    cols = [ind for ind in data.index if (('rank' in ind) | 
                                      ('mean' in ind) &
                                      ('time' not in ind))]
    return data.loc[cols, :]




def fold_scores(data, score):
    '''
    Returns fold scores, mean score, std for both train and test sets
    
    Inputs: data = columns of interest from GridSearchCV of 
                   type DataFrame
            score = str, name of score given to GridSearchCV object
    Output: DataFrame containing mean score, fold scores, and std
            for both train and test sets
    '''
#     creates string to find index entries
    test_str = 'test_' + score 
    train_str = 'train_' + score
    
#     creates list of index names
    test_ind = [ind for ind in data.index if ((test_str in ind) & 
                                               ('rank' not in ind))]
    train_ind = [ind for ind in data.index if ((train_str in ind) & 
                                               ('rank' not in ind))]
    
#     Creates new index values for future DataFrames
    new_test_ind = {ind: ind.replace("test_", "") for ind in test_ind}
    new_train_ind = {ind: ind.replace("train_", "") for ind in train_ind}
    
#     Creates empty dataframe
    df = pd.DataFrame()
    
#     Iterates through columns in case of multiply iterations having the same score
#     Places test and train scores for each iteration next to eachother
    for col in data.columns:
#     Creates individual DataFrames for test and train
        test_df = data.loc[test_ind, col].rename(index=new_test_ind)
        test_df.rename('test_'+str(col), inplace=True)
        
        train_df = data.loc[train_ind, col].rename(index=new_train_ind)    
        train_df.rename('train_'+str(col), inplace=True)
        
#     Concats DataFrames
        df = pd.concat([df, test_df, train_df], axis=1)
    
#     returnes DataFrame
    return df





def quick_metrics(y_true, y_pred):
    '''
    Displays confusion matrix and classification report
    
    Inputs: y_true = real labels
            y_pred = predicted labels
    '''
    
#     Generates confusion matrix
    cm = confusion_matrix(y_true, y_pred)
#     Creates and displays dataframe for easier viewing of confusion matrix
    display(pd.DataFrame(data=cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1']))    
    print()
#     creates classification_report dictionary
    cr_dict = classification_report(y_true, y_pred, output_dict=True)
#     displays metrics for our non-vaccinated predictions
    acc = cr_dict['accuracy']
    print(f'Accuracy: {acc}')
    display(cr_dict['1'])