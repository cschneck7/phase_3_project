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


class RandomImputer(BaseEstimator, TransformerMixin):
    '''
    Randomly imputes values for missing data in new columns.
    Values are based off of existing values and rates of occurences.
    
    Initialized with optional argument columns which specify
    columns to be transformed, if left to default value 'all_missing_columns', all
    columns with missing values will be filled.
    '''
    
#     Initializes class object
    def __init__(self, missing_columns='all_missing_columns'):
        self.missing_columns = missing_columns
 

    def fit(self, X, y=None):
#     Finds column names containing missing values if missing_columns equals all_missing_columns
        if self.missing_columns == 'all_missing_columns':
            nan_amount = X.isna().sum()
            self.missing_columns = list(nan_amount[nan_amount>0].index)
#     Handles if single column entered as string
#     str type changed to list for future for loop operation
        elif type(self.missing_columns) == str:
            self.missing_columns = [self.missing_columns]

#     Initializes empty dict which will contain distribution of existing values for columns
#     in missing_columns
        feature_value_info = {}
#     Iterates through missing_columns finding value distributions 
        for col in self.missing_columns:
            feature_value_info[col] = X.loc[X[col].notnull(), col].value_counts(normalize=True)

#     Saves distributions as parameter
        self.feature_value_info = feature_value_info
        return self
            
    
    def transform(self, X, y=None):
#         Sets random seed for random seed generation for iterative
#         calls to random.choice in for loop
        np.random.seed(7337)
#         random seeds generation for loop
        rand_seeds = np.random.randint(0, 10e3, len(self.missing_columns), 'int64')
        
        df = X.copy()
        
#     Iterates through missing columns
        for i, col in enumerate(self.missing_columns):
#     Creates copy of column to have values imputed into
            df[col+'_imp'] = df[col]
#     Finds number of missing values in column
            number_missing = df[col].isnull().sum()
#     Sets random seed for random.choice
            np.random.seed(rand_seeds[i])
#     Randomly Imputes observed values replacing all missing information
            df.loc[df[col].isnull(), col+'_imp'] = np.random.choice(self.feature_value_info[col].index, 
                                                                    number_missing, 
                                                                    replace = True,
                                                                    p = self.feature_value_info[col])

#     Creates column index variable to be called to set DataFrame index
        self.features_out = df.columns
        
        return df
    
#     Returns final columns index
    def get_features_out(self):
        return self.features_out
    
    
    
class IterativeClassification(BaseEstimator, TransformerMixin):
    '''
    Uses an iterative DecisionTreeClassifier to fill in missing values.
    
    __init__ :
        Input: max_depth = max_depth parameter of DecisionTreeClassifier
               class_order = 'many_first' or 'less_first'
                             default = 'many_first', determines which order
                             or columns to iterate through based off of quantity
                             of missing values in columns
    
    fit : 
        Input: X = DataFrame with missing values
               y = None
               
    transform :
        Input: X = DataFrame with missing values to be transformed
        
    get_features_out :
        Returns column names for final transformed DataFrame
    '''
    
    def __init__(self, max_depth=None, class_order='many_first', num_cols=None, cat_cols=None):
        self.max_depth=max_depth
        self.class_order=class_order
        self.num_cols=num_cols
        self.cat_cols=cat_cols

    def fit(self, X, y=None):
        '''
        Iteratively fits DecisionTreeClassifier models to pd.DataFrame X
        
        1. Sorts columns with missing values by quantity, the order set
        by parameter class_order with options 'many_first' and 'less_first'.
        
        2. Finds the leftover features that didn't have any missing values
        and don't need to be classified from DataFrame X, sorts these to prevent
        indexing issues later on.
        
        3 Creates a list of all features being used to fit the 
        DecisionTreeClassifier, input DataFrame X should be in a format resulting
        from being tranformed using the RandomImputer tranformer. The features 
        resulting from this transformer have had variables randomly imputed and 
        the feature names ending with '_imp' to signify their transformation. 
        The input DataFrame X contains all the features from the original DataFrame
        with missing values plus the imputed features.
        
        4. Creates a DataFrame copy using the list from step 3.
        
        5. Iterates through the columns that had missing values,
        using their imputed copy as a target variable and the rest of 
        the features from the DataFrame created in step 4 to fit a 
        DecisionTreeClassifier. The remaining features have to be 
        OneHotEncoded before used in Classifier. This classifier is then 
        used to predict the variables for those that were originally 
        missing and replacing them in the DataFrame created in step 4.
        
        The fifth step is then repeated for each column while saving the 
        DecisionTreeClassifier at each step to be used later in the 
        transform method.
        '''
        
#         Finds amount of nans in columns
#         Orders columns to be predicted by class_order parameter
        nan_amount = X.isna().sum()
        if self.class_order == 'many_first':
            self.missing_columns = list(nan_amount[nan_amount>0].sort_values(ascending=False).index)
        elif self.class_order == 'less_first':
            self.missing_columns = list(nan_amount[nan_amount>0].sort_values().index)
        else:
            sys.exit('''Incorrect input for class_order parameter.
                 Parameter only accepts values ('many_first', or 'less_first')''')

#         Finds features in dataframe that aren't included in missing_columns
        leftover_features = list(set(X.columns) - set(self.missing_columns) - 
                                {col+'_imp' for col in self.missing_columns})
        leftover_features.sort()
        self.leftover_features = leftover_features
        
#         Sets missing_columns predictive features with _imp extension matching fitting from RandomImputer class
#         Adds leftover features to create dataframe with all predictive features
        self.pred_features = [col+'_imp' for col in self.missing_columns] + leftover_features
        pred_df = X[self.pred_features].copy()
        
#         Creates empty dictionairies for fitted models, and their accuracy scores
        models = {}
        accuracy_scores = {}
        
#         Iterates through missing_columns, creating a model, updating iterations target
#         feature with predicted values at each iteration
        for col in self.missing_columns:
#             creates dataframe for this iterations predictives features
#             This involves ohe categorical features
            temp_df = self._step_features(pred_df, col)
#             Initializes and fits DecisionTreeClassifier
            dt = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
            dt.fit(temp_df, pred_df[col+'_imp'])
            
#             imports predicted values into missing locations
            pred_df.loc[X[col].isnull(), col+'_imp'] = dt.predict(temp_df)[X[col].isnull()]
            
#             Saves this iterations model and accuracy score
            models[col] = dt
            accuracy_scores[col] = accuracy_score(X.loc[X[col].notnull(), col],
                                                dt.predict(temp_df)[X[col].notnull()])    
        self.dt_models = models
        self.fit_accuracy_scores = accuracy_scores
        
        return self
    
    
    def transform(self, X, y=None):
        '''
        Transforms X using models fit in fit method
        
        1. Creates prediction df with all columns that will be classified
        
        2. Iterates through the columns with missing values in the same manner
           as the fit method, except only tranforming the target using the
           fitted classifiers from the fit method.
           
        3. Creates Final DataFrame containing classified columns, plus columns
           that weren't originally missing values
        '''
#         Initializes dataframe containing determined values
        det_df = pd.DataFrame(columns = ['Det_'+col for col in self.missing_columns])
#         Creates dataframe of predictive features
        pred_df = X[self.pred_features].copy()
        
#         Iterates through missing columns to transform with fitted models
        for col in self.missing_columns:
#             Fills determined column with values
            det_df['Det_'+col] = X[col+'_imp']
#             creates dataframe for this iterations predictives features
#             This involves ohe categorical features 
            temp_df = self._step_features(pred_df, col)
#             Loads this iterations fitted model
            dt_model = self.dt_models[col]
#            imports predicted values into missing locations for predictive and determined dataframe
            det_df.loc[X[col].isnull(), 'Det_'+col] = dt_model.predict(temp_df)[X[col].isnull()]
            pred_df.loc[X[col].isnull(), col+'_imp'] = dt_model.predict(temp_df)[X[col].isnull()]

#         Creates transfromed df determined features as well as nontransformed features
        final_df = pd.concat([det_df, X[self.leftover_features]], axis=1)
        
#         Saves transformed models feature names        
        self.features_out = final_df.columns
    
        return final_df
    
#     Returns transformed models feature names
    def get_features_out(self):
        return self.features_out
    
#     Creates dataframe for the current iteration
    def _step_features(self, X, col):
        '''
        Takes current iterations target variable, and creates
        predictive dataframe with proper feature names 
        and OneHotEncodes categorical columns
        
        inputs: X = feature dataframe
                col = Current iterations target variable
                
        output: DataFrame with OneHotEncoded categorical features
                and correct feature names matching those from the
                RandomImputer class
        '''
#         Checks if categorical features exist
        if self.cat_cols != None:
#             adds _imp to end of feature name if included in missing_columns
            imp_cat_cols = self._imp_columns(self.cat_cols)
#             creates list of columns to be OneHotEncoded
            ohe_cols = list(set(imp_cat_cols) - {col+'_imp'})
#             Sorts in alphabetical order for consistency
            ohe_cols.sort()
#             Initializes, fits and transforms OneHotEncoder
            ohe = OneHotEncoder(sparse=False) 
            ohe_features = ohe.fit_transform(X[ohe_cols])
#             Creates dataframe with OneHotEncoded features
            ohe_df = pd.DataFrame(ohe_features,
                        columns=ohe.get_feature_names_out(ohe_cols),
                        index=X.index)
#             Checks if there are numerical columns    
            if self.num_cols != None:
#                 adds _imp to end of feature name if included in missing_columns
                imp_num_cols = self._imp_columns(self.num_cols)
#                 Creates list of numerical column names and sorts them
                num_columns = list(set(imp_num_cols) - {col+'_imp'})
                num_columns.sort()
#                 creates dataframe with ohe and num features    
                temp_df = pd.concat([X[num_columns], ohe_df], axis=1)
#             Creates dataframe if there are no numerical features
            else:
                temp_df = ohe_df
#         Checks if their are numerical features, if no categorical features            
        elif self.num_cols != None:
#             adds _imp to end of feature name if included in missing_columns
            imp_num_cols = self._imp_columns(self.num_cols)
#             Creates list of numerical column names and sorts them
            num_columns = list(set(imp_num_cols) - {col+'_imp'})
            num_columns.sort()
            
#             creates dataframe with numerical features with altered names
            temp_df = X[num_columns].copy()
#         If no features were assigned to either num_cols or cat_cols
        else: 
            sys.exit('''Need to assign both or one of num_cols and cat_cols''')
            
        return temp_df
    
    def _imp_columns(self, columns):
        '''
        Adds _imp to end of feature name if included in missing_columns.
        missing_columns should be features transformed through the RandomImputer
        class
        
        input: columns = columns to have feature names tranformed if applicable
        ouput: list of new column names
        '''
        imp_cols = [col+'_imp' if col in self.missing_columns else col for col in columns]
        return imp_cols
    
    
    
    
    
    
    
    