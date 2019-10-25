"""
@file_name : DataUtils.py
@author : Srihari Seshadri
@description : This file contains methods for data wrangling and other array/set operations
@date : 01-29-2019
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd 
import numpy as np
import os
import pickle

# Main

# Multiple categories
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, do not try to understand it (yet).

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.pipeline import FeatureUnion


def num_nan_rows(df):
    return df.shape[0] - df.dropna().shape[0]


def find_common_elems(list_of_lists):
    """
    Find common elems in all the lists given in the list of lists using sets (intersection)
    :param list_of_lists: list of lists of elements
    :return: list of common elements
    """
    common_elems = list_of_lists[0]
    for i in range(1, len(list_of_lists)):
        common_elems = list(set(common_elems).intersection(list_of_lists[i]))
    return common_elems


def find_unique_elems(list_of_lists):
    """
    Find unique non overlapping elems in all the lists given in the list of lists using sets
    :param list_of_lists: list of lists of elements
    :return: list of unique elements
    """
    unique_elems = set(list_of_lists[0])
    for i in range(1, len(list_of_lists)):
        unique_elems = set(list_of_lists[i]) ^ unique_elems
    return list(unique_elems)


def analyse_nans(df):
    """
    Returns a dataframe with details on how many NaNs and where from a given dataframe
    :param df: dataframe
    :return: NaN dataframe
    """
    temp_df = pd.DataFrame(columns=df.columns,
                           index=["total", "percentage", "idx_list"])
    for col in df.columns:
        idxes = df[col].isnull()
        num_nans = idxes.sum(axis = 0)
        nan_pct = 100*np.round(num_nans/df.shape[0], 3)
        temp_df[col] = [num_nans, nan_pct,
                        df.index[idxes.values == True].tolist()]
    return temp_df
    
    
def describe_unique(df, colname, filter_unnecessary=True):
    """
    Describes all the unique elements in a column
    :param df: Dataframe
    :param colname: Column name
    :param filter_unnecessary: IF True, IF all values are unique, returns nothing
    :return: prints details on all the unique elements in column
    """
    print("Column name : ", colname)
    unique_elems = pd.unique(df[colname])
    types_of_data = [type(x) for x in unique_elems]
    if filter_unnecessary:
        if len(unique_elems) == df.shape[0]:
            print("All values are unique.")
            return
    print("Number of unique elems : ", len(unique_elems))
    print("Types of data in col :", set(types_of_data))
    for idx, uel in zip(range(0, len(unique_elems)), unique_elems):
        print("  ", str(idx)+".", type(uel), "\t",uel)


def merge_replace(left, right, left_on, right_on, how, drop_list):
    """
    Merges 2 datasets and drops the columns specified in the list
    :param left: left dataframe
    :param right: right dataframe
    :param left_on: left_on key
    :param right_on: right_on key
    :param how: "inner", "outer", "left", "right"
    :param drop_list: list of cols to drop
    :return: merged dataframe
    """
    left = pd.merge(left=left, right=right, left_on=left_on, right_on=right_on, how=how)
    left.drop(drop_list, axis=1, inplace=True)
    return left


def cartesian_product(left, right):
    """
    Performs cartesian product of 2 dataframes
    :param left: left dataframe
    :param right: right dataframe
    :return: cartesian product of leftxright dataframe
    """
    la, lb = len(left), len(right)
    ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])

    return pd.DataFrame(
        np.column_stack([left.values[ia2.ravel()], right.values[ib2.ravel()]]))
		
# ----------------------------------------------------------------------------------------------------------------------
# FILE UTILS
# ----------------------------------------------------------------------------------------------------------------------

def save_to_disk(obj, filename):
    try:
        with open(filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)


def load_from_disk(filename):
    try:
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)        
            return b
    except Exception as e:
        print(e)    


# ----------------------------------------------------------------------------------------------------------------------
# ENCODING UTILS
# ----------------------------------------------------------------------------------------------------------------------

def get_label_encoded(df, colname, inplace=True):
    """
    Returns label encoded column appended to the data frame.
    New column is pre-pended with "le_" followed by @colname
    :param df: data frame
    :param colname: name of the column to encode
    :param inplace: if True, replaces the original columns instead of making new ones
    :return: updated dataframe
    """
    
    # Sanity check
    if colname not in df.columns:
        raise ValueError("Column not in Dataframe!")
        return data
    
    le = LabelEncoder()
    le.fit(df[colname])
    le_colname = colname
    if not inplace:
        le_colname = "le_" + le_colname
    df[le_colname] = le.transform(df[colname])
    return df, le


def labelencode_collist(df, collist, inplace=True):
    """
    Returns label encoded columns appended to the data frame.
    New columns are pre-pended with "le_" followed by @colname
    :param df: data frame
    :param collist: list with names of the columns to encode
    :param inplace: if True, replaces the original columns instead of making new ones
    :return: updated dataframe and dict of colname:encoder
    """
    
    encoder_list= {}
    
    for col in collist:
        if col not in df.columns:
            continue
        df, le = get_label_encoded(df, col, inplace)
        encoder_list[col] = le
        
    return df, encoder_list


def get_onehot_encoded(df, colname, drop_original=True):
    """
    Returns One Hot Encoded columns appended to the data frame.
    New columns are pre-pended with @colname followed by encoded class label
    :param df: data frame
    :param colname: name of the column to encode
    :param drop_original: if True, drops original column
    :return: updated dataframe 
    """
    # Sanity check
    if colname not in df.columns:
        raise ValueError("Column not in Dataframe!")
        return data
    
    ohe = OneHotEncoder(categorical_features=[0])
    vec = np.asarray(df[colname].values).reshape(-1,1)
    out = ohe.fit_transform(vec).toarray()
    # Drop the first column - dummy variable trap
    out = out[:,1:]
    # Join with the original data frame
    dfOneHot = pd.DataFrame(out, 
                            columns=[colname+"_"+str(int(i)) for i in range(out.shape[1])], 
                            index=df.index)
    df = pd.concat([df, dfOneHot], axis=1)
    
    if drop_original:
        df.drop(colname, axis=1, inplace=True)
    
    return df, ohe


def onehotencode_collist(df, collist, drop_original=True):
    """
    Returns One Hot Encoded columns appended to the data frame.
    New columns are pre-pended with @colname followed by encoded class label
    :param df: data frame
    :param collist: list with names of the columns to encode
    :param drop_original: if True, drops original column
    :return: updated dataframe and dict of colname:encoder
    """
    
    encoder_list= {}
    
    for col in collist:
        if col not in df.columns:
            continue
        df, ohe = get_onehot_encoded(df, col, drop_original)
        encoder_list[col] = ohe
        
    return df, encoder_list



# ----------------------------------------------------------------------------------------------------------------------
# SCALING UTILS
# ----------------------------------------------------------------------------------------------------------------------

def scale_collist(df, collist):
    
    scaler_list = {}
    
    for col in collist:
        if col not in df.columns:
            continue
        
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
        scaler_list[col] = scaler
        
    return df, scaler_list