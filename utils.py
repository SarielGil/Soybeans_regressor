import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from pycaret.regression import *
from mrmr import mrmr_regression


target = 'Sitlav - Soybeans - Yield - KG/Ha'

def filter_nans(df, row_trashold=0.5 ,col_trashold=0.7):
    df = df.dropna(subset=['plot code']).drop(columns= ['Season code'])
    row_nan_trashold = df.shape[1]*row_trashold
    df = df.dropna(axis=0, thresh=int(row_nan_trashold))
    col_nan_trashold = df.shape[0]*col_trashold
    df = df.dropna(axis=1, thresh=int(col_nan_trashold))
    return df 

def clean_target_colunm(df, target, mul_std = 2.5):
    target_std = df[target].std()
    target_mean = df[target].mean()
    low_trashold = target_mean-mul_std*target_std
    high_trashold = target_mean+mul_std*target_std
    df  = df.loc[(df[target]>low_trashold) & (df[target]< high_trashold)]
    return df 

def plot_corr_marix(df):
    corr_matrix = df.round(2).corr()
    mask = np.tril(np.ones_like(df.corr()))
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(corr_matrix, cmap="coolwarm", mask=mask)
    return corr_matrix

def create_bins_for_stratification(df, target_col, q=10):
    return pd.qcut(df[target_col], q=q, labels=False)

def change_object_col_to_cat(df):
    df[df.select_dtypes(include=['object', 'category']).columns] = df.select_dtypes(include=['object', 'category']).astype('category')
    return df

def run_kfold_metrics(df,target_col,cat_column,round_numeric_val):
    r2 = []
    mae = []
    rmse = []
    df = change_object_col_to_cat(df)
    X = df.drop(columns=['plot code',target_col]).round(round_numeric_val).reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)
    lgbm_reg = lgbm.LGBMRegressor(random_state=0, categorical_feature=cat_column)

    y_bins_cat = create_bins_for_stratification(df, target_col, q=10)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins_cat)):
        train_data, y_train= X.iloc[train_idx], y[train_idx]
        val_data, y_test = X.iloc[val_idx], y[val_idx]

        lgbm_reg.fit(train_data, y_train)
        y_pred = lgbm_reg.predict(val_data)
        r2.append(r2_score(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))
        rmse.append(mean_squared_error(y_test, y_pred, squared=False))
        
    print("Mean R-squared: ", np.mean(r2))
    print("MAE: ", np.mean(mae))
    print("RMSE: ", np.mean(rmse))

def print_metrics(y_true, y_pred):
  r2 = r2_score(y_true, y_pred)
  print("R-squared: ", r2)
  mae = mean_absolute_error(y_true, y_pred)
  print("MAE: ", mae)
  rmse = mean_squared_error(y_true, y_pred, squared=False)
  print("RMSE: ", rmse)

def feature_selection(df, target='Sitlav - Soybeans - Yield - KG/Ha', number_of_features=10):
    X1 = df.drop(columns=['plot code','CLASSE_DOM'])
    Y1 = df[target]
    return mrmr_regression(X=X1, y=Y1, K=number_of_features)


def creat_pycaret_models(df, target, features=None):
    if features is None:
        features = df.columns
    else:
        features = features+[target]
    exp = setup(df[features], target = target, session_id = 125)
    model_results = exp.compare_models()
    return exp, model_results

