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
from typing import Union, List,Optional,Tuple, Any



def filter_nans(df: pd.DataFrame, row_threshold: float = 0.5, col_threshold: float = 0.7) -> pd.DataFrame:
    """
    Filter out rows and columns containing too many NaNs.

    Args:
        df (pd.DataFrame): The input DataFrame.
        row_trashold (float): The minimum proportion of non-NaN values required for a row to be kept. Default is 0.5.
        col_trashold (float): The minimum proportion of non-NaN values required for a column to be kept. Default is 0.7.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    df = df.dropna(subset=['plot code']).drop(columns=['Season code'])
    row_nan_trashold = df.shape[1] * row_threshold
    df = df.dropna(axis=0, thresh=int(row_nan_trashold))
    col_nan_trashold = df.shape[0] * col_threshold
    df = df.dropna(axis=1, thresh=int(col_nan_trashold))
    return df


def clean_target_column(df: pd.DataFrame, target: str, mul_std: float = 2.5) -> pd.DataFrame:
    """
    Remove outliers from the target column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The name of the target column.
        mul_std (float): The number of standard deviations beyond which a value is considered an outlier. Default is 2.5.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    target_std = df[target].std()
    target_mean = df[target].mean()
    low_trashold = target_mean - mul_std * target_std
    high_trashold = target_mean + mul_std * target_std
    df = df.loc[(df[target] > low_trashold) & (df[target] < high_trashold)]
    return df


def plot_corr_marix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Plot the correlation matrix of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The correlation matrix of the DataFrame.
    """
    corr_matrix = df.round(2).corr()
    mask = np.tril(np.ones_like(df.corr()))
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr_matrix, cmap="coolwarm", mask=mask)
    return corr_matrix


def create_bins_for_stratification(df: pd.DataFrame, target_col: str, q: int = 10) -> pd.Series:
    """
    Create bins for stratification.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        q (int): The number of quantiles to use when creating bins. Default is 10.

    Returns:
        pd.Series: The created bins.
    """
    return pd.qcut(df[target_col], q=q, labels=False)

def change_object_col_to_cat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns in a Pandas DataFrame to categorical columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to modify.
    
    Returns:
    pd.DataFrame: The modified DataFrame with object columns converted to categorical columns.
    """
    df[df.select_dtypes(include=['object', 'category']).columns] = df.select_dtypes(include=['object', 'category']).astype('category')
    return df

def run_kfold_metrics(df: pd.DataFrame, target_col: str, cat_column: List[str], round_numeric_val: int) -> None:
    """
    Compute the mean R-squared, mean absolute error, and root mean squared error for a dataset using K-fold cross-validation.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to use for K-fold cross-validation.
    target_col (str): The name of the target column in the DataFrame.
    cat_column (List[str]): A list of categorical column names in the DataFrame.
    round_numeric_val (int): The number of decimal places to round numeric values in the DataFrame.
    
    Returns:
    None: The function does not return anything, but prints the computed metrics to the console.
    """
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


def print_metrics(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> None:
    """
    Print the R-squared, mean absolute error, and root mean squared error for predicted values compared to true values.
    
    Parameters:
    y_true (pd.Series or np.ndarray): The true values for a target variable.
    y_pred (pd.Series or np.ndarray): The predicted values for the target variable.
    
    Returns:
    None: The function does not return anything, but prints the computed metrics to the console.
    """
    r2 = r2_score(y_true, y_pred)
    print("R-squared: ", r2)
    mae = mean_absolute_error(y_true, y_pred)
    print("MAE: ", mae)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print("RMSE: ", rmse)


def feature_selection(df: pd.DataFrame, target: str ='Sitlav - Soybeans - Yield - KG/Ha', number_of_features: int =10) -> List[str]:
    """
    Perform feature selection using the mRMR algorithm.
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to perform feature selection on.
    target : str, optional
        The name of the target column to predict, by default 'Sitlav - Soybeans - Yield - KG/Ha'.
    number_of_features : int, optional
        The number of features to select, by default 10.

    Returns
    -------
    List[str]
        A list containing the names of the selected features.

    """
    X1 = df.drop(columns=['plot code'])#,'CLASSE_DOM','Season code'], errors = 'ignore')
    Y1 = df[target]
    selected_columns = mrmr_regression(X=X1, y=Y1, K=number_of_features, cat_features=['CLASSE_DOM','Season code'])
    df = df[selected_columns + [target]]
    return df 


def creat_pycaret_models(df: pd.DataFrame, target: str, features: Optional[List[str]]=None) -> Tuple[Any, pd.DataFrame]:
    """
    Create and compare several regression models using PyCaret.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to perform model training on.
    target : str
        The name of the target column to predict.
    features : List[str], optional
        The list of feature names to use for training the models. If None, all columns will be used, by default None.

    Returns
    -------
    Tuple[Any, pandas.DataFrame]
        A tuple containing the trained PyCaret experiment and the resulting model metrics dataframe.

    """
    if features is None:
        features = df.columns
    else:
        features = features+[target]
    exp = setup(df[features], target = target, session_id = 125)
    model_results = exp.compare_models()
    return exp, model_results



