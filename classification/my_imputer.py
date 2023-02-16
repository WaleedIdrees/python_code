import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

class MixImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to impute missing values in a Pandas DataFrame.
    It imputes missing values using KNN imputation for both numerical and categorical features.
    """
    
    def __init__(self, impute_missing=True):
        """
        Initialize the MixImputer object.
        
        Parameters:
        - impute_missing (bool): If True, impute missing values. Otherwise, return the original DataFrame.
        """
        self.impute_missing = impute_missing
        
    def fit(self, X, y=None):
        """
        Fit the MixImputer object. 
        
        Parameters:
        - X (pandas.DataFrame): The input DataFrame.
        - y (None): Ignored argument.
        
        Returns:
        - self: The MixImputer object.
        """
        return self
    
    def transform(self, X):
        """
        Transform the input DataFrame by imputing missing values and returning a new DataFrame.
        
        Parameters:
        - X (pandas.DataFrame): The input DataFrame.
        
        Returns:
        - X (pandas.DataFrame): The transformed DataFrame with imputed missing values.
        """
        # Copy the input DataFrame to avoid modifying the original DataFrame
        X = X.copy()
        
        # Get the columns with categorical features
        cols = X.select_dtypes(["object","category"]).columns.tolist()
        
        if self.impute_missing:
            # Encode the categorical features using LabelEncoder
            # LabelEncoder converts the categorical values to numerical labels (e.g., 0, 1, 2)
            encoder = LabelEncoder()
            mapped_dic = {} # A dictionary to store the mapping of original values to encoded values
            
            for col in cols:
                X[[col]] = X[[col]].apply(lambda series: pd.Series(
                    encoder.fit_transform(series[series.notnull()]),
                    index=series[series.notnull()].index
                ))
                encode_dic = { x: y for x,y in zip(X[col][X[col].notnull()].unique(), encoder.classes_)}
                mapped_dic[col] = encode_dic
            
            # Impute missing values using KNN imputer
            imp_knn = IterativeImputer(estimator=KNeighborsRegressor(n_jobs=-1, n_neighbors=X.shape[1]), 
                                       max_iter=100, random_state=0)
            X = pd.DataFrame(imp_knn.fit_transform(X), columns=X.columns.tolist())
            
            # Round off the categorical columns to the nearest integer
            X[cols] = X[cols].round(0).astype("int")
            
            # Convert the encoded values back to original values
            for col in cols:
                X[col] = X[col].map(mapped_dic[col])
                
        return X