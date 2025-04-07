import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DateExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["timestamp"] = pd.to_datetime(X["timestamp"])
        X["year"] = X["timestamp"].dt.year
        X["month"] = X["timestamp"].dt.month
        X.drop(columns=["timestamp"], inplace=True)
        return X

class OutlierDetector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.q1 = X["amount"].quantile(0.25)
        self.q3 = X["amount"].quantile(0.75)
        self.iqr = self.q3 - self.q1
        self.lower_bound = self.q1 - 1.5 * self.iqr
        self.upper_bound = self.q3 + 1.5 * self.iqr
        return self

    def transform(self, X):
        X = X.copy()
        X["amount_outlier"] = ((X["amount"] < self.lower_bound) |
                               (X["amount"] > self.upper_bound)).astype(int)
        return X

class CategoryImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.category_medians = X.groupby('category')['amount'].median().to_dict()
        self.global_median = X['amount'].median()
        return self

    def transform(self, X):
        X = X.copy()
        for category, median in self.category_medians.items():
            mask = (X['category'] == category) & (X['amount'].isna())
            X.loc[mask, 'amount'] = median
        X['amount'] = X['amount'].fillna(self.global_median)
        return X

class MerchantImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'merchant' in X.columns:
            X['merchant'] = X['merchant'].fillna('Unknown')
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, initial_drop=None, final_drop=None):
        self.initial_drop = initial_drop if initial_drop else []
        self.final_drop = final_drop if final_drop else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.initial_drop:
            if col in X.columns:
                X.drop(columns=col, inplace=True)
        for col in self.final_drop:
            if col in X.columns:
                X.drop(columns=col, inplace=True)
        return X

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = [
                "category", "payment_method", "account_type", "device_info"
            ]
            numerical_columns = ["amount"]
            initial_drop = ["transaction_id", "user_id", "location", "fraud_reason"]
            final_drop = ["merchant"]

            preprocessing_pipeline = Pipeline([
                ('initial_column_dropper', ColumnDropper(initial_drop=initial_drop)),
                ('merchant_imputer', MerchantImputer()),
                ('category_imputer', CategoryImputer()),
                ('date_extractor', DateExtractor()),
                ('outlier_detector', OutlierDetector()),
                ('final_column_dropper', ColumnDropper(final_drop=final_drop))
            ])

            columns_after_preprocessing = [
                "category", "payment_method", "account_type", "device_info",
                "amount", "year", "month", "amount_outlier"
            ]

            feature_engineering = ColumnTransformer([
                ('cat_encoder', OneHotEncoder(sparse_output=False, drop='first'), 
                 [col for col in categorical_columns if col in columns_after_preprocessing]),
                ('num_cols', 'passthrough', 
                 [col for col in numerical_columns + ['year', 'month', 'amount_outlier'] 
                  if col in columns_after_preprocessing])
            ])

            full_pipeline = Pipeline([
                ('preprocessing', preprocessing_pipeline),
                ('feature_engineering', feature_engineering)
            ])

            logging.info("Pipeline created successfully")
            return full_pipeline

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "is_fraud"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing pipeline on training data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info("Training data transformed")

            logging.info("Applying preprocessing pipeline on test data")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Test data transformed")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessing object saved")

            return (
                input_feature_train_arr, 
                input_feature_test_arr, 
                target_feature_train_df.to_numpy(), 
                target_feature_test_df.to_numpy(), 
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)
