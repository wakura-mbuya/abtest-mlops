import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from logger import Logger
from sklearn.model_selection import cross_validate


class Preprocess:

    def __init__(self) -> None:
        """Initilize class."""
        try:
            self.logger = Logger("preprocess.log").get_app_logger()
            self.logger.info(
                'Successfully Instantiated preprocess Class Object')
        except Exception:
            self.logger.exception(
                'Failed to Instantiate Preprocessing Class Object')
            sys.exit(1)

    def get_numerical_columns(self, df):
        """Get numerical columns from dataframe."""
        try:
            self.logger.info('Getting Numerical Columns from Dataframe')
            return df.select_dtypes(
                exclude="object").columns.tolist()
        except Exception:
            self.logger.exception(
                'Failed to get Numerical Columns from Dataframe')
            sys.exit(1)

    def get_categorical_columns(self, df):
        """Get categorical columns from dataframe."""
        try:
            self.logger.info('Getting Categorical Columns from Dataframe')
            return df.select_dtypes(
                include="object").columns.tolist()
        except Exception:
            self.logger.exception(
                'Failed to get Categorical Columns from Dataframe')
            sys.exit(1)

    def get_missing_values(self, df):
        """Get missing values from dataframe."""
        try:
            self.logger.info('Getting Missing Values from Dataframe')
            return df.isnull().sum()
        except Exception:
            self.logger.exception(
                'Failed to get Missing Values from Dataframe')
            sys.exit(1)

    def convert_to_datetime(self, df, column):
        """Convert column to datetime."""
        try:
            self.logger.info('Converting Column to Datetime')
            df[column] = pd.to_datetime(df[column])
            return df
        except Exception:
            self.logger.exception(
                'Failed to convert Column to Datetime')
            sys.exit(1)

    