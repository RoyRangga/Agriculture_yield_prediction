import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
#columns transformer penting untuk membuat sebuah  pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
#            numerical_columns=[
#                'Tn', 'Tx', 'Tavg',
#                'RH_avg', 'RR', 'ss',
#                'ff_x', 'ddd_x', 'ff_avg', 
#                'tahun','luar_wilayah_hektar']
            numerical_columns=['Tavg','RR','ss','tahun','luar_wilayah_hektar']
            categorical_columns=['province_name', 'nama_komoditas']

            num_pipeline=Pipeline(
                steps=[
                    #("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    #("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    #("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"numerical columns: {numerical_columns}")
            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_columns),
                    ("categorical_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformation_object()

            target_column_name="hasil_panen"
#            numerical_columns=['Tn', 'Tx', 'Tavg','RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg','tahun','luar_wilayah_hektar']
            numerical_columns=['Tavg','RR','ss','tahun','luar_wilayah_hektar']
            categorical_columns=['province_name', 'nama_komoditas']

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info("applying preprocessing object on training and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df).toarray()

            train_arr=np.c_[
                input_feature_train_arr, target_feature_train_df
            ]
            test_arr=np.c_[
                input_feature_test_arr, target_feature_test_df
            ]
            logging.info("saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)