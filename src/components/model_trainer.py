import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("splitting training and testing input data")
            X_train, y_train, X_test, y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                #"K-Neighbors regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }
            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32]
                },                
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32]
                }
            }

            model_report, best_params =evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models, param = params)
           ##retrieving best score from dictionary of model_report
            best_model_score=max(sorted(model_report.values()))
            ##retrieving best score'names from dictionary of model_report
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info("best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                )

#            print("Model Report:")
#            for model_name, r2_score in model_report.items():
#                best_parameter = best_params[model_name] 
#                print(f"{model_name}: R2 Score = {r2_score}, Best Params: {best_params[model_name]}")
#                print(f"{model_name}: R2 Score = {r2_score}, Best Params: {best_parameter}")


            predicted=best_model.predict(X_test)
            r2_squared=r2_score(y_test,predicted)

#            print("Best Model:", best_model_name)
#            print("Best Model R2 Score:", best_model_score)
#            print("Best Model Parameters:", best_params[best_model_name])
#            print("R2 Squared:", r2_squared)
            parameters=best_model.get_params()
            data_parameters=pd.DataFrame()
            data_parameters["parameter_name"]=parameters.keys()
            data_parameters["argument_name"]=list(parameters.values())

            return r2_squared, model_report, data_parameters

        except Exception as e:
            raise CustomException(e, sys)