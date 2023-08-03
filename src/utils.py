import os 
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle

def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report={}
        best_parameters = {}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            grid_search=GridSearchCV(model,para,cv=3)
            grid_search.fit(X_train,y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train,y_train)

            y_train_predict=model.predict(X_train)
            y_test_predict=model.predict(X_test)
            train_model_score=r2_score(y_train,y_train_predict)
            test_model_score=r2_score(y_test,y_test_predict)
#            test_model_rmse=mean_squared_error(y_test,y_test_predict)
            report[list(models.keys())[i]]=test_model_score
#            report[list(param.keys())[i]]=grid_search.best_params_
#            best_parameters[list(models.keys())[i]] = grid_search.best_params_
#            best_parameter=grid_search.best_params_

        return report, test_model_score, 
    
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)