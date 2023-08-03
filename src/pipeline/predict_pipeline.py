import sys
import os
import pandas as pd
import numpy  as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 province_name:str,
                 nama_komoditas:str,
#                 Tn:float,
#                 Tx:float,
                 Tavg:float,
#                 RH_avg:float,
                 RR:float,
                 ss:float,
#                 ff_x:float,
#                 ddd_x:float,
#                 ff_avg:float,
                 luar_wilayah_hektar:float,
                 tahun:float):
        
        self.province_name=province_name
        self.nama_komoditas=nama_komoditas
#        self.Tn=Tn
#        self.Tx=Tx
        self.Tavg=Tavg
#        self.RH_avg=RH_avg
        self.RR=RR
        self.ss=ss
#        self.ff_x=ff_x
#        self.ddd_x=ddd_x
#        self.ff_avg=ff_avg
        self.luar_wilayah_hektar=luar_wilayah_hektar
        self.tahun=tahun
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "province_name":[self.province_name],
                "nama_komoditas":[self.nama_komoditas],
#                "Tn":[self.Tn],
#                "Tx":[self.Tx],
                "Tavg":[self.Tavg],
#                "RH_avg":[self.RH_avg],
                "RR":[self.RR],
                "ss":[self.ss],
#                "ff_x":[self.ff_x],
#                "ddd_x":[self.ddd_x],
#                "ff_avg":[self.ff_avg],
                "luar_wilayah_hektar":[self.luar_wilayah_hektar],
                "tahun":[self.tahun]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)