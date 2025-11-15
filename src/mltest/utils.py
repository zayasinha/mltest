import os
import sys
from src.mltest.exception import CustomException
from src.mltest.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np


load_dotenv()

host = os.getenv("host")
user = os.getenv("user")    
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    logging.info("Reading data from SQL database")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db 
        )
        logging.info(f"Database connection established successfully: {mydb}")
        df=pd.read_sql_query("select * from bmw",mydb)
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex)
    
def save_object(file_path:str, obj:object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    