import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
def read_dir_csv(path="train"):

    csv_list = os.listdir(path)

    df = pd.DataFrame()

    for file in csv_list:
        if file.endswith(".csv"):
            df_temp = pd.read_csv(f"{path}/{file}")
            df = pd.concat([df, df_temp])
    return df

def mean_10min(df):
    """
    將資料按時間區做平均
    """
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    location = df["LocationCode"].unique()
    new_df = pd.DataFrame()
    
    for l in location:
        l_df = df[df["LocationCode"] == l]
        l_df.set_index('DateTime', inplace=True)
        l_df = l_df.resample('10min').mean().dropna()  
        l_df = l_df.reset_index()

        new_df = pd.concat([new_df, l_df], ignore_index=True)
    
    return new_df

def spilt_data_with_datetime(df: pd.DataFrame, location_ori):
    datetime_list = list(df['DateTime'])
    

    data_df = df.drop(columns=['DateTime', 'Power(mW)', 'WindSpeed(m/s)', 'Pressure(hpa)'])
    label_df = df['Power(mW)']

    data_label_list = []
    last_index = 0

    start_time = []

    for i in range(1, len(datetime_list) - 1):
        date = datetime_list[i]
        last_date = datetime_list[i - 1]
        if date.day != last_date.day or \
            date.hour * 60 + date.minute - 10 != last_date.hour * 60 + last_date.minute or \
                location_ori[i] != location_ori[i - 1]:
            
            start_time.append(datetime_list[last_index].hour)
            data = torch.from_numpy(np.array(data_df.iloc[last_index: i]))
            label = torch.from_numpy(np.array(label_df.iloc[last_index: i]))

            data_label_list.append((len(data), data, label))
            last_index = i
    
    return data_label_list, start_time


