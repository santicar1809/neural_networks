import pandas as pd

def load_datasets():
    
    data=pd.read_csv('./files/datasets/input/faces/labels.csv')
    return data