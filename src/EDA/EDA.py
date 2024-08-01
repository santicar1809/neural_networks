import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import os
import numpy as np
import pandas as pd

def eda_report(data,features,target):
    
    describe_result=data.describe()
    
    eda_path = './files/datasets/modeling_output/figures/'

    if not os.path.exists(eda_path):
        os.makedirs(eda_path)

    # Exporting the file
    with open('./files/datasets/modeling_output/reports/describe.txt', 'w') as f:
        f.write(describe_result.to_string())

    # Exporting general info
    with open('./files/datasets/modeling_output/reports/info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__
    
    fig1,ax1=plt.subplots()    
    ax1.hist(data['real_age'])
    ax1.set_title('Distribución de la edad')
    plt.show()
    fig1.savefig(eda_path+'fig1.png')
    
    fig2,ax2=plt.subplots()
    ax2.boxplot(data['real_age'])
    ax2.set_title('Distribución de la edad')
    plt.show()
    fig2.savefig(eda_path+'fig2.png')
    
    # muestra 16 imágenes
    fig = plt.figure(figsize=(10,10))
    for i in range(15):
        fig.add_subplot(4, 4, i+1)
        plt.imshow(features[i])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    fig.savefig(eda_path+'fig3.png')
    
    print(target[:15])
    target_1=pd.Series(target[:15])
    target_1.to_csv('./files/datasets/modeling_output/reports/target.txt', sep='\t', index=False)