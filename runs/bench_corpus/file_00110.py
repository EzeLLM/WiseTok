# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:11:33 2020

@author: Carlos Jose Munoz
"""

import numpy as np;
import matplotlib.pyplot as plt;
from csv import reader as reader_csv;
import scipy.signal as signal;
import scipy.io as sio;

mat_contents = sio.loadmat('ecg.mat')

print('the loaded keys are: '+ str(mat_contents.keys()));


data=mat_contents['ecg']

sio.savemat('C:/Users/Carlos Jose Munoz/Desktop/semestres/Senal_filtrada.mat',{'senalfiltrada': data})

sensores,puntos,ensayos=data.shape
data=np.reshape(data,(sensores,puntos*ensayos),order = 'F')

data=open('P1_RAWEEG_2018-11-15_Ensayo_1min.txt',"r");
lines=reader_csv(data);

row_number=0;
header= '';
channels =11;
header_size=6;

data=[];

for row in lines:
    if row_number < header_size:
        header=header+row[0]+'\n';
        row_number = row_number+1;
        print(row);
    else:
        temp=[];
        counter=0;
        for column in row:
            if counter ==0:
                counter = counter+1;
                continue;
            elif counter == channels+1:
                break;
            else: 
                temp.append(float(column));
                
            counter = counter+1;
        data.append(temp);

biosignal= np.asarray(data, order = 'C');
biosignal = np.transpose(biosignal);

t=np.linspace(0, 32, 8074);
dc=10
for i in range (7):
    plt.plot( biosignal [i,:]+dc*i, linewidth = 0.1); #igual para todas
    #plt.title ("EEG")
plt.title ("EEG")
plt.savefig('foo.png')
plt.show();