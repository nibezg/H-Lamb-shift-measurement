#%%
import numpy as np
from numpy import random
import os
import datetime as dt
import time

#%%

#%%
x = np.random.uniform(-0.1,0.1)

#%%
x
#%%
if x > 0:
    pos_bool = True
else:
    pos_bool = False

x_abs = np.abs(x)

rounded_blind_kHz = np.round(x_abs*1E3, decimals=1)

rep_blind = str(int(rounded_blind_kHz*10))

freq_kHz_arr = np.zeros((len(rep_blind),1))

for i in range(freq_kHz_arr.shape[0]):
    freq_kHz_arr[i] = int(rep_blind[len(rep_blind)-1-i])* 10**(i-1)

for i in range(freq_kHz_arr.shape[0])[1:]:
    freq_kHz_arr[i] = np.sum(freq_kHz_arr[i-1:i+1])

if not(pos_bool):
    freq_kHz_arr = np.append(freq_kHz_arr, freq_kHz_arr[-1]*(-1))

else:
    freq_kHz_arr = np.append(freq_kHz_arr, freq_kHz_arr[-1])
#%%

freq_kHz = 909845.5
sigma_freq_kHz = 8.1

lamb_shift_ans_arr = freq_kHz_arr + freq_kHz

#%%
lamb_shift_ans_arr
#%%
for i in range(lamb_shift_ans_arr.shape[0]):
    time.sleep(i)
    print(lamb_shift_ans_arr[i])
