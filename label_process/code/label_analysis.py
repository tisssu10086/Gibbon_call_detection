import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

'''This script is to analys the label on duration distribution and interval distribution'''


file_dic = pickle.load(open('../../label/cross_val_label_dic/all_file_name.p', 'rb'))

# analysis the duration distribution in label
duration_dist = np.array([])

for file_name in file_dic:
    gibbon_timestamps = pd.read_csv('../../label/processed_label/' + file_name + '.data', sep=',')
    duration = np.asarray(gibbon_timestamps['Duration'])
    duration_dist = np.concatenate((duration_dist, duration), axis = 0)

duration_dist = duration_dist.astype(int)
np.savetxt('../label_analysis_result/gibbon_call_duration.csv', duration_dist, delimiter= ',')

plt.cla()
plt.hist(duration_dist,range(12))
plt.xlabel('Duration of gibbon phrase',fontsize=14)
plt.ylabel('Number of gibbon phrase',fontsize=14)
plt.xticks(range(12))
# plt.title('Distribution of gibbon call duration',fontsize=16)
plt.savefig('../label_analysis_result/gibbon_call_duration.png')


#analys the interval distribution between adjcent gibbon calls within 11s which is maximum duration of a gibbon call
interval_dist = []

for file_name in file_dic:
    gibbon_timestamps = pd.read_csv('../../label/processed_label/' + file_name + '.data', sep=',')
    for i in gibbon_timestamps.index:
        if i+1 in gibbon_timestamps.index:
            interval_dist.append(gibbon_timestamps['Start'][i+1] - gibbon_timestamps['End'][i])

interval_dist = np.asarray(interval_dist).astype(int)
interval_dist = interval_dist[(interval_dist <= 11)]
plt.cla()
plt.hist(interval_dist)
plt.xlabel('Duration of interval between adjcent gibbon call within 11s',fontsize=14)
plt.ylabel('Adjcent gibbon call interval counts',fontsize=14)
plt.title('Distribution of interval duration between adjcent gibbon calls within 11s')
plt.xticks(range(11))
plt.savefig('../label_analysis_result/interval_dist_11s.png')

