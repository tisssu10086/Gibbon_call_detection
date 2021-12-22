import os
import numpy as np
import pandas as pd

'''This script is for preprocessing the label, finding the mistake in it and stroe label in a unified format in processed_label dic'''

file_dic_Extra = os.listdir('../../label/Extra_Labels')
file_dic_Train = os.listdir('../../label/Train_labels')
file_dic_Test = os.listdir('../../label/Test_labels')

#store the gibbon call duration distribution
duration_dist = np.array([])
duration_dist2 = np.array([])

for file_name in file_dic_Extra: # go through the Extra_Labels dictionary
    if file_name[0] == 'g':
        gibbon_timestamps = pd.read_csv('../../label/Extra_Labels/' + file_name, sep=',')
        duration = np.asarray(gibbon_timestamps['Duration'])
        duration_dist = np.concatenate((duration_dist, duration), axis = 0)
        # test the whether the duration equals to 'end' - 'start'
        duration2 = np.asarray(gibbon_timestamps['End'] - gibbon_timestamps['Start'])
        duration_dist2 = np.concatenate((duration_dist2, duration2), axis = 0)  
        if duration.size != 0 :
            if min(duration) <= 0:
                print(file_name, 'has wrong record')  
        gibbon_timestamps.to_csv('../../label/processed_label/' + file_name[2:], index = 0)

for file_name in file_dic_Train: # go through the Train_Labels dictionary
    if file_name[0] == 'g':
        gibbon_timestamps = pd.read_csv('../../label/Train_Labels/' + file_name, sep=',')
        duration = np.asarray(gibbon_timestamps['Duration'])
        duration_dist = np.concatenate((duration_dist, duration), axis = 0)
        # test the whether the duration equals to 'end' - 'start'
        duration2 = np.asarray(gibbon_timestamps['End'] - gibbon_timestamps['Start'])
        duration_dist2 = np.concatenate((duration_dist2, duration2), axis = 0)
        if duration.size != 0:
            if min(duration) <= 0:
                print(file_name, 'has wrong record')  
        gibbon_timestamps.to_csv('../../label/processed_label/' + file_name[2:], index = 0)

# result show that duration equals to 'end' - 'start'
test_duration = duration_dist2 == duration_dist
duration_test_result = np.where(test_duration == False)
if duration_test_result[0].size == 0:
    print('duration equals to end - star')
else:
    print('duration record typo exist')

for file_name in file_dic_Test: # go through the Test_Labels dictionary and save data to processed label dictionary
    gibbon_timestamps = pd.read_csv('../../label/Test_Labels/' + file_name, sep=',')
    gibbon_timestamps['End'] = gibbon_timestamps['Start'] + gibbon_timestamps['Duration']
    gibbon_timestamps = gibbon_timestamps[['Start', 'End', 'Duration']]
    if duration.size != 0 :
        if min(duration) <= 0:
            print(file_name, 'has wrong record')  
    gibbon_timestamps.to_csv('../../label/processed_label/' + file_name[:-9] + '.data', index = 0)


# g_HGSM3BD_0+1_20160305_060000.data has wrong record
# g_HGSM3AC_0+1_20160312_055400.data has wrong record
# this two file has minus or equals to zero duration because of typo, these error have been fixed in processed-label manually.