import gc
import os
import pickle 
import numpy as np
from pathlib import Path

import dataset
import label_matching
import post_process
import svm_cv



#load file dictionary
file_dic = []
for file_name in Path('../../label/processed_label').glob('*.data'):
    file_dic.append(file_name.stem)

file_dic.sort()


#data parameter
data_p = dataset.Hparams(overwritten = False, seq_len = 1, train_hop_len = 1, test_hop_len = 1)

#evaluate network with different postprocessing
svm_evaluation = svm_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter_train= data_p, data_parameter_predict= data_p, model_path= '../model')
svm_evaluation.cv_post_train_test()

svm_result_summary = {}
svm_result_summary['svm_result'] = svm_evaluation.svm_result
svm_result_summary['hmm_bino_result'] = svm_evaluation.hmm_bino_result
pickle.dump(svm_result_summary, open('../results/svm_results_summary.p', 'wb'))
# svm_result = pickle.load(open('../results/svm_results_summary.p', 'rb'))





#test model performance with different simulation data
#################################################################################
#pitch shift experiments
pitch_shift_results = {}
pitch_shift_set = [-1, -0.5, 0.5, 1]
for pitch_shift in  pitch_shift_set:
    simulation_p = dataset.Hparams(overwritten = True, shift_step = pitch_shift, seq_len = 1, train_hop_len = 1, test_hop_len = 1)

    svm_evaluation = svm_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter_train= data_p, augment_train = None,
                                            data_parameter_predict= simulation_p, augment_predict = dataset.Pitch_shift(simulation_p), model_path= '../model')
    svm_evaluation.cv_post_evaluate()

    pitch_shift_results['shift_step_{:.3f}_svm_result'.format(pitch_shift)] = svm_evaluation.svm_result
    pitch_shift_results['shift_step_{:.3f}_hmm_bino_result'.format(pitch_shift)] = svm_evaluation.hmm_bino_result

print(pitch_shift_results)
pickle.dump(pitch_shift_results, open('../results/pitch_shift_results.p', 'wb'))




#time stretch experiments
time_stretch_results = {}
time_stretch_set = [0.81, 0.93, 1.07, 1.23]
for time_stretch in  time_stretch_set:
    simulation_p = dataset.Hparams(overwritten = True, stretch_rate = time_stretch, seq_len = 1, train_hop_len = 1, test_hop_len = 1)

    svm_evaluation = svm_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter_train= data_p, augment_train= None,
                                            data_parameter_predict = simulation_p, augment_predict= dataset.Time_stretch(simulation_p), model_path= '../model')
    svm_evaluation.cv_post_evaluate()

    time_stretch_results['stretch_rate_{:.3f}_svm_result'.format(time_stretch)] = svm_evaluation.svm_result
    time_stretch_results['stretch_rate_{:.3f}_hmm_bino_result'.format(time_stretch)] = svm_evaluation.hmm_bino_result

print(time_stretch_results)
pickle.dump(time_stretch_results, open('../results/time_stretch_results.p', 'wb'))




#crop experiments
crop_results = {}
crop_set = [0.9, 0.8, 0.7, 0.6]
for crop in  crop_set:
    simulation_p = dataset.Hparams(overwritten = True, crop_rate = crop, seq_len = 1, train_hop_len = 1, test_hop_len = 1)

    svm_evaluation = svm_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter_train= data_p, augment_train=None,
                                            data_parameter_predict = simulation_p, augment_predict= dataset.Cropping(simulation_p), model_path= '../model')
    svm_evaluation.cv_post_evaluate()

    crop_results['crop_rate_{:.3f}_svm_result'.format(crop)] = svm_evaluation.svm_result
    crop_results['crop_rate_{:.3f}_hmm_bino_result'.format(crop)] = svm_evaluation.hmm_bino_result

print(crop_results)
pickle.dump(crop_results, open('../results/crop_results.p', 'wb'))






