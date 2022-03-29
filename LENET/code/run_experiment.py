import matplotlib.pyplot as plt
import gc
import os
import pickle 
import numpy as np
from pathlib import Path

import nn_function
import dataset
import label_matching
import post_process
import cnn_function
import nn_cv




#load file dictionary
file_dic = []
for file_name in Path('../../label/processed_label').glob('*.data'):
    file_dic.append(file_name.stem)

file_dic.sort()


#train network
data_p = dataset.Hparams(overwritten = False, seq_len = 1, train_hop_len = 1, test_hop_len = 1)
model_p = nn_function.Hparams(num_epochs = 30, 
                            batch_size = 1024, 
                            learning_rate = 1e-3, 
                            lr_decay_step_size = 30, 
                            lr_decay_rate =  1,
                            weight_decay_rate = 1e-1,
                            early_stop_patience = 30,
                            num_classes = 2)


cv_network_train_test = nn_cv.Network_CV(cv_file_dic= file_dic,
                                    cv_fold_number = 4, 
                                    data_parameter = data_p, 
                                    model_parameter = model_p, 
                                    seed = 0, 
                                    cv_filesplit_savedic = '../../label/cross_val_label_dic', 
                                    cv_filesplit_overwrite = False,
                                    network_model = cnn_function.LENET(),
                                    model_save_dic = '../model/cnn_model',
                                    trainning_visulise_dic = '../train_visulise',
                                    train_result_dic = '../results',
                                    test_result_dic = '../results')

#train and test network
cv_network_train_test.cv_train()
cv_network_train_test.cv_test()

print(cv_network_train_test.train_time)
print(cv_network_train_test.test_time)


#evaluate network with different postprocessing
lenet_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= data_p, model_parameter= model_p, model_path= '../model')
lenet_evaluation.cv_post_train_test()

lenet_result_summary = {}
lenet_result_summary['nn_result'] = lenet_evaluation.nn_result
lenet_result_summary['threshold_result'] = lenet_evaluation.threshold_result
lenet_result_summary['average_result'] = lenet_evaluation.average_result
lenet_result_summary['hmm_bino_result'] = lenet_evaluation.hmm_bino_result
lenet_result_summary['hmm_gmm_result'] = lenet_evaluation.hmm_gmm_result
lenet_result_summary['hmm_bino_threshold_result'] = lenet_evaluation.hmm_bino_threshold_result
pickle.dump(lenet_result_summary, open('../results/lenet_results_summary.p', 'wb'))






#test model performance with different simulation data
#################################################################################
#pitch shift experiments
pitch_shift_results = {}
pitch_shift_set = [-1, -0.5, 0.5, 1]
for pitch_shift in  pitch_shift_set:
    simulation_p = dataset.Hparams(overwritten = True, shift_step = pitch_shift, seq_len = 1, train_hop_len = 1, test_hop_len = 1)
    lenet_evaluation = nn_cv.CV_evaluation(cv_file_dic = file_dic, data_parameter = simulation_p, model_parameter = model_p, 
                                            model_path = '../model', augment = dataset.Pitch_shift(simulation_p))
    lenet_evaluation.cv_post_evaluate()

    pitch_shift_results['shift_step_{:.3f}_nn_result'.format(pitch_shift)] = lenet_evaluation.nn_result
    pitch_shift_results['shift_step_{:.3f}_threshold_result'.format(pitch_shift)] = lenet_evaluation.threshold_result
    pitch_shift_results['shift_step_{:.3f}_average_result'.format(pitch_shift)] = lenet_evaluation.average_result
    pitch_shift_results['shift_step_{:.3f}_hmm_bino_result'.format(pitch_shift)] = lenet_evaluation.hmm_bino_result
    pitch_shift_results['shift_step_{:.3f}_hmm_gmm_result'.format(pitch_shift)] = lenet_evaluation.hmm_gmm_result
    pitch_shift_results['shift_step_{:.3f}_hmm_bino_threshold_result'.format(pitch_shift)] = lenet_evaluation.hmm_bino_threshold_result

print(pitch_shift_results)
pickle.dump(pitch_shift_results, open('../results/pitch_shift_results.p', 'wb'))



#time stretch experiments
time_stretch_results = {}
time_stretch_set = [0.81, 0.93, 1.07, 1.23]
for time_stretch in  time_stretch_set:
    simulation_p = dataset.Hparams(overwritten = True, stretch_rate = time_stretch, seq_len = 1, train_hop_len = 1, test_hop_len = 1)
    lenet_evaluation = nn_cv.CV_evaluation(cv_file_dic = file_dic, data_parameter = simulation_p, model_parameter = model_p, 
                                            model_path = '../model', augment= dataset.Time_stretch(simulation_p))
    lenet_evaluation.cv_post_evaluate()

    time_stretch_results['stretch_rate_{:.3f}_nn_result'.format(time_stretch)] = lenet_evaluation.nn_result
    time_stretch_results['stretch_rate_{:.3f}_threshold_result'.format(time_stretch)] = lenet_evaluation.threshold_result
    time_stretch_results['stretch_rate_{:.3f}_average_result'.format(time_stretch)] = lenet_evaluation.average_result
    time_stretch_results['stretch_rate_{:.3f}_hmm_bino_result'.format(time_stretch)] = lenet_evaluation.hmm_bino_result
    time_stretch_results['stretch_rate_{:.3f}_hmm_gmm_result'.format(time_stretch)] = lenet_evaluation.hmm_gmm_result
    time_stretch_results['stretch_rate_{:.3f}_hmm_bino_threshold_result'.format(time_stretch)] = lenet_evaluation.hmm_bino_threshold_result

print(time_stretch_results)
pickle.dump(time_stretch_results, open('../results/time_stretch_results.p', 'wb'))



#cropping experiments
crop_results = {}
crop_set = [0.9, 0.8, 0.7, 0.6]
for crop in  crop_set:
    simulation_p = dataset.Hparams(overwritten = True, crop_rate = crop, seq_len = 1, train_hop_len = 1, test_hop_len = 1)
    lenet_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= simulation_p, model_parameter= model_p,
                                            model_path= '../model', augment= dataset.Cropping(simulation_p))
    lenet_evaluation.cv_post_evaluate()

    crop_results['crop_rate_{:.3f}_nn_result'.format(crop)] = lenet_evaluation.nn_result
    crop_results['crop_rate_{:.3f}_threshold_result'.format(crop)] = lenet_evaluation.threshold_result
    crop_results['crop_rate_{:.3f}_average_result'.format(crop)] = lenet_evaluation.average_result
    crop_results['crop_rate_{:.3f}_hmm_bino_result'.format(crop)] = lenet_evaluation.hmm_bino_result
    crop_results['crop_rate_{:.3f}_hmm_gmm_result'.format(crop)] = lenet_evaluation.hmm_gmm_result
    crop_results['crop_rate_{:.3f}_hmm_bino_threshold_result'.format(crop)] = lenet_evaluation.hmm_bino_threshold_result

print(crop_results)
pickle.dump(crop_results, open('../results/crop_results.p', 'wb'))






