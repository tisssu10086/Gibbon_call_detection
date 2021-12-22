from pathlib import Path
import gc
import os
import pickle 
import numpy as np


import nn_function
import dataset
import label_matching
import post_process
import cnn_gru_structure
import nn_cv


##############################################
#load file dictionary
file_dic = []
for file_name in Path('../../label/processed_label').glob('*.data'):
    file_dic.append(file_name.stem)

file_dic.sort()


#train network
data_p = dataset.Hparams(overwritten = True)
model_p = nn_function.Hparams()

cv_network_train_test = nn_cv.Network_CV(cv_file_dic = file_dic, 
                                        cv_fold_number = 4, 
                                        data_parameter = data_p, 
                                        model_parameter = model_p, 
                                        seed = 42, 
                                        cv_filesplit_savedic = '../../label/cross_val_label_dic', 
                                        cv_filesplit_overwrite = False,
                                        network_model = cnn_gru_structure.CRNN(1, 256),
                                        model_save_dic = '../model/crnn_model',
                                        trainning_visulise_dic = '../train_visulise',
                                        train_result_dic = '../crnn_results',
                                        test_result_dic = '../crnn_results')
#train and test network
cv_network_train_test.cv_train()
cv_network_train_test.cv_test()




#evaluate network with different postprocessing
crnn_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= data_p, model_parameter= model_p, model_path= '../model')
crnn_evaluation.cv_post_train_test()

crnn_result_summary = {}
crnn_result_summary['nn_result'] = crnn_evaluation.nn_result
crnn_result_summary['threshold_result'] = crnn_evaluation.threshold_result
crnn_result_summary['average_result'] = crnn_evaluation.average_result
crnn_result_summary['hmm_bino_result'] = crnn_evaluation.hmm_bino_result
crnn_result_summary['hmm_gmm_result'] = crnn_evaluation.hmm_gmm_result
crnn_result_summary['hmm_bino_threshold_result'] = crnn_evaluation.hmm_bino_threshold_result
pickle.dump(crnn_result_summary, open('../crnn_results/crnn_results_summary.p', 'wb'))










#test model performance with different simulation data
#################################################################################
#pitch shift experiments
pitch_shift_results = {}
pitch_shift_set = [-1, -0.5, 0.5, 1]
for pitch_shift in  pitch_shift_set:
    simulation_p = dataset.Hparams(overwritten = True, shift_step = pitch_shift)
    crnn_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= simulation_p, model_parameter= model_p, 
                                            model_path= '../model', augment= dataset.Pitch_shift(simulation_p))
    crnn_evaluation.cv_post_evaluate()

    pitch_shift_results['shift_step_{:.3f}_nn_result'.format(pitch_shift)] = crnn_evaluation.nn_result
    pitch_shift_results['shift_step_{:.3f}_threshold_result'.format(pitch_shift)] = crnn_evaluation.threshold_result
    pitch_shift_results['shift_step_{:.3f}_average_result'.format(pitch_shift)] = crnn_evaluation.average_result
    pitch_shift_results['shift_step_{:.3f}_hmm_bino_result'.format(pitch_shift)] = crnn_evaluation.hmm_bino_result
    pitch_shift_results['shift_step_{:.3f}_hmm_gmm_result'.format(pitch_shift)] = crnn_evaluation.hmm_gmm_result
    pitch_shift_results['shift_step_{:.3f}_hmm_bino_threshold_result'.format(pitch_shift)] = crnn_evaluation.hmm_bino_threshold_result

print(pitch_shift_results)
pickle.dump(pitch_shift_results, open('../crnn_results/pitch_shift_results.p', 'wb'))
# pitch_shift_results = pickle.load(open('../crnn_results/pitch_shift_results.p', 'rb'))


#time stretch experiments
time_stretch_results = {}
time_stretch_set = [0.81, 0.93, 1.07, 1.23]
for time_stretch in  time_stretch_set:
    simulation_p = dataset.Hparams(overwritten = True, stretch_rate = time_stretch)
    crnn_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= simulation_p, model_parameter= model_p,
                                            model_path= '../model', augment= dataset.Time_stretch(simulation_p))
    crnn_evaluation.cv_post_evaluate()

    time_stretch_results['stretch_rate_{:.3f}_nn_result'.format(time_stretch)] = crnn_evaluation.nn_result
    time_stretch_results['stretch_rate_{:.3f}_threshold_result'.format(time_stretch)] = crnn_evaluation.threshold_result
    time_stretch_results['stretch_rate_{:.3f}_average_result'.format(time_stretch)] = crnn_evaluation.average_result
    time_stretch_results['stretch_rate_{:.3f}_hmm_bino_result'.format(time_stretch)] = crnn_evaluation.hmm_bino_result
    time_stretch_results['stretch_rate_{:.3f}_hmm_gmm_result'.format(time_stretch)] = crnn_evaluation.hmm_gmm_result
    time_stretch_results['stretch_rate_{:.3f}_hmm_bino_threshold_result'.format(time_stretch)] = crnn_evaluation.hmm_bino_threshold_result

print(time_stretch_results)
pickle.dump(time_stretch_results, open('../crnn_results/time_stretch_results.p', 'wb'))
# time_stretch_results = pickle.load(open('../crnn_results/time_stretch_results.p', 'rb'))



#cropping experiments
crop_results = {}
crop_set = [0.9, 0.8, 0.7, 0.6]
for crop in  crop_set:
    simulation_p = dataset.Hparams(overwritten = True, crop_rate = crop)
    crnn_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= simulation_p, model_parameter= model_p,
                                            model_path= '../model', augment= dataset.Cropping(simulation_p))
    crnn_evaluation.cv_post_evaluate()

    crop_results['crop_rate_{:.3f}_nn_result'.format(crop)] = crnn_evaluation.nn_result
    crop_results['crop_rate_{:.3f}_threshold_result'.format(crop)] = crnn_evaluation.threshold_result
    crop_results['crop_rate_{:.3f}_average_result'.format(crop)] = crnn_evaluation.average_result
    crop_results['crop_rate_{:.3f}_hmm_bino_result'.format(crop)] = crnn_evaluation.hmm_bino_result
    crop_results['crop_rate_{:.3f}_hmm_gmm_result'.format(crop)] = crnn_evaluation.hmm_gmm_result
    crop_results['crop_rate_{:.3f}_hmm_bino_threshold_result'.format(crop)] = crnn_evaluation.hmm_bino_threshold_result

print(crop_results)
pickle.dump(crop_results, open('../crnn_results/crop_results.p', 'wb'))
# crop_results = pickle.load(open('../crnn_results/crop_results.p', 'rb'))





# #volume change experiment
# volume_change_results = {}
# volume_change_set = [-10, -5, 5, 10]
# for volume_change in volume_change_set:
#     simulation_p = dataset.Hparams(overwritten = True, volume_change = volume_change)
#     crnn_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= simulation_p, model_parameter= model_p,
#                                             model_path= '../model/crnn_model', augment= dataset.Volume_changing(simulation_p))

#     volume_change_results['volume_change_{:.3f}_nn_result'.format(volume_change)] = crnn_evaluation.nn_result
#     volume_change_results['volume_change_{:.3f}_threshold_result'.format(volume_change)] = crnn_evaluation.threshold_result
#     volume_change_results['volume_change_{:.3f}_average_result'.format(volume_change)] = crnn_evaluation.average_result
#     volume_change_results['volume_change_{:.3f}_hmm_bino_result'.format(volume_change)] = crnn_evaluation.hmm_bino_result
#     volume_change_results['volume_change_{:.3f}_hmm_gmm_result'.format(volume_change)] = crnn_evaluation.hmm_gmm_result
#     volume_change_results['volume_change_{:.3f}_hmm_bino_threshold_result'.format(volume_change)] = crnn_evaluation.hmm_bino_threshold_result

# print(volume_change_results)
# pickle.dump(volume_change_results, open('../crnn_results/volume_change_results.p', 'wb'))


























