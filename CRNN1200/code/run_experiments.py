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
import cnn_gru_structure
import nn_cv


##############################################
#load file dictionary
file_dic = []
for file_name in Path('../../label/processed_label').glob('*.data'):
    file_dic.append(file_name.stem)

file_dic.sort()


#train network
data_p = dataset.Hparams(overwritten = False,  seq_len = 1200, train_hop_len = 600, test_hop_len = 1200)
model_p = nn_function.Hparams(num_epochs = 60, 
                            batch_size = 2, 
                            learning_rate = 1e-3, 
                            lr_decay_step_size = 60, 
                            lr_decay_rate =  1,
                            weight_decay_rate = 0,
                            early_stop_patience = 60,
                            num_classes = 2)

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
# test_result = pickle.load(open('../crnn_results/test_conf_matrix_sum.p', 'rb'))



#evaluate network with different postprocessing
crnn_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= data_p, model_parameter= model_p, model_path= '../model/crnn_model')
crnn_evaluation.cv_nn_evaluate()

crnn_result_summary = {}
crnn_result_summary['nn_result'] = crnn_evaluation.nn_result
crnn_result_summary['threshold_result'] = crnn_evaluation.threshold_result
crnn_result_summary['average_result'] = crnn_evaluation.average_result
crnn_result_summary['hmm_bino_result'] = crnn_evaluation.hmm_bino_result
crnn_result_summary['hmm_gmm_result'] = crnn_evaluation.hmm_gmm_result
crnn_result_summary['hmm_bino_threshold_result'] = crnn_evaluation.hmm_bino_threshold_result
pickle.dump(crnn_result_summary, open('../crnn_results/crnn_results_summary.p', 'wb'))

































# #test model performance with different simulation data
# #################################################################################
# #pitch shift experiments
# pitch_shift_results = {}
# pitch_shift_set = [-1, -0.5, 0.5, 1]
# for pitch_shift in  pitch_shift_set:
#     simulation_p = dataset.Hparams(overwritten = True, shift_step = pitch_shift)

#     crnn_evaluation = CV_evaluation(cv_file_dic= file_dic, data_parameter= simulation_p, augment= dataset.Pitch_shift(simulation_p))
#     crnn_evaluation.cv_nn_evaluate()

#     pitch_shift_results['shift_step_{:.3f}_nn_result'.format(pitch_shift)] = crnn_evaluation.nn_result
#     pitch_shift_results['shift_step_{:.3f}_threshold_result'.format(pitch_shift)] = crnn_evaluation.threshold_result
#     pitch_shift_results['shift_step_{:.3f}_average_result'.format(pitch_shift)] = crnn_evaluation.average_result
#     pitch_shift_results['shift_step_{:.3f}_hmm_bino_result'.format(pitch_shift)] = crnn_evaluation.hmm_bino_result
#     pitch_shift_results['shift_step_{:.3f}_hmm_gmm_result'.format(pitch_shift)] = crnn_evaluation.hmm_gmm_result
#     pitch_shift_results['shift_step_{:.3f}_hmm_bino_threshold_result'.format(pitch_shift)] = crnn_evaluation.hmm_bino_threshold_result

# print(pitch_shift_results)
# pickle.dump(pitch_shift_results, open('../crnn_results/pitch_shift_results.p', 'wb'))




# #time stretch experiments
# time_stretch_results = {}
# time_stretch_set = [0.83, 0.91, 1.11, 1.25]
# for time_stretch in  time_stretch_set:
#     simulation_p = dataset.Hparams(overwritten = True, stretch_rate = time_stretch)

#     crnn_evaluation = CV_evaluation(cv_file_dic= file_dic, data_parameter= simulation_p, augment= dataset.Time_stretch(simulation_p))
#     crnn_evaluation.cv_nn_evaluate()

#     time_stretch_results['stretch_rate_{:.3f}_nn_result'.format(time_stretch)] = crnn_evaluation.nn_result
#     time_stretch_results['stretch_rate_{:.3f}_threshold_result'.format(time_stretch)] = crnn_evaluation.threshold_result
#     time_stretch_results['stretch_rate_{:.3f}_average_result'.format(time_stretch)] = crnn_evaluation.average_result
#     time_stretch_results['stretch_rate_{:.3f}_hmm_bino_result'.format(time_stretch)] = crnn_evaluation.hmm_bino_result
#     time_stretch_results['stretch_rate_{:.3f}_hmm_gmm_result'.format(time_stretch)] = crnn_evaluation.hmm_gmm_result
#     time_stretch_results['stretch_rate_{:.3f}_hmm_bino_threshold_result'.format(time_stretch)] = crnn_evaluation.hmm_bino_threshold_result

# print(time_stretch_results)
# pickle.dump(time_stretch_results, open('../crnn_results/time_stretch_results.p', 'wb'))







    # #initilize performance metrics
    # metrics = label_matching.result_analysis(0, 'hit_match')

    # for i in range(K_FOLD):

    #     train_test_split = dataset.cross_valid(seed = 42, split_number = 4, fold_needed = i, file_dic = file_dic, 
    #                                     save_path = Path('../../label/cross_val_label_dic'), overwritten = False, verbose = False)


    #     test_set = dataset.gibbon_dataset(cfg = simulation_p, dataset_type =  'test', dataset_usage= 'nonoverlap_pred', domain_transform = dataset.SpecImg_transform(simulation_p),
    #                             augment = dataset.Pitch_shift(simulation_p) ,pytorch_X_transform = dataset.Pytorch_data_transform(), pytorch_Y_transform = dataset.Pytorch_label_transform(), 
    #                             train_test_split = train_test_split)



    #     print('predict begin')
    #     crnn_predict = nn_function.model_predict(model_p, pred_seq_len = simulation_p.audio_len, seed = 42, model_path =  '../model/crnn_model' + str(i) + '.pt')
    #     crnn_predict.test_predict(test_set)
    #     #calculate and save test result
    #     metrics(crnn_predict.test_bino_predict['label'], crnn_predict.test_bino_predict['pred'])


    # metrics.result_process()
    # print(metrics.result_summary)
    # pitch_shift_results['pitch shift step{:.3f}'.format(pitch_shift)] = metrics.result_summary





# ###############################################################################################
# #time stretch experiments
# model_p = nn_function.Hparams()
# time_stretch_results = {}
# time_stretch_set = [0.83, 0.91, 1.11, 1.25]
# for time_stretch in  time_stretch_set:
#     simulation_p = dataset.Hparams(overwritten = True, stretch_rate = time_stretch)
#     #initilize performance metrics
#     metrics = label_matching.result_analysis(0, 'hit_match')

#     for i in range(K_FOLD):

#         train_test_split = dataset.cross_valid(seed = 42, split_number = 4, fold_needed = i, file_dic = file_dic, 
#                                         save_path = Path('../../label/cross_val_label_dic'), overwritten = False, verbose = False)


#         test_set = dataset.gibbon_dataset(cfg = simulation_p, dataset_type =  'test', dataset_usage= 'nonoverlap_pred', domain_transform = dataset.SpecImg_transform(simulation_p),
#                                 augment = dataset.Time_stretch(simulation_p) ,pytorch_X_transform = dataset.Pytorch_data_transform(), pytorch_Y_transform = dataset.Pytorch_label_transform(), 
#                                 train_test_split = train_test_split)



#         print('predict begin')
#         crnn_predict = nn_function.model_predict(model_p, pred_seq_len = simulation_p.audio_len, seed = 42, model_path =  '../model/crnn_model' + str(i) + '.pt')
#         crnn_predict.test_predict(test_set)
#         #calculate and save test result
#         metrics(crnn_predict.test_bino_predict['label'], crnn_predict.test_bino_predict['pred'])


#     metrics.result_process()
#     print(metrics.result_summary)
#     time_stretch_results['time stretch rate{:.3f}'.format(time_stretch)] = metrics.result_summary

# pickle.dump(time_stretch_results, open('../crnn_results/time_stretch_results.p', 'wb'))



# ###############################################################################################
# #crop experiments
# model_p = nn_function.Hparams()
# crop_results = {}
# crop_set = [1, 0.95, 0.9, 0.85, 0.8]
# for crop_rate in  crop_set:
#     simulation_p = dataset.Hparams(overwritten = True, crop_rate = crop_rate)
#     #initilize performance metrics
#     metrics = label_matching.result_analysis(0, 'hit_match')

#     for i in range(K_FOLD):

#         train_test_split = dataset.cross_valid(seed = 42, split_number = 4, fold_needed = i, file_dic = file_dic, 
#                                         save_path = Path('../../label/cross_val_label_dic'), overwritten = False, verbose = False)


#         test_set = dataset.gibbon_dataset(cfg = simulation_p, dataset_type =  'test', dataset_usage= 'nonoverlap_pred', domain_transform = dataset.SpecImg_transform(simulation_p),
#                                 augment = dataset.Cropping(simulation_p) ,pytorch_X_transform = dataset.Pytorch_data_transform(), pytorch_Y_transform = dataset.Pytorch_label_transform(), 
#                                 train_test_split = train_test_split)



#         print('predict begin')
#         crnn_predict = nn_function.model_predict(model_p, pred_seq_len = simulation_p.audio_len, seed = 42, model_path =  '../model/crnn_model' + str(i) + '.pt')
#         crnn_predict.test_predict(test_set)
#         #calculate and save test result
#         metrics(crnn_predict.test_bino_predict['label'], crnn_predict.test_bino_predict['pred'])


#     metrics.result_process()
#     print(metrics.result_summary)
#     crop_results['crop rate{:.2f}'.format(crop_rate)] = metrics.result_summary

# pickle.dump(crop_results, open('../crnn_results/crop_results.p', 'wb'))

















# average_processing_result = pickle.load(open('../crnn_results/crnn_results_average.p', 'rb'))
# threshold_result = pickle.load(open('../crnn_results/crnn_results_change_threshold.p', 'rb'))
# crnn_raw_results = pickle.load(open('../crnn_results/crnn_result_summary.p', 'rb'))
# pitch_shift_results = pickle.load(open('../crnn_results/pitch_shift_results.p', 'rb'))
# time_stretch_results = pickle.load(open('../crnn_results/time_stretch_results.p', 'rb'))
# crop_results = pickle.load(open('../crnn_results/crop_results.p', 'rb'))
