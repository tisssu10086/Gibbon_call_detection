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
                            learning_rate = 5e-4, 
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
                                    network_model = cnn_function.CNN(),
                                    model_save_dic = '../model/cnn_model',
                                    trainning_visulise_dic = '../train_visulise',
                                    train_result_dic = '../results',
                                    test_result_dic = '../results')

#train and test network
cv_network_train_test.cv_train()
cv_network_train_test.cv_test()




#evaluate network with different postprocessing
vgg_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= data_p, model_parameter= model_p, model_path= '../model/cnn_model')
vgg_evaluation.cv_nn_evaluate()

vgg_result_summary = {}
vgg_result_summary['nn_result'] = vgg_evaluation.nn_result
vgg_result_summary['threshold_result'] = vgg_evaluation.threshold_result
vgg_result_summary['average_result'] = vgg_evaluation.average_result
vgg_result_summary['hmm_bino_result'] = vgg_evaluation.hmm_bino_result
vgg_result_summary['hmm_gmm_result'] = vgg_evaluation.hmm_gmm_result
vgg_result_summary['hmm_bino_threshold_result'] = vgg_evaluation.hmm_bino_threshold_result
pickle.dump(vgg_result_summary, open('../results/vgg_results_summary.p', 'wb'))
















