import matplotlib.pyplot as plt
import gc
import os
import pickle 
import numpy as np
from pathlib import Path
import time

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
data_p = dataset.Hparams(overwritten = False,  seq_len = 4, train_hop_len = 2, test_hop_len = 4)
model_p = nn_function.Hparams(num_epochs = 20, 
                            batch_size = 512, 
                            learning_rate = 1e-3, 
                            lr_decay_step_size = 20, 
                            lr_decay_rate =  1,
                            weight_decay_rate = 0,
                            early_stop_patience = 20,
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
start_time = time.time()
cv_network_train_test.cv_train()
end_time = time.time()
print(end_time - start_time)
cv_network_train_test.cv_test()




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














