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
data_p = dataset.Hparams(overwritten = True, seq_len = 1, train_hop_len = 1, test_hop_len = 1)
model_p = nn_function.Hparams(num_epochs = 40, 
                            batch_size = 1024, 
                            learning_rate = 1e-3, 
                            lr_decay_step_size = 40, 
                            lr_decay_rate =  1,
                            weight_decay_rate = 1e-1,
                            early_stop_patience = 40,
                            num_classes = 2)

# model_p.init_learning_rate = model_p.learning_rate
Train_proportion = [0.75, 0.5, 0.25]

for i, j in enumerate(Train_proportion):
    # model_p.learning_rate = model_p.init_learning_rate * j
    cv_network_train_test = nn_cv.Network_CV(cv_file_dic= file_dic,
                                        cv_fold_number = 4, 
                                        data_parameter = data_p, 
                                        model_parameter = model_p, 
                                        seed = 42, 
                                        train_val_proportion= j,
                                        valid_proportion= 0.3,
                                        cv_filesplit_savedic = '../cross_val_proportion', 
                                        cv_filesplit_overwrite = True,
                                        network_model = cnn_function.LENET(),
                                        model_save_dic = '../model/proportion_' + str(i) + '/cnn_model',
                                        trainning_visulise_dic = '../train_visulise/proportion_' + str(i),
                                        train_result_dic = '../results/proportion_' + str(i),
                                        test_result_dic = '../results/proportion_' + str(i))

    #train and test network
    cv_network_train_test.cv_train()
    cv_network_train_test.cv_test()




    #evaluate network with different postprocessing
    lenet_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= data_p, model_parameter= model_p, 
                                            seed= 42, train_val_proportion= j, valid_proportion= 0.3,
                                            cv_filesplit_savedic = '../cross_val_proportion', cv_filesplit_overwrite = False,
                                            model_path= '../model/proportion_' + str(i))
    lenet_evaluation.cv_post_train_test()

    lenet_result_summary = {}
    lenet_result_summary['nn_result'] = lenet_evaluation.nn_result
    lenet_result_summary['threshold_result'] = lenet_evaluation.threshold_result
    lenet_result_summary['average_result'] = lenet_evaluation.average_result
    lenet_result_summary['hmm_bino_result'] = lenet_evaluation.hmm_bino_result
    lenet_result_summary['hmm_gmm_result'] = lenet_evaluation.hmm_gmm_result
    lenet_result_summary['hmm_bino_threshold_result'] = lenet_evaluation.hmm_bino_threshold_result
    pickle.dump(lenet_result_summary, open('../results/proportion_' + str(i) + '/lenet_results_summary.p', 'wb'))



for i in range(3):
    result = pickle.load(open('../results/proportion_' + str(i) + '/lenet_results_summary.p', 'rb'))
    print(result, '\n')

