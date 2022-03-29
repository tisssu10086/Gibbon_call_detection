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
model_p.init_learning_rate = model_p.learning_rate
Train_proportion = [0.75, 0.5, 0.25]

for i, j in enumerate(Train_proportion):
    model_p.learning_rate = model_p.init_learning_rate * j
    cv_network_train_test = nn_cv.Network_CV(cv_file_dic = file_dic, 
                                            cv_fold_number = 4, 
                                            data_parameter = data_p, 
                                            model_parameter = model_p, 
                                            seed = 42, 
                                            train_val_proportion= j,
                                            valid_proportion= 0.3,
                                            cv_filesplit_savedic = '../cross_val_proportion', 
                                            cv_filesplit_overwrite = True,
                                            network_model = cnn_gru_structure.CRNN(1, 256),
                                            model_save_dic = '../model/proportion_' + str(i) + '/crnn_model',
                                            trainning_visulise_dic = '../train_visulise/proportion_' + str(i),
                                            train_result_dic = '../crnn_results/proportion_' + str(i),
                                            test_result_dic = '../crnn_results/proportion_' + str(i))
    #train and test network
    cv_network_train_test.cv_train()
    cv_network_train_test.cv_test()


    #evaluate network with different postprocessing
    crnn_evaluation = nn_cv.CV_evaluation(cv_file_dic= file_dic, data_parameter= data_p, model_parameter= model_p,
                                            seed= 42, train_val_proportion= j, valid_proportion= 0.3,
                                            cv_filesplit_savedic = '../cross_val_proportion', cv_filesplit_overwrite = False,
                                            model_path= '../model/proportion_' + str(i))


    crnn_evaluation.cv_post_train_test()

    crnn_result_summary = {}
    crnn_result_summary['nn_result'] = crnn_evaluation.nn_result
    crnn_result_summary['threshold_result'] = crnn_evaluation.threshold_result
    crnn_result_summary['average_result'] = crnn_evaluation.average_result
    crnn_result_summary['hmm_bino_result'] = crnn_evaluation.hmm_bino_result
    crnn_result_summary['hmm_gmm_result'] = crnn_evaluation.hmm_gmm_result
    crnn_result_summary['hmm_bino_threshold_result'] = crnn_evaluation.hmm_bino_threshold_result
    pickle.dump(crnn_result_summary, open('../crnn_results/proportion_' + str(i) + '/crnn_results_summary.p', 'wb'))
    print(crnn_result_summary)





for i in range(3):
    result = pickle.load(open('../crnn_results/proportion_' + str(i) + '/crnn_results_summary.p', 'rb'))
    print(result, '\n')


























