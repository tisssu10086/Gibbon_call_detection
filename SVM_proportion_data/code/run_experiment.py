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

# file_dic.sort()


#data parameter
data_p = dataset.Hparams(overwritten = True, seq_len = 1, train_hop_len = 1, test_hop_len = 1)

Train_proportion = [0.75, 0.5, 0.25]

#evaluate network with different postprocessing
for i, j in enumerate(Train_proportion):
    svm_evaluation = svm_cv.CV_evaluation(cv_file_dic= file_dic, 
                                            data_parameter_train= data_p, 
                                            data_parameter_predict= data_p, 
                                            model_path= '../model/proportion_' + str(i),
                                            seed = 42, 
                                            train_val_proportion = j,
                                            valid_proportion = 0.3,
                                            cv_filesplit_savedic = '../cross_val_proportion', 
                                            cv_filesplit_overwrite = True)
    svm_evaluation.cv_post_train_test()

    svm_result_summary = {}
    svm_result_summary['svm_result'] = svm_evaluation.svm_result
    svm_result_summary['hmm_bino_result'] = svm_evaluation.hmm_bino_result
    pickle.dump(svm_result_summary, open('../results/proportion_' + str(i) + '/svm_results_summary.p', 'wb'))
    print(svm_result_summary)
















