import matplotlib.pyplot as plt
import gc
import os
import pickle 
import numpy as np
from pathlib import Path


import svm_function
import dataset
import label_matching
import post_process




class CV_evaluation:
    '''Evaluate the model with different evaluaition metrics with vary post hoc method'''
    def __init__(self, 
                cv_file_dic = None, 
                cv_fold_number = 4, 
                data_parameter_train = None, 
                data_parameter_predict = None,
                seed = 42, 
                train_val_proportion = 1,
                valid_proportion = 0.3,
                cv_filesplit_savedic = '../../label/cross_val_label_dic', 
                cv_filesplit_overwrite = False,
                hmm_bino_post_process: bool = True,
                augment_train = None,
                augment_predict = None,
                model_path = None,
                verbose: bool = True):

                self.cv_file_dic = cv_file_dic
                self.cv_fold_number = cv_fold_number
                self.data_parameter_train = data_parameter_train
                self.data_parameter_predict = data_parameter_predict
                self.seed = seed
                self.train_val_proportion = train_val_proportion
                self.valid_proportion = valid_proportion
                self.cv_filesplit_savedic = cv_filesplit_savedic
                self.cv_filesplit_overwrite = cv_filesplit_overwrite
                self.hmm_bino_post_process = hmm_bino_post_process      
                self.augment_train = augment_train
                self.augment_predict = augment_predict
                self.model_path = model_path
                self.verbose = verbose

                self.domain_transform_train = dataset.MFCC_transform(data_parameter_train)
                self.domain_transform_predict = dataset.MFCC_transform(data_parameter_predict)
                self.pytorch_X_transform = None
                self.pytorch_Y_transform = None

                self.svm_result = None
                self.hmm_bino_result = None




    def cv_post_train_test(self):
        '''train hmm-bino and output method performance with different evaluation method'''

        K_FOLD = self.cv_fold_number
        seed = self.seed
        file_dic = self.cv_file_dic
        train_val_proportion = self.train_val_proportion
        valid_proportion = self.valid_proportion
        cv_save_dic = Path(self.cv_filesplit_savedic)
        cv_overwrite = self.cv_filesplit_overwrite

        data_p_train = self.data_parameter_train
        domain_transform_train = self.domain_transform_train
        augment_train = self.augment_train
        model_path = self.model_path

        p_x_t = self.pytorch_X_transform
        p_y_t = self.pytorch_Y_transform
        h_b_pp = self.hmm_bino_post_process
        verbose = self.verbose


        #initilize performance metrics
        metrics_svm = label_matching.result_analysis(0, 'hit_match')
        best_c_value_set = []
        if h_b_pp:
            metrics_hmm_bino = label_matching.result_analysis(0, 'hit_match')
            model_path_h = model_path + '/hmm_model'
        


        for i in range(K_FOLD):
            train_test_split = dataset.cross_valid(seed = seed, split_number = K_FOLD, fold_needed = i, file_dic = file_dic, 
                                            save_path = cv_save_dic, overwritten = cv_overwrite, verbose = False,
                                            train_val_proportion= train_val_proportion, valid_proportion= valid_proportion)

            #data for train svm and hmm
            train_set = dataset.gibbon_dataset(cfg = data_p_train, dataset_type =  'train', dataset_usage= 'overlap_train', domain_transform = domain_transform_train,
                                    augment = augment_train, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)
            val_set = dataset.gibbon_dataset(cfg = data_p_train, dataset_type =  'valid', dataset_usage= 'overlap_train', domain_transform = domain_transform_train,
                                    augment = augment_train, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)
            test_set = dataset.gibbon_dataset(cfg = data_p_train, dataset_type =  'test', dataset_usage= 'overlap_train', domain_transform = domain_transform_train,
                                    augment = augment_train, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)         



            #predict label
            if verbose:
                print('predict begin')
            svm_predict = svm_function.SVM_process(seed = seed, pred_seq_len= data_p_train.audio_len)
            svm_predict(train_set, val_set, test_set)
            best_c_value_set.append(svm_predict.best_c_value)

            #calculate and save test result
            metrics_svm(svm_predict.test_bino_predict['label'], svm_predict.test_bino_predict['pred'])


            #post processing and result saving
            if verbose:
                print('post processing begin')


            if h_b_pp:
                hmm_post_processing = post_process.hmm_post_process(model_path = model_path_h + str(i))
                hmm_post_processing.train_hmm_bino(svm_predict.train_bino_predict, svm_predict.val_bino_predict)
                hmm_post_processing.predict_hmm_bino(svm_predict.test_bino_predict)
                metrics_hmm_bino(hmm_post_processing.hmm_bino_pred_label['label'], hmm_post_processing.hmm_bino_pred_label['pred'])




        metrics_svm.result_process()
        self.svm_result = metrics_svm.result_summary
        self.svm_result['c_value_set'] = best_c_value_set
        if verbose:
            print('metrics_svm:', self.svm_result)

        if h_b_pp:
            metrics_hmm_bino.result_process()
            self.hmm_bino_result = metrics_hmm_bino.result_summary
            if verbose:
                print('metrics_hmm_bino:', self.hmm_bino_result)








    def cv_post_evaluate(self):
        '''output method performance with different evaluation method'''

        K_FOLD = self.cv_fold_number
        seed = self.seed
        train_val_proportion = self.train_val_proportion
        valid_proportion = self.valid_proportion
        file_dic = self.cv_file_dic
        cv_save_dic = Path(self.cv_filesplit_savedic)
        cv_overwrite = self.cv_filesplit_overwrite

        data_p_train = self.data_parameter_train
        data_p_predict = self.data_parameter_predict
        domain_transform_train = self.domain_transform_train
        domain_transform_predict = self.domain_transform_predict
        augment_train = self.augment_train
        augment_predict = self.augment_predict
        model_path = self.model_path

        p_x_t = self.pytorch_X_transform
        p_y_t = self.pytorch_Y_transform
        h_b_pp = self.hmm_bino_post_process
        verbose = self.verbose


        #initilize performance metrics
        metrics_svm = label_matching.result_analysis(0, 'hit_match')
        best_c_value_set = []
        if h_b_pp:
            metrics_hmm_bino = label_matching.result_analysis(0, 'hit_match')
            model_path_h = model_path + '/hmm_model'
        


        for i in range(K_FOLD):
            train_test_split = dataset.cross_valid(seed = seed, split_number = K_FOLD, fold_needed = i, file_dic = file_dic, 
                                            save_path = cv_save_dic, overwritten = cv_overwrite, verbose = False,
                                            train_val_proportion= train_val_proportion, valid_proportion= valid_proportion)

            #data for train svm
            train_set = dataset.gibbon_dataset(cfg = data_p_train, dataset_type =  'train', dataset_usage= 'overlap_train', domain_transform = domain_transform_train,
                                    augment = augment_train, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)
            val_set = dataset.gibbon_dataset(cfg = data_p_train, dataset_type =  'valid', dataset_usage= 'overlap_train', domain_transform = domain_transform_train,
                                    augment = augment_train, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)

            #data for ptrdict svm
            predict_test = dataset.gibbon_dataset(cfg = data_p_predict, dataset_type =  'test', dataset_usage= 'nonoverlap_pred', domain_transform = domain_transform_predict,
                                    augment = augment_predict, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)



            #predict label
            if verbose:
                print('predict begin')
            svm_predict = svm_function.SVM_process(seed = seed, pred_seq_len= data_p_predict.audio_len)
            svm_predict(train_set, val_set, predict_test)
            best_c_value_set.append(svm_predict.best_c_value)

            #calculate and save test result
            metrics_svm(svm_predict.test_bino_predict['label'], svm_predict.test_bino_predict['pred'])


            #post processing and result saving
            if verbose:
                print('post processing begin')


            if h_b_pp:
                hmm_post_processing = post_process.hmm_post_process(model_path = model_path_h + str(i))
                hmm_post_processing.predict_hmm_bino(svm_predict.test_bino_predict)
                metrics_hmm_bino(hmm_post_processing.hmm_bino_pred_label['label'], hmm_post_processing.hmm_bino_pred_label['pred'])


        metrics_svm.result_process()
        self.svm_result = metrics_svm.result_summary
        self.svm_result['c_value_set'] = best_c_value_set
        if verbose:
            print('metrics_svm:', self.svm_result)

        if h_b_pp:
            metrics_hmm_bino.result_process()
            self.hmm_bino_result = metrics_hmm_bino.result_summary
            if verbose:
                print('metrics_hmm_bino:', self.hmm_bino_result)

