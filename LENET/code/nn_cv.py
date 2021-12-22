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




class Network_CV:
    def __init__(self, 
                cv_file_dic = None, 
                cv_fold_number = 4, 
                data_parameter = None, 
                model_parameter = None, 
                seed = 42, 
                cv_filesplit_savedic = '../../label/cross_val_label_dic', 
                cv_filesplit_overwrite = False,
                network_model = None,
                model_save_dic = None,
                trainning_visulise_dic = None,
                train_result_dic = None,
                test_result_dic = None):

                self.cv_file_dic = cv_file_dic
                self.cv_fold_number = cv_fold_number
                self.data_parameter = data_parameter
                self.model_parameter = model_parameter 
                self.seed = seed
                self.cv_filesplit_savedic = cv_filesplit_savedic
                self.cv_filesplit_overwrite = cv_filesplit_overwrite
                self.network_model = network_model
                self.model_save_dic = model_save_dic
                self.train_vs_dic = trainning_visulise_dic
                self.train_result_dic = train_result_dic
                self.test_result_dic = test_result_dic

                self.dl_process = nn_function.DL_process(model_parameter, seed = seed)
                self.domain_transform = dataset.SpecImg_transform(data_parameter)
                self.augment = None
                self.pytorch_X_transform = dataset.Pytorch_data_transform()
                self.pytorch_Y_transform = dataset.Pytorch_label_transform()


    def cv_train(self):
        '''train the CRNN with cross validation'''
        K_FOLD = self.cv_fold_number
        data_p = self.data_parameter
        seed = self.seed
        file_dic = self.cv_file_dic
        cv_save_dic = Path(self.cv_filesplit_savedic)
        cv_overwrite = self.cv_filesplit_overwrite
        dl_process = self.dl_process
        domain_transform = self.domain_transform
        augment = self.augment
        p_x_t = self.pytorch_X_transform
        p_y_t = self.pytorch_Y_transform
        nn_model = self.network_model
        model_save_dic = self.model_save_dic
        train_vs_dic = Path(self.train_vs_dic)
        train_result_dic = Path(self.train_result_dic)

        runing_loss_curve_all = []
        runing_accu_curve_all = []  
        val_loss_curve_all = []
        val_accu_curve_all = []
        conf_matrix_set_all = []


        for i in range(K_FOLD):
            print('fold number:', i)
            train_test_split = dataset.cross_valid(seed = seed, split_number = K_FOLD, fold_needed = i, file_dic = file_dic, 
                                            save_path = cv_save_dic, overwritten = cv_overwrite, verbose = False)

            train_set = dataset.gibbon_dataset(cfg = data_p, dataset_type =  'train', dataset_usage= 'overlap_train', domain_transform = domain_transform,
                                    augment = augment, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)

            valid_set = dataset.gibbon_dataset(cfg = data_p, dataset_type =  'valid', dataset_usage= 'overlap_train', domain_transform = domain_transform,
                                    augment = augment, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)

            print('trainning begin')
            runing_loss_curve, runing_accu_curve, val_loss_curve, val_accu_curve, conf_matrix_set = dl_process.train_nn(nn_model = nn_model, 
                                                                                                                        train_data_label = train_set,
                                                                                                                        val_data_label = valid_set,
                                                                                                                        save_path = model_save_dic + str(i) + '.pt')
            runing_loss_curve_all.append(runing_loss_curve)
            runing_accu_curve_all.append(runing_accu_curve)
            val_loss_curve_all.append(val_loss_curve)
            val_accu_curve_all.append(val_accu_curve)
            conf_matrix_set_all.append(conf_matrix_set)
            #visulise the loss and accu curve for each cross validation procedure
            plt.cla()
            plt.plot(runing_loss_curve)
            plt.savefig(train_vs_dic / Path('runing_loss_curve%d.png'%i))
            plt.cla()
            plt.plot(val_loss_curve)
            plt.savefig(train_vs_dic / Path('val_loss_curve%d.png'%i))
            plt.cla()
            plt.plot(runing_accu_curve)
            plt.savefig(train_vs_dic / Path('runing_accu_curve%d.png'%i))
            plt.cla()
            plt.plot(val_accu_curve)
            plt.savefig(train_vs_dic / Path('val_accu_curve%d.png'%i))
            # release the storage
            del train_set, valid_set
            gc.collect()


        # save the training and validation result
        pickle.dump(runing_loss_curve_all, open(train_result_dic / Path('running_loss.p'), 'wb'))
        pickle.dump(runing_accu_curve_all, open(train_result_dic / Path('running_accu.p'), 'wb'))
        pickle.dump(val_loss_curve_all, open(train_result_dic / Path('val_loss.p'), 'wb'))
        pickle.dump(val_accu_curve_all, open(train_result_dic / Path('val_accu.p'), 'wb'))
        conf_matrix_set_all = np.asarray(conf_matrix_set_all)
        pickle.dump(conf_matrix_set_all, open(train_result_dic / Path('val_conf_matrix_set.p'), 'wb'))
        conf_matrix_set_all_sum = conf_matrix_set_all.sum(axis = 0)
        pickle.dump(conf_matrix_set_all_sum, open(train_result_dic / Path('val_conf_matrix_sum.p'), 'wb'))



    def cv_test(self):
        '''test the crnn model '''

        #initialize hyper parameter
        data_p = self.data_parameter
        #initilize processer
        dl_process = self.dl_process
        K_FOLD = self.cv_fold_number
        seed = self.seed
        file_dic = self.cv_file_dic
        cv_save_dic = Path(self.cv_filesplit_savedic)
        cv_overwrite = self.cv_filesplit_overwrite
        dl_process = self.dl_process
        domain_transform = self.domain_transform
        augment = self.augment
        p_x_t = self.pytorch_X_transform
        p_y_t = self.pytorch_Y_transform
        model_save_dic = self.model_save_dic
        test_result_dic = Path(self.test_result_dic)


        test_loss_all = []
        test_accu_all = []
        test_conf_matrix_all = []


        for i in range(K_FOLD):

            train_test_split = dataset.cross_valid(seed = seed, split_number = K_FOLD, fold_needed = i, file_dic = file_dic, 
                                            save_path = cv_save_dic, overwritten = cv_overwrite, verbose = False)


            test_set = dataset.gibbon_dataset(cfg = data_p, dataset_type =  'test', dataset_usage= 'nonoverlap_pred', domain_transform = domain_transform,
                                    augment = augment, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)

            print('test begin')
            #call the eval_crnn function
            test_loss, test_correct, test_conf_matrix = dl_process.eval_nn(test_data_label = test_set, model_path = model_save_dic + str(i) + '.pt')
            test_loss_all.append(test_loss)
            test_accu_all.append(test_correct)
            test_conf_matrix_all.append(test_conf_matrix)
            del test_set
            gc.collect()

        #store the test output
        pickle.dump(test_loss_all, open(test_result_dic / Path('test_loss.p'), 'wb'))
        pickle.dump(test_accu_all, open(test_result_dic / Path('test_accu.p'), 'wb'))
        test_conf_matrix_all  = np.asarray(test_conf_matrix_all)
        test_conf_matrix_sum = test_conf_matrix_all.sum(axis = 0)
        pickle.dump(test_conf_matrix_all, open(test_result_dic / Path('test_conf_matrix_set.p'), 'wb'))
        pickle.dump(test_conf_matrix_sum, open(test_result_dic / Path('test_conf_matrix_sum.p'), 'wb'))
        print(test_conf_matrix_sum)





















##################################################################################################


class CV_evaluation:
    '''Evaluate the model with different evaluaition metrics with vary post hoc method'''
    def __init__(self, 
                cv_file_dic = None, 
                cv_fold_number = 4, 
                data_parameter = None, 
                model_parameter = None, 
                seed = 42, 
                cv_filesplit_savedic = '../../label/cross_val_label_dic', 
                cv_filesplit_overwrite = False,
                model_path = None,
                threshold_post_process: bool = True,
                average_post_process: bool = True,
                hmm_bino_post_process: bool = True,
                hmm_gmm_post_process: bool = True,
                hmm_bino_threshold_post_process: bool = True,
                augment = None,
                verbose: bool = True):

                self.cv_file_dic = cv_file_dic
                self.cv_fold_number = cv_fold_number
                self.data_parameter = data_parameter
                self.model_parameter = model_parameter 
                self.seed = seed
                self.cv_filesplit_savedic = cv_filesplit_savedic
                self.cv_filesplit_overwrite = cv_filesplit_overwrite
                self.model_path = model_path
                self.threshold_post_process = threshold_post_process
                self.average_post_process = average_post_process
                self.hmm_bino_post_process = hmm_bino_post_process
                self.hmm_gmm_post_process = hmm_gmm_post_process    
                self.hmm_bino_threshold_post_process = hmm_bino_threshold_post_process            
                self.augment = augment
                self.verbose = verbose

                self.domain_transform = dataset.SpecImg_transform(data_parameter)
                self.pytorch_X_transform = dataset.Pytorch_data_transform()
                self.pytorch_Y_transform = dataset.Pytorch_label_transform()

                self.nn_result = None
                self.threshold_result = None
                self.average_result = None
                self.hmm_bino_result = None
                self.hmm_gmm_result = None
                self.hmm_bino_threshold_result = None

                if threshold_post_process == False and hmm_bino_threshold_post_process == True:
                    print('error, hmm threshold process must come with threshold post process')
                    hmm_bino_threshold_post_process = False


    def cv_post_train_test(self):
        '''train post processing and output method performance with different evaluation method'''

        K_FOLD = self.cv_fold_number
        seed = self.seed
        file_dic = self.cv_file_dic
        cv_save_dic = Path(self.cv_filesplit_savedic)
        cv_overwrite = self.cv_filesplit_overwrite
        model_p = self.model_parameter
        data_p = self.data_parameter
        domain_transform = self.domain_transform
        augment = self.augment
        p_x_t = self.pytorch_X_transform
        p_y_t = self.pytorch_Y_transform
        model_path = self.model_path
        t_pp = self.threshold_post_process
        a_pp =self.average_post_process
        h_b_pp = self.hmm_bino_post_process
        h_g_pp = self.hmm_gmm_post_process
        h_b_t_pp = self.hmm_bino_threshold_post_process
        verbose = self.verbose


        #initilize performance metrics
        metrics_nn = label_matching.result_analysis(0, 'hit_match')
        model_path_nn = model_path + '/cnn_model'
        if t_pp:
            metrics_threshold = label_matching.result_analysis(0, 'hit_match')
            model_path_t = model_path + '/threshold_model'
        if a_pp:
            metrics_average = label_matching.result_analysis(0, 'hit_match')
            model_path_a = model_path +'/average_model'
        if h_b_pp:
            metrics_hmm_bino = label_matching.result_analysis(0, 'hit_match')
            model_path_h = model_path+ '/hmm_model'
        if h_g_pp:
            metrics_hmm_gmm = label_matching.result_analysis(0, 'hit_match')
            model_path_h = model_path+ '/hmm_model'
        if h_b_t_pp:
            metrics_hmm_bino_threshold = label_matching.result_analysis(0, 'hit_match')
            model_path_h_t  = model_path + '/hmm_threshold_model'


        for i in range(K_FOLD):
            train_test_split = dataset.cross_valid(seed = seed, split_number = K_FOLD, fold_needed = i, file_dic = file_dic, 
                                            save_path = cv_save_dic, overwritten = cv_overwrite, verbose = False)

            test_set = dataset.gibbon_dataset(cfg = data_p, dataset_type =  'test', dataset_usage= 'nonoverlap_pred', domain_transform = domain_transform,
                                    augment = augment, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)
            train_set = dataset.gibbon_dataset(cfg = data_p, dataset_type =  'train', dataset_usage= 'nonoverlap_pred', domain_transform = domain_transform,
                                    augment = augment, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)
            val_set = dataset.gibbon_dataset(cfg = data_p, dataset_type =  'valid', dataset_usage= 'nonoverlap_pred', domain_transform = domain_transform,
                                    augment = augment, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)


            #predict label
            if verbose:
                print('predict begin')
            nn_predict = nn_function.model_predict(model_p, pred_seq_len = data_p.audio_len, seed = 42, model_path =  model_path_nn + str(i) + '.pt')
            nn_predict.test_predict(test_set)
            nn_predict.train_val_predict(train_set, val_set)

            #calculate and save test result
            metrics_nn(nn_predict.test_bino_predict['label'], nn_predict.test_bino_predict['pred'])

            #post processing and result saving
            if verbose:
                print('post processing begin')


            if t_pp:
                threshold_processsing = post_process.threshold_post_process(model_path = model_path_t + str(i) + '.p')
                threshold_processsing.train_threshold(nn_predict.train_prob_predict, nn_predict.val_prob_predict, nn_predict.test_prob_predict)
                metrics_threshold(threshold_processsing.test_bino_predict['label'], threshold_processsing.test_bino_predict['pred'])
                if h_b_t_pp:
                    # no need for .p
                    hmm_threshold_postprocessing = post_process.hmm_post_process(model_path = model_path_h_t + str(i))
                    hmm_threshold_postprocessing.train_hmm_bino(threshold_processsing.train_bino_predict, threshold_processsing.val_bino_predict)
                    hmm_threshold_postprocessing.predict_hmm_bino(threshold_processsing.test_bino_predict)
                    metrics_hmm_bino_threshold(hmm_threshold_postprocessing.hmm_bino_pred_label['label'], hmm_threshold_postprocessing.hmm_bino_pred_label['pred'])
            if a_pp:
                average_processing = post_process.average_postprocess(n_jobs = 6, model_path = model_path_a + str(i) + '.p')
                average_processing.train_average(nn_predict.train_prob_predict, nn_predict.val_prob_predict)
                average_processing.predict_average(nn_predict.test_prob_predict)
                metrics_average(average_processing.test_bino_predict['label'], average_processing.test_bino_predict['pred'])
            if h_b_pp or h_g_pp:
                # no need for .p
                hmm_post_processing = post_process.hmm_post_process(model_path = model_path_h + str(i))
                if h_b_pp:
                    hmm_post_processing.train_hmm_bino(nn_predict.train_bino_predict, nn_predict.val_bino_predict)
                    hmm_post_processing.predict_hmm_bino(nn_predict.test_bino_predict)
                    metrics_hmm_bino(hmm_post_processing.hmm_bino_pred_label['label'], hmm_post_processing.hmm_bino_pred_label['pred'])
                if h_g_pp:
                    hmm_post_processing.train_hmm_gmm(nn_predict.train_raw_predict, nn_predict.val_raw_predict)
                    hmm_post_processing.predict_hmm_gmm(nn_predict.test_raw_predict)
                    metrics_hmm_gmm(hmm_post_processing.hmm_gmm_pred_label['label'], hmm_post_processing.hmm_gmm_pred_label['pred'])



        metrics_nn.result_process()
        self.nn_result = metrics_nn.result_summary
        if verbose:
            print('metrics_nn:', self.nn_result)

        if t_pp:
            metrics_threshold.result_process()
            self.threshold_result = metrics_threshold.result_summary
            if verbose:
                print('metrics_threshold', self.threshold_result)


        if a_pp:
            metrics_average.result_process()
            self.average_result = metrics_average.result_summary
            if verbose:
                print('metrics_average', self.average_result)

        if h_b_pp:
            metrics_hmm_bino.result_process()
            self.hmm_bino_result = metrics_hmm_bino.result_summary
            if verbose:
                print('metrics_hmm_bino:', self.hmm_bino_result)

        if h_g_pp:
            metrics_hmm_gmm.result_process()
            self.hmm_gmm_result = metrics_hmm_gmm.result_summary
            if verbose:
                print('metrics_hmm_gmm:',  self.hmm_gmm_result)

        if h_b_t_pp:
            metrics_hmm_bino_threshold.result_process()
            self.hmm_bino_threshold_result = metrics_hmm_bino_threshold.result_summary
            if verbose:
                print('metrics_hmm_bino_threshold:', self.hmm_bino_threshold_result)





    def cv_post_evaluate(self):
        '''output method performance with different evaluation method'''

        K_FOLD = self.cv_fold_number
        seed = self.seed
        file_dic = self.cv_file_dic
        cv_save_dic = Path(self.cv_filesplit_savedic)
        cv_overwrite = self.cv_filesplit_overwrite
        model_p = self.model_parameter
        data_p = self.data_parameter
        domain_transform = self.domain_transform
        augment = self.augment
        p_x_t = self.pytorch_X_transform
        p_y_t = self.pytorch_Y_transform
        model_path = self.model_path
        t_pp = self.threshold_post_process
        a_pp =self.average_post_process
        h_b_pp = self.hmm_bino_post_process
        h_g_pp = self.hmm_gmm_post_process
        h_b_t_pp = self.hmm_bino_threshold_post_process
        verbose = self.verbose


        #initilize performance metrics
        metrics_nn = label_matching.result_analysis(0, 'hit_match')
        model_path_nn = model_path + '/cnn_model'
        if t_pp:
            metrics_threshold = label_matching.result_analysis(0, 'hit_match')
            model_path_t = model_path + '/threshold_model'
        if a_pp:
            metrics_average = label_matching.result_analysis(0, 'hit_match')
            model_path_a = model_path +'/average_model'
        if h_b_pp:
            metrics_hmm_bino = label_matching.result_analysis(0, 'hit_match')
            model_path_h = model_path+ '/hmm_model'
        if h_g_pp:
            metrics_hmm_gmm = label_matching.result_analysis(0, 'hit_match')
            model_path_h = model_path+ '/hmm_model'
        if h_b_t_pp:
            metrics_hmm_bino_threshold = label_matching.result_analysis(0, 'hit_match')
            model_path_h_t  = model_path + '/hmm_threshold_model'


        for i in range(K_FOLD):
            train_test_split = dataset.cross_valid(seed = seed, split_number = K_FOLD, fold_needed = i, file_dic = file_dic, 
                                            save_path = cv_save_dic, overwritten = cv_overwrite, verbose = False)

            test_set = dataset.gibbon_dataset(cfg = data_p, dataset_type =  'test', dataset_usage= 'nonoverlap_pred', domain_transform = domain_transform,
                                    augment = augment, pytorch_X_transform = p_x_t, pytorch_Y_transform = p_y_t, 
                                    train_test_split = train_test_split)



            #predict label
            if verbose:
                print('predict begin')
            nn_predict = nn_function.model_predict(model_p, pred_seq_len = data_p.audio_len, seed = 42, model_path =  model_path_nn + str(i) + '.pt')
            nn_predict.test_predict(test_set)

            #calculate and save test result
            metrics_nn(nn_predict.test_bino_predict['label'], nn_predict.test_bino_predict['pred'])

            #post processing and result saving
            if verbose:
                print('post processing begin')


            if t_pp:
                threshold_processsing = post_process.threshold_post_process(model_path = model_path_t + str(i) + '.p')
                threshold_processsing.predict_threshold(nn_predict.test_prob_predict)
                metrics_threshold(threshold_processsing.test_bino_predict['label'], threshold_processsing.test_bino_predict['pred'])
                if h_b_t_pp:
                    hmm_threshold_postprocessing = post_process.hmm_post_process(model_path = model_path_h_t + str(i))
                    hmm_threshold_postprocessing.predict_hmm_bino(threshold_processsing.test_bino_predict)
                    metrics_hmm_bino_threshold(hmm_threshold_postprocessing.hmm_bino_pred_label['label'], hmm_threshold_postprocessing.hmm_bino_pred_label['pred'])
            if a_pp:
                average_processing = post_process.average_postprocess(n_jobs = 6, model_path = model_path_a + str(i) + '.p')
                average_processing.predict_average(nn_predict.test_prob_predict)
                metrics_average(average_processing.test_bino_predict['label'], average_processing.test_bino_predict['pred'])
            if h_b_pp or h_g_pp:
                hmm_post_processing = post_process.hmm_post_process(model_path = model_path_h + str(i))
                if h_b_pp:
                    hmm_post_processing.predict_hmm_bino(nn_predict.test_bino_predict)
                    metrics_hmm_bino(hmm_post_processing.hmm_bino_pred_label['label'], hmm_post_processing.hmm_bino_pred_label['pred'])
                if h_g_pp:
                    hmm_post_processing.predict_hmm_gmm(nn_predict.test_raw_predict)
                    metrics_hmm_gmm(hmm_post_processing.hmm_gmm_pred_label['label'], hmm_post_processing.hmm_gmm_pred_label['pred'])



        metrics_nn.result_process()
        self.nn_result = metrics_nn.result_summary
        if verbose:
            print('metrics_nn:', self.nn_result)

        if t_pp:
            metrics_threshold.result_process()
            self.threshold_result = metrics_threshold.result_summary
            if verbose:
                print('metrics_threshold', self.threshold_result)


        if a_pp:
            metrics_average.result_process()
            self.average_result = metrics_average.result_summary
            if verbose:
                print('metrics_average', self.average_result)

        if h_b_pp:
            metrics_hmm_bino.result_process()
            self.hmm_bino_result = metrics_hmm_bino.result_summary
            if verbose:
                print('metrics_hmm_bino:', self.hmm_bino_result)

        if h_g_pp:
            metrics_hmm_gmm.result_process()
            self.hmm_gmm_result = metrics_hmm_gmm.result_summary
            if verbose:
                print('metrics_hmm_gmm:',  self.hmm_gmm_result)

        if h_b_t_pp:
            metrics_hmm_bino_threshold.result_process()
            self.hmm_bino_threshold_result = metrics_hmm_bino_threshold.result_summary
            if verbose:
                print('metrics_hmm_bino_threshold:', self.hmm_bino_threshold_result)















