import pickle
import numpy as np
# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import os
from cuml.svm import SVC

class SVM_process:
    def __init__(self, seed = 42, pred_seq_len = 28800):
        self.seed = seed
        self.pred_seq_len = pred_seq_len
        self.best_c_value = None
        self.train_bino_predict = None
        self.val_bino_predict = None
        self.test_bino_predict = None


    @staticmethod
    def cal_F_score(label_set, pred_set):
        conf_matrix = np.zeros((2,2))
        if label_set.ndim == 1:
            label_set = label_set[None, :]
            pred_set = pred_set[None, :]
        for preds, labels in zip(label_set, pred_set):
            for p, t in zip(preds, labels):
                conf_matrix[p, t] += 1
        F_score = 2 * conf_matrix[1,1]/(2 * conf_matrix[1,1] + conf_matrix[0,1] + conf_matrix[1,0])
        return F_score


    def __call__(self, train_data_label, val_data_label, test_data_label):
        seed = self.seed
        pred_seq_len = self.pred_seq_len

        train_data = train_data_label.X_batch_data
        train_label = train_data_label.Y_batch_label.squeeze(1).astype('int')
        val_data = val_data_label.X_batch_data
        val_label = val_data_label.Y_batch_label.squeeze(1).astype('int')
        test_data = test_data_label.X_batch_data
        test_label = test_data_label.Y_batch_label.squeeze(1).astype('int')

        shuffled_train_data, shuffled_train_label = shuffle(train_data, train_label, random_state = seed)
        shuffled_val_data, shuffled_val_label = shuffle(val_data, val_label, random_state = seed)

        # use zero-mean normalization
        train_scaler = StandardScaler()
        scaled_shuffled_train_data = train_scaler.fit_transform(shuffled_train_data) 
        scaled_val_data = train_scaler.transform(val_data)

        # train svm classifier, use valid set to find the best parameter
        f_score_set = []
        C_set = np.logspace(0, 5, 6, base= 10)
        valid_predict_set = []
        for c_value in  C_set:
            svc_mfcc_grid = SVC(C = c_value, kernel='rbf', class_weight= None, random_state = seed, cache_size = 1024, tol = 1e-3, gamma = 'scale')
            # train model
            svc_mfcc_grid.fit(scaled_shuffled_train_data, shuffled_train_label)
            val_predict = svc_mfcc_grid.predict(scaled_val_data).astype('int')
            f_score_set.append(self.cal_F_score(val_label, val_predict))
            valid_predict_set.append(val_predict)

        best_index = f_score_set.index(max(f_score_set))
        best_c_value = C_set[best_index]
        self.best_c_value = best_c_value
        #get the prediction for validation set
        val_predict = valid_predict_set[best_index]


        #retrain the svm with train and valid set
        tv_data = np.concatenate((shuffled_train_data, shuffled_val_data), axis = 0)
        tv_label = np.concatenate((shuffled_train_label, shuffled_val_label), axis = 0)

        scaler = StandardScaler()
        scaled_tv_data = scaler.fit_transform(tv_data)
        scaled_test_data = scaler.transform(test_data)
        scaled_train_data = scaler.transform(train_data)


        svc_mfcc = SVC(C = self.best_c_value, kernel='rbf', class_weight= None, random_state = seed, cache_size = 1024, tol = 1e-3, gamma = 'scale')
        #retrain model
        svc_mfcc.fit(scaled_tv_data, tv_label)
        #predict 
        test_predict = svc_mfcc.predict(scaled_test_data).astype('int')
        train_predict = svc_mfcc.predict(scaled_train_data).astype('int')


        self.train_bino_predict = {'pred': train_predict.reshape(-1, pred_seq_len), 'label': train_label.reshape(-1, pred_seq_len)}
        self.val_bino_predict = {'pred': val_predict.reshape(-1, pred_seq_len), 'label': val_label.reshape(-1, pred_seq_len)}
        self.test_bino_predict = {'pred': test_predict.reshape(-1, pred_seq_len), 'label': test_label.reshape(-1, pred_seq_len)}






