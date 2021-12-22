import numpy as np
import pickle
from hmmlearn import hmm
import pandas as pd
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

class hmm_post_process:
    def __init__(self, model_path, hidden_state_number: int = 2, observe_state_number: int = 2):

        self.hidden_state_number = hidden_state_number
        self.observe_state_number = observe_state_number
        self.hmm_bino_pred_label = None
        self.hmm_gmm_pred_label = None
        self.model_path = model_path


    def cal_tr_matrix(self, label_set):

        def transition_matrix(current, future, tr_matrix):
            tr_matrix[current, future] += 1
            return tr_matrix

        num_state = self.hidden_state_number
        tr_matrix = np.zeros((num_state, num_state))

        if label_set.ndim == 1:
            label_set = label_set[None, :]
        for labels in label_set:
            for j in range(len(labels) -1):
                transition_matrix(labels[j], labels[j+1], tr_matrix)
        #normalise transition matrix
        tr_matrix = tr_matrix / np.sum(tr_matrix, axis= 1)[:, None]
        return tr_matrix



    def cal_pi_vector(self, label_set):
        num_state = self.hidden_state_number
        pi_vector = np.zeros(num_state)

        if label_set.ndim == 1:
            label_set = label_set[None, :]
        for labels in label_set:
            pi_vector[labels[0]] += 1
        #normalise initial state vector
        pi_vector = pi_vector/pi_vector.sum()
        return pi_vector




    def cal_em_matrix(self, pred_set, label_set):

        def emmmision_matrix(latent, seen, em_matrix):
            em_matrix[latent, seen] += 1
            return em_matrix

        num_ob = self.observe_state_number
        num_hid = self.hidden_state_number

        em_matrix = np.zeros((num_hid, num_ob))
        assert pred_set.shape== label_set.shape

        if label_set.ndim == 1:
            label_set = label_set[None, :]
            pred_set = pred_set[None, :]

        for i in range(label_set.shape[0]):
            for j in range(label_set.shape[1]):
                emmmision_matrix(label_set[i][j], pred_set[i][j], em_matrix)
        #normalize emission matrix
        em_matrix = em_matrix / np.sum(em_matrix, axis= 1)[:, None]
        return em_matrix



    def cal_em_dist(self, pred_set, label_set, num_mix):
        '''Now can only calculate gmm with two hidden state'''
        assert label_set.shape == pred_set.shape

        if label_set.ndim == 1:
            label_set = label_set[None, :]
            pred_set = pred_set[None, :]

        all_pos_val = []
        all_neg_val = []

        for i in range(label_set.shape[0]):
            for j in range(label_set.shape[1]):
                if label_set[i][j] == 0:
                    all_neg_val.append(pred_set[i][j])
                else:
                    all_pos_val.append(pred_set[i][j])

        #calculate mean, cov and weight for negative component in gaussian mixed model             
        all_neg_val = np.asarray(all_neg_val)
        all_neg_val = all_neg_val[:, np.newaxis]
        gmm_neg = GaussianMixture(n_components=num_mix, random_state= 42, covariance_type = 'diag', reg_covar = 5e-5).fit(all_neg_val)
        bic_neg = gmm_neg.bic(all_neg_val)
        weights_neg = gmm_neg.weights_
        weights_neg = weights_neg[np.newaxis,:]
        mean_neg = gmm_neg.means_
        mean_neg = mean_neg[np.newaxis,:]
        cov_neg = gmm_neg.covariances_
        cov_neg = cov_neg[np.newaxis,:]

        #calculate mean, cov and weight for positive component in gaussian mixed model  
        all_pos_val = np.asarray(all_pos_val)
        all_pos_val = all_pos_val[:, np.newaxis]
        gmm_pos = GaussianMixture(n_components=num_mix, random_state= 42, covariance_type = 'diag', reg_covar = 5e-5).fit(all_pos_val)
        bic_pos = gmm_pos.bic(all_pos_val)
        weights_pos = gmm_pos.weights_
        weights_pos = weights_pos[np.newaxis,:]
        mean_pos = gmm_pos.means_
        mean_pos = mean_pos[np.newaxis,:]
        cov_pos = gmm_pos.covariances_
        cov_pos = cov_pos[np.newaxis,:]

        gmm_weights = np.concatenate((weights_neg, weights_pos), axis = 0)
        gmm_means = np.concatenate((mean_neg, mean_pos), axis = 0)
        gmm_covs = np.concatenate((cov_neg, cov_pos), axis = 0)
        bic = (bic_neg + bic_pos) / 2

        gmm_params = {'weights': gmm_weights, 'means': gmm_means, 'covs': gmm_covs, 'bic': bic, 'num_mix': num_mix}

        # return all_pos_val, all_neg_val, gmm_weights, gmm_means, gmm_covs
        return gmm_params




    def train_hmm_bino(self, train_bino_predict_label, val_bino_predict_label):
        model_path = self.model_path
        train_predict = train_bino_predict_label['pred']
        train_label = train_bino_predict_label['label']
        val_predict = val_bino_predict_label['pred']
        val_label = val_bino_predict_label['label']

        train_val_predict = np.concatenate((train_predict, val_predict), axis= 0)
        train_val_label = np.concatenate((train_label, val_label), axis = 0)

        model = {}
        model['transition_matrix'] = self.cal_tr_matrix(train_val_label)
        model['pi_vector'] = self.cal_pi_vector(train_val_label)
        model['emission_matrix'] = self.cal_em_matrix(train_val_predict, train_val_label)
        pickle.dump(model, open(model_path + '_bino.p', 'wb'))




    def train_hmm_gmm(self, train_raw_predict_label, val_raw_predict_label):
        model_path = self.model_path
        train_predict = train_raw_predict_label['pred']
        train_label = train_raw_predict_label['label']
        val_predict = val_raw_predict_label['pred']
        val_label = val_raw_predict_label['label']

        train_val_predict = np.concatenate((train_predict, val_predict), axis= 0)
        train_val_label = np.concatenate((train_label, val_label), axis = 0)


        #choose mix number with bic
        different_mix_result = []
        different_mix_bic = []
        NUM_MIX = np.logspace(0, 10, 11, base= 2).astype('int')
        for num_mix in NUM_MIX:
            mix_result = self.cal_em_dist(train_val_predict, train_val_label, num_mix)
            different_mix_result.append(mix_result)
            different_mix_bic.append(mix_result['bic'])
        #find the mixture with minimum bic 
        best_index = different_mix_bic.index(min(different_mix_bic))
        emission_dist = different_mix_result[best_index]

        model = {}
        model['transition_matrix'] = self.cal_tr_matrix(train_val_label)
        model['pi_vector'] = self.cal_pi_vector(train_val_label)
        model['emission_dist'] = emission_dist
        pickle.dump(model, open(model_path + '_gmm.p', 'wb'))





    def predict_hmm_bino(self, test_bino_predict_label):
        num_state = self.hidden_state_number
        test_predict = test_bino_predict_label['pred']
        test_label = test_bino_predict_label['label']
        model_p = pickle.load(open(self.model_path + '_bino.p', 'rb'))

        model = hmm.MultinomialHMM(n_components=num_state)
        model.startprob_= model_p['pi_vector']
        model.transmat_= model_p['transition_matrix']
        model.emissionprob_= model_p['emission_matrix']


        if test_predict.ndim ==1:
            test_predict = test_predict[None, :]

        hmm_bino_pred = []
        for predict_seq in test_predict:
            predict_seq = predict_seq.reshape(-1, 1)
            hmm_pred_seq = model.predict(predict_seq)
            hmm_bino_pred.append(hmm_pred_seq)

        hmm_bino_pred = np.asarray(hmm_bino_pred)
        self.hmm_bino_pred_label = {'pred': hmm_bino_pred, 'label': test_label}




    def predict_hmm_gmm(self, test_raw_predict_label):
        num_state = self.hidden_state_number
        test_predict = test_raw_predict_label['pred']
        test_label = test_raw_predict_label['label']
        model_p = pickle.load(open(self.model_path + '_gmm.p', 'rb'))

        model = hmm.GMMHMM(n_components=num_state, n_mix= model_p['emission_dist']['num_mix'])
        model.startprob_= model_p['pi_vector']
        model.transmat_= model_p['transition_matrix']
        model.weights_= model_p['emission_dist']['weights']
        model.means_ = model_p['emission_dist']['means']
        model.covars_ = model_p['emission_dist']['covs']


        if test_predict.ndim ==1:
            test_predict = test_predict[None, :]

        hmm_gmm_pred = []
        for predict_seq in test_predict:
            predict_seq = predict_seq.reshape(-1, 1)
            hmm_pred_seq = model.predict(predict_seq)
            hmm_gmm_pred.append(hmm_pred_seq)

        hmm_gmm_pred = np.asarray(hmm_gmm_pred)
        self.hmm_gmm_pred_label = {'pred': hmm_gmm_pred, 'label': test_label}







class average_postprocess:

    def __init__(self, n_jobs, model_path):
        self.n_jobs = n_jobs
        self.test_bino_predict = None
        self.model_path = model_path


    def moving_average_process(self, pred_set, average_step):
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'same') / w

        if pred_set.ndim == 1:
            pred_set = pred_set[None, :]

        pred_average = Parallel(n_jobs = self.n_jobs)(delayed(moving_average)(pred_seq, average_step) for pred_seq in pred_set)
        pred_average = np.asarray(pred_average)
        return pred_average


    @staticmethod
    def binarize(prob_set, threshold):
        low = 0
        up = 1
        return np.where(prob_set>threshold, up, low)

    # 绘制混淆矩阵 draw confusion matrix ( 对整个数组生效 )
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


    def train_average(self, train_prob_predict_label, val_prob_predict_label):
        model_path = self.model_path

        train_predict = train_prob_predict_label['pred']
        train_label = train_prob_predict_label['label']
        val_predict = val_prob_predict_label['pred']
        val_label = val_prob_predict_label['label']

        train_val_predict = np.concatenate((train_predict, val_predict), axis= 0)
        train_val_label = np.concatenate((train_label, val_label), axis = 0)


        #grid search for the threshold and moving average step
        threshold_set = np.linspace(1, 9, 9)/10
        average_step_set = np.logspace(0, 10, 11, base= 2).astype('int')
        f_score_set = np.empty((threshold_set.shape[0], average_step_set.shape[0]))
        for i in range(threshold_set.shape[0]):
            for j in  range(average_step_set.shape[0]):
                train_val_average = self.moving_average_process(train_val_predict, average_step_set[j])
                train_val_bino = self.binarize(train_val_average, threshold_set[i])
                f_score_set[i, j] = self.cal_F_score(train_val_label, train_val_bino)
        #the np.where output is an numpy array [best_threshold_index], [best_step_index]
        best_threshold_index, best_step_index = np.where(f_score_set == np.max(f_score_set))
        best_threshold = threshold_set[best_threshold_index[0]]
        best_average_step = average_step_set[best_step_index[0]]

        model = {}
        model['best_threshold'] = best_threshold
        model['best_average_step'] = best_average_step
        pickle.dump(model, open(model_path, 'wb'))




    def predict_average(self, test_prob_predict_label):
        model_p = pickle.load(open(self.model_path, 'rb'))

        test_predict = test_prob_predict_label['pred']
        test_label = test_prob_predict_label['label']

        best_average_step = model_p['best_average_step']
        best_threshold = model_p['best_threshold']
        test_average = self.moving_average_process(test_predict, best_average_step)
        test_bino = self.binarize(test_average, best_threshold)

        self.test_bino_predict = {'pred': test_bino, 'label': test_label}

    # def average_predict(self, test_prob_predict_label, average_step):
    #     test_predict = test_prob_predict_label['pred']
    #     test_label = test_prob_predict_label['label']

    #     test_average = self.moving_average_process(test_predict, average_step)
    #     test_bino = self.binarize(test_average, 0.5)
    #     test_bino_predict = {'pred': test_bino, 'label': test_label}
    #     return test_bino_predict








class threshold_post_process:

    def __init__(self, model_path):
        self.train_bino_predict = None
        self.val_bino_predict = None
        self.test_bino_predict = None
        self.model_path = model_path

    @staticmethod
    def binarize(prob_set, threshold):
        low = 0
        up = 1
        return np.where(prob_set>threshold, up, low)

    # 绘制混淆矩阵 draw confusion matrix ( 对整个数组生效 )
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


    def train_threshold(self, train_prob_predict_label, val_prob_predict_label, test_prob_predict_label):
        model_path = self.model_path

        train_predict = train_prob_predict_label['pred']
        train_label = train_prob_predict_label['label']
        val_predict = val_prob_predict_label['pred']
        val_label = val_prob_predict_label['label']
        test_predict = test_prob_predict_label['pred']
        test_label = test_prob_predict_label['label']

        train_val_predict = np.concatenate((train_predict, val_predict), axis= 0)
        train_val_label = np.concatenate((train_label, val_label), axis = 0)

        #grid search for the threshold
        f_score_set = []
        threshold_set = np.linspace(1, 9, 9)/10
        for threshold in threshold_set:
            train_val_bino = self.binarize(train_val_predict, threshold)
            f_score_set.append(self.cal_F_score(train_val_label, train_val_bino))
        best_index = f_score_set.index(max(f_score_set))
        best_threshold = threshold_set[best_index]

        model  = {}
        model['best_threshold'] = best_threshold
        pickle.dump(model, open(model_path, 'wb'))

        self.train_bino_predict = {'pred': self.binarize(train_predict, best_threshold), 'label': train_label}
        self.val_bino_predict = {'pred': self.binarize(val_predict, best_threshold), 'label': val_label}
        self.test_bino_predict = {'pred': self.binarize(test_predict, best_threshold), 'label': test_label}





    def predict_threshold(self, test_prob_predict_label):
        model_p = pickle.load(open(self.model_path, 'rb'))
        best_threshold = model_p['best_threshold']


        test_predict = test_prob_predict_label['pred']
        test_label = test_prob_predict_label['label']

        self.test_bino_predict = {'pred': self.binarize(test_predict, best_threshold), 'label': test_label}