'''This file is to matching the label with prediction and calculate evaluation metric for the algorithm'''
import numpy as np


class result_analysis:
    '''match_type: 'iou_match' or 'hit_match', iou_match can find match pairs with maximum iou value, 
    while hit_macth can find matched pairs with maximum matched number'''
    def __init__(self, iou_threshold: float, match_type: str):
        self.iou_threshold = iou_threshold
        self.match_type = match_type
        self.conf_mat = np.zeros((2,2))
        self.all_label_num = 0
        self.all_pred_num = 0
        self.all_match_num = 0
        self.all_match_iou = 0
        self.all_pred_event = []
        self.all_label_event = []



    def __call__(self, label_set, pred_set):
        if label_set.ndim == 1:
            label_set = label_set[np.newaxis, :]
            pred_set = pred_set[np.newaxis, :]

        for label_seq, pred_seq in zip(label_set, pred_set):
            # print(label_seq.shape)
            # print(pred_seq.shape)
            match_mat, label_num, predict_num = self.Matching_matrix(label_seq, pred_seq, self.iou_threshold)
            #chech whether predict event and label event are matched
            pred_event = match_mat[:, :, 1].sum(axis = 0)
            label_event = match_mat[:, :, 1].sum(axis = 1)
            #change value not 0 to 1, keep 0
            pred_event = pred_event.astype(bool).astype(int)
            label_event = label_event.astype(bool).astype(int)
            self.all_pred_event.extend(pred_event)
            self.all_label_event.extend(label_event)

            if self.match_type == 'hit_match':
                match_number, match_iou = self.hit_matching(match_mat)
            elif self.match_type == 'iou_match':
                match_number, match_iou = self.iou_matching(match_mat)
            else:
                print('match type must be hit_matching or iou_matching')
            self.all_label_num += label_num
            self.all_pred_num += predict_num
            self.all_match_num += match_number
            self.all_match_iou += match_iou
            self.confusion_matrix(label_seq, pred_seq, self.conf_mat)


    def result_process(self):
        self.result_summary = {'confusion matrix': self.conf_mat, 'segment precision': self.conf_mat[1,1] / (self.conf_mat[1,0] + self.conf_mat[1,1]),
                                'sgement recall': self.conf_mat[1,1] / (self.conf_mat[0,1] + self.conf_mat[1,1]), 
                                'sgement F-score': 2 * self.conf_mat[1,1]/(2 * self.conf_mat[1,1] + self.conf_mat[0,1] + self.conf_mat[1,0]),
                                'number of label': self.all_label_num, 'number of prediction': self.all_pred_num, 'number of matching event': self.all_match_num,
                                'total matching iou': self.all_match_iou, 'encounter number error' : abs(self.all_pred_num - self.all_label_num),
                                'encounter error rate': abs(self.all_pred_num - self.all_label_num)/ self.all_label_num,
                                'event precision': self.all_match_num/ self.all_pred_num, 'event recall': self.all_match_num/ self.all_label_num,
                                'event F-score': 2*self.all_match_num/ (self.all_label_num + self.all_pred_num),
                                'pred event': np.asarray(self.all_pred_event), 'label event': np.asarray(self.all_label_event)}





    # 绘制混淆矩阵 draw confusion matrix ( 对整个数组生效 )
    @staticmethod
    def confusion_matrix(labels, preds, conf_matrix):
        for p, t in zip(preds, labels):
            conf_matrix[p, t] += 1
        return conf_matrix

    @staticmethod
    def Matching_matrix(label_seq, pred_seq, iou_threshold):
        '''This funciton is to build a matching matrix between prediction and label sequence based on IOU threshold, the matrix will 
            be in three dimension, where row indew coordinate stands for label and column index stands for prediction, the value in first layer of
            z dimension stands for IOU between corresponding label and prediction, and the second layer of z dimension stands for 
            whether the corresponding label and prediction are seen as matching (TP) with certain IOU threshold. 
            This function will also return the number of label event and prediction event'''
        #make sure that the prediction seq and label seq are numpy array
        pred_seq = np.asarray(pred_seq)
        label_seq = np.asarray(label_seq)
        #seperate all labels in the label sequence and store in list 
        if label_seq[0] == 1:
            count = 0
            label_chunk = [[0]]
        elif label_seq[0] == 0:
            count = -1
            label_chunk = []
        for i in range(1 , label_seq.shape[0]):
            if label_seq[i] == 1 and label_seq[i-1] != 1:
                label_chunk.append([i])
                count += 1
            elif label_seq[i] ==1 and label_seq[i-1] == 1:
                label_chunk[count].append(i)
        #seperate all prediction in the pred sequence and store in list
        if pred_seq[0] == 1:
            count = 0
            pred_chunk = [[0]]
        elif pred_seq[0] == 0:
            count = -1
            pred_chunk = []
        for i in range(1 , pred_seq.shape[0]):
            if pred_seq[i] == 1 and pred_seq[i-1] != 1:
                pred_chunk.append([i])
                count += 1
            elif pred_seq[i] ==1 and pred_seq[i-1] == 1:
                pred_chunk[count].append(i)
        #build the zero value matrix with right shape
        matching_mat = np.zeros((len(label_chunk), len(pred_chunk), 2))
        #calculating IOU of the seperate # test the funtion
    # label_seq = np.array((1,0,0,0,1,1,0,0,1,1,1,0,1))  
    # pred_seq = np.array((1,1,0,0,0,1,1,0,0,1,1,0,0))
    # Matching_matrix(label_seq, pred_seq, 0)label and prediction and send value to the first layer of matrix
        for i in range(len(label_chunk)):
            for j in range(len(pred_chunk)):
                intersection = list(set(label_chunk[i]) & set(pred_chunk[j]))
                union = list(set(label_chunk[i]) | set(pred_chunk[j]))
                matching_mat[i,j,0] = len(intersection)/len(union)
                #decide whether counted as a hitting based on IOU value and threshold and send value to the second layer of matrix    
                if matching_mat[i,j,0] > iou_threshold:
                    matching_mat[i,j,1] = 1
        return matching_mat, len(label_chunk), len(pred_chunk)


    @staticmethod
    def hit_matching(matching_mat):
        '''This funtion is to find the hit based matching number (TP) search with sencond layer of matching matrix'''
        if matching_mat.size == 0:
            best_match_number = 0
            best_match_iou = 0
        else:
            for i in range(1, min(matching_mat.shape[0], matching_mat.shape[1])):
                for j in range(i, matching_mat.shape[0]):
                    #find the sub area for the maximum matching number and add to the current column number
                    matching_mat[j,i,1] += np.max(matching_mat[:j, :i, 1])
                    #find the maximum accumulate iou value correspond to maximum matching number and add to current column iou number
                    index = np.where(matching_mat[:j, :i, 1]==np.max(matching_mat[:j, :i, 1]))
                    matching_mat[j,i,0] +=  max(matching_mat[index[0], index[1], 0]) 
                for k in range(i+1, matching_mat.shape[1]):
                    #find the sub area for the maximum matching number and add to the current row number
                    matching_mat[i,k,1] += np.max(matching_mat[:i, :k, 1])
                    #find the maximum accumulate iou value correspond to maximum matching number and add to current row iou number
                    index = np.where(matching_mat[:i, :k, 1]==np.max(matching_mat[:i, :k, 1]))         
                    matching_mat[i,k,0] +=  max(matching_mat[index[0], index[1], 0])  
            #find the best matching number and corresponding IOU value
            best_match_number = np.max(matching_mat[:, :, 1])
            best_index = np.where(matching_mat[:, :, 1]==np.max(matching_mat[:, :, 1]))
            best_match_iou = max(matching_mat[best_index[0], best_index[1], 0])   
        return best_match_number, best_match_iou


    @staticmethod
    def iou_matching(matching_mat):
        '''This funtion is to find the IOU based matching number (TP) search with first layer of matching matrix'''
        if matching_mat.size == 0:
            best_match_number = 0
            best_match_iou = 0
        else:
            for i in range(1, min(matching_mat.shape[0], matching_mat.shape[1])):
                for j in range(i, matching_mat.shape[0]):
                    #find the sub area for the maximum matching IOU and add to the current column IOU
                    matching_mat[j,i,0] += np.max(matching_mat[:j, :i, 0])
                    #find the maximum accumulate iou value correspond to maximum matching number and add to current column number
                    index = np.where(matching_mat[:j, :i, 0]==np.max(matching_mat[:j, :i, 0]))
                    matching_mat[j,i,1] +=  max(matching_mat[index[0], index[1], 1]) 
                for k in range(i+1, matching_mat.shape[1]):
                    #find the sub area for the maximum matching IOU and add to the current row IOU
                    matching_mat[i,k,0] += np.max(matching_mat[:i, :k, 0])
                    #find the maximum accumulate iou value correspond to maximum matching number and add to current row number
                    index = np.where(matching_mat[:i, :k, 0]==np.max(matching_mat[:i, :k, 0]))         
                    matching_mat[i,k,1] +=  max(matching_mat[index[0], index[1], 1])  
            #find the best matching IOU value and corresponding matching number
            best_match_iou = np.max(matching_mat[:, :, 0])
            best_index = np.where(matching_mat[:, :, 0]==np.max(matching_mat[:, :, 0]))
            best_match_number= max(matching_mat[best_index[0], best_index[1], 1])   
        return best_match_number, best_match_iou




