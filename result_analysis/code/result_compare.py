import pickle
import pandas as pd
import numpy as np



crnn2_result = pickle.load(open('../../CRNN2/crnn_results/crnn_results_summary.p', 'rb'))
crnn4_result = pickle.load(open('../../CRNN4/crnn_results/crnn_results_summary.p', 'rb'))
crnn6_result = pickle.load(open('../../CRNN6/crnn_results/crnn_results_summary.p', 'rb'))
crnn8_result = pickle.load(open('../../CRNN8/crnn_results/crnn_results_summary.p', 'rb'))
crnn10_result = pickle.load(open('../../CRNN10/crnn_results/crnn_results_summary.p', 'rb'))
crnn40_result = pickle.load(open('../../CRNN40/crnn_results/crnn_results_summary.p', 'rb'))
crnn100_result = pickle.load(open('../../CRNN100/crnn_results/crnn_results_summary.p', 'rb'))
crnn400_result = pickle.load(open('../../CRNN400/crnn_results/crnn_results_summary.p', 'rb'))
crnn1200_result = pickle.load(open('../../CRNN1200/crnn_results/crnn_results_summary.p', 'rb'))
vgg_result = pickle.load(open('../../VGG/results/vgg_results_summary.p', 'rb'))
lenet_result = pickle.load(open('../../LENET/results/lenet_results_summary.p', 'rb'))
svm_result = pickle.load(open('../../SVM/results/svm_results_summary.p', 'rb'))


result_summary = {'crnn2': pd.DataFrame(crnn2_result), 'crnn4': pd.DataFrame(crnn4_result), 'crnn6': pd.DataFrame(crnn6_result),
                                'crnn8': pd.DataFrame(crnn8_result), 'crnn10': pd.DataFrame(crnn10_result), 'crnn40': pd.DataFrame(crnn40_result),
                                'crnn100': pd.DataFrame(crnn100_result), 'crnn400': pd.DataFrame(crnn400_result), 'crnn1200': pd.DataFrame(crnn1200_result),
                                'vgg': pd.DataFrame(vgg_result), 'lenet': pd.DataFrame(lenet_result), 'svm': pd.DataFrame(svm_result)}
result_summary = pd.concat(result_summary)
result_summary.to_csv('../result/result_summary.csv', sep = ',')




crnn_pitch_shift = pickle.load(open('../../CRNN400/crnn_results/pitch_shift_results.p', 'rb'))
crnn_time_stretch = pickle.load(open('../../CRNN400/crnn_results/time_stretch_results.p', 'rb'))
crnn_crop = pickle.load(open('../../CRNN400/crnn_results/crop_results.p', 'rb'))
lenet_pitch_shift = pickle.load(open('../../LENET/results/pitch_shift_results.p', 'rb'))
lenet_time_stretch = pickle.load(open('../../LENET/results/time_stretch_results.p', 'rb'))
lenet_crop = pickle.load(open('../../LENET/results/crop_results.p', 'rb'))
svm_pitch_shift = pickle.load(open('../../SVM/results/pitch_shift_results.p', 'rb'))
svm_time_stretch = pickle.load(open('../../SVM/results/time_stretch_results.p', 'rb'))
svm_crop = pickle.load(open('../../SVM/results/crop_results.p', 'rb'))

simulation_summary = {'crnn_picth_shift': pd.DataFrame(crnn_pitch_shift), 'crnn_time_stretch': pd.DataFrame(crnn_time_stretch),
                        'crnn_crop': pd.DataFrame(crnn_crop), 'lenet_picth_shift': pd.DataFrame(lenet_pitch_shift),
                        'lenet_time_stretch': pd.DataFrame(lenet_time_stretch), 'lenet_crop': pd.DataFrame(lenet_crop),
                        'svm_pitch_shift': pd.DataFrame(svm_pitch_shift), 'svm_time_stretch': pd.DataFrame(svm_time_stretch),
                        'svm_crop': pd.DataFrame(svm_crop)}

simulation_summary = pd.concat(simulation_summary)
simulation_summary.to_csv('../result/simulation_summary.csv', sep = ',')








###############################################3
crnn_result = pickle.load(open('../../CRNN400/crnn_results/crnn_results_summary.p', 'rb'))
lenet_result = pickle.load(open('../../LENET/results/lenet_results_summary.p', 'rb'))
svm_result = pickle.load(open('../../SVM/results/svm_results_summary.p', 'rb'))



result_event = {'crnn': crnn_result['threshold_result']['label event'], 'lenet_hmm_bino': lenet_result['hmm_bino_threshold_result']['label event'],
                 'lenet_hmm_gmm': lenet_result['hmm_gmm_result']['label event'], 'lenet': lenet_result['threshold_result']['label event'], 
                 'svm_hmm_bino': svm_result['hmm_bino_result']['label event'], 'svm': svm_result['svm_result']['label event'], 
                 'crnn_hmm': crnn_result['hmm_bino_threshold_result']['label event']}#add a crnn hmm result 



result_event = pd.DataFrame(result_event)
result_event.to_csv('../result/result_event2.csv', sep = ',', index= False)





#####################################################
crnn_result_proportion_075 = pickle.load(open('../../CRNN400_proportion_data/crnn_results/proportion_0/crnn_results_summary.p', 'rb'))
crnn_result_proportion_050 = pickle.load(open('../../CRNN400_proportion_data/crnn_results/proportion_1/crnn_results_summary.p', 'rb'))
crnn_result_proportion_025 = pickle.load(open('../../CRNN400_proportion_data/crnn_results/proportion_2/crnn_results_summary.p', 'rb'))

lenet_result_proportion_075 = pickle.load(open('../../LENET_proportion_data/results/proportion_0/lenet_results_summary.p', 'rb'))
lenet_result_proportion_050 = pickle.load(open('../../LENET_proportion_data/results/proportion_1/lenet_results_summary.p', 'rb'))
lenet_result_proportion_025 = pickle.load(open('../../LENET_proportion_data/results/proportion_2/lenet_results_summary.p', 'rb'))

svm_result_proportion_075 = pickle.load(open('../../SVM_proportion_data/results/proportion_0/svm_results_summary.p', 'rb'))
svm_result_proportion_050 = pickle.load(open('../../SVM_proportion_data/results/proportion_1/svm_results_summary.p', 'rb'))
svm_result_proportion_025 = pickle.load(open('../../SVM_proportion_data/results/proportion_2/svm_results_summary.p', 'rb'))

proportion_data_summary = {'crnn_proportion_075': pd.DataFrame(crnn_result_proportion_075), 
                            'crnn_proportion_050': pd.DataFrame(crnn_result_proportion_050), 
                            'crnn_proportion_025': pd.DataFrame(crnn_result_proportion_025),                    
                            'lenet_proportion_075': pd.DataFrame(lenet_result_proportion_075), 
                            'lenet_proportion_050': pd.DataFrame(lenet_result_proportion_050),
                            'lenet_proportion_025': pd.DataFrame(lenet_result_proportion_025),
                            'svm_proportion_075': pd.DataFrame(svm_result_proportion_075), 
                            'svm_proportion_050': pd.DataFrame(svm_result_proportion_050), 
                            'svm_proportion_025': pd.DataFrame(svm_result_proportion_025)}

proportion_data_summary = pd.concat(proportion_data_summary)
proportion_data_summary.to_csv('../result/proportion_data_summary.csv', sep = ',')




################################################################################
#calculate variance 
crnn2_result = pickle.load(open('../../CRNN2/crnn_results/test_conf_matrix_set.p', 'rb'))
crnn4_result = pickle.load(open('../../CRNN4/crnn_results/test_conf_matrix_set.p', 'rb'))
crnn6_result = pickle.load(open('../../CRNN6/crnn_results/test_conf_matrix_set.p', 'rb'))
crnn8_result = pickle.load(open('../../CRNN8/crnn_results/test_conf_matrix_set.p', 'rb'))
crnn10_result = pickle.load(open('../../CRNN10/crnn_results/test_conf_matrix_set.p', 'rb'))
crnn40_result = pickle.load(open('../../CRNN40/crnn_results/test_conf_matrix_set.p', 'rb'))
crnn100_result = pickle.load(open('../../CRNN100/crnn_results/test_conf_matrix_set.p', 'rb'))
crnn400_result = pickle.load(open('../../CRNN400/crnn_results/test_conf_matrix_set.p', 'rb'))
crnn1200_result = pickle.load(open('../../CRNN1200/crnn_results/test_conf_matrix_set.p', 'rb'))


def cal_result_var(result_set, fold_number):
    def cal_precision(result_matrix):
        precision = result_matrix[1,1]   / (result_matrix[1,0] + result_matrix[1,1])
        return precision
    def cal_recall(result_matrix):
        recall = result_matrix[1,1]   / (result_matrix[0,1] + result_matrix[1,1])
        return recall
    def cal_f_score(result_matrix):
        f_score = 2 * result_matrix[1,1]/(2 * result_matrix[1,1] + result_matrix[0,1] + result_matrix[1,0])
        return f_score
    precision_set = []
    recall_set = []
    f_set = []
    for i in range(fold_number):
        precision_set.append(cal_precision(result_set[i]))
        recall_set.append(cal_recall(result_set[i]))
        f_set.append(cal_f_score(result_set[i]))
    # print(precision_set)
    # print(recall_set)
    # print(f_set)
    precision_var = np.var(precision_set)
    recall_var = np.var(recall_set)
    f_score_var = np.var(f_set)
    return precision_var, recall_var, f_score_var

crnn2_var = cal_result_var(crnn2_result, 4)
crnn4_var = cal_result_var(crnn4_result, 4)
crnn6_var = cal_result_var(crnn6_result, 4)
crnn8_var = cal_result_var(crnn8_result, 4)
crnn10_var = cal_result_var(crnn10_result, 4)
crnn40_var = cal_result_var(crnn40_result, 4)
crnn100_var = cal_result_var(crnn100_result, 4)
crnn400_var = cal_result_var(crnn400_result, 4)
crnn1200_var = cal_result_var(crnn1200_result, 4)




