import pickle
import pandas as pd




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
                 'svm_hmm_bino': svm_result['hmm_bino_result']['label event'], 'svm': svm_result['svm_result']['label event']}



result_event = pd.DataFrame(result_event)
result_event.to_csv('../result/result_event.csv', sep = ',', index= False)