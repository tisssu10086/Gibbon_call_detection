# install.packages('cowplot')
library(ggplot2)
library(tidyr)
library(dplyr)
library(cowplot)

result_summary = read.csv('../result/result_summary.csv',  na.strings=c("","NA"), stringsAsFactors = F)
summary_col_name = names(result_summary)
summary_col_name[1] = 'method'
summary_col_name[2] = 'metrics'
names(result_summary) = summary_col_name 
# fix typo, have to make string as factor to fix typo
result_summary[result_summary == 'sgement recall'] = 'segment recall'
result_summary[result_summary == 'sgement F-score'] = 'segment F-score'

result_summary_long = gather(result_summary, result_type, value_set, -method, -metrics, na.rm = T)
result_summary = spread(result_summary_long, metrics, value_set)
write.csv(result_summary, '../result/result_summary_reformat.csv', row.names = F)

result_summary_unite = unite(result_summary_long, col = method, method, result_type)
result_summary_segment = subset(result_summary_unite, metrics == 'segment precision' 
                                     | metrics == 'segment recall' 
                                     | metrics == 'segment F-score')
result_summary_segment$value_set = as.numeric(result_summary_segment$value_set) * 100

result_summary_event = subset(result_summary_unite, metrics == 'event precision' 
                              | metrics == 'event recall' 
                              | metrics == 'event F-score')
result_summary_event$value_set = as.numeric(result_summary_event$value_set) * 100

result_summary_encounter = subset(result_summary_unite, metrics == 'encounter error rate')
result_summary_encounter$value_set = as.numeric(result_summary_encounter$value_set) * 100



##########################################################
#draw plot for segment result
pdf(file="results_segment.pdf",h=5,w=7)
#discrete x value need declare the group variable
p1 = ggplot(data = result_summary_segment, mapping = aes(x = method, y = value_set, colour = metrics, group = metrics))
p1 = p1 + geom_point(data = result_summary_segment[result_summary_segment['method'] == 'crnn400_threshold_result',], aes(x = method, y = value_set))

p1 = p1 + geom_line(data = result_summary_segment[result_summary_segment['method'] == 'lenet_hmm_bino_threshold_result'
                                                       | result_summary_segment['method'] == 'lenet_hmm_gmm_result'
                                                       | result_summary_segment['method'] == 'lenet_threshold_result',], aes(x = method, y = value_set))
p1 = p1 + geom_point(data = result_summary_segment[result_summary_segment['method'] == 'lenet_hmm_bino_threshold_result'
                                                        | result_summary_segment['method'] == 'lenet_hmm_gmm_result'
                                                        | result_summary_segment['method'] == 'lenet_threshold_result',], aes(x = method, y = value_set))

p1 = p1 + geom_line(data = result_summary_segment[result_summary_segment['method'] == 'vgg_hmm_bino_threshold_result'
                                                       | result_summary_segment['method'] == 'vgg_hmm_gmm_result'
                                                       | result_summary_segment['method'] == 'vgg_threshold_result',], aes(x = method, y = value_set))
p1 = p1 + geom_point(data = result_summary_segment[result_summary_segment['method'] == 'vgg_hmm_bino_threshold_result'
                                                       | result_summary_segment['method'] == 'vgg_hmm_gmm_result'
                                                       | result_summary_segment['method'] == 'vgg_threshold_result',], aes(x = method, y = value_set))

p1 = p1 + geom_line(data = result_summary_segment[result_summary_segment['method'] == 'svm_hmm_bino_result'
                                                       | result_summary_segment['method'] == 'svm_svm_result',], aes(x = method, y = value_set))
p1 = p1 + geom_point(data = result_summary_segment[result_summary_segment['method'] == 'svm_hmm_bino_result'
                                                        | result_summary_segment['method'] == 'svm_svm_result',], aes(x = method, y = value_set))

p1 = p1 + theme(axis.text.x = element_text(angle = 45, hjust = 1))
p1 = p1 + labs(y = 'Percentage(%)', x = 'Comparison methods')

level_order = c('crnn400_threshold_result',
                'lenet_hmm_bino_threshold_result',
                'lenet_hmm_gmm_result',
                'lenet_threshold_result',
                'vgg_hmm_bino_threshold_result',
                'vgg_hmm_gmm_result',
                'vgg_threshold_result',
                'svm_hmm_bino_result',
                'svm_svm_result')

level_name = c('crnn400_threshold_result' = '1.CRNN-400',
               'lenet_hmm_bino_threshold_result' = '2.LeNet+HMM(Bern)',
               'lenet_hmm_gmm_result' = '3.LeNet+HMM(GMM)',
               'lenet_threshold_result' = '4.LeNet',
               'vgg_hmm_bino_threshold_result' = '5.VGGNet+HMM(Bern)',
               'vgg_hmm_gmm_result' = '6.VGGNet+HMM(GMM)',
               'vgg_threshold_result' = '7.VGGNet',
               'svm_hmm_bino_result' = '8.MFCC-SVM+HMM(Bern)',
               'svm_svm_result' = '9.MFCC-SVM')

p1 = p1 + scale_x_discrete(labels = level_name, limits = level_order)
p1 = p1 + scale_colour_discrete(name = 'Metrics' ,labels = c('Segment F-score', 'Segment precision', 'Segment recall'))
p1
dev.off()



#################################################################
#draw plot for crnn length comparison
pdf(file="crnn_comparison.pdf",h=5,w=7)
p2 = ggplot(data = result_summary_segment, mapping = aes(x = method, y = value_set, colour = metrics, group = metrics))
p2 = p2 + geom_point(data = result_summary_segment[result_summary_segment['method'] == 'crnn2_threshold_result'
                                                   |result_summary_segment['method'] == 'crnn4_threshold_result'
                                                   |result_summary_segment['method'] == 'crnn6_threshold_result'
                                                   |result_summary_segment['method'] == 'crnn8_threshold_result'
                                                   |result_summary_segment['method'] == 'crnn10_threshold_result'
                                                   |result_summary_segment['method'] == 'crnn40_threshold_result'
                                                   |result_summary_segment['method'] == 'crnn100_threshold_result'
                                                   |result_summary_segment['method'] == 'crnn400_threshold_result'
                                                   |result_summary_segment['method'] == 'crnn1200_threshold_result',], aes(x = method, y = value_set))
p2 = p2 +  geom_line(data = result_summary_segment[result_summary_segment['method'] == 'crnn2_threshold_result'
                                                 |result_summary_segment['method'] == 'crnn4_threshold_result'
                                                 |result_summary_segment['method'] == 'crnn6_threshold_result'
                                                 |result_summary_segment['method'] == 'crnn8_threshold_result'
                                                 |result_summary_segment['method'] == 'crnn10_threshold_result'
                                                 |result_summary_segment['method'] == 'crnn40_threshold_result'
                                                 |result_summary_segment['method'] == 'crnn100_threshold_result'
                                                 |result_summary_segment['method'] == 'crnn400_threshold_result'
                                                 |result_summary_segment['method'] == 'crnn1200_threshold_result',], aes(x = method, y = value_set))
p2 = p2 + theme(axis.text.x = element_text(angle = 0, hjust = 1))
p2 = p2 + labs(y = 'Percentage(%)', x = 'Sequence length of the CRNN')
level_order = c('crnn2_threshold_result',
                'crnn4_threshold_result',
                'crnn6_threshold_result',
                'crnn8_threshold_result',
                'crnn10_threshold_result',
                'crnn40_threshold_result',
                'crnn100_threshold_result',
                'crnn400_threshold_result',
                'crnn1200_threshold_result')

level_name = c('crnn2_threshold_result' = '2',
                'crnn4_threshold_result' = '4',
                'crnn6_threshold_result' = '6',
                'crnn8_threshold_result' = '8',
                'crnn10_threshold_result' = '10',
                'crnn40_threshold_result' = '40',
                'crnn100_threshold_result' = '100',
                'crnn400_threshold_result'= '400',
                'crnn1200_threshold_result' = '1200')

p2 = p2 + scale_x_discrete(labels = level_name, limits = level_order)
p2 = p2 + scale_colour_discrete(name = 'Metrics' ,labels = c('Segment F-score', 'Segment precision', 'Segment recall'))
p2
dev.off()



##########################################################
#draw plot for event result
pdf(file="results_event.pdf",h=5,w=7)
p3 = ggplot(data = result_summary_event, mapping = aes(x = method, y = value_set, colour = metrics, group = metrics))
p3 = p3 + geom_point(data = result_summary_event[result_summary_event['method'] == 'crnn400_threshold_result',], aes(x = method, y = value_set))

p3 = p3 + geom_line(data = result_summary_event[result_summary_event['method'] == 'lenet_hmm_bino_threshold_result'
                                                  | result_summary_event['method'] == 'lenet_hmm_gmm_result'
                                                  | result_summary_event['method'] == 'lenet_threshold_result',], aes(x = method, y = value_set))
p3 = p3 + geom_point(data = result_summary_event[result_summary_event['method'] == 'lenet_hmm_bino_threshold_result'
                                                   | result_summary_event['method'] == 'lenet_hmm_gmm_result'
                                                   | result_summary_event['method'] == 'lenet_threshold_result',], aes(x = method, y = value_set))

p3 = p3 + geom_line(data = result_summary_event[result_summary_event['method'] == 'vgg_hmm_bino_threshold_result'
                                                  | result_summary_event['method'] == 'vgg_hmm_gmm_result'
                                                  | result_summary_event['method'] == 'vgg_threshold_result',], aes(x = method, y = value_set))
p3 = p3 + geom_point(data = result_summary_event[result_summary_event['method'] == 'vgg_hmm_bino_threshold_result'
                                                   | result_summary_event['method'] == 'vgg_hmm_gmm_result'
                                                   | result_summary_event['method'] == 'vgg_threshold_result',], aes(x = method, y = value_set))

p3 = p3 + geom_line(data = result_summary_event[result_summary_event['method'] == 'svm_hmm_bino_result'
                                                  | result_summary_event['method'] == 'svm_svm_result',], aes(x = method, y = value_set))
p3 = p3 + geom_point(data = result_summary_event[result_summary_event['method'] == 'svm_hmm_bino_result'
                                                   | result_summary_event['method'] == 'svm_svm_result',], aes(x = method, y = value_set))

p3 = p3 + theme(axis.text.x = element_text(angle = 45, hjust = 1))
p3 = p3 + labs(y = 'Percentage(%)', x = 'Comparison methods')

level_order = c('crnn400_threshold_result',
                'lenet_hmm_bino_threshold_result',
                'lenet_hmm_gmm_result',
                'lenet_threshold_result',
                'vgg_hmm_bino_threshold_result',
                'vgg_hmm_gmm_result',
                'vgg_threshold_result',
                'svm_hmm_bino_result',
                'svm_svm_result')

level_name = c('crnn400_threshold_result' = '1.CRNN-400',
               'lenet_hmm_bino_threshold_result' = '2.LeNet+HMM(Bern)',
               'lenet_hmm_gmm_result' = '3.LeNet+HMM(GMM)',
               'lenet_threshold_result' = '4.LeNet',
               'vgg_hmm_bino_threshold_result' = '5.VGGNet+HMM(Bern)',
               'vgg_hmm_gmm_result' = '6.VGGNet+HMM(GMM)',
               'vgg_threshold_result' = '7.VGGNet',
               'svm_hmm_bino_result' = '8.MFCC-SVM+HMM(Bern)',
               'svm_svm_result' = '9.MFCC-SVM')

p3 = p3 + scale_x_discrete(labels = level_name, limits = level_order)
p3 = p3 + scale_colour_discrete(name = 'Metrics' ,labels = c('Phrase F-score', 'Phrase precision', 'Phrase recall'))
p3
dev.off()




######################################################################3
#draw encounter rate plot
pdf(file="results_encounter_error.pdf",h=5,w=7)
p4 = ggplot(data = result_summary_encounter, mapping = aes(x = method, y = value_set, group = metrics))
p4 = p4 + geom_point(data = result_summary_encounter[result_summary_encounter['method'] == 'crnn400_threshold_result',], aes(x = method, y = value_set))

p4 = p4 + geom_line(data = result_summary_encounter[result_summary_encounter['method'] == 'lenet_hmm_bino_threshold_result'
                                                | result_summary_encounter['method'] == 'lenet_hmm_gmm_result'
                                                | result_summary_encounter['method'] == 'lenet_threshold_result',], aes(x = method, y = value_set, colour = 'red'))
p4 = p4 + geom_point(data = result_summary_encounter[result_summary_encounter['method'] == 'lenet_hmm_bino_threshold_result'
                                                 | result_summary_encounter['method'] == 'lenet_hmm_gmm_result'
                                                 | result_summary_encounter['method'] == 'lenet_threshold_result',], aes(x = method, y = value_set))

p4 = p4 + geom_line(data = result_summary_encounter[result_summary_encounter['method'] == 'vgg_hmm_bino_threshold_result'
                                                | result_summary_encounter['method'] == 'vgg_hmm_gmm_result'
                                                | result_summary_encounter['method'] == 'vgg_threshold_result',], aes(x = method, y = value_set, colour = 'red'))
p4 = p4 + geom_point(data = result_summary_encounter[result_summary_encounter['method'] == 'vgg_hmm_bino_threshold_result'
                                                 | result_summary_encounter['method'] == 'vgg_hmm_gmm_result'
                                                 | result_summary_encounter['method'] == 'vgg_threshold_result',], aes(x = method, y = value_set))

p4 = p4 + geom_line(data = result_summary_encounter[result_summary_encounter['method'] == 'svm_hmm_bino_result'
                                                | result_summary_encounter['method'] == 'svm_svm_result',], aes(x = method, y = value_set, colour = 'red'))
p4 = p4 + geom_point(data = result_summary_encounter[result_summary_encounter['method'] == 'svm_hmm_bino_result'
                                                 | result_summary_encounter['method'] == 'svm_svm_result',], aes(x = method, y = value_set))

p4 = p4 + theme(axis.text.x = element_text(angle = 45, hjust = 1))
p4 = p4 + labs(y = 'Percentage(%)', x = 'Comparison methods')

level_order = c('crnn400_threshold_result',
                'lenet_hmm_gmm_result',
                'lenet_hmm_bino_threshold_result',
                'lenet_threshold_result',
                'vgg_hmm_gmm_result',
                'vgg_hmm_bino_threshold_result',
                'vgg_threshold_result',
                'svm_hmm_bino_result',
                'svm_svm_result')

level_name = c('crnn400_threshold_result' = '1.CRNN-400',
               'lenet_hmm_gmm_result' = '2.LeNet+HMM(GMM)',
               'lenet_hmm_bino_threshold_result' = '3.LeNet+HMM(Bern)',
               'lenet_threshold_result' = '4.LeNet',
               'vgg_hmm_gmm_result' = '5.VGGNet+HMM(GMM)',
               'vgg_hmm_bino_threshold_result' = '6.VGGNet+HMM(Bern)',
               'vgg_threshold_result' = '7.VGGNet',
               'svm_hmm_bino_result' = '8.MFCC-SVM+HMM(Bern)',
               'svm_svm_result' = '9.MFCC-SVM')

p4 = p4 + scale_x_discrete(labels = level_name, limits = level_order)
p4 = p4 + theme(legend.position = "none")
p4
dev.off()









#########################################################################################################################
#load the data twice to avoid colname change
simulation_summary = read.csv('../result/simulation_summary.csv', na.strings = c("", "NA"), header = F, stringsAsFactors = F)
simulation_name = as.matrix(simulation_summary[1,])
simulation_name[1] = 'method'
simulation_name[2] = 'metrics'
simulation_summary = read.csv('../result/simulation_summary.csv', na.strings = c("", "NA"), stringsAsFactors = F)
names(simulation_summary) = simulation_name
#fix typo 
simulation_summary[simulation_summary == 'sgement recall'] = 'segment recall'
simulation_summary[simulation_summary == 'sgement F-score'] = 'segment F-score'
simulation_summary_long = gather(simulation_summary, simulation_config, value_set, -method, -metrics, na.rm = T)
simulation_summary = spread(simulation_summary_long, metrics, value_set)
write.csv(simulation_summary, '../result/simulation_summray_reformat.csv', row.names = F)


simulation_compare = subset(simulation_summary_long, metrics == 'segment F-score' | metrics == 'event F-score')
simulation_separate = extract(simulation_compare, col="method", into = c("method","simulation"), regex="([[:alpha:]]+)_([a-zA-Z0-9_]+)")
# fix typo
simulation_separate[simulation_separate == 'picth_shift'] = 'pitch_shift'
# regrex split 
simulation_separate = extract(simulation_separate, col="simulation_config", 
                              into = c("simulation_config","result_type"), regex="([a-zA-Z_]+_[0-9_[:punct:]]+)_([a-zA-Z_]+)")
simulation_draw = unite(simulation_separate, col = method, method, result_type)
simulation_draw = subset(simulation_draw, method == 'crnn_threshold_result'
                         | method == 'lenet_threshold_result'
                         | method == 'lenet_hmm_bino_threshold_result'
                         | method == 'svm_svm_result'
                         | method == 'svm_hmm_bino_result')

#add result for origin data
origin_result = subset(result_summary_unite, method == 'crnn400_threshold_result'
                         |method == 'lenet_hmm_bino_threshold_result'
                         |method =='lenet_threshold_result'
                         |method == 'svm_hmm_bino_result'
                         |method == 'svm_svm_result')
origin_result = subset(origin_result, metrics == 'segment F-score'| metrics == 'event F-score')
origin_result[origin_result == 'crnn400_threshold_result'] = 'crnn_threshold_result'
origin_result$simulation_config = 'Raw'
origin_result_crop = origin_result
origin_result_crop$simulation = 'crop'
origin_result_ts = origin_result
origin_result_ts$simulation = 'time_stretch'
origin_result_ps = origin_result
origin_result_ps$simulation = 'pitch_shift'
origin_result =  merge(origin_result_ts, origin_result_crop, all = TRUE)
origin_result = merge(origin_result, origin_result_ps, all = TRUE)
simulation_draw = merge(simulation_draw,origin_result, all = TRUE)

#modify word
simulation_draw$value_set = as.numeric(simulation_draw$value_set) * 100
simulation_draw[simulation_draw == 'segment F-score'] = 'Segment F-score'
simulation_draw[simulation_draw == 'event F-score'] = 'Phrase F-score'
simulation_draw[simulation_draw == 'time_stretch'] = 'Time stretch'
simulation_draw[simulation_draw == 'pitch_shift'] = 'Pitch shift'
simulation_draw[simulation_draw == 'crop'] = 'Random crop'




pdf(file="results_simulation.pdf",h=6,w=8)
p5 = ggplot(data = simulation_draw, mapping = aes(x = reorder(simulation_config, -value_set), y = value_set, colour = method, group = method))
p5 = p5 + geom_line() + geom_point()

p5 = p5 + facet_wrap(~metrics + simulation, nrow = 2, scales = 'free_x')

level_name = c('crop_rate_0.600' = '0.4',
               'crop_rate_0.700' = '0.3',
               'crop_rate_0.800' = '0.2',
               'crop_rate_0.900' = '0.1',
               'shift_step_-1.000' = '-1.0',
               'shift_step_-0.500' = '-0.5',
               'shift_step_0.500' = '0.5',
               'shift_step_1.000' = '1.0',
               'stretch_rate_0.810' = '0.81',
               'stretch_rate_0.930' = '0.93',
               'stretch_rate_1.070' = '1.07',
               'stretch_rate_1.230' = '1.23')
        
p5 = p5 + scale_x_discrete(labels = level_name)
p5 = p5 + labs(y = 'Percentage(%)', x = 'Simulation configuration')
label_name = c('CRNN-400',
               'LeNet+HMM(Bern)',
               'LeNet',
               'MFCC-SVM+HMM(Bern)',
               'MFCC-SVM')
p5 = p5 + scale_colour_discrete(name = 'Methods' ,labels = label_name)
p5
dev.off()





##############################
result_event = read.csv('../result/result_event.csv')
signal_strength_event = read.csv('../result/signal_strength_event.csv')
snr = data.frame(x = signal_strength_event$sig_rms_origin - signal_strength_event$sig_rms_neg) 

result_event = bind_cols(result_event, snr)
result_col_name = c('CRNN-400',
               'LeNet+HMM(Bern)',
               'LeNet+HMM(GMM)',
               'LeNet',
               'MFCC-SVM+HMM(Bern)',
               'MFCC-SVM',
               'SNR')
names(result_event) = result_col_name
result_event = gather(result_event, method,  value_set, -SNR)

pdf(file="detection_vs_snr.pdf",h=5,w=12)
p = ggplot(result_event, aes(SNR, fill = factor(value_set))) 
p = p + geom_histogram(bins = 50, mapping = aes(fill =  factor(value_set)), position = 'fill', na.rm = TRUE )
p = p + facet_wrap(~method, nrow = 1, scales = 'free_x')
p = p + labs(y = 'Proportion', x = 'SNR')
p = p + scale_fill_discrete(name = 'Label phrase states' , labels = c('Undetected', 'Detected'))
p
dev.off()


################################
signal_strength_event = read.csv('../result/signal_strength_event.csv')
snr = data.frame(x = signal_strength_event$sig_rms_origin - signal_strength_event$sig_rms_neg) 
duration = read.csv('../../label_process/label_analysis_result/gibbon_call_duration.csv', header = F)

p1 = ggplot(snr, aes(x)) 
p1 = p1 + geom_histogram(bins = 60, fill = '#3182bd')
p1 = p1 + labs(y = 'Number of gibbon phrase', x = 'SNR (dB)')
p1

p2 = ggplot(duration, aes(V1))
p2 = p2 + geom_histogram(binwidth = 1, fill = '#3182bd')
p2 = p2 + labs(y = 'Number of gibbon phrase', x = 'Duration of gibbon phrase (s)')
p2

p3 = plot_grid(p2, p1, labels = c("A", "B"), rel_widths = c(2, 3), align = 'h')
save_plot("data_description.pdf", p3)






