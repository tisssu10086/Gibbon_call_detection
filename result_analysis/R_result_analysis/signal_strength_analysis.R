
library(ggplot2)
library(tidyr)
library(dplyr)
library(cowplot)

result_event = read.csv('../result/result_event2.csv')
result_event = subset(result_event, select = c('crnn', 'lenet_hmm_bino'))
signal_strength_event = read.csv('../result/signal_strength_event.csv')


# signal_strength_event$snr_rms = signal_strength_event$sig_rms_origin - signal_strength_event$sig_rms_neg
# signal_strength_event$snr_max = signal_strength_event$sig_max_origin - signal_strength_event$sig_max_neg

signal_strength_event$sig_rms_cali = 10*log10(10**(signal_strength_event$sig_rms_origin/10) - 
                                                10**(signal_strength_event$sig_rms_neg/10)) 

signal_strength_event$sig_max_cali = 10*log10(10**(signal_strength_event$sig_max_origin/10) - 
                                                10**(signal_strength_event$sig_max_neg/10))

signal_strength_event = subset(signal_strength_event, select = -c(sig_max_neg, sig_rms_neg))



result_event = bind_cols(result_event, signal_strength_event)

result_event = gather(result_event, method,  bino_predict, -sig_max_origin, -sig_rms_origin, 
                      -sig_rms_cali, -sig_max_cali)

result_event = gather(result_event, signal_type, sig_value, -method, -bino_predict)



pdf(file="detection_vs_signal_strength.pdf",h=15,w=6)
p = ggplot(result_event, aes(sig_value, fill = factor(bino_predict))) 
p = p + geom_histogram(bins = 50, mapping = aes(fill =  factor(bino_predict)), position = 'fill', na.rm = TRUE )
p = p + facet_wrap(~signal_type + method, nrow = 6, scales = 'free_x')
p = p + labs(y = 'Proportion', x = 'Signal_strength_value')
p = p + scale_fill_discrete(name = 'Label phrase states' , labels = c('Undetected', 'Detected'))
p
dev.off()
