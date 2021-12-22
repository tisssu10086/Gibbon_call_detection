import gc
import sys
import pickle 
import numpy as np
import pandas as pd
import typing
import copy
import librosa
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

import dataset


def get_animal_sound(ani_file_dic: typing.List[Path], label_path: Path, audio_path: Path, save_path: Path) -> None:
    """Input a list of file name and save a list contains all gibbon call within given file"""
    gibbon_lib = []
    for file_name in ani_file_dic:
        sound, sr = librosa.load(audio_path/ file_name.with_suffix('.wav'), sr = None)
        table = pd.read_csv(label_path / file_name.with_suffix('.data'), sep = ',')
        for i in table.index:
            start_frame = table['Start'][i]*sr
            end_frame = table['End'][i]*sr
            gibbon_lib.append(sound[start_frame:end_frame])
    pickle.dump(gibbon_lib, open(save_path, 'wb'))
    del gibbon_lib
    del sound
    gc.collect()



def get_negative_correspond(ani_file_dic: typing.List[Path], label_path: Path, audio_path: Path, save_path: Path) -> None:
    gibbon_neg_lib = []
    for file_name in ani_file_dic:
        sound, sr = librosa.load(audio_path/ file_name.with_suffix('.wav'), sr = None)
        table = pd.read_csv(label_path / file_name.with_suffix('.data'), sep = ',')
        for i in table.index:
            # get one second before the positive sound
            start_frame = (table['Start'][i]-1)* sr
            end_frame = table['Start'][i] *sr
            gibbon_neg_lib.append(sound[start_frame: end_frame])
    pickle.dump(gibbon_neg_lib, open(save_path, 'wb'))
    del gibbon_neg_lib
    del sound
    gc.collect()




################################################################################
def butter_bandpass_filter(data, lowcut, highcut, sr, order=8):
   
    def butter_bandpass(lowcut, highcut, sr, order=8):
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], 'bandpass')
        return b, a

    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    y = filtfilt(b, a, data)
    return y


def band_filter_gibbon(Gibbon_lib: typing.List[np.ndarray], lowcut: int, highcut: int, sr: int, order: int) -> typing.List[np.ndarray]: 
    gibbon_lib = copy.deepcopy(Gibbon_lib)
    for i, gibbon_call in enumerate(gibbon_lib):
        gibbon_lib[i] = butter_bandpass_filter(gibbon_call, lowcut, highcut, sr, order)
    # pickle.dump(gibbon_lib, open(save_dic/ Path(str(lowcut) + '_' + str(highcut)+ '_gibbon.p'), 'wb'))
    return gibbon_lib



def cal_sig_strength(gibbon_lib: typing.List[np.ndarray], str_type: str)-> np.ndarray:
    '''str_type: max_str or rms_str'''
    sig_str = []
    if str_type == 'max_str':
        for gibbon_call in gibbon_lib:
            sig_str.append(max(abs(gibbon_call)))
    elif str_type == 'rms_str':
        for gibbon_call in gibbon_lib:
            sig_str.append(np.sqrt(np.mean(np.square(gibbon_call))))
    else:
        print('str_type must be max_str or rms_str')
        sys.exit()
    sig_str = np.asarray(sig_str)
    str_db = 20* np.log10(sig_str)
    return str_db





#####################################################################
#load file dictionary
file_dic = []
for file_name in Path('../../label/processed_label').glob('*.data'):
    file_dic.append(file_name.stem)

file_dic.sort()

test_set = []
K_FOLD = 4
for i in range(K_FOLD):
    train_test_split = dataset.cross_valid(seed = 42, split_number = K_FOLD, fold_needed = i, file_dic = file_dic, 
                                save_path = '../../label/cross_val_label_dic', overwritten = False, verbose = False)
    test_set.extend(train_test_split['test'])

ani_file_dic = []
for file_name in test_set:
    ani_file_dic.append(Path(file_name))


label_path = Path('../../label/processed_label')
audio_path = Path('../../data/raw_data')
save_path_gibbon = Path('../data/gibbon_library.p')
save_path_neg_gibbon = Path('../data/gibbon_neg_library.p')

get_animal_sound(ani_file_dic, label_path = label_path, audio_path = audio_path, save_path = save_path_gibbon)
get_negative_correspond(ani_file_dic, label_path = label_path, audio_path = audio_path, save_path = save_path_neg_gibbon)


###############################################################################
gibbon_lib = pickle.load(open('../data/gibbon_library.p', 'rb'))
gibbon_neg_lib = pickle.load(open('../data/gibbon_neg_library.p', 'rb'))

gibbon1113 = band_filter_gibbon(gibbon_lib, 1100, 1300, 9600, 8)
sig_max_origin = cal_sig_strength(gibbon1113, 'max_str')
sig_rms_origin = cal_sig_strength(gibbon1113, 'rms_str')

neg_1113 = band_filter_gibbon(gibbon_neg_lib, 1100, 1300, 9600, 8)
sig_max_neg = cal_sig_strength(neg_1113, 'max_str')
sig_rms_neg = cal_sig_strength(neg_1113, 'rms_str')


sig_data = {'sig_max_origin': sig_max_origin, 'sig_rms_origin': sig_rms_origin, 'sig_max_neg': sig_max_neg, 'sig_rms_neg': sig_rms_neg}
sig_data = pd.DataFrame(sig_data)
sig_data.to_csv('../result/signal_strength_event.csv', sep = ',', index = False)














































