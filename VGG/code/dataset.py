import pandas as pd
import numpy as np
import os
import pickle
import librosa
import scipy
import time
from skimage import io, transform
import imageio


from pathlib import Path
import typing
import abc
import torch
from functools import lru_cache
import inspect

from joblib import Parallel, delayed

class Hparams:

    audio_len: int = 28800
    seq_len: int = 400
    train_hop_len: int = 200
    test_hop_len: int = 400
    sample_rate: int = 9600

    stretch_rate: float = 1 
    shift_step: float = 0 
    crop_rate: float = 1
    volume_change: float = 0


    raw_data_path: str =  '../../data/raw_data'
    processed_label_path: str =  '../../label/processed_label'
    data_save_dic: str = '../data'
    overwritten: bool = True
    n_jobs: int = 6

    fft_seq_len = 512
    fft_hop_len = 256
    n_mels = 32 #different from the other crnn with 36 mel banks
    f_min = 1000
    f_max = 2000
    visulise = False

    seed = 42

    

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    #make parameter class iterable
    def __iter__(self):
        def f(obj):
            return {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("__") and not inspect.isfunction(v)
            }
        #vars()返回类属性的__dict__, 而实例化self后， __class__仍然存在实例化前的dict（全局变量），而self本身的__dict__属性值包括__init__时传入对象
        return iter({**f(self.__class__), **f(self)}.items())



class Transforms:
    @abc.abstractmethod
    def __call__(self,data):
        """Add diferent augment to audio data"""
        raise NonImplementedError




class SpecImg_transform(Transforms):
    def __init__(self, cfg):
        self.cfg = cfg
    
    def generate_spectrogram(self, audio):
        fft_seq_len = self.cfg.fft_seq_len
        fft_hop_len = self.cfg.fft_hop_len
        n_mels = self.cfg.n_mels
        f_min = self.cfg.f_min
        f_max = self.cfg.f_max
        sr = self.cfg.sample_rate
        seq_len = self.cfg.seq_len


        mel_spec = librosa.feature.melspectrogram(audio, n_fft=fft_seq_len, hop_length=fft_hop_len, n_mels=n_mels, sr= sr, power=1.0, fmin = f_min, fmax=f_max)
        log_mel_spec = librosa.amplitude_to_db(mel_spec)      
        log_mel_spec = transform.resize(log_mel_spec, (n_mels, n_mels* seq_len)) ####resize from original size to (number of mel banks, number of mel banks * sequence length)(32,76) to (32,64)
        log_mel_spec = (log_mel_spec- log_mel_spec.min())/(log_mel_spec.max() - log_mel_spec.min())
        return log_mel_spec
    

    def __call__(self, audios: np.ndarray, data_name: Path, dataset_usage: str)-> np.ndarray:

        data_save_dic = self.cfg.data_save_dic
        overwritten = self.cfg.overwritten
        n_mels = self.cfg.n_mels
        seq_len = self.cfg.seq_len
        n_jobs = self.cfg.n_jobs
        visulise = self.cfg.visulise


        # different hop length and file dic for training(validation) data and testing data
        if dataset_usage == 'overlap_train':
            save_dic = Path('overlap_data/img_data')
        elif dataset_usage == 'nonoverlap_pred':
            save_dic = Path('nonoverlap_data/img_data')
        else:
            print('data usage error')

        # check whether the dic been already produced 
        total_save_dic = Path(data_save_dic) / save_dic
        if not total_save_dic.exists():
            Path.mkdir(total_save_dic, parents = True)

        # check whether the file been already produced 
        total_save_path = total_save_dic / data_name.with_suffix('.p')
        if total_save_path.exists() and not overwritten:
            # print('load img data')
            chunk_img = pickle.load(open(total_save_path, 'rb'))
            assert chunk_img.shape[1] == n_mels, chunk_img.shape[1]
            assert chunk_img.shape[2] == n_mels * seq_len, chunk_img.shape[2]
        else:
            # # print('generate new img data')
            # chunk_img = []
            # for audio in audios:
            #     mel_spec = librosa.feature.melspectrogram(audio, n_fft=fft_seq_len, hop_length=fft_hop_len, n_mels=n_mels, sr= sr, power=1.0, fmin = f_min, fmax=f_max)
            #     log_mel_spec = librosa.amplitude_to_db(mel_spec)      
            #     log_mel_spec = transform.resize(log_mel_spec, (n_mels, n_mels* seq_len)) ####resize from original size to (number of mel banks, number of mel banks * sequence length)(32,76) to (32,64)
            #     log_mel_spec = (log_mel_spec- log_mel_spec.min())/(log_mel_spec.max() - log_mel_spec.min())
            #     # print(log_mel_spec.shape)
            #     chunk_img.append(log_mel_spec)
            chunk_img = Parallel(n_jobs= n_jobs)(delayed(self.generate_spectrogram)(audio) for audio in audios)
            chunk_img = np.asarray(chunk_img)
            pickle.dump(chunk_img, open(total_save_path, 'wb'))
        if visulise:
            visulise_dic = Path('visulise') / data_name.stem
            total_visulise_dic = total_save_dic / visulise_dic 
            if not total_visulise_dic.exists():
                Path.mkdir(total_visulise_dic, parents= True)
            for i in range(chunk_img.shape[0]):
                imageio.imwrite(total_visulise_dic / Path('%d.png'%i),chunk_img[i])
        return chunk_img




class Pitch_shift(Transforms):

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, audios: np.ndarray, data_name: Path, dataset_usage: str):

        data_save_dic = self.cfg.data_save_dic
        overwritten = self.cfg.overwritten
        sr = self.cfg.sample_rate
        shift_step = self.cfg.shift_step
        n_jobs = self.cfg.n_jobs

        # different file dic for training(validation) data and testing data
        if dataset_usage == ('overlap_train'):
            save_dic = Path('overlap_data/pitch_shift_sound')
        elif dataset_usage == 'nonoverlap_pred':
            save_dic = Path('nonoverlap_data/pitch_shift_sound')
        else:
            print('data usage error')

        # check whether the dic been already produced 
        total_save_dic = Path(data_save_dic) / save_dic
        if not total_save_dic.exists():
            Path.mkdir(total_save_dic, parents = True)

        # check whether the file been already produced 
        total_save_path = total_save_dic / data_name.with_suffix('.p')
        if total_save_path.exists() and not overwritten:
            shift_sounds = pickle.load(open(total_save_path, 'rb'))
        else:
            # shift_sounds = []
            # for audio in audios:
            #     shift_audio = librosa.effects.pitch_shift(audio, sr, shift_step)
            #     shift_sounds.append(shift_audio)
            shift_sounds = Parallel(n_jobs = n_jobs)(delayed(librosa.effects.pitch_shift)(audio, sr, shift_step) for audio in audios)
            shift_sounds = np.asarray(shift_sounds)
            pickle.dump(shift_sounds, open(total_save_path, 'wb'))
        return shift_sounds



class Time_stretch(Transforms):
    #time stretching in crnn input level, not in every segment level

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, audios: np.ndarray, data_name: Path, dataset_usage: str):

        data_save_dic = self.cfg.data_save_dic
        overwritten = self.cfg.overwritten
        stretch_rate = self.cfg.stretch_rate
        n_jobs = self.cfg.n_jobs

        # different file dic for training(validation) data and testing data
        if dataset_usage == ('overlap_train'):
            save_dic = Path('overlap_data/time_stretch_sound')
        elif dataset_usage == 'nonoverlap_pred':
            save_dic = Path('nonoverlap_data/time_stretch_sound')
        else:
            print('data usage error')

        # check whether the dic been already produced 
        total_save_dic = Path(data_save_dic) / save_dic
        if not total_save_dic.exists():
            Path.mkdir(total_save_dic, parents = True)

        # check whether the file been already produced 
        total_save_path = total_save_dic / data_name.with_suffix('.p')
        if total_save_path.exists() and not overwritten:
            stretch_sounds = pickle.load(open(total_save_path, 'rb'))
        else:
            # stretch_sounds = []
            # for audio in audios:
            #     stretch_audio = librosa.effects.time_stretch(audio, stretch_rate)
            #     stretch_sounds.append(stretch_audio)
            stretch_sounds = Parallel(n_jobs = n_jobs)(delayed(librosa.effects.time_stretch)(audio, stretch_rate) for audio in audios)
            stretch_sounds = np.asarray(stretch_sounds)
            pickle.dump(stretch_sounds, open(total_save_path, 'wb'))
        return stretch_sounds




# class Volume_changing(Transforms):
# # 不会对结果产生影响 由于标准化

#     def __init__(self, cfg):
#         self.cfg = cfg

#     def __call__(self, audios: np.ndarray, data_name: Path, dataset_usage: str):

#         data_save_dic = self.cfg.data_save_dic
#         overwritten = self.cfg.overwritten
#         vol_change = self.cfg.volume_change

#         # different file dic for training(validation) data and testing data
#         if dataset_usage == ('overlap_train'):
#             save_dic = Path('overlap_data/vol_change_sound')
#         elif dataset_usage == 'nonoverlap_pred':
#             save_dic = Path('nonoverlap_data/vol_change_sound')
#         else:
#             print('data usage error')

#         # check whether the dic been already produced 
#         total_save_dic = Path(data_save_dic) / save_dic
#         if not total_save_dic.exists():
#             Path.mkdir(total_save_dic, parents = True)

#         # check whether the file been already produced 
#         total_save_path = total_save_dic / data_name.with_suffix('.p')
#         if total_save_path.exists() and not overwritten:
#             vol_change_sounds = pickle.load(open(total_save_path, 'rb'))
#         else:
#             vol_change_sounds = audios * np.power(10, vol_change / 20)
#             pickle.dump(vol_change_sounds, open(total_save_path, 'wb'))
#         return vol_change_sounds



class Cropping(Transforms):

    def __init__(self, cfg):
        self.cfg = cfg

    def crop_seq(self, audio, start_position, end_position):
        sr = self.cfg.sample_rate
        seq_len = self.cfg.seq_len

        audio_reshape = audio.reshape((seq_len, sr))
        crop_audio = audio_reshape[:, start_position: end_position]
        audio_reverse = crop_audio.reshape(-1)
        return audio_reverse


    def __call__(self, audios: np.ndarray, data_name: Path, dataset_usage: str):

        data_save_dic = self.cfg.data_save_dic
        overwritten = self.cfg.overwritten
        crop_rate = self.cfg.crop_rate
        sr = self.cfg.sample_rate
        seq_len = self.cfg.seq_len
        seed = self.cfg.seed
        n_jobs = self.cfg.n_jobs
        np.random.seed(seed)

        seq_frames = sr * seq_len
        assert seq_frames == audios.shape[1], audios.shape[1]

        crop_length = int(np.ceil(sr * crop_rate))
        start_position = np.random.randint(sr - crop_length + 1)
        end_position = start_position + crop_length
        new_seq_frames = crop_length * seq_len

        # different file dic for training(validation) data and testing data
        if dataset_usage == ('overlap_train'):
            save_dic = Path('overlap_data/crop_sound')
        elif dataset_usage == 'nonoverlap_pred':
            save_dic = Path('nonoverlap_data/crop_sound')
        else:
            print('data usage error')

        # check whether the dic been already produced 
        total_save_dic = Path(data_save_dic) / save_dic
        if not total_save_dic.exists():
            Path.mkdir(total_save_dic, parents = True)

        # check whether the file been already produced 
        total_save_path = total_save_dic / data_name.with_suffix('.p')
        if total_save_path.exists() and not overwritten:
            crop_sounds = pickle.load(open(total_save_path, 'rb'))        
        else:
            crop_sounds = Parallel(n_jobs = n_jobs)(delayed(self.crop_seq)(audio, start_position, end_position) for audio in audios)
            crop_sounds = np.asarray(crop_sounds)
            pickle.dump(crop_sounds, open(total_save_path, 'wb'))
        assert crop_sounds.shape[1] == new_seq_frames, crop_sounds.shape[1]
        return crop_sounds




class Pytorch_data_transform(Transforms):
    def __call__(self, np_data: np.ndarray) -> torch.float32:
        tf_data = torch.from_numpy(np_data).unsqueeze(1)
        return tf_data


class Pytorch_label_transform(Transforms):
    def __call__(self, np_label: np.ndarray) -> torch.float32:
        tf_label = torch.from_numpy(np_label).type(torch.FloatTensor)
        return tf_label






class gibbon_dataset(torch.utils.data.Dataset):
    '''dataset_usage: 'overlap_train' or 'nonoverlap_pred' '''
    def __init__(
        self, 
        cfg, 
        dataset_type: str = 'test',
        dataset_usage: str = 'overlap_train',
        domain_transform: Transforms = None,
        augment: Transforms = None,
        pytorch_X_transform: Transforms = None, 
        pytorch_Y_transform: Transforms = None, 
        train_test_split: dict = None):

        self.cfg = cfg
        self.dataset_type = dataset_type
        self.dataset_usage = dataset_usage
        self.domain_transform = domain_transform
        self.augment = augment
        self.p_x_t = pytorch_X_transform
        self.p_y_t = pytorch_Y_transform
        self.used_set = train_test_split[dataset_type]
        self.X_batch_data, self.Y_batch_label = self.load_data()




    
    def load_data(self):
        raw_data_path = Path(self.cfg.raw_data_path)
        processed_label_path = Path(self.cfg.processed_label_path)

        X_batch_data = []
        Y_batch_label = []
        for i in self.used_set:
            #split audio into chunks
            audio, sample_rate = librosa.load(raw_data_path / Path(i).with_suffix('.wav'), sr = None)
            # check whether sample rate correct
            assert self.cfg.sample_rate == sample_rate, sample_rate
            X_data = self.extract_all_chunks(audio, Path(i))
            #argument data
            if self.augment:
                X_data = self.augment(audios = X_data, data_name = Path(i), dataset_usage = self.dataset_usage)
            #change sound to image
            if self.domain_transform:
                X_data = self.domain_transform(audios = X_data, data_name= Path(i), dataset_usage= self.dataset_usage) # [number_of_img * 32 * (32*seq_len)]
            #change table to on hot label
            table = pd.read_csv(processed_label_path / Path(i).with_suffix('.data'), sep= ',')
            Y_label = self.table_to_labels(table)
            #add data and label into batch
            X_batch_data.extend(X_data)
            Y_batch_label.extend(Y_label)
        X_batch_data = np.asarray(X_batch_data)
        Y_batch_label = np.asarray(Y_batch_label)
        
        #give the number of total segment data points
        self.segment_len = Y_batch_label.size
        # transfer data for neural network usage
        if self.p_x_t:
            X_batch_data = self.p_x_t(X_batch_data)
        if self.p_y_t:
            Y_batch_label = self.p_y_t(Y_batch_label)   
        # print(X_batch_data.shape)   
        # print(Y_batch_label.shape) 
        assert X_batch_data.shape[0] == Y_batch_label.shape[0]
        #check the number of x_data batch comparaing to __len__()
        self.data_len = X_batch_data.shape[0]
        return X_batch_data, Y_batch_label



    def __getitem__(self, index):
        data = self.X_batch_data[index]
        label = self.Y_batch_label[index]
        return data, label


    def __len__(self):
        audio_len = self.cfg.audio_len
        seq_len = self.cfg.seq_len

        if self.dataset_usage == 'overlap_train':
            hop_len = self.cfg.train_hop_len
        elif self.dataset_usage == 'nonoverlap_pred':
            hop_len = self.cfg.test_hop_len
        else:
            print('data usage error')
        return len(self.used_set) * np.floor(((audio_len - seq_len) / hop_len) + 1).astype('int')



    def extract_all_chunks(self, data: np.ndarray, data_name: Path):
        """Cut audio into segments based on every seq length and hop length"""
        audio_len = self.cfg.audio_len
        data_save_dic = self.cfg.data_save_dic
        overwritten = self.cfg.overwritten
        seq_len = self.cfg.seq_len
        sr = self.cfg.sample_rate


        #check whether data length corrects
        total_frames = data.shape[0]
        assert total_frames == audio_len * sr, total_frames
    
        # different hop length and file dic for training(validation) data and testing data
        if self.dataset_usage == 'overlap_train':
            hop_len = self.cfg.train_hop_len
            save_dic = Path('overlap_data/sound_data')
        elif self.dataset_usage == 'nonoverlap_pred':
            hop_len = self.cfg.test_hop_len
            save_dic = Path('nonoverlap_data/sound_data')
        else:
            print('data usage error')

        # check whether the dic been already produced 
        total_save_dic = Path(data_save_dic) / save_dic
        if not total_save_dic.exists():
            Path.mkdir(total_save_dic, parents = True)

        # check whether the file been already produced 
        total_save_path = total_save_dic / data_name.with_suffix('.p')
        if total_save_path.exists() and not overwritten:
            # print('load sound data')
            chunk_extracted = pickle.load(open(total_save_path, 'rb'))
            assert chunk_extracted.shape[1] == seq_len * sr, chunk_extracted.shape[1]
        else:
            # print('generate new sound data')
            seq_frames = seq_len * sr
            #add chunks into chunk_extracted
            chunk_extracted = []
            jump = 0
            while True:
                start_position = (jump * hop_len * sr)
                end_position = start_position + seq_frames
                jump = jump + 1
                if end_position > total_frames:
                    break         
                # Append the audio data
                chunk_extracted.append(data[int(start_position):int(end_position)])
            chunk_extracted = np.asarray(chunk_extracted)
            pickle.dump(chunk_extracted, open(total_save_path, 'wb'))
            
        return chunk_extracted #(#chunks, seq_len)



    def table_to_labels(self, labels_file: pd.DataFrame):
        """creating one hot label by seconds"""
        audio_len = self.cfg.audio_len
        seq_len = self.cfg.seq_len

        # different hop length and training(validation) label and testing label
        if self.dataset_usage == 'overlap_train':
            hop_len = self.cfg.train_hop_len
        elif self.dataset_usage == 'nonoverlap_pred':
            hop_len = self.cfg.test_hop_len
        else:
            print('data usage error')
        
        labels = np.zeros(int(audio_len))
        for i in labels_file.index:
            start_time = labels_file['Start'][i]
            end_time = labels_file['End'][i]
            # print("start time ", start_time)
            # print("end time ", end_time)
            labels[start_time : end_time] = 1

        #change file label into batch label
        batch_label = []
        j = 0
        while True:
            batch_label.append(labels[j:(j+ seq_len)])
            j += hop_len
            if j + seq_len > audio_len:
                break
        batch_label = np.asarray(batch_label)
        return batch_label





############################################################



#making cross validation label file name set for further use
from sklearn.model_selection import KFold
import pickle
import numpy as np
import random
from pathlib import Path





def cross_valid(seed, split_number, fold_needed, file_dic: list, save_path: Path, overwritten: bool = False, verbose: bool = False):
    file_dic = np.asarray(file_dic)
    random.seed(seed)
    random.shuffle(file_dic)
    count = 0
    kf = KFold(n_splits = split_number)
    for train_index, test_index in kf.split(file_dic):
        train_test_split = {}
        train_val_set, test_set = file_dic[train_index], file_dic[test_index]

        random.shuffle(train_val_set)
        train_set = train_val_set[:-7]
        val_set = train_val_set[-7:]
        train_test_split['train'] = train_set
        train_test_split['valid'] = val_set
        train_test_split['test'] = test_set
        if verbose:
            print(train_test_split)

        total_save_path = save_path / Path('train_test_split%d.p'%count)
        if not overwritten and total_save_path.exists():
            if verbose:
                print('Already exist:', total_save_path)
        else:
            pickle.dump(train_test_split, open(total_save_path, 'wb'))
        count += 1
               
    #load required train test split         
    train_test_split_need = pickle.load(open(save_path / Path('train_test_split%d.p'%fold_needed), 'rb'))
    return train_test_split_need








##########################################################################################

# #test the code above

# file_dic = []
# for file_name in Path('../../label/processed_label').glob('*.data'):
#     file_dic.append(file_name.stem)
# #之后把这个给uncommon掉
# # file_dic.sort()


# train_test_split0 = cross_valid(seed = 42, split_number = 4, fold_needed = 0, file_dic = file_dic, 
#                                 save_path = Path('../../label/cross_val_label_dic'), overwritten = False, verbose = False)



# p = Hparams(overwritten = True)

# data_set = gibbon_dataset(cfg = p, 
#         dataset_type =  'test',
#         dataset_usage= 'nonoverlap_pred',
#         domain_transform = SpecImg_transform(p),
#         augment = None,
#         pytorch_X_transform = Pytorch_data_transform(), 
#         pytorch_Y_transform = Pytorch_label_transform(), 
#         train_test_split = train_test_split0)

# data_set = gibbon_dataset(cfg = p, 
#         dataset_type =  'train',
#         dataset_usage= 'overlap_train',
#         domain_transform = SpecImg_transform(p),
#         augment = None,
#         pytorch_X_transform = Pytorch_data_transform(), 
#         pytorch_Y_transform = Pytorch_label_transform(), 
#         train_test_split = train_test_split0)



# data_set = gibbon_dataset(cfg = p, 
#         dataset_type =  'valid',
#         dataset_usage= 'nonoverlap_pred',
#         domain_transform = SpecImg_transform(p),
#         augment = None,
#         pytorch_X_transform = Pytorch_data_transform(), 
#         pytorch_Y_transform = Pytorch_label_transform(), 
#         train_test_split = train_test_split0)


