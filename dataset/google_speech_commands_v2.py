import os
import librosa
from torch.utils.data import Dataset
import numpy as np
import soundfile
from transformers import AutoFeatureExtractor
import torchaudio, torch

class Google_Speech_Commands_v2(Dataset):
    """
    Lazy-loading version of GSC v2 dataset.
    """

    def __init__(self, data_path, max_len_AST, split, apply_SpecAug=False, few_shot=False, samples_per_class=1):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"`split` arg ({split}) must be train/valid/test.")
            
        self.data_path = os.path.expanduser(data_path)
        self.max_len_AST = max_len_AST
        self.split = split
        
        self.apply_SpecAug = apply_SpecAug
        self.freq_mask = 24
        self.time_mask = 80
        
        # Initialize the processor once here so it can be used inside __getitem__
        self.processor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", 
            max_length=self.max_len_AST
        )
        
        # self.x_paths stores ONLY the file strings, avoiding the RAM spike
        self.x_paths, self.y = self.get_data()
        
        if few_shot:
            self.x_paths, self.y = self.get_few_shot_data(samples_per_class)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        # 1. LAZY LOAD: Read the audio file from disk only when requested
        pathh = self.x_paths[index]
        wav, sampling_rate = soundfile.read(pathh)
        
        # 2. Extract features on the fly
        fbank = self.processor(wav, sampling_rate=16000, return_tensors='pt')['input_values'].squeeze(0)
        
        # 3. Apply SpecAugment if required
        if self.apply_SpecAug:
            freqm = torchaudio.transforms.FrequencyMasking(self.freq_mask)
            timem = torchaudio.transforms.TimeMasking(self.time_mask)
            
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)
            fbank = freqm(fbank)
            fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)
            
        return fbank, self.y[index]
    
    def get_few_shot_data(self, samples_per_class: int):
        x_few, y_few = [], []
        
        total_classes = np.unique(self.y)
        
        for class_ in total_classes:
            cap = 0
            for index in range(len(self.y)):
                if self.y[index] == class_:
                    x_few.append(self.x_paths[index])
                    y_few.append(self.y[index])
                    
                    cap += 1
                    if cap == samples_per_class: break
        return x_few, y_few
    
    def get_data(self):
        if self.split == 'valid': 
            list_name = 'validation_list.txt'
        elif self.split == 'test':
            list_name = 'testing_list.txt'
        else: # train needs both lists.
            list_test_name = 'testing_list.txt'
            list_valid_name = 'validation_list.txt'
        
        x_paths, y = [], []
        
        if self.split in ['valid','test']:
            with open(os.path.join(self.data_path, 'speech_commands_v0.02', list_name)) as f:
                lines = f.readlines()
            
            for line in lines:
                pathh = os.path.join(self.data_path, 'speech_commands_v0.02', line.strip())
                
                # Append string path instead of processing
                x_paths.append(pathh)
                y.append(self.class_ids[line.split('/')[0]])
                
            return x_paths, np.array(y)
        
        else:
            # Using sets for O(1) lookup speeds up initialization significantly
            with open(os.path.join(self.data_path, 'speech_commands_v0.02', list_valid_name)) as f:
                lines_valid = set([x.strip() for x in f.readlines()])
            with open(os.path.join(self.data_path, 'speech_commands_v0.02', list_test_name)) as f:
                lines_test = set([x.strip() for x in f.readlines()])
            
            for class_id in self.class_ids:
                list_files = os.listdir(os.path.join(self.data_path, 'speech_commands_v0.02', class_id))
                
                for file_class in list_files:
                    file_class_ = class_id + '/' + file_class
                   
                    if file_class_ in lines_valid or file_class_ in lines_test:
                        continue
                    
                    pathh = os.path.join(self.data_path, 'speech_commands_v0.02', class_id, file_class)
                    
                    # Append string path instead of processing
                    x_paths.append(pathh)
                    y.append(self.class_ids[class_id])
                
            return x_paths, np.array(y)

    @property
    def class_ids(self):
        return {
             'backward': 0, 'bed': 1, 'bird': 2, 'cat': 3, 'dog': 4,
             'down': 5, 'eight': 6, 'five': 7, 'follow': 8, 'forward': 9,
             'four': 10, 'go': 11, 'happy': 12, 'house': 13, 'learn': 14,
             'left': 15, 'marvin': 16, 'nine': 17, 'no': 18, 'off': 19,
             'on': 20, 'one': 21, 'right': 22, 'seven': 23, 'sheila': 24,
             'six': 25, 'stop': 26, 'three': 27, 'tree': 28, 'two': 29,
             'up': 30, 'visual': 31, 'wow': 32, 'yes': 33, 'zero': 34,
        }