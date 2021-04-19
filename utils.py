from __future__ import print_function, division

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import subprocess
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from speechbrain.pretrained import SpeakerRecognition
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment, notebook

torchaudio.set_audio_backend("soundfile")

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module to load preocmputed xvectors using ECAP-TDNN on Voxconverse Dataset
VoxConverse_Xvectors_Precomputed = {}
VoxConverse_Xvectors_Precomputed[250, 1500] = "https://www.dropbox.com/s/3rfyz2fxm8ic9cu/VoxConverse_Xvectors_250_1500.zip?dl=1"
VoxConverse_Xvectors_Precomputed[750, 1500] = "https://www.dropbox.com/s/yic3oeq7qd5rnpf/VoxConverse_Xvectors_750_1500.zip?dl=1"
VoxConverse_Xvectors_Precomputed[500, 1000] = "https://www.dropbox.com/s/2w93hho2ixzl2je/VoxConverse_Xvectors_500_1000.zip?dl=1"
VoxConverse_Xvectors_Precomputed[25, 1500] = "https://www.dropbox.com/s/qj6dm1io9tfgm0j/VoxConverse_Xvectors_25_1500.zip?dl=1"

def downloadZipAndExtract(url, save_dir):
    subprocess.check_output(["wget", "-O", "tmp.zip", url])
    subprocess.check_output(["unzip", "-o", "tmp.zip", "-d", save_dir])
    subprocess.check_output(["rm", "tmp.zip"])
    
    print("Download and Extraction Complete")
    

def read_audio(path: str,
               target_sr: int = 16000):

    assert torchaudio.get_audio_backend() == 'soundfile'
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=target_sr)
        wav = transform(wav)
        sr = target_sr

    assert sr == target_sr
    return wav.squeeze(0)
    
# Creating test and train splits for VoxConverse dataset...
VoxConverseSplits = {}
VoxConverseSplits['test'] = [173, 155,  37,  54, 198, 119, 204,  53,   4,  60, 181, 158,  13,
                              61,  80, 176,  14,   8, 130, 166,  31, 163, 133,  75,   2,  19,
                              12, 107,  22,  88,  69, 150,  59, 135,  40,  35, 190, 134,  81,
                              48,  45, 142, 103,  30,  24, 194, 172, 179,  87, 215]

VoxConverseSplits['train'] = [ 15,  76,  94,  96, 162, 211,  68,  56, 212,  84,  46,  93, 118,
                                148,   0,  21, 151, 209, 156, 113, 120,  66, 203, 138, 121, 132,
                                16, 145, 193, 175,  50,  38,  27, 109,  71,  92, 192, 157, 116,
                                18, 199, 104, 122, 105, 112,  85,  20, 123, 152,  36,  78, 139,
                                74,  32, 129, 174, 165,  26, 106, 108, 186, 154, 127,  33, 164,
                                183, 187, 197,  34,   9, 136, 141, 178, 101, 205, 159,  65,   6,
                                57, 170, 177, 100, 206,  98,  47, 143,   1, 210, 202, 146, 189,
                                117,  29,  25,   5,  39, 131, 144,  42,  89,  67,  64, 111, 114,
                                140, 149,  51,  55,   3, 169,  17,  95, 213,  91, 208, 171, 102,
                                191,  79,  10,   7,  11, 167,  41,  83,  73, 188, 126, 201,  49,
                                184,  23, 207, 200,  70,  99, 180, 128,  77, 115,  86,  58, 161,
                                153, 147,  62, 160,  52, 214, 125, 168, 185,  82,  97, 196,  90,
                                43,  63, 124,  28,  44, 110, 195,  72, 182, 137]
                                
VoxConverseSplits['full'] = [*range(216)]
                

# Data pipeline
class DiarizationDataSet(Dataset):
    def __init__(self, 
                 root_dir='./audio/', 
                 label_dir='./voxconverse/dev/',
                 xvectors_dir=None,
                 vad_dir=None,
                 sr=16000, 
                 window_len=240, 
                 window_step=120, 
                 transform=None,
                 batch_size_for_ecapa=512,
                 vad_step=4,
                 split='full'):
        
        """
        Args:
        - root_dir (string): Local directory of the audio files
        - audioFilelist (string): txt file with audio file list
        - label_dir (string): Local directory of the rttm label files
        - sr (int): Sample rate for audio signal, default 16kHz
        - window_len (int): Length of each segment of audio signal in milliseconds
        - window_step (int): Length between two window_len in milliseconds
        - mel_transform (callable, optional): Parameters of mel transform. None signifies no transform
        - batch_size_for_ecapa (int): Size of batches used while applying pretrained speechbrain ECAPA model

        """

        self.root_dir = root_dir
        self.filelist = np.array(sorted(os.listdir(root_dir)))[VoxConverseSplits[split]]
        self.split = split
        self.label_dir = label_dir
        self.sr = sr
        self.win_len = window_len
        self.win_step = window_step
        self.transform = transform
        self.batch_size_for_ecapa = batch_size_for_ecapa
        self.vad_step = vad_step
        self.vad_dir = vad_dir
        
        if xvectors_dir:
            self.xvectors_dir = xvectors_dir
        elif (window_step, window_len) in list(VoxConverse_Xvectors_Precomputed.keys()):
            print("Precomputed X-vectors exists!\nWill use precomputed features...")
            print("\nDownloading precomputed features...")
            downloadZipAndExtract(VoxConverse_Xvectors_Precomputed[(window_step, window_len)], "./")
            self.xvectors_dir = "./VoxConverse_Xvectors/"
        else:
            self.xvectors_dir = xvectors_dir
            
        with torch.no_grad():
            # Load ECAPA-TDNN x-vector based pre-trained model on speaker verification task (latest x-vector system)
            # https://arxiv.org/pdf/2005.07143.pdf
            if self.xvectors_dir == None:
                self.ECAPA = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
            
            # Load VAD Model
            # https://github.com/snakers4/silero-vad
            if self.vad_dir == None:
                self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=True)
                (self.get_speech_ts, _, _, _, _, _) = utils

    def __len__(self):
        return len(self.filelist)
  
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = os.path.join(self.root_dir, self.filelist[idx])
        label_path = os.path.join(self.label_dir, self.filelist[idx][:-4]+'.rttm')

        # Torch array of audio signal
        audio = read_audio(audio_path, target_sr=self.sr)

        if self.transform:
            audio = self.transform(audio.detach().cpu().numpy())
        
        # Window len and Window step in frames
        win_len = self.win_len*(self.sr//1000)
        win_step = self.win_step*(self.sr//1000)
        
        if self.xvectors_dir != None:
            audio_segment_path = os.path.join(self.xvectors_dir, self.filelist[idx][:-4]+'.npy')
            audio_segments = torch.Tensor(np.load(audio_segment_path))
        else:
            # Pad and create audio segments
            audio_vec = audio.reshape(1, audio.shape[0])
            audio_vec = F.pad(input=audio_vec, pad=(win_len//2, win_len//2, 0, 0), mode='constant', value=0)
    
            audio_segments = []
            for i in range(win_len//2, audio_vec.shape[1]-win_len//2, win_step):
                audio_segments.append(audio_vec[:, i-win_len//2:i+win_len//2])
    
            audio_segments = torch.vstack(audio_segments)
            
            # Compute ECAPA-TDNN x-vectors for the audio signal
            with torch.no_grad():
                Xt = []
                for i in range(audio_segments.shape[0]//self.batch_size_for_ecapa):
                    Xt.append(self.ECAPA.encode_batch(audio_segments[i*self.batch_size_for_ecapa:(i+1)*self.batch_size_for_ecapa])[:,0,:])
        
                if audio_segments.shape[0]%self.batch_size_for_ecapa != 0:
                    Xt.append(self.ECAPA.encode_batch(audio_segments[(audio_segments.shape[0]//self.batch_size_for_ecapa)*self.batch_size_for_ecapa:])[:,0,:])
        
                audio_segments = torch.vstack(Xt)
            
        NumWin = len(audio_segments)

        # VAD timestamps
        if self.vad_dir != None:
            vad_path = os.path.join(self.vad_dir, self.filelist[idx][:-4]+'.npy')
            speech_timestamps = np.load(vad_path, allow_pickle=True)
        else:
            speech_timestamps = self.get_speech_ts(audio, self.vad_model, num_steps=self.vad_step)
            
        speech_segments = torch.zeros(NumWin)
        for i in speech_timestamps:
            speech_segments[i['start']//win_step:i['end']//win_step] = 1

        # Load diarization labels from RTTM file
        labels_data = self.read_rttm(label_path)
        numspks = np.unique(labels_data[:, 2]).shape[0]

        diarization_segments = torch.zeros((NumWin, numspks))
        for i in labels_data:
            diarization_segments[i[0]//win_step:i[1]//win_step, i[2]] = 1

        return audio_segments, diarization_segments, speech_segments, label_path

    def read_rttm(self, path):
        '''
        Read RTTM Diarization file
        '''

        rttm_out = [] # returns list of list containing start frame, end frame, spkid
        with open(path, "r") as f:
            for line in f:
                entry = line[:-1].split()
                indexes = [0, 1, 2, 5, 6, 8, 9]
                for index in sorted(indexes, reverse=True):
                    del entry[index]

                entry[0] = int(float(entry[0])*self.sr) # Start frame
                entry[1] = entry[0] + int(float(entry[1])*self.sr) # End frame
                entry[2] = int(entry[2][-2:]) # Label
                rttm_out.append(entry)
                
            # Sort rttm list according to start frame
            rttm_out.sort(key = lambda x: x[0])

        return np.array(rttm_out)

def make_rttm(out_dir, name, labels, win_step):
    '''
    Create RTTM Diarization files for non-overlapping speaker labels in var 'labels'.
    - win_step -- in ms
    - out_dir -- location where to save rttm file
    - name -- name of the rttm file

    Returns:
    path where the rttm file is saved.
    '''

    rttm_out = []
    flag = 1
    for i in range(len(labels)):
        if flag==1:
            if labels[i]>-1:
                start_i = i
                end_i = i
                flag = 0
        else:
            if labels[i] == labels[end_i]:
                end_i = i
            else:
                start = max(0, (start_i-0.5)*win_step/1000.0)
                end = (end_i+0.5)*win_step/1000.0
                label = labels[start_i]

                rttm_out.append(['SPEAKER', name, '1', str(start), str(end-start), '<NA>', '<NA>', 'spk' + str(int(label)), '<NA>', '<NA>'])

                if labels[i] == -1:
                    flag = 1
                else:
                    start_i = i
                    end_i = i

    np.savetxt(out_dir + name + ".rttm", rttm_out, fmt='%s')

    return out_dir + name + ".rttm"
    
def get_metrics(groundtruth_path, hypothesis_path):
    '''
    groundtruth_path = directory of groundtruth rttm files
    hypothesis_path = directory of hypothesis rttm files
    '''
    metric = DiarizationErrorRate()
    
    gt_filepath = sorted(os.listdir(path = groundtruth_path))
    hp_filepath = sorted(os.listdir(path = hypothesis_path))
    
    for filename in tqdm(hp_filepath):
        groundtruth = load_rttm(groundtruth_path + filename)[filename[:-5]]
        hypothesis = load_rttm(hypothesis_path + filename)[filename[:-5]]
        metric(groundtruth, hypothesis)
    
    return metric