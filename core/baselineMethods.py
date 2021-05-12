from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import os
import torch
from torch.utils.data import Dataset, DataLoader
from speechbrain.pretrained import SpeakerRecognition
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, SpectralClustering
from sklearn import decomposition
from tqdm.auto import tqdm
from spectralcluster import SpectralClusterer
import shutil

from core.utils import DiarizationDataSet, make_rttm


def diarizationOracleNumSpkrs(audio_dataset, method="KMeans", hypothesis_dir="./rttm_output/"):
    '''
    Compute diarization labels based on oracle number of speakers. Used as an optimal benchmark for performance.
    '''

    try:
        shutil.rmtree(hypothesis_dir)
    except:
        pass

    os.makedirs(hypothesis_dir, exist_ok=True)

    for i in tqdm(range(len(audio_dataset))):
        # Get data sample
        audio_segments, diarization_segments, speech_segments, rttm_path = audio_dataset[i]

        # extract indexes where vad labelles the audio as speech signal
        speech_idx = np.argwhere(speech_segments==1).reshape(-1)

        # Data centering
        Xt = audio_segments[speech_idx].detach().cpu().numpy()
        X = Xt - Xt.mean(axis=0)
        X = X/X.std(axis=0)

        # Extract 'k' principle components. K = is the oracle number of speakers
        pca = decomposition.PCA(n_components=diarization_segments.shape[1])
        pca.fit(X)
        X_pca = pca.transform(X)

        if method=="Spectral":
            clustering = SpectralClustering(n_clusters=diarization_segments.shape[1],
                                            assign_labels="discretize",
                                            random_state=0)
        elif method=="KMeans":
            clustering = KMeans(n_clusters=diarization_segments.shape[1],
                                init="k-means++",
                                max_iter=300,
                                random_state=0)

        # Applying cluster labels
        plabels = clustering.fit_predict(X_pca)

        # assign "-1" to non speech regions and cluster labels to speech regions
        diarization_prediction = np.zeros(diarization_segments.shape[0]+1)-1
        diarization_prediction[:-1][speech_idx] = plabels.copy()

        # Create RTTM file to compute DER with original diarization result
        name = rttm_path.split(sep="/")[-1][:-5]
        rttm_path_h = make_rttm(hypothesis_dir, name, diarization_prediction, audio_dataset.win_step)

    return hypothesis_dir

def diarizationEigenGapNumSpkrs(audio_dataset, hypothesis_dir="./rttm_output/"):
    '''
    Compute diarization labels based on oracle number of speakers. Used as an optimal benchmark for performance.
    '''

    try:
        shutil.rmtree(hypothesis_dir)
    except:
        pass

    os.makedirs(hypothesis_dir, exist_ok=True)

    # Parameters are tuned to achieve good eigen-gap based prediction for number of speakers
    clusterer = SpectralClusterer(min_clusters=1,
                            max_clusters=100,
                            p_percentile=0.9,
                            gaussian_blur_sigma=2,
                            stop_eigenvalue=1e-2)

    for i in tqdm(range(len(audio_dataset))):
        # Get data sample
        audio_segments, diarization_segments, speech_segments, rttm_path = audio_dataset[i]

        # extract indexes where vad labelles the audio as speech signal
        speech_idx = np.argwhere(speech_segments==1).reshape(-1)

        # Data centering
        Xt = audio_segments[speech_idx].detach().cpu().numpy()
        X = Xt - Xt.mean(axis=0)
        X = X/X.std(axis=0)

        # Applying cluster labels
        plabels = clusterer.predict(X)

        # assign "-1" to non speech regions and cluster labels to speech regions
        diarization_prediction = np.zeros(diarization_segments.shape[0]+1)-1
        diarization_prediction[:-1][speech_idx] = plabels.copy()

        # Create RTTM file to compute DER with original diarization result
        name = rttm_path.split(sep="/")[-1][:-5]
        rttm_path_h = make_rttm(hypothesis_dir, name, diarization_prediction, audio_dataset.win_step)

    return hypothesis_dir
