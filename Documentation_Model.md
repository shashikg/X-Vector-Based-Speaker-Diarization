# Documentation: EE698R DEC based Diarization Model

This speaker diarization model uses [Deep Embedding Clustering][dec] with a deep neural network initialized via 
a Residual Autoencoder to assign speaker labels to segments of the raw audio signal.
Clustering is perfomed on x-vectors extracted using [Desplanques et al.][desplanques]'s ECAPA-TDNN framework. 
We use [Silero-VAD][vad] for voice audio detection.

**Baseline Model:** Spectral clustering is used for audio-label assignment.

## DataSet
Model is tested on [VoxConverse][voxconverse] dataset (total 216 audio files). We randomly split the dataset into two parts: ‘test’ and ‘train’ with test data having 50 audio files.

## ipynb Notebook Files
- **DEC_ResAE.ipynb:** To evaluate the DER score for the DEC models described in the report. Use the link available in Tutorial section to open it on google colab
- **ExtractVAD.ipynb:** Used to extract and save all the VAD mapping for the audio files in VoxConverse dataset.
- **ExtractXvectors.ipynb:** Used to precompute X-vectors for the audio files in VoxConverse dataset and save it into a zip file to use it in the DiarizationDataset.
- **Baseline.ipynb:** To evaluate the DER score for the baseline models described in the report. Use the link available in the Tutorial section to open it on google colab.

## Tutorial
**DEC Speaker Diarization** \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shashikg/speaker_diarization_ee698/blob/main/DEC_ResAE.ipynb)

**Baseline Speaker Diarization** \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shashikg/speaker_diarization_ee698/blob/main/Baseline.ipynb)

## API Documentation
### Index
- [Defined in: utils.py](#utils.py)
  - [class DiarizationDataset](#diarizationdataset)
    - [\_\_getitem\_\_](#getitem)
    - [read\_rttm](#read_rttm)
  - [func make\_rttm](#make_rttm)
- [Defined in: baselineMethods.py](#baselineMethods.py)
  - [func get\_metrics](#get_metrics)
  - [func diarizationOracleNumSpkrs](#diarizationOracleNumSpkrs)
  - [func diarizationEigenGapNumSpkrs](#diarizationEigenGapNumSpkrs)
- [Defined in: optimumSpeaker.py](#optimumSpeaker.py)
  - [class eigengap](#eigengap)
    - [\_get\_refinement\_operator](#getrefinementoperator)

---
## <a name =  'utils.py'></a> Defined in: utils.py
### <a name = 'diarizationdataset'></a> class DiarizationDataset() 
_Defined in utils.py_
```sh
class DiarizationDataset(root_dir='./audio/', 
                 label_dir='./voxconverse/dev/',
                 xvectors_dir=None,
                 vad_dir=None,
                 sr=16000, 
                 window_len=240, 
                 window_step=120, 
                 transform=None,
                 batch_size_for_ecapa=512,
                 vad_step=4,
                 split='full')
```
Create an abstract class for loading dataset. This class applies the necessary pre-processing and x-vector feature extraction methods to return the audio file as a bunch of segmented x-vector features to use it directly in the clustering algorithm to predict speaker labels. The module uses the pre-computed X-vectors if available otherwise extract it during the runtime.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`root_dir:`                     |  _str_, Directory containing the audio wav files 
`label_dir:`                    |  _str_, Directory containing the rttm label files
`xvectors_dir:`                 |  _str_, Directory containing the precomputed xvectors for audio segments,<br /> default = None
`vad_dir=None:`                 |  _str_, Directory containing the precomputer audio speech detection timestamps, <br /> default = None
`sr:`                           |  _int_, Sampling rate of the audio signal 
`window_len:`                   |  _int_, Window length (in ms) of each of the audio segments to be passed for feature extraction
`window_step:`                  |  _int_, Step (in ms) between two windows of audio segments to be passed for feature extraction
`transform:`                    |  _list_, List of transforms like `mel-transform` to be performed on audio while preprocessing, <br /> default = None
`batch_size_for_ecapa:`         |  _int_, Batch size of audio segments while performing feature extraction using ECAPA-TDNN
`vad_step:`                     |  _int_, Number of windows to split each audio chunk into. Argument used by Silero-VAD module
`split:`                        |  _str_, Argument defining type of split of dataset, <br /> default = 'full' indicates no split

**Class Functions:**

1. <a name = 'getitem'></a> **\_\_getitem\_\_:**
```def __getitem__(self, idx)```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`idx:`                          |  _int_, Index to the required audio in the list of audio in root directory

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`audio_segments:`               |  _torch.Tensor_, (n_windows, features_len) Tensor of feature vectors of each audio segment window
`diarization_segments:`         |  _torch.Tensor_, (n_windows, n_spks) Tensor containing ground truth of speaker labels, <br /> 1 if i-th window has j-th speaker speaking, else 0
`audio_segments:`               |  _torch.Tensor_, (n_windows, features_len) Tensor of feature vectors of each audio segment window
`speech_segments:`              |  _torch.Tensor_, (n_windows,) Tensor with i-th value 1 if VAD returns presence of speech audio in i-th window, else 0
`label_path:`                   |  _str_, Path of the rttm file containing labels for the 'idx' wav audio

2. <a name = 'read_rttm'></a> **read\_rttm:**
```def read_rttm(self, path)```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`path:`                         |  _str_, Path to the RTTM diarization file

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`rttm_out:`                     |  _numpy.array_, (..., 3) Array with column 1 holding start time of speaker, column 2 holding end time of speaker, and column 3 holding speaker label

---
### <a name = 'make_rttm'></a> def make\_rttm()
```sh
def make_rttm(out_dir, name, labels, win_step):
```
_Defined in utils.py_

Create RTTM Diarization files for non-overlapping speaker labels in var `labels`. Assumes non-speech part to have value `-1` and speech part to have some speaker label `(0, 1, 2, ...)`.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`out_dir:`                      |  _str_, Directory where the output RTTM diarization files to be saved
`name:`                         |  _str_, name for the audio files for which diarization was predicted
`labels:`                       |  _int_, Speaker/ Non-speech labels assigned to different audio segments based on the win\_step used to extract feature vectors
`win_step:`                     |  _int_, Step (in ms) between two windows of audio segments used for feature extraction

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`return variable:`              |  _str_, Path to the saved RTTM diarization file

---
### <a name = 'get_metrics'></a> def get\_metrics()
```sh
def get_metrics(groundtruth_path, hypothesis_path):
```
_Defined in utils.py_

Evaluate the diarization results of all the predicted RTTM files present in hypothesis directory to the grountruth RTTM files present in groundtruth directory.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`groundtruth_path:`             |  _str_, directory of groundtruth rttm files
`hypothesis_path:`              |  _str_, directory of hypothesis rttm files

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`metric:`                       |  _pyannote.metrics_, Pyannote metric class having diarization DERs for all the files.

---
## <a name = 'baselineMethods.py'></a> Defined in baselineMethods.py
### <a name = 'diarizationOracleNumSpkrs'></a> def diarizationOracleNumSpkrs()
```sh
def diarizationOracleNumSpkrs(audio_dataset, method="KMeans"):
```
_Defined in baselineMethods.py_

Predict the diarization labels using the oracle number of speakers for all the audio files in audio\_dataset with KMeans/ Spectral clustering algorithm. 

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`audio_dataset:`                |  _utils.DiarizationDataset_, Diarization dataset
`method:`                       |  _str_, Name of the method to be used for clustering part. Supports: "KMeans" or "Spectral"

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`hypothesis_dir:`               |  _str_, Directory where all the predicted RTTM diarization files are saved

---
### <a name = 'diarizationEigenGapNumSpkrs'></a> def diarizationEigenGapNumSpkrs()
```sh
def diarizationEigenGapNumSpkrs(audio_dataset):
```
_Defined in baselineMethods.py_

Predict the diarization labels using for all the audio files in audio\_dataset with Spectral clustering algorithm. It uses Eigen principle to predict the optimal number of speakers. The module uses already implented spectral algorithm from here: [https://github.com/wq2012/SpectralCluster](https://github.com/wq2012/SpectralCluster)

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`audio_dataset:`                |  _utils.DiarizationDataset_, Diarization dataset

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`hypothesis_dir:`               |  _str_, Directory where all the predicted RTTM diarization files are saved

---
## <a name = 'optimumSpeaker.py'></a> Defined in optimumSpeaker.py
Inspired from [https://github.com/wq2012/SpectralCluster](https://github.com/wq2012/SpectralCluster)
### <a name = 'eigengap'></a> class eigengap()
```sh
class eigengap(min_clusters=1, 
               max_clusters=100, 
               p_percentile=0.9, 
               gaussian_blur_sigma=2, 
               stop_eigenvalue=1e-2,
               thresholding_soft_multiplier=0.01, 
               thresholding_with_row_max=True)
```
_Defined in optimumSpeaker.py_

Utility function to decide the optimal number of speakers for clustering based on maximization of eigen-gap of the affinity matrix

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`min_clusters:`                 | _int_, Minimum number of output clusters
`max_clusters:`                 | _int_, Maximum number of output clusters 
`p_percentile:`                 | _float_, Parameter to computing p-th percentile for percentile based thresholding
`gaussian_blur_sigma:`          | _float_, sigma value for scipy gaussian filter
`stop_eigenvalue:`              | _float_, minimum value of eigenvalue of Affinity matrix for its eigenvector to be considered in clustering
`thresholding_soft_mutiplier:`  | _float_, factor to multiply to cells with value less than threshold in row/percentile thresholding. Parameter value of 0.0 turn cells less than threshold to zero in the matrix 
`thresholding_with_row_max:`    | _bool_, True for row thresholding, False for percentile thresholding

**Class Functions:**

1. <a name = 'getrefinementoperator'></a> **\_get\_refinement\_operator:**
```def _get_refinement_operator(self, name)```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`name:`                         | _str_, Get the named refinement operator. Available refinements- `'CropDiagonal'`, `'GaussianBlur'`, `'RowWiseThreshold'`, `'Symmetrize'`, `'Diffuse'`, `'RowWiseNormalize'`

---
[//]: #
[dec]: <https://arxiv.org/abs/1511.06335>
[desplanques]: <https://arxiv.org/abs/2005.07143v1>
[vad]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
[voxconverse]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
