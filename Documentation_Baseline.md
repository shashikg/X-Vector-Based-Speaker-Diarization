# Documentation: EE698R Baseline Model

The Baseline model for speaker diarization uses PyTorch based [Silero-VAD][vad] for Audio detection and [Desplanques et al.][desplanques]'s ECAPA-TDNN for x-vector 
feature extraction. Spectral clustering is used for audio-label assignment.

## DataSet
Model is tested on [VoxConverse][voxconverse] dataset

## Tutorial
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D23rxcCqZe78hUeJORv5nu8efTUMifws)

## API Documentation
### Index
- [class DiarizationDataset](#diarizationdataset)
  - [\_\_getitem\_\_](#getitem)


### <a name = 'diarizationdataset'></a> class DiarizationDataset()
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
Create an abstract class for loading dataset

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`root_dir:`                     |  _str_, Directory containing the audio wav files 
`label_dir:`                    |  _str_, Directory containing the rttm label files
`xvectors_dir:`                 |  _str_, Directory containing the precomputed xvectors for audio segments,<br /> default = None
`vad_dir=None:`                 |  _str_, Directory containing the precomputer audio speech detection timestamps, <br /> default = None
`sr:`                           |  _int_, Sampling rate of the audio signal 
`window_len:`                   |  _int_, Window length of each of the audio segments to be passed for feature extraction
`window_step:`                  |  _int_, Step between two windows of audio segments to be passed for feature extraction
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
`speech_segments:`              |  _torch.Tensor_, (n_windows,) Tensor with i-th value 1 if VAD returns speech in i-th window, else 0
`label_path:`                   |  _str_, Path of the rttm file containing labels for the 'idx' wav audio

[//]: #
[desplanques]: <https://arxiv.org/abs/2005.07143v1>
[vad]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
[voxconverse]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
