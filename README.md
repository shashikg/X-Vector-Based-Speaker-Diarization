# Deep Embedding Clustering for Speaker Diarization

Course project for EE698 (2020-21 Sem 2)

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

- [Defined in: utils.py](https://github.com/shashikg/speaker_diarization_ee698/wiki/utils)
  - [class DiarizationDataset](https://github.com/shashikg/speaker_diarization_ee698/wiki/utils#diarizationdataset)
    - [func \_\_getitem\_\_](https://github.com/shashikg/speaker_diarization_ee698/wiki/utils#getitem)
    - [func read\_rttm](https://github.com/shashikg/speaker_diarization_ee698/wiki/utils#read_rttm)
  - [func make\_rttm](https://github.com/shashikg/speaker_diarization_ee698/wiki/utils#make_rttm)
- [Defined in: baselineMethods.py](https://github.com/shashikg/speaker_diarization_ee698/wiki/baselineMethods)
  - [func get\_metrics](https://github.com/shashikg/speaker_diarization_ee698/wiki/baselineMethods#get_metrics)
  - [func diarizationOracleNumSpkrs](https://github.com/shashikg/speaker_diarization_ee698/wiki/baselineMethods#diarizationOracleNumSpkrs)
  - [func diarizationEigenGapNumSpkrs](https://github.com/shashikg/speaker_diarization_ee698/wiki/baselineMethods#diarizationEigenGapNumSpkrs)
- [Defined in: optimumSpeaker.py](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker)
  - [class eigengap](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#eigengap)
    - [func \_get\_refinement\_operator](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#getrefinementoperator)
    - [func compute\_affinity\_matrix](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#computeaffinitymatrix)
    - [func compute\_sorted\_eigenvectors](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#computesortedeigenvectors)
    - [func compute\_number\_of\_clusters](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#computenumberofclusters)
  - [class AffinityRefinementOperation](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#affinityrefinementoperation)
    - [func check\_input](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#checkinput)
    - [func refine](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#refine)
  - [class CropDiagonal](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#Cropdiagonal)
    - [func refine](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#refineCropdiagonal)
  - [class GaussianBlur](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#gaussianblur)
    - [func refine](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#refinegaussianblur)
  - [class RowWiseThreshold](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#rowwisethreshold)
    - [func refine](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#refinerowwisethreshold)
  - [class Symmetrize](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#symmetrize)
    - [func refine](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#refinesymmetrize)
  - [class Diffuse](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#diffuse)
    - [func refine](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#refinediffuse)
  - [class RowWiseNormalize](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#rowwisenormalize)
    - [func refine](https://github.com/shashikg/speaker_diarization_ee698/wiki/optimumSpeaker#refinerowwisenormalize)
- [Defined in: DEC.py](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC)
  - [class ResidualAutoEncoder](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC#residualautoencoder)
  - [func load\_encoder](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC#loadencoder)
  - [class ClusteringModule](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC#clusteringmodule)
    - [func init\_centroid](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC#initcentroid)
  - [class DEC](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC#dec)
    - [func fit](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC#fit)
    - [func predict](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC#predict)
    - [func clusterAccuracy](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC#clusteraccuracy)
  - [func diarizationDEC](https://github.com/shashikg/speaker_diarization_ee698/wiki/DEC#diarizationDEC)
- [Defined in: colab_demo_utils.py](https://github.com/shashikg/speaker_diarization_ee698/wiki/colab_demo_utils)
  - [func downloadYouTube](https://github.com/shashikg/speaker_diarization_ee698/wiki/colab_demo_utils#downloadYouTube)
  - [func loadVideoFile](https://github.com/shashikg/speaker_diarization_ee698/wiki/colab_demo_utils#loadVideoFile)
  - [func read\_rttm](https://github.com/shashikg/speaker_diarization_ee698/wiki/colab_demo_utils#read_rttm)
  - [func combine\_audio](https://github.com/shashikg/speaker_diarization_ee698/wiki/colab_demo_utils#combine_audio)
  - [func createAnnotatedVideo](https://github.com/shashikg/speaker_diarization_ee698/wiki/colab_demo_utils#createAnnotatedVideo)


---
[//]: #
[dec]: <https://arxiv.org/abs/1511.06335>
[desplanques]: <https://arxiv.org/abs/2005.07143v1>
[vad]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
[voxconverse]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
