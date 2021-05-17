# Deep Embedding Clustering for Speaker Diarization

Course project for EE698 (2020-21 Sem 2)

This speaker diarization model uses [Deep Embedding Clustering][dec] with a deep neural network initialized via
a Autoencoder to assign speaker labels to segments of the raw audio signal.
Clustering is perfomed on x-vectors extracted using [Desplanques et al.][desplanques]'s ECAPA-TDNN framework.
We use [Silero-VAD][vad] for voice audio detection.

**Baseline Model:** Spectral clustering is used for audio-label assignment.

## DataSet
Model is tested on [VoxConverse][voxconverse] dataset (total 216 audio files). We randomly split the dataset into two parts: ‘test’ and ‘train’ with test data having 50 audio files. We also tested the model on [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) test dataset (total 16 audio files).

## Results
### VoxConverse
Methods                          |     DER
-------------------------------  | -----------
Spectral Clustering              | 17.76
Ours                             | 12.99
Spectral Clustering (Oracle VAD) | 17.98
**Ours (Oracle VAD)**            | **11.70**

### AMI Corpus
Methods                          |     DER
-------------------------------  | -----------
Spectral Clustering              | 21.99
Ours                             | 23.39
Spectral Clustering (Oracle VAD) | 14.96
**Ours (Oracle VAD)**            | **13.14**

### Demo on random YouTube file
Original Video Link: [here](https://www.youtube.com/watch?v=4-mvb-8FHPo)\
Diarization Output Link: [here](http://www.youtube.com/watch?v=NH9Glqdu0gw "Demo Speaker Diarization by Team TensorSlow")

https://user-images.githubusercontent.com/45726064/117953334-8d48e200-b333-11eb-9bab-3e6529b83f57.mp4

![hypothesis](https://user-images.githubusercontent.com/45726064/117957270-8623d300-b337-11eb-9e4c-15751fb2ac9e.png)

## ipynb Notebook Files
- **DEC_ResAE.ipynb:** To evaluate the DER score for the DEC models described in the report. Use the link available in Tutorial section to open it on google colab
- **ExtractVAD.ipynb:** Used to extract and save all the VAD mapping for the audio files in VoxConverse dataset.
- **ExtractXvectors.ipynb:** Used to precompute X-vectors for the audio files in VoxConverse dataset and save it into a zip file to use it in the DiarizationDataset.
- **Baseline.ipynb:** To evaluate the DER score for the baseline models described in the report. Use the link available in the Tutorial section to open it on google colab.

## Live Demo on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w1-BD2XLW3oz6kG5YqNZEMoalgIKqp8b?usp=sharing)

## API Documentation
### Index

- [Defined in: utils.py](documentations/Documentation_Model.md/#utils.py)
  - [class DiarizationDataset](documentations/Documentation_Model.md/utils#diarizationdataset)
    - [func \_\_getitem\_\_](documentations/Documentation_Model.md/utils#getitem)
    - [func read\_rttm](documentations/Documentation_Model.md/utils#read_rttm)
  - [func make\_rttm](documentations/Documentation_Model.md/utils#make_rttm)
- [Defined in: baselineMethods.py](documentations/Documentation_Model.md/#baselineMethods.py)
  - [func get\_metrics](documentations/Documentation_Model.md/baselineMethods#get_metrics)
  - [func diarizationOracleNumSpkrs](documentations/Documentation_Model.md/baselineMethods#diarizationOracleNumSpkrs)
  - [func diarizationEigenGapNumSpkrs](documentations/Documentation_Model.md/baselineMethods#diarizationEigenGapNumSpkrs)
- [Defined in: optimumSpeaker.py](documentations/Documentation_Model.md/#optimumSpeaker.py)
  - [class eigengap](documentations/Documentation_Model.md/optimumSpeaker#eigengap)
    - [func \_get\_refinement\_operator](documentations/Documentation_Model.md/optimumSpeaker#getrefinementoperator)
    - [func compute\_affinity\_matrix](documentations/Documentation_Model.md/optimumSpeaker#computeaffinitymatrix)
    - [func compute\_sorted\_eigenvectors](documentations/Documentation_Model.md/optimumSpeaker#computesortedeigenvectors)
    - [func compute\_number\_of\_clusters](documentations/Documentation_Model.md/optimumSpeaker#computenumberofclusters)
  - [class AffinityRefinementOperation](documentations/Documentation_Model.md/optimumSpeaker#affinityrefinementoperation)
    - [func check\_input](documentations/Documentation_Model.md/optimumSpeaker#checkinput)
    - [func refine](documentations/Documentation_Model.md/optimumSpeaker#refine)
  - [class CropDiagonal](documentations/Documentation_Model.md/optimumSpeaker#Cropdiagonal)
    - [func refine](documentations/Documentation_Model.md/optimumSpeaker#refineCropdiagonal)
  - [class GaussianBlur](documentations/Documentation_Model.md/optimumSpeaker#gaussianblur)
    - [func refine](documentations/Documentation_Model.md/optimumSpeaker#refinegaussianblur)
  - [class RowWiseThreshold](documentations/Documentation_Model.md/optimumSpeaker#rowwisethreshold)
    - [func refine](documentations/Documentation_Model.md/optimumSpeaker#refinerowwisethreshold)
  - [class Symmetrize](documentations/Documentation_Model.md/optimumSpeaker#symmetrize)
    - [func refine](documentations/Documentation_Model.md/optimumSpeaker#refinesymmetrize)
  - [class Diffuse](documentations/Documentation_Model.md/optimumSpeaker#diffuse)
    - [func refine](documentations/Documentation_Model.md/optimumSpeaker#refinediffuse)
  - [class RowWiseNormalize](documentations/Documentation_Model.md/optimumSpeaker#rowwisenormalize)
    - [func refine](documentations/Documentation_Model.md/optimumSpeaker#refinerowwisenormalize)
- [Defined in: DEC.py](documentations/Documentation_Model.md/#DEC.py)
  - [class ResidualAutoEncoder](documentations/Documentation_Model.md/DEC#residualautoencoder)
  - [func load\_encoder](documentations/Documentation_Model.md/DEC#loadencoder)
  - [class ClusteringModule](documentations/Documentation_Model.md/DEC#clusteringmodule)
    - [func init\_centroid](documentations/Documentation_Model.md/DEC#initcentroid)
  - [class DEC](documentations/Documentation_Model.md/DEC#dec)
    - [func fit](documentations/Documentation_Model.md/DEC#fit)
    - [func predict](documentations/Documentation_Model.md/DEC#predict)
    - [func clusterAccuracy](documentations/Documentation_Model.md/DEC#clusteraccuracy)
  - [func diarizationDEC](documentations/Documentation_Model.md/DEC#diarizationDEC)
- [Defined in: colab_demo_utils.py](documentations/Documentation_Model.md/#colab_demo_utils.py)
  - [func downloadYouTube](documentations/Documentation_Model.md/colab_demo_utils#downloadYouTube)
  - [func loadVideoFile](documentations/Documentation_Model.md/colab_demo_utils#loadVideoFile)
  - [func read\_rttm](documentations/Documentation_Model.md/colab_demo_utils#read_rttm)
  - [func combine\_audio](documentations/Documentation_Model.md/colab_demo_utils#combine_audio)
  - [func createAnnotatedVideo](documentations/Documentation_Model.md/colab_demo_utils#createAnnotatedVideo)


---
[//]: #
[dec]: <https://arxiv.org/abs/1511.06335>
[desplanques]: <https://arxiv.org/abs/2005.07143v1>
[vad]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
[voxconverse]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
