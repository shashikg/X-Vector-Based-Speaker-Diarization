# Deep Embedding Clustering for Speaker Diarization

**Team Name:** TensorSlow

**Members:** Aditya Singh ([@adityajaas](https://github.com/adityajaas)) and Shashi Kant Gupta ([@shashikg](https://github.com/shashikg))

**Report:** It is available [here](EE698R_TensorSlow_report.pdf). It contains 4 pages of main text + 1 references page + 2 pages of supplementary materials.

This speaker diarization model uses [Deep Embedding Clustering][dec] with a deep neural network initialized via
an Autoencoder to assign speaker labels to segments of the raw audio signal. Clustering is perfomed on x-vectors extracted using [Desplanques et al.][desplanques]'s ECAPA-TDNN framework. We use [Silero-VAD][vad] for voice audio detection.

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

![hypothesis](https://user-images.githubusercontent.com/45726064/118474684-f3b17400-b728-11eb-83eb-c329722f9707.png)

## Live Demo on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w1-BD2XLW3oz6kG5YqNZEMoalgIKqp8b?usp=sharing)

## ipynb Notebook Files
- **Baseline<DATASET_NAME>.ipynb:** To evaluate the DER score for the baseline models described in the report.
- **Compare_Spectral_vs_DEC_<DATASET_PARAM>.ipynb:** To evaluate the DER score for the DEC models described in the report and compare it against the Spectral clustering method.
- **utilities/TrainAutoEncoder.ipynb:** Output notebook file for training the AutoEncoder of the DEC model.
- **utilities/ExtractVAD.ipynb:** Used to extract and save all the VAD mapping for the audio files.
- **utilities/ExtractXvectors.ipynb:** Used to precompute X-vectors for the audio files and save it into a zip file to use it in the DiarizationDataset.

## API Documentation
Documentation and details about functions isnide the core module.
### Index

- [Defined in: utils.py](documentations/Documentation_Model.md#utils.py)
  - [class DiarizationDataset](documentations/Documentation_Model.md#diarizationdataset)
    - [func \_\_getitem\_\_](documentations/Documentation_Model.md#getitem)
    - [func read\_rttm](documentations/Documentation_Model.md#read_rttm)
  - [func make\_rttm](documentations/Documentation_Model.md#make_rttm)
  - [func get\_metrics](documentations/Documentation_Model.md#get_metrics)
  - [func plot\_annot](documentations/Documentation_Model.md#plot_annot)
- [Defined in: baselineMethods.py](documentations/Documentation_Model.md#baselineMethods.py)
  - [func diarizationOracleNumSpkrs](documentations/Documentation_Model.md#diarizationOracleNumSpkrs)
  - [func diarizationEigenGapNumSpkrs](documentations/Documentation_Model.md#diarizationEigenGapNumSpkrs)
- [Defined in: optimumSpeaker.py](documentations/Documentation_Model.md#optimumSpeaker.py)
  - [class eigengap](documentations/Documentation_Model.md#eigengap)
    - [func \_get\_refinement\_operator](documentations/Documentation_Model.md#getrefinementoperator)
    - [func compute\_affinity\_matrix](documentations/Documentation_Model.md#computeaffinitymatrix)
    - [func compute\_sorted\_eigenvectors](documentations/Documentation_Model.md#computesortedeigenvectors)
    - [func compute\_number\_of\_clusters](documentations/Documentation_Model.md#computenumberofclusters)
    - [func find](documentations/Documentation_Model.md#find)
  - [class AffinityRefinementOperation](documentations/Documentation_Model.md#affinityrefinementoperation)
    - [func check\_input](documentations/Documentation_Model.md#checkinput)
    - [func refine](documentations/Documentation_Model.md#refine)
  - [class CropDiagonal](documentations/Documentation_Model.md#Cropdiagonal)
    - [func refine](documentations/Documentation_Model.md#refineCropdiagonal)
  - [class GaussianBlur](documentations/Documentation_Model.md#gaussianblur)
    - [func refine](documentations/Documentation_Model.md#refinegaussianblur)
  - [class RowWiseThreshold](documentations/Documentation_Model.md#rowwisethreshold)
    - [func refine](documentations/Documentation_Model.md#refinerowwisethreshold)
  - [class Symmetrize](documentations/Documentation_Model.md#symmetrize)
    - [func refine](documentations/Documentation_Model.md#refinesymmetrize)
  - [class Diffuse](documentations/Documentation_Model.md#diffuse)
    - [func refine](documentations/Documentation_Model.md#refinediffuse)
  - [class RowWiseNormalize](documentations/Documentation_Model.md#rowwisenormalize)
    - [func refine](documentations/Documentation_Model.md#refinerowwisenormalize)
- [Defined in: DEC.py](documentations/Documentation_Model.md#DEC.py)
  - [class ResidualAutoEncoder](documentations/Documentation_Model.md#residualautoencoder)
  - [func load\_encoder](documentations/Documentation_Model.md#loadencoder)
  - [class ClusteringModule](documentations/Documentation_Model.md#clusteringmodule)
    - [func init\_centroid](documentations/Documentation_Model.md#initcentroid)
  - [class DEC](documentations/Documentation_Model.md#dec)
    - [func fit](documentations/Documentation_Model.md#fit)
    - [func predict](documentations/Documentation_Model.md#predict)
    - [func clusterAccuracy](documentations/Documentation_Model.md#clusteraccuracy)
  - [func diarizationDEC](documentations/Documentation_Model.md#diarizationDEC)
- [Defined in: colab_demo_utils.py](documentations/Documentation_Model.md#colab_demo_utils.py)
  - [func downloadYouTube](documentations/Documentation_Model.md#downloadYouTube)
  - [func loadVideoFile](documentations/Documentation_Model.md#loadVideoFile)
  - [func read\_rttm](documentations/Documentation_Model.md#read_rttm)
  - [func combine\_audio](documentations/Documentation_Model.md#combine_audio)
  - [func createAnnotatedVideo](documentations/Documentation_Model.md#createAnnotatedVideo)


---
[//]: #
[dec]: <https://arxiv.org/abs/1511.06335>
[desplanques]: <https://arxiv.org/abs/2005.07143v1>
[vad]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
[voxconverse]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
