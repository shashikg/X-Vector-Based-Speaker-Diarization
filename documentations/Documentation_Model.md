# Documentation: EE698R DEC based Diarization Model

**Team Name:** TensorSlow

**Members:** Aditya Singh ([@adityajaas](https://github.com/adityajaas)) and Shashi Kant Gupta ([@shashikg](https://github.com/shashikg))


This speaker diarization model uses [Deep Embedding Clustering][dec] with a deep neural network initialized via 
a Residual Autoencoder to assign speaker labels to segments of the raw audio signal.
Clustering is perfomed on x-vectors extracted using [Desplanques et al.][desplanques]'s ECAPA-TDNN framework. 
We use [Silero-VAD][vad] for voice audio detection.

**Baseline Model:** Spectral clustering is used for audio-label assignment.

## API Documentation
### Index
 - [Defined in: utils.py](#utils.py)
   - [class DiarizationDataset](#diarizationdataset)
     - [func \_\_getitem\_\_](#getitem)
     - [func read\_rttm](#read_rttm)
   - [func make\_rttm](#make_rttm)
   - [func get\_metrics](#get_metrics)
   - [func plot\_annot](#plot_annot)
- [Defined in: baselineMethods.py](#baselineMethods.py)
  - [func diarizationOracleNumSpkrs](#diarizationOracleNumSpkrs)
  - [func diarizationEigenGapNumSpkrs](#diarizationEigenGapNumSpkrs)
- [Defined in: optimumSpeaker.py](#optimumSpeaker.py)
  - [class eigengap](#eigengap)
    - [func \_get\_refinement\_operator](#getrefinementoperator)
    - [func compute\_affinity\_matrix](#computeaffinitymatrix)
    - [func compute\_sorted\_eigenvectors](#computesortedeigenvectors)
    - [func compute\_number\_of\_clusters](#computenumberofclusters)
    - [func find](#find)
  - [class AffinityRefinementOperation](#affinityrefinementoperation)
    - [func check\_input](#checkinput)
    - [func refine](#refine)
  - [class CropDiagonal](#Cropdiagonal)
    - [func refine](#refineCropdiagonal)
  - [class GaussianBlur](#gaussianblur)
    - [func refine](#refinegaussianblur)
  - [class RowWiseThreshold](#rowwisethreshold)
    - [func refine](#refinerowwisethreshold)
  - [class Symmetrize](#symmetrize)
    - [func refine](#refinesymmetrize)
  - [class Diffuse](#diffuse)
    - [func refine](#refinediffuse)
  - [class RowWiseNormalize](#rowwisenormalize)
    - [func refine](#refinerowwisenormalize)
- [Defined in: DEC.py](#DEC.py)
  - [class ResidualAutoEncoder](#residualautoencoder)
  - [func load\_encoder](#loadencoder)
  - [class ClusteringModule](#clusteringmodule)
    - [func init\_centroid](#initcentroid)
  - [class DEC](#dec)
    - [func fit](#fit)
    - [func predict](#predict)
    - [func clusterAccuracy](#clusteraccuracy)
  - [func diarizationDEC](#diarizationDEC)
- [Defined in: colab\_demo\_utils.py](#colab_demo_utils.py)
  - [func downloadYouTube](#downloadYouTube)
  - [func loadVideoFile](#loadVideoFile)
  - [func read\_rttm](#read_rttm)
  - [func combine\_audio](#combine_audio)
  - [func createAnnotatedVideo](#createAnnotatedVideo)
---
[//]: # (======================================================for utils.py=========================================================)
# <a name =  'utils.py'></a> Defined in: utils.py

### <a name = 'diarizationdataset'></a> class DiarizationDataset() 
_Defined in utils.py_
```sh
class DiarizationDataset(dataset_name=None
                 data_dir=None, 
                 sr=16000, 
                 window_len=240, 
                 window_step=120, 
                 transform=None,
                 batch_size_for_ecapa=512,
                 vad_step=4,
                 split='full',
                 use_precomputed_vad= True,
                 use_oracle_vad= False,
                 skip_overlap= True)
```
Create an abstract class for loading the dataset. This class applies the necessary pre-processing and x-vector feature extraction methods to return the audio file as a bunch of segmented x-vector features to use it directly in the clustering algorithm to predict speaker labels. The module uses the pre-computed X-vectors if available otherwise extract it during the runtime.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`dataset_name:`                 |  _str_, Name of the pre-existing dataset to use. Options: `ami`, `ami_dev`, `voxconverse`
`data_dir:`                     |  _str_, Directory for any dataset other the options specified in `dataset_name`. <br />  Both `dataset_name` and `data_dir` cannot be None 
`sr:`                           |  _int_, Sampling rate of the audio signal 
`window_len:`                   |  _int_, Window length (in ms) of each of the audio segments to be passed for feature extraction
`window_step:`                  |  _int_, Step (in ms) between two windows of audio segments to be passed for feature extraction
`transform:`                    |  _list_, List of transforms like `mel-transform` to be performed on audio while preprocessing, <br /> default = None
`batch_size_for_ecapa:`         |  _int_, Batch size of audio segments while performing feature extraction using ECAPA-TDNN
`vad_step:`                     |  _int_, Number of windows to split each audio chunk into. Argument used by Silero-VAD module
`split:`                        |  _str_, Argument defining type of split of dataset, <br /> default = 'full' indicates no split
`use_precomputed_vad:`           | _bool_, If True, downloads precomputed Voice Activity Detection label output for the dataset. Only available for dataset options specified in `dataset_name`
`use_oracle_vad:`                | _bool_, If True, model does Voice Activity Detection directly from groundtruth rttm files bypassing the Silero VAD module.
`skip_overlap:`                  | _bool_, If True, model skips the windows with multiple speakers speaking by inspecting the groundtruth rttm files

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
`rttm_out:`                     |  _numpy.ndarray_, (..., 3) Array with column 1 holding start time of speaker, column 2 holding end time of speaker, and column 3 holding speaker label

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
def get_metrics(groundtruth_path, hypothesis_path, collar=0.25, skip_overlap=True):
```
_Defined in utils.py_

Evaluate the diarization results of all the predicted RTTM files present in hypothesis directory to the grountruth RTTM files present in groundtruth directory.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`groundtruth_path:`             |  _str_, directory of groundtruth rttm files
`hypothesis_path:`              |  _str_, directory of hypothesis rttm files
`collar:`                        |  _float_, Duration (in seconds) of collars removed from evaluation around boundaries of reference segments
`skip_overlap:`                  |  _bool_, If True, calculates Diarization Error Rate ignoring the overlapped region

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`metric:`                       |  _pyannote.metrics_, Pyannote metric class having diarization DERs for all the files.

---

### <a name = 'plot_annot'></a> def plot\_annot()
```sh
def plot_annot(name="IS1009a", collar=0.25, skip_overlap=True, groundtruth_path=None, hypothesis_path=None):
```
_Defined in utils.py_

Calculate the Diarization Error Rate for filename specified, and print the groundtruth and hypothesis time series plot.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`name:`                         |  _str_, Name of the file whose time series plot is to be generated. File must be present in the `hypothesis_path` folder
`collar:`                        |  _float_, Duration (in seconds) of collars removed from evaluation around boundaries of reference segments
`skip_overlap:`                  |  _bool_, If True, calculates Diarization Error Rate ignoring the overlapped region
`groundtruth_path:`             |  _str_, Directory of groundtruth rttm files
`hypothesis_path:`              |  _str_, Directory of hypothesis rttm files

---
---
[//]: # (===================================================for baselineMethods.py======================================================)

# <a name = 'baselineMethods.py'></a> Defined in baselineMethods.py

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
[//]: # (==================================================for optimumSpeaker.py=====================================================)

# <a name = 'optimumSpeaker.py'></a> Defined in optimumSpeaker.py
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
`gaussian_blur_sigma:`          | _float_, sigma value for standard deviation of gaussian kernel in scipy gaussian filter
`stop_eigenvalue:`              | _float_, Minimum value of eigenvalue of Affinity matrix for its eigenvector to be considered in clustering
`thresholding_soft_mutiplier:`  | _float_, Factor to multiply to cells with value less than threshold in row/percentile thresholding. Parameter value of 0.0 turn cells less than threshold to zero in the matrix 
`thresholding_with_row_max:`    | _bool_, True for row-max thresholding, False for percentile thresholding

**Class Functions:**

1. <a name = 'getrefinementoperator'></a> **\_get\_refinement\_operator:**
```sh
def _get_refinement_operator(self, name)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`name:`                         | _str_, Get the input refinement operator. Available refinements- `'CropDiagonal'`, `'GaussianBlur'`, `'RowWiseThreshold'`, `'Symmetrize'`, `'Diffuse'`, `'RowWiseNormalize'`

**Returns:**
Variable                                                                                                           | Detail
-------------------------------------------------------------------------------------------------------            | -----------
 `CropDiagonal()`/`GaussianBlur()`/<br>`RowWiseThreshold()`/`Symmetrize()`/ <br>`Diffuse()`/`RowWiseNormalize()`   | _optimumSpeaker.AffinityRefinementOperation_, Returns specified refinement method class

2. <a name = 'computeaffinitymatrix'></a> **compute\_affinity\_matrix:**
```sh
def compute_affinity_matrix(self, X)
```
Compute the affinity matrix for a matrix X with row as each instance and column as features by calculating cosine similarity between pair of l2 normalized columns of X

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            |  _numpy.ndarray_, (n_windows, n_features) Input matrix with column as features to compute affinity matrix between pair of columns

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`affinity:`                     |  _numpy.ndarray_, (n_windows, n_windows) Symmetric array with (i,j)th value equal to cosine similiarity between i-th and j-th row

3. <a name = 'computesortedeigenvectors'></a> **compute\_sorted\_eigenvectors:**
```sh
def compute_sorted_eigenvectors(self, A)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`A:`                            |  _numpy.ndarray_, (n_windows, n_windows) Symmetric array with (i,j)th value equal to cosine similiarity between i-th and j-th row

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`w:`                            |  _numpy.ndarray_, Decreasing order sorted eigen values of affinity matrix A
`v:`                            |  _numpy.ndarray_, Eigen vectors corresponding to eigen values returned

4. <a name = 'computenumberofclusters'></a> **compute\_number\_of\_clusters:**
```sh
def compute_number_of_clusters(self, eigenvalues, max_clusters, stop_eigenvalue)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`eigenvalues:`                  |  _numpy.ndarray_, Decreasing order sorted eigen values of affinity matrix between different windows
`max_clusters:`                 |  _int_, Maximum number of clusters required. Default `'None'` puts no such limit to the number of clusters
`stop_eigenvalue:`              |  _float_, Minimum value of eigenvalue to be considered for deciding number of clusters. Eigenvalues below this value are discarded

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`max_delta_index:`              |  _int_, Index to the eigenvalue such that eigen gap is maximized. It gives the number of clusters determined by the function

5. <a name = 'find'></a> **find**
```sh
def find(self, X)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            |  _numpy.ndarray_, (n_windows, n_features) Input matrix with column as features to compute affinity matrix between pair of columns

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`k:`                            |  _int_, Number of clusters calculated after creating the affinity matrix, applying refinements, and using eigen-gap maximization. `self.min_clusters` ≤ `k` ≤ `self.max_clusters`

---

### <a name = 'affinityrefinementoperation'></a> class AffinityRefinementOperation()
```sh
class AffinityRefinementOperation(metaclass=abc.ABCMeta)
```
_Defined in optimumSpeaker.py_

Meta class to the refinement operation classes passed as input to be perfomed on the data

**Class Functions:**

1. <a name = 'checkinput'></a> **check\_input:**
```sh
def check_input(self, X)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            | _numpy.ndarray_, Input array to be refined by refinement operators 

**Returns:**
Variable                        | Detail
------------------------        | -----------
`ValueError()`\ `TypeError()` | _ValueError/TypeError_, Type Error if X is not a numpy array. Value error if X is not a 2D square matrix

2. <a name = 'refine'></a> **refine:**
```sh
def refine(self, X)
```
Abstract function redefined in various child classes of class AffinityRefinementOperation

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            | _numpy.ndarray_, Input array to be refined by refinement operators 

---

### <a name = 'Cropdiagonal'></a> class CropDiagonal()
```sh
class Cropdiagonal(AffinityRefinementOperation)
```
_Defined in optimumSpeaker.py_

Operator to replace diagonal element by the max non-diagonal value of row. Post operation, the matrix has similar properties to a standard Laplacian matrix.
This also helps to avoid the bias during Gaussian blur and normalization.

**Class Functions:**

1. <a name = 'refineCropdiagonal'></a> **refine:**
```sh
def refine(self, X)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            | _numpy.ndarray_, Input array to be refined by refinement operators 

**Returns:**
Variable                        | Detail
------------------------        | -----------
`Y:`                             | _numpy.ndarray_, Output array with Crop diagonal refinement applied

---

### <a name = 'gaussianblur'></a> class GaussianBlur()
```sh
class GaussianBlur(AffinityRefinementOperation)
      def __init__(self, sigma = 1)
```
_Defined in optimumSpeaker.py_

Operator to apply gaussian filter to the input array. Uses scipy.ndimage.gaussian_filter

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`sigma:`                        | _float_, Standard deviation for Gaussian kernel

**Class Functions:**

1. <a name = 'refinegaussianblur'></a> **refine:**
```sh
def refine(self, X)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            | _numpy.ndarray_, Input array to be refined by refinement operators 

**Returns:**
Variable                        | Detail
------------------------        | -----------
`Y:`                             | _numpy.ndarray_, Output array with gaussian filter applied

---

### <a name = 'rowwisethreshold'></a> class RowWiseThreshold()
```sh
class RowWiseThreshold(AffinityRefinementOperation)
      def __init__(self,
                 p_percentile=0.95,
                 thresholding_soft_multiplier=0.01,
                 thresholding_with_row_max=False)
```
_Defined in optimumSpeaker.py_

Operator to apply row wise thresholding based on either percentile or row-max thresholding. 

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`p_percentile:`                 | _float_, Standard deviation for Gaussian kernel
`thresholding_soft_multiplier:` | _float_, Factor to multiply to cells with value less than threshold in row/percentile thresholding. Parameter value of 0.0 turn cells less than threshold to zero in the matrix 
`thresholding_with_row_max:`    | _bool_, `True` applies row-max based thresholding, `False` applies percentile based thresholding

**Class Functions:**

1. <a name = 'refinerowwisethreshold'></a> **refine:**
```sh
def refine(self, X)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            | _numpy.ndarray_, Input array to be refined by refinement operators 

**Returns:**
Variable                        | Detail
------------------------        | -----------
`Y:`                             | _numpy.ndarray_, Output array with row wise threshold applied

---

### <a name = 'symmetrize'></a> class Symmetrize()
```sh
class Cropdiagonal(AffinityRefinementOperation)
```
_Defined in optimumSpeaker.py_

Operator to return a symmetric matrix based on max{ X, X<sup>T</sup> } from a given input matrix X.

**Class Functions:**

1. <a name = 'refinesymmetrize'></a> **refine:**
```sh
def refine(self, X)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            | _numpy.ndarray_, Input array to be used to create a symmetric matrix

**Returns:**
Variable                        | Detail
------------------------        | -----------
`Y:`                             | _numpy.ndarray_, Output symmetric array 

---

### <a name = 'diffuse'></a> class Diffuse()
```sh
class Diffuse(AffinityRefinementOperation)
```
_Defined in optimumSpeaker.py_

Operator to return a diffused symmetric matrix X<sup>T</sup>X from a given input matrix X.

**Class Functions:**

1. <a name = 'refinediffuse'></a> **refine:**
```sh
def refine(self, X)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            | _numpy.ndarray_, Input array to be used to create a diffused symmetric matrix

**Returns:**
Variable                        | Detail
------------------------        | -----------
`Y:`                             | _numpy.ndarray_, Output diffused symmetric array 

---
### <a name = 'rowwisenormalize'></a> class RowWiseNormalize()
```sh
class RowWiseNormalize(AffinityRefinementOperation)
```
_Defined in optimumSpeaker.py_

Operator to normalize each row of input matrix X by the maximum value in the corresponding rows.

**Class Functions:**

1. <a name = 'refinerowwisenormalize'></a> **refine:**
```sh
def refine(self, X)
```

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`X:`                            | _numpy.ndarray_, Input array to be row normalized

**Returns:**
Variable                        | Detail
------------------------        | -----------
`Y:`                             | _numpy.ndarray_, Output row normalized array

---

[//]: # (======================================================for DEC.py=========================================================)
# <a name = 'DEC.py'></a> Defined in DEC.py

### <a name = 'residualautoencoder'></a> class ResidualAutoEncoder()
```sh
class ResidualAutoEncoder(ip_features,
                          hidden_dims=[500, 500, 2000, 30]))
```
_Defined in DEC.py_

Create a torch.nn.Module for a deep autoencoder composed of Residual Neural Network (ResNet) bloacks as the encoder and decoder layer. Activation used is ReLU. The bottleneck encoder output and final decoder output are not activated to avoid data loss due to ReLU activation.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`ip_features:`                  |  _int_, Input features size
`hidden_dims:`                  |  _list of int_, List of hidden dimension features. Last element on the list is the output dimension of bottleneck of the autoencoder

**Returns:**
Variable                        | Detail
------------------------        | -----------
`z:`                            | _torch.Tensor_, Output from the bottle encoder of the deep autoencoder network.
`xo:`                           | _list of torch.Tensor_, Output from each encoder except the bottle encoder of the deep autoencoder. First item of the list is the input given to the system.
`xr:`                           | _list of torch.Tensor_, Reconstruction of inputs to each encoder layer of autoencoder. xr is reversed so that i-th item in list xr is the reconstruction of i-th item in list xo. Eg. First item of xo is the input to the ResidualAutoEncoder network, and first item of xr is the reconstruction from the ResidualAutoEncoder network.

---
### <a name = 'loadencoder'></a> def load\_encoder()
```sh
def load_encoder():
```
_Defined in DEC.py_

Load weights from the ResidualAutoEncoder trained on the training data.

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`model:`                        |  _ResidualAutoEncoder, Model with input feature size of 192, and hidden layers of size 500, 500, 2000, 30. Weights of the model initialized to weight of the autoencoder trained on training data.

---
### <a name = 'clusteringmodule'></a> class ClusteringModule()
```sh
class ClusteringModule(nn.Module):
    def __init__(self,
                 num_clusters,
                 encoder, data,
                 cinit = "KMeans"):
```
_Defined in DEC.py_

Clustering module of the deep embedding clustering (DEC) algorithm. It uses the trained encoder of the ResidualAutoEncoder to initialize the DEC Clustering network. Kmeans is used to initialize centroids in the latent space.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`num_clusters:`                 |  _str_, Number of clusters to create from the algorithm
`encoder:`                      |  _nn.Module_, Pre-trained encoder for intializing the centroids. Encoder tranforms data to the latent space for clustering
`cinit:`                        |  _str_, Initialization method of centroids of clusters. Default `KMeans`

**Returns:**
Variable                        | Detail
------------------------        | -----------
`q:`                            | _torch.Tensor_, Tensor of similarity between embedding points z_i and centroid mu_j. Assumes Student's t distribution as the kernel
`p:`                            | _torch.Tensor_, Tensor of target distribution based on soft assignment of q_i
`xo[0]`                         | _torch.Tensor_, Input data to the ResidualAutoEncoder
`xr[0]`                         | -torch.Tensor_, Reconstructed input by the ResidualAutoEncoder

**Class Functions:**

1. <a name = 'initcentroid'></a> **init\_centroid:**
```sh
def init_centroid(self,
                  data,
                  method = "KMeans")
```
Returns clustered data after calculating the optimal number of speakers using eigen-gap method, and then clustering the data based on the method specified.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`data:`                         | _torch.Tensor_, Input data to be clustered
`method:`                       | _numpy.ndarray_, Clustering method. Default `KMeans`. Options `KMeans`/`Spectral`

**Returns:**
Variable                        | Detail
------------------------        | -----------
`output:`                       | _torch.Tensor_, Tensor containing intialized centroids for the dataset

---
### <a name = 'dec'></a> class DEC()
```sh
class DEC(self,
          num_clusters,
          encoder, data,
          cinit = "KMeans"):
```
_Defined in DEC.py_

Deep embedding clustering (DEC) algorithm. It uses the trained encoder of the ResidualAutoEncoder to initialize the DEC Clustering network. It calls ClusteringModule class to initialize the centroids.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`encoder:`                      |  _nn.Module_, Pre-trained encoder for intializing the centroids. Encoder tranforms data to the latent space for clustering
`num_clusters:`                 |  _str_, Number of clusters to create from the algorithm. Default `None` uses eigengap to determine number of clusters
`cinit:`                        |  _str_, Initialization method of centroids of clusters. Default `KMeans`. Options `KMeans`/`Spectral`


**Class Functions:**

1. <a name = 'fit'></a> **fit:**
```sh
def fit(self,
        data,
        y_true = None,
        niter = 150,
        lrEnc = 1e-4,
        lrCC = 1e-4,
        verbose = False)
```

Trains the algorithm by measuring the KL Divergence between target and observed distributions. Also updates the ResidualAutoEncoder using MSE loss in parallel to improve the latent space project of the data for better clustering. Both the updates use the Adams optimizer and the objective function is a linear combination of KL Divergence between target and observed distribution, and MSE Loss between input data and its reconstruction by the ResidualAutoEncoder.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`data:`                         | _torch.Tensor_, Input data to be clustered
`y_true:`                       | _numpy.ndarray_, True labels of the data we aim to cluster. `predict()` and `clusterAccuracy()` functions are invoked only if y_true is not `None`
`niter`                         | _int_, Number of epochs to train the model for
`lrEnc`                         | _float_, Learning rate for updating the encoder
`lrCC`                          | _float_, Learning rate for updating the cluster centres
`verbose`                       | _bool_, `True` value activates the tqdm progress bar while training. `False` returns no updates when training

2. <a name = 'predict'></a> **predict:**
```def predict(self, data)```

Predict the cluster label to the data by inspecting the label about which the observed distribution is maximized.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`data:`                         | _torch.Tensor_, Input data to be labels after clustering

**Returns:**
Variable                        | Detail
------------------------        | -----------
`y_pred:`                       | _numpy.ndarray_, Soft prediction labels of the data


3. <a name = 'clusteraccuracy'></a> **clusterAccuracy:**
```def clusterAccuracy(self, y_pred, y_true)```

Predict the cluster labels accuracy as the maximum accuracy between y_pred and y_true for all the permutation of y_pred. This permutation is found by linear_sum_assignment optimization function of scipy.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`y_pred:`                       | _numpy.ndarray_, Prediction of the labels by DEC algorithm
`y_true`                        | _numpy.ndarray_, True labels of the data

**Returns:**
Variable                        | Detail
------------------------        | -----------
`accuracy:`                     | _float_, Cluster assignment accuracy
`reassignment:`                 | _dict_, dictionary with key as rows and value as cols indices for the optimal assignment

---
### <a name = 'diarizationDEC'></a> def diarizationDEC()
```sh
def diarizationDEC(audio_dataset,
                   num_spkr = None,
                   hypothesis_dir = None)
```
_Defined in DEC.py_

Compute diarization labels based on oracle number of speakers if `num_spkr = 'oracle'`. Used as an optimal benchmark for performance of DEC. If `num_spkr = None`, uses eigen-gap maximization in the ClusteringModule to determine the number of speakers.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`audio_dataset:`                |  _utils.DiarizationDataset_, Test diarization dataset
`num_spkr:`                     |  _str_,  `None` for calculating the optimal number of speakers from eigen-gap maximization. `oracle` for using the number of speakers in each window given with the data.
`hypothesis_dir:`               |  _str_, Directory to store the predicted speaker labels in the audio segments in an rttm file. `None` stores it in `./rttm_output/` directory

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`hypothesis_dir:`               |  _str_, Directory to the rttm files containing predicted speaker labels with their timestamps

---
# <a name =  'colab_demo_utils.py'></a> Defined in: colab\_demo\_utils.py

### <a name = 'downloadYouTube'></a> def downloadYouTube()
```sh
def downloadYouTube(videourl, path):
```
_Defined in colab\_demo\_utils.py_

Download video from YouTube in .mp4 format using Video URL.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`videourl:`                     |  _str_, URL of the YouTube video to download
`path:`                         |  _str_, directory to save the YouTube video. If directory does not exist, it is created.

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`save_dir:`                     |  _str_, Save directory location

---

### <a name = 'loadVideoFile'></a> def loadVideoFile()
```sh
def loadVideoFile(playvideo_file=False):
```
_Defined in colab\_demo\_utils.py_

Load video file either from YouTube or from your local directory into your current session working directory. Also extracts and stores its audio file in .wav format.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`playvideo_file:`               |  _bool_, If True, plays the video after loading in the working directory. Default=`False`

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`video_dir:`                    |  _str_, Returns the path to the saved video

---

### <a name = 'read_rttm'></a> def read\_rttm()
```sh
def read_rttm(path):
```
_Defined in colab\_demo\_utils.py_

Create hypothesis labels for each window using .rttm file.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`path:`                         |  _str_, Path to the rttm file

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`hypothesis_labels:`            |  _numpy.ndarray_, (n_instances, 3) i-th row's first, second and third column contains start, end, and speaker id of the i-th instance of speech.

---

### <a name = 'combine_audio'></a> def combine\_audio()
```sh
def combine_audio(vidname, audname, outname, fps):
```
_Defined in colab\_demo\_utils.py_

Combine cv2 processed silent video with its audio file to output the complete annotated video.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`vidname:`                      |  _str_, Path to the silent video
`audname:`                      |  _str_, Path to the audio file to be attached
`outname:`                      |  _str_, Output video file name
`fps:`                          |  _int_, Frame rate of the video
---

### <a name = 'createAnnotatedVideo'></a> def createAnnotatedVideo()
```sh
def createAnnotatedVideo(audio_dataset, hypothesis_dir):
```
_Defined in colab\_demo\_utils.py_

Use cv2 to put annotations in the video using the hypothesis labels.

**Parameters:**
Argument                        | Detail
------------------------------- | ------------
`audio_dataset:`                 |  _utils.DiarizationDataset_, Dataset pipeline
`hypothesis_dir:`               |  _str_, Path to the directory with hypothesis labels rttm files

**Returns:**
Variable                        | Detail
------------------------------- | ------------
`op_video_name:`                |  _str_, Annotated output video filename

---

[//]: #
[dec]: <https://arxiv.org/abs/1511.06335>
[desplanques]: <https://arxiv.org/abs/2005.07143v1>
[vad]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
[voxconverse]: <https://pytorch.org/hub/snakers4_silero-vad_vad/>
