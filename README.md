# MRC
Code for "Multiscale reduction clustering of vibration signals for unsupervised diagnosis of machine faults''

# Please cite this paper if the code is helpful
Yifan Wu, Chuan Li, Shuai Yang, Yun Bai,
Multiscale reduction clustering of vibration signals for unsupervised diagnosis of machine faults,
Applied Soft Computing,
Volume 142,
2023,
110358,
ISSN 1568-4946,
https://doi.org/10.1016/j.asoc.2023.110358.
(https://www.sciencedirect.com/science/article/pii/S1568494623003769)

# Abstract
Fault diagnosis is of great importance for the intelligent health management of mechanical systems. For engineering applications, it is very difficult to collect and label vibration signals corresponding to machine faults. Due to the complicated operational environment, moreover, useful and critical features are often covered by surrounding noise. For those reasons, a multiscale reduction clustering (MRC) method is proposed for the unsupervised diagnosis of machine faults. In the present approach, vibration signals were collected to generate multiscale convolutional representation through a one-dimensional convolutional neural network without prior knowledge of signal processing techniques. Chosen by a convolutional encoder, the dimensionality of the multiscale convolutional representation was reduced for improving the clustering capability. During this dimensionality reduction, a loss function was proposed to optimize the network and speed up the convergence of the diagnostics. The proposed method was evaluated by two benchmark datasets and an experimental setup. With the present method, the clustering accuracy and normalized mutual information for three datasets are all over 0.91 and 0.82, respectively. Results show that the addressed MRC has superior diagnosis ability under the unsupervised fashion compared to other state-of-the-art models. It is proved that MRC is robust for the diagnosis tasks under different working conditions.
# Keywords
Unsupervised learning; Feature extraction; Convolutional neural network; Multiscale reduction clustering; Fault diagnosis

## Requirements
* Python version : 3.7
* cuda==10.0
* cudnn
* tensorflow-gpu==1.14.0
* keras


## Basic Usage

* Parameters can be speicified in main1.py, mian2.py.
* Normalized data: 3Dprinter.h5. 
* Others: the source link:  http://www.52phm.cn (Jiangnan University, China)

