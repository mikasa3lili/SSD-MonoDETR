# SSD-MonoDETR

Official implementation of the paper 'SSD-MonoDETR: Supervised Scale-aware Deformable Transformer for Monocular 3D Object Detection'.
                               
![image](https://github.com/mikasa3lili/SSD-MonoDETR/blob/main/pipeline.png)

# Abstract

Transformer-based methods have demonstrated superior performance for monocular 3D object detection recently, which aims at predicting 3D attributes from a single 2D image. Most existing transformer-based methods leverage both visual and depth representations to explore valuable query points on objects, and the quality of the learned query points has a great impact on detection accuracy. Unfortunately, existing unsupervised attention mechanisms in transformers are prone to generate low-quality query features due to inaccurate receptive fields, especially on hard objects. To tackle this problem, this paper proposes a novel ``Supervised Scale-aware Deformable Attention'' (SSDA) for monocular 3D object detection. Specifically, SSDA presets several masks with different scales and utilizes depth and visual features to adaptively learn a scale-aware filter for object query augmentation. Imposing the scale awareness, SSDA could well predict the accurate receptive field of an object query to support robust query feature generation. Aside from this, SSDA is assigned with a Weighted Scale Matching (WSM) loss to supervise scale prediction, which presents more confident results as compared to the unsupervised attention mechanisms. Extensive experiments on the KITTI benchmark demonstrate that SSDA significantly improves the detection accuracy, especially on moderate and hard objects, yielding state-of-the-art performance as compared to the existing approaches. 

# Requirements
(Our code is tested on:)    
Pyton 3.8    
Pytorch 1.10.1+cu111    

# Installation
a. Clone this repository:     
`git clone https://github.com/mikasa3lili/SSD-MonoDETR`      
`cd SSD-MonoDETR`    

b. Create a conda environment and install:  
`conda create -n ssd-monodetr python=3.8`    
`conda activate ssd-monodetr`    
`conda install pytorch torchvision cudatoolkit`  
`pip install -r requirements.txt`  
`cd lib/models/monodetr/ops/`  
`bash make.sh`  
`cd ../../../..`  
`mkdir logs`  

# Data Preparation  
You shoud download the KITTI, Waymo datasets, and follow the OpenPCDet(https://github.com/open-mmlab/OpenPCDet) to generate data infos.    
These datasets shold have the following organization:    
KITTI:    
`├── data`    
`│   ├── kitti`  
`│   │   │── ImageSets`  
`│   │   │── training`  
`│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)`  
`│   │   │── testing`  
`│   │   │   ├──calib & velodyne & image_2`  

Waymo:  
`├── data`  
`│   ├── waymo`  
`│   │   │── ImageSets`  
`│   │   │── raw_data`  
`│   │   │   │── segment-xxxxxxxx.tfrecord`  
`|   |   |   |── ...`  
`|   |   |── waymo_processed_data_v0_5_0`  
`│   │   │   │── segment-xxxxxxxx/`  
`|   |   |   |── ...`  
`│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/  (old, for single-frame)`  
`│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl  (old, for single-frame)`  
`│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional, old, for single-frame)`  
`│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)`  
`│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)`   

Then, please follow the [DEVIANT](https://github.com/abhi1kumar/DEVIANT) to further process the Waymo dataset.

# Training and Testing   
`cd tools` 
train:
`python train_val.py --config ${CONFIG_FILE}`
test:
`python train_val.py --config ${CONFIG_FILE} -e` 

# Acknowledgement
Our project is developed based on [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR), and the data processing of Waymo is follow the [DEVIANT](https://github.com/abhi1kumar/DEVIANT) Thanks for their excellence works!
