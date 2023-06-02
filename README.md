# SSD-MonoDETR

Official implementation of the paper 'SSD-MonoDETR: Supervised Scale-aware Deformable Transformer for Monocular 3D Object Detection'.

                                 😄We will release code and checkpoints in the future.
# Abstract

Transformer-based methods have demonstrated superior performance for monocular 3D object detection recently, which aims at predicting 3D attributes from a single 2D image. Most existing transformer-based methods leverage both visual and depth representations to explore valuable query points on objects, and the quality of the learned query points has a great impact on detection accuracy. Unfortunately, existing unsupervised attention mechanisms in transformers are prone to generate low-quality query features due to inaccurate receptive fields, especially on hard objects. To tackle this problem, this paper proposes a novel ``Supervised Scale-aware Deformable Attention'' (SSDA) for monocular 3D object detection. Specifically, SSDA presets several masks with different scales and utilizes depth and visual features to adaptively learn a scale-aware filter for object query augmentation. Imposing the scale awareness, SSDA could well predict the accurate receptive field of an object query to support robust query feature generation. Aside from this, SSDA is assigned with a Weighted Scale Matching (WSM) loss to supervise scale prediction, which presents more confident results as compared to the unsupervised attention mechanisms. Extensive experiments on the KITTI benchmark demonstrate that SSDA significantly improves the detection accuracy, especially on moderate and hard objects, yielding state-of-the-art performance as compared to the existing approaches. 

# Acknowledgement
Our project is developed based on [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR). Thanks for this excellence work!
