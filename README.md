# SSD-MonoDETR

Official implementation of the paper 'SSD-MonoDTR: Supervised Scale-constrained Deformable Transformer for Monocular 3D Object Detection'.

😄We will release code and checkpoints in the future.

Abstract

Transformer-based methods have demonstrated superior performance for monocular 3D object detection recently, which predicts 3D attributes from a single 2D image. Most existing transformer-based methods leverage visual and depth representations to explore valuable query points on objects, and the quality of the learned queries has a great impact on detection accuracy. Unfortunately, existing unsupervised attention mechanisms in transformer are prone to generate low-quality query features due to inaccurate receptive fields, especially on hard objects. To tackle this problem, this paper proposes a novel “Supervised Scale-constrained Deformable Attention” (SSDA) for monocular 3D object detection. Specifically, SSDA presets several masks with different scales and utilizes depth and visual features to predict the local feature for each query. Imposing the scale constraint, SSDA could well predict the accurate receptive field of a query to support robust query feature generation. What is more, SSDA is assigned with a Weighted Scale Matching (WSM) loss to supervise scale prediction, which presents more confident results as compared to the unsupervised attention mechanisms. Extensive experiments on KITTI demonstrate that SSDA significantly improves the detection accuracy especially on moderate and hard objects, yielding SOTA performance as compared to the existing approaches. 
