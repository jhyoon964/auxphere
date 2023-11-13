# Unified Auxiliary Restoration Network for Robust Multimodal 3D Object Detection in Adverse Conditions

(abstract) The fusion of LiDAR and camera sensors offers remarkable results in multimodal 3D object detection with enhanced performance. However, existing fusion methods are primarily designed considering ideal data, ignoring the practical challenges of sensor specification and environmental variations encountered in autonomous driving. Thus, these methods often exhibit a significant performance degradation when faced with adverse conditions, such as sparse point cloud and inclement weather. To address these multiple adverse conditions simultaneously, we present the first attempt to apply auxiliary restoration networks in multimodal 3D object detection. These networks restore degraded point clouds and images, ensuring the primary multimodal detection network obtains higher quality features in a unified form. Especially, we propose a spherical domain point upsampler based on bilateral interpolation and a refinement network with a dilated pyramid block. Additionally, for efficient fusion with restored point clouds and images, we suggest a graph-based detector with a unified loss function, including auxiliary, contrastive, and difficulty losses. The experimental results demonstrate that the proposed approach prevents a performance decline in adverse conditions and outperforms state-of-the-art methods. The source code with the pretrained models
is available at https://anonymous.4open.science/r/auxphere-B613/.


## Training & Testing
```
# Train
python train.py

# Test
python val.py
