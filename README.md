# 3D object detection
(Abstract) The fusion of LiDAR and camera sensors offers remarkable results in multimodal 3D object detection with enhanced performance. However, existing fusion methods are primarily designed considering ideal data, ignoring the practical challenges of sensor specification and environmental variations encountered in autonomous driving. Thus, these methods often exhibit a significant performance degradation when faced with adverse conditions, such as sparse point cloud and inclement weather. To address these multiple adverse conditions simultaneously, we present the first attempt to apply auxiliary restoration networks in multimodal 3D object detection. These networks restore degraded point cloud and image, ensuring the primary multimodal detection network obtains higher quality features in a unified form. Especially, we propose a spherical domain point upsampler based on bilateral point generation and an adjustment network with a horizontal alignment block. Additionally, for efficient fusion with restored point cloud and image, we suggest a graph detector with a unified loss function, including auxiliary, contrastive, and difficulty losses. The experimental results demonstrate that the proposed approach prevents a performance decline in adverse conditions and outperforms state-of-the-art methods.

![Fig2 (9)](https://github.com/user-attachments/assets/ed608847-b08a-4e77-9280-8a1685a702e8)



pretrained model - [auxphere](https://drive.google.com/file/d/1EhKkVmRDsRobxodYBijvgfNCETBf80vA/view?usp=sharing)

Public dataset links

KITTI dataset - https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

CADC dataset - http://cadcd.uwaterloo.ca/

Dense dataset - https://www.uni-ulm.de/in/mrm/forschung/datensaetze-1/dense-datasets/

## Training & Testing
```
# Train
bash scripts/dist_train.sh

# Test
bash scripts/dist_test.sh
