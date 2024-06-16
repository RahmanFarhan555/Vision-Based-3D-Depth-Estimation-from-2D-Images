# Vision-Based-3D-Depth-Estimation-from-2D-Images
1 Introduction
The goal of this project is to use 2D object detections from images to estimate the 3D
depth of objects, specifically cars, using a subset of the KITTI dataset. The object
detection is performed using YOLOv8x, and the depth estimation leverages the intrinsic
matrix of the camera setup provided by the KITTI dataset. This report outlines the
steps taken, the mathematical approach for depth estimation, the results obtained, and
the analysis of discrepancies between estimated distances and ground truth values.

2 Methodology

The algorithm begins by importing necessary libraries such as os for file system interactions, numpy for numerical operations, cv2 for computer vision tasks, and matplotlib.pyplot for image visualization. It then initializes a pre-trained YOLOv8 model
using the Ultralytics library for object detection. Directories are set up to store input
images, calibration matrices, ground truth labels, and annotated output images. Key
constants are defined, including the class ID for ”car” in the COCO dataset. Functions
are created to load intrinsic camera matrices, ground truth labels, and calculate Intersection over Union (IoU) for evaluation. The algorithm iterates over image files, loading
each image, its corresponding intrinsic matrix, and ground truth labels. Object detection
is performed on the images, with detected cars analyzed to estimate distances from the
camera and compare with ground truth. Annotated images are then generated, displaying bounding boxes, midpoints, annotations, and distance estimates, which are saved to
the output directory.

2.1 Geometrical/mathematical reasoning of distance calculation

To calculate the distance to a detected car, we employ the intrinsic camera matrix and
geometric principles. Initially, the coordinates of the detected bounding box representing
the car are extracted from the image. Subsequently, the midpoint of the lower bound
of this bounding box is computed, serving as the focal point for distance estimation.
Utilizing the intrinsic matrix, a 2D homogeneous coordinate vector is constructed to
represent this midpoint. By applying the inverse of the intrinsic matrix to this vector,
a direction vector from the camera to the midpoint in camera coordinates is derived.
This directional vector is then intersected with the ground plane, typically defined at
a known height above the ground level (e.g., -1.65 meters), producing a 3D point in
space. Finally, the Euclidean distance from the camera to this intersection point on
the ground plane is calculated, representing the estimated distance to the detected car.
This comprehensive process effectively utilizes the intrinsic camera matrix and geometric
reasoning to accurately estimate distances to detected objects in the image.

2.2 Matching Detections with Ground Truth:
To match detections with ground truth objects, we calculate the Intersection over Union
(IoU) between each detected bounding box and the ground truth bounding boxes.
The IoU measures the overlap between two bounding boxes, providing a metric for
their similarity.
We select the ground truth box with the highest IoU for each detected car, considering
it as the matched ground truth object.
If the IoU is below a certain threshold, or if no ground truth box is available, we
consider the detection as a false positive.

![image](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/299ea3a2-0e97-4b1f-9269-93ed7f596e42)
![006037](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/dc8617ae-8c0b-4792-9972-775ed077ebdf)
![006042](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/818e1d9d-0b58-4a6d-8d65-886815232373)
![006048](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/56b6e26b-eb5e-4370-bffa-7cd6690b553b)
![006054](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/fbfab3e7-cde8-4ff6-b920-93f82fa4abe6)
![006059](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/4ba24f1f-e802-497c-87b0-9ccc4a1ce280)
![006067](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/20656e06-d96a-41b1-9d1d-e8d4485e3d06)
![006097](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/e7935a54-48ed-4e64-a435-161d68396283)
![006098](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/2e127488-91d3-4aea-a48c-e73e6a2c1bc1)
![006121](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/99511c3e-68eb-4b34-be64-e44502532ddb)
![006130](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/be00ee71-e58c-4323-a536-46396bfb26fc)
![006206](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/6a94aef0-e3d1-4c29-9520-462368592266)
![006211](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/763bf073-d1f9-4ebf-96bf-0f01f98faa56)
![006227](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/c476751a-64af-4b24-a8d5-69175d9b00f0)
![006253](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/3281b918-4826-4bb9-be9d-8f325002a31c)
![006291](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/2d3cc55c-4ff8-4727-9e21-b0188ee48791)
![006310](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/e761d423-f288-4231-920a-333c05b7ab0d)
![006312](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/32311b94-cdab-4491-8ab3-e582f11ac391)
![006315](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/9cc61131-8abb-4c32-9d8d-811785f84662)
![006329](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/edec89de-06ca-4aa3-9a83-6b911b96bcbb)
![006374](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/a4ed5df5-169a-45f2-b76d-4164842cce7d)
![plot](https://github.com/RahmanFarhan555/Vision-Based-3D-Depth-Estimation-from-2D-Images/assets/170820777/a21e6732-9a64-4c99-8e41-83501138a4cc)




















