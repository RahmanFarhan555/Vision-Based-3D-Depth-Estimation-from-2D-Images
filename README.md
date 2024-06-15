# Vision-Based-3D-Depth-Estimation-from-2D-Images
The goal of this project is to use 2D object detections from images to estimate the 3D
depth of objects, specifically cars, using a subset of the KITTI dataset. The object
detection is performed using YOLOv8x, and the depth estimation leverages the intrinsic
matrix of the camera setup provided by the KITTI dataset. This report outlines the
steps taken, the mathematical approach for depth estimation, the results obtained, and
the analysis of discrepancies between estimated distances and ground truth values.
