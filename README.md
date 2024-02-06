# Scene Classification for ICESat2 Data
## Requirements
1. scipy
2. sklearn
3. networkx
4. tensorflow
5. spektral
6. cv2
7. PIL
8. open3d
9. numba
10. findpeaks
## How to Use
### Start from Scratch (see [train.py]())
1. Prepare and put training samples in [train_data()], where each training sample is an ATL03 data scene (in the polar stereographic coordinate system) with a specific label (zero or one)
2. Use [feature_augmentation.py]() to convert training samples into three feature spaces.
3. Use scene_classification_network.py to train and save a model.
### Use a Saved Model (see [test.py]())
1.
2.
