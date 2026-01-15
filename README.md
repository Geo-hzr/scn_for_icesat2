# Scene Classification Network for ICESat-2 Data
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
### Start from Scratch (see [train.py](train.py))
1. Prepare and put training samples in [train_data](train_data), where each training sample is an ATL03 data scene (in the polar stereographic coordinate system) with a specific label (positive or negative).
2. Use [feature_augmentation.py](feature_augmentation.py) to convert training samples into three feature spaces.
3. Use [scene_classification_network.py](scene_classification_network.py) to train and save a model.
### Use a Saved Model (see [test.py](test.py))
1. Load a model from [saved_model](saved_model).
2. Prepare and put a track of ATL03 data in [test_data](test_data).
3. Use [presegmentation.py](presegmentation.py) to segment a track of ATL03 data into separate scenes, whose labels are predicted by a loaded model.
## Citation
Please cite our paper if you use this project in your research: [A Novel Deep Learning-Based Approach for Rift and Iceberg Recognition From ICESat-2 Data](https://ieeexplore.ieee.org/abstract/document/10480730), Z. Huang, S. Wang, R. B. Alley, A. Li, and B. R. Parizek, IEEE Transactions on Geoscience and Remote Sensing, 2024, doi: 10.1109/TGRS.2024.3382573.
