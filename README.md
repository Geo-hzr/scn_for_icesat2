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
### Start from Scratch (see [train.py](https://github.com/Geo-hzr/scn_for_icesat2/blob/ee227c3052bab78a452e2bc39852928d71e9d7e4/train.py))
1. Prepare and put training samples in [train_data](https://github.com/Geo-hzr/scn_for_icesat2/tree/978083bf0dfa33463d1fb02c94136d46dbf0e4b7/train_data), where each training sample is an ATL03 data scene (in the polar stereographic coordinate system) with a specific label (positive or negative).
2. Use [feature_augmentation.py](https://github.com/Geo-hzr/scn_for_icesat2/blob/b0acfa0f9d35da797265e784f2206450e592cf8e/feature_augmentation.py) to convert training samples into three feature spaces.
3. Use [scene_classification_network.py](https://github.com/Geo-hzr/scn_for_icesat2/blob/2797be3034080082df7a31db64854aa6309beddd/scene_classification_network.py) to train and save a model.
### Use a Saved Model (see [test.py](https://github.com/Geo-hzr/scn_for_icesat2/blob/e27259ea55871109d2c8418840655981c84b9abe/test.py))
1. Load a model from [saved_model](https://github.com/Geo-hzr/scn_for_icesat2/tree/94ba76c584f326ac8921ec6d5db64ddbcd4caefb/saved_model).
2. Prepare and put a track of ATL03 data in [test_data](https://github.com/Geo-hzr/scn_for_icesat2/tree/0397e3edb822745721e36bbe7f5712dc6ba9c8e7/test_data).
3. Use [presegmentation.py](https://github.com/Geo-hzr/scn_for_icesat2/blob/58a4b7bf0014d4b5a4ad2cb987d73c109a8ded9f/presegmentation.py) to segment a track of ATL03 data into separate scenes, whose labels will be predicted by a loaded model.
