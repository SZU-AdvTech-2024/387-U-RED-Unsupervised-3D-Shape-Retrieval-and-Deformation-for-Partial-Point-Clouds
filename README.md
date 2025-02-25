## Data download and preprocessing details
Please follow [joint_learning_retrieval_deformation](https://github.com/mikacuy/joint_learning_retrieval_deformation) to prepare the dataset and reorder the directory like:
```
your_base_dir/
    data_aabb_all_models/
    data_aabb_constraints_keypoint/
    dis_mat/
    generated_datasplits/
    partnet_rgb_masks_chair/
    partnet_rgb_masks_storagefurniture/
    partnet_rgb_masks_table/
```

## Setup
```
pip install -r requirements.txt
```


## Configs
Modify the config files in folder 'config'. Complete the value occupied by 'xxx' including "base_dir" and "log_path" in training config and "dm_model_path", "re_model_path", "base_dir" and "log_path" in testing config. During testing phase, keep "dm_model_path" and "re_model_path" exactly the same as the trained model. If you want to change the category, just fix "category" into "storagefurniture" or "table".

## Train
```
CUDA_VISIBLE_DEVICES=0 python engine/train.py config/config_train_chair.json
```

## Test
```
CUDA_VISIBLE_DEVICES=0 python engine/test.py config/config_test_chair.json
```

## Visualization
```
CUDA_VISIBLE_DEVICES=0 python engine/vis.py config/config_vis_chair.json
```