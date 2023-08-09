# Exploiting Cross-Modality Self-Attention for Enhanced Person Re-Identification Across Modalities

## Environmental requirements:

PyTorch >= 1.2.0

ignite >= 0.2.1

torchvision >= 0.4.0

apex == 0.1

## Quick start
1. Modify the path to datasets:

The path to datasets can be modified in the following file:

```shell
./configs/default/dataset.py
```

2. Training:

To train the model, you can use following command:

SYSU-MM01:
```Shell
python train.py --cfg ./configs/SYSU.yml
```

RegDB:
```Shell
python train.py --cfg ./configs/RegDB.yml
```

## Reference:
[DoubtedSteam/MPANet](https://github.com/DoubtedSteam/MPANet)

