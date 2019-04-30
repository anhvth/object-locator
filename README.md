# object-locator
Tensorflow implementation for Weighted Hausdorff Distance: A Loss Function For Object Localization https://arxiv.org/abs/1806.07564


# Requerements 
```
tensorflow=1.13.1
opencv
tqdm
```
# Training
```
conda activate object-locator
python train.py
```
to monitor the training process, by default the summary dir is set to $args.outputdir/summary
```
tensorboard --logdir training_checkpoints/summary/
```

