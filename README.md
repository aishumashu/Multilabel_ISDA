# Multilabel_ISDA
## Baseline
**BCELoss**

```
python multilabel.py -a resnet50 --pretrained --dist-url 'tcp://127.0.0.1:51444' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data1/lyurenqin/workspace/dataset/VOC_yolo
```

## ISDA
**weighted var**

```
python multilabel_isda_wandb.py -a resnet50 --pretrained --dist-url 'tcp://127.0.0.1:51444' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --run-name RUN_NAME --wandb-offline /data1/lyurenqin/workspace/dataset/VOC_yolo
```

**inverted weighted var**

```
python multilabel_isda_wandb.py -a resnet50 --pretrained --dist-url 'tcp://127.0.0.1:51444' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --var-type 'inverted-weighted-var' --run-name RUN_NAME --wandb-offline /data1/lyurenqin/workspace/dataset/VOC_yolo
```

