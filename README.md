# Multilabel_ISDA
## Baseline
**BCELoss**
    python multilabel.py -a resnet50 --dist-url 'tcp://127.0.0.1:51444' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data1/lyurenqin/workspace/dataset/VOC_yolo
## ISDA
**weighted var**
    python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:51444' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data1/lyurenqin/workspace/dataset/VOC_yolo
