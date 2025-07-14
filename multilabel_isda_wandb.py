import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import wandb
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, Dataset
from torchmetrics.classification import MultilabelAveragePrecision
from utils.yolo_dataloader import YOLOMultiLabelDataset
from utils.feature import ModelWithFeatures
from utils.BCE_ISDA import ISDA_BCELoss


model_names = sorted(
    name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", nargs="?", default="imagenet", help="path to dataset (default: imagenet)")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
)
parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="lr"
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
parser.add_argument(
    "--dist-url", default="tcp://224.66.41.62:23456", type=str, help="url used to set up distributed training"
)
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--no-accel", action="store_true", help="disables accelerator")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument("--dummy", action="store_true", help="use fake data to benchmark")
parser.add_argument("--project-name", default="multilabel-isda", type=str, help="wandb project name")
parser.add_argument("--run-name", default=None, type=str, help="wandb run name")
parser.add_argument("--wandb-offline", action="store_true", help="run wandb in offline mode")

best_acc1 = 0


def compute_gradient_norm(model):
    """计算模型参数的梯度范数"""
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    return (total_norm**0.5) if param_count > 0 else 0.0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU. This will completely " "disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn(
                "nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'"
            )
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # Initialize wandb
    if args.rank == 0 or not args.distributed:
        wandb_mode = "offline" if args.wandb_offline else "online"
        wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={
                "architecture": args.arch,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "pretrained": args.pretrained,
                "distributed": args.distributed,
                "gpu_count": ngpus_per_node if args.distributed else 1,
            },
            mode=wandb_mode,
        )

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        if args.gpu is not None:
            torch.accelerator.set_device_index(args.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # 修改最后一层以适应多标签分类
    # 首先需要知道类别数量，从classes.txt文件读取
    if not args.dummy:
        class_names_file = os.path.join(args.data, "classes.txt")
        if os.path.exists(class_names_file):
            with open(class_names_file, "r", encoding="utf-8") as f:
                num_classes = len([line.strip() for line in f.readlines()])
        else:
            print(f"Warning: {class_names_file} not found, using default 1000 classes")
            num_classes = 1000
    else:
        num_classes = 1000

    # 修改模型的最后一层
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # 使用包装器来支持特征提取
    model = ModelWithFeatures(model)

    # 获取特征维度
    if hasattr(model.model, "fc"):
        feature_num = model.model.fc.in_features
    elif hasattr(model.model, "classifier"):
        if isinstance(model.model.classifier, nn.Sequential):
            feature_num = model.model.classifier[-1].in_features
        else:
            feature_num = model.model.classifier.in_features
    else:
        feature_num = 2048  # 默认值，适用于大多数ResNet

    if not use_accel:
        print("using CPU, this will be slow")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if device.type == "cuda":
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(device)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == "cuda":
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion_bce = nn.BCEWithLogitsLoss().to(device)  # 多标签分类使用BCE损失
    criterion_isda = ISDA_BCELoss(feature_num, num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f"{device.type}:{args.gpu}"
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            criterion_isda = checkpoint["criterion_isda"]
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        # YOLO格式数据集路径
        train_img_dir = os.path.join(args.data, "images", "train")
        train_label_dir = os.path.join(args.data, "labels", "train")
        val_img_dir = os.path.join(args.data, "images", "val")
        val_label_dir = os.path.join(args.data, "labels", "val")
        class_names_file = os.path.join(args.data, "classes.txt")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = YOLOMultiLabelDataset(
            train_img_dir,
            train_label_dir,
            class_names_file,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        val_dataset = YOLOMultiLabelDataset(
            val_img_dir,
            val_label_dir,
            class_names_file,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    if args.evaluate:
        validate(val_loader, model, criterion_bce, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion_isda, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion_bce, args)

        scheduler.step()

        # wandb logging for epoch-level metrics
        if (args.rank == 0 or not args.distributed) and wandb.run is not None:
            wandb.log({"epoch/mAP": acc1, "epoch/learning_rate": scheduler.get_last_lr()[0], "epoch/epoch": epoch})

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "criterion_isda": criterion_isda,
                },
                is_best,
            )

            # Save model artifact to wandb when we have a new best model
            if is_best and (args.rank == 0 or not args.distributed) and wandb.run is not None:
                artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
                artifact.add_file("model_best.pth.tar")
                wandb.log_artifact(artifact)

    # Finish wandb logging
    if (args.rank == 0 or not args.distributed) and wandb.run is not None:
        wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, device, args):

    use_accel = not args.no_accel and torch.accelerator.is_available()

    batch_time = AverageMeter("Time", use_accel, ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", use_accel, ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", use_accel, ":.4e", Summary.NONE)
    mAP = AverageMeter("mAP", use_accel, ":6.2f", Summary.NONE)
    f1_overall = AverageMeter("F1-Overall", use_accel, ":6.2f", Summary.NONE)
    f1_classwise = AverageMeter("F1-Classwise", use_accel, ":6.2f", Summary.NONE)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mAP, f1_overall, f1_classwise],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output and features
        output, features = model(images, isda=True)
        loss = criterion(model, output, features, target, 7.5 * (epoch / args.epochs))
        # measure accuracy and record loss
        acc1, acc2, acc3 = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        mAP.update(acc1[0], images.size(0))
        f1_overall.update(acc2[0], images.size(0))
        f1_classwise.update(acc3[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # 计算梯度范数
        grad_norm = compute_gradient_norm(model)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

            # wandb logging for training metrics
            if (args.rank == 0 or not args.distributed) and wandb.run is not None:
                step = epoch * len(train_loader) + i
                wandb.log(
                    {
                        "train/loss": losses.avg,
                        "train/mAP": mAP.avg,
                        "train/f1_overall": f1_overall.avg,
                        "train/f1_classwise": f1_classwise.avg,
                        "train/batch_time": batch_time.avg,
                        "train/data_time": data_time.avg,
                        "train/gradient_norm": grad_norm,
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )


def validate(val_loader, model, criterion, args):

    use_accel = not args.no_accel and torch.accelerator.is_available()

    def run_validate(loader, base_progress=0):

        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if use_accel:
                    if args.gpu is not None and device.type == "cuda":
                        torch.accelerator.set_device_index(args.gpu)
                        images = images.cuda(args.gpu, non_blocking=True)
                        target = target.cuda(args.gpu, non_blocking=True)
                    else:
                        images = images.to(device)
                        target = target.to(device)
                # compute output and features
                output, features = model(images, isda=True)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc2, acc3 = accuracy(output, target)
                losses.update(loss.item(), images.size(0))
                mAP.update(acc1[0], images.size(0))
                f1_overall.update(acc2[0], images.size(0))
                f1_classwise.update(acc3[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter("Time", use_accel, ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", use_accel, ":.4e", Summary.NONE)
    mAP = AverageMeter("mAP", use_accel, ":6.2f", Summary.AVERAGE)
    f1_overall = AverageMeter("F1-Overall", use_accel, ":6.2f", Summary.AVERAGE)
    f1_classwise = AverageMeter("F1-Classwise", use_accel, ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, mAP, f1_overall, f1_classwise],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        mAP.all_reduce()
        f1_overall.all_reduce()
        f1_classwise.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(
            val_loader.dataset, range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset))
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    # wandb logging for validation metrics
    if (args.rank == 0 or not args.distributed) and wandb.run is not None:
        wandb.log(
            {
                "val/loss": losses.avg,
                "val/mAP": mAP.avg,
                "val/f1_overall": f1_overall.avg,
                "val/f1_classwise": f1_classwise.avg,
                "val/batch_time": batch_time.avg,
            }
        )

    return mAP.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, use_accel, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target):
    """Computes multi-label classification metrics: mAP, Overall F1, and Class-wise F1"""
    with torch.no_grad():
        batch_size = target.size(0)
        num_classes = target.size(1)

        # 使用sigmoid激活函数
        sigmoid_output = torch.sigmoid(output)
        predictions = sigmoid_output > 0.5

        # 转换为布尔型以便计算
        target_bool = target.bool()
        predictions_bool = predictions.bool()

        # 计算mAP (平均精度)
        metric = MultilabelAveragePrecision(num_labels=num_classes, average="macro", thresholds=None)
        mAP = metric(output, target.int()) * 100.0

        # 计算Overall F1 (micro-average)
        tp_overall = (predictions_bool & target_bool).sum().float()
        fp_overall = (predictions_bool & ~target_bool).sum().float()
        fn_overall = (~predictions_bool & target_bool).sum().float()

        precision_overall = tp_overall / (tp_overall + fp_overall + 1e-8)
        recall_overall = tp_overall / (tp_overall + fn_overall + 1e-8)
        f1_overall = 2 * precision_overall * recall_overall / (precision_overall + recall_overall + 1e-8)
        f1_overall = f1_overall * 100.0

        # 计算Class-wise F1 (macro-average)
        tp_classwise = (predictions_bool & target_bool).sum(dim=0).float()
        fp_classwise = (predictions_bool & ~target_bool).sum(dim=0).float()
        fn_classwise = (~predictions_bool & target_bool).sum(dim=0).float()
        precision_classwise = tp_classwise / (tp_classwise + fp_classwise + 1e-8)
        recall_classwise = tp_classwise / (tp_classwise + fn_classwise + 1e-8)
        f1_classwise = (
            2 * precision_classwise * recall_classwise / (precision_classwise + recall_classwise + 1e-8)
        ).mean() * 100.0

        # 返回三个值：mAP, Overall F1, Class-wise F1
        return [mAP], [f1_overall], [f1_classwise]


if __name__ == "__main__":
    main()
