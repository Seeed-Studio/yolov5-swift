import random
import warnings
from copy import deepcopy

import argparse
import math
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import yaml
from datetime import datetime
from pathlib import Path
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.std import tqdm

from models.common import Classify
from models.yolo import DetectMultiBackend, Model
from utils.general import LOGGER, colorstr, one_cycle, two_linear, increment_path, check_img_size
from utils.torch_utils import select_device, de_parallel, ModelEMA, model_info

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def train(opt, hyp, trainset, valset, nc, names
          ):
    save_dir, evolve, batch_size, epochs = Path(opt.save_dir), opt.evolve, opt.batch_size, opt.epochs

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    device = select_device(opt.device)
    cuda = device.type != 'cpu'

    # Model
    if opt.cfg.startswith('yolov5m'):
        # YOLOv5 Classifier
        if opt.weights:
            model = DetectMultiBackend(weights=opt.weights, device=torch.device('cuda:0'))
        else:
            model = torch.hub.load('ultralytics/yolov5', opt.cfg, pretrained=True, autoshape=False)
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:12] if opt.cfg.endswith('6') else model.model[:8]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else sum(x.in_channels for x in m.m)  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
    elif opt.cfg in torch.hub.list('rwightman/gen-efficientnet-pytorch'):  # i.e. efficientnet_b0
        model = torch.hub.load('rwightman/gen-efficientnet-pytorch', opt.cfg, pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, nc)
    elif opt.cfg.startswith('resnet'):
        model = torchvision.models.resnet18(True)
        model.fc.out_features = nc
    elif opt.cfg.endswith('.yaml'):
        model = Model(cfg=opt.cfg, nc=nc)
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else sum(x.in_channels for x in m.m)  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
    else:  # try torchvision
        model = torchvision.models.__dict__[opt.cfg](pretrained=True)
        model.fc = nn.Linear(model.fc.weight.shape[1], nc)

    model_info(model, img_size=opt.imgsz)
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.optimizer == 'Adam':
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
    optimizer = SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    del g0, g1, g2

    # Image size
    gs = 32  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = names
    model.stride = torch.tensor([8, 16, 32])

    model.to(device)
    ema = ModelEMA(model)
    scaler = amp.GradScaler(enabled=cuda)

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    elif opt.two_linear_lr:
        lf = two_linear(hyp['lrf'], epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    nbs = 64
    nb = len(trainset)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    best_fitness = 0.0
    last_opt_step = -1

    criterion = nn.CrossEntropyLoss()  # define loss function
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"{'epoch':10s}{'gpu_mem':10s}{'train_loss':12s}{'accuracy':12s}")
        mloss = 0.0  # mean loss
        model.train()
        pbar = tqdm(enumerate(trainset), total=len(trainset), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        # Test

        lab = []
        pre = []
        # test(model, valset, device, epoch, epochs, criterion, pbar=pbar, )  # test

        model.train()
        for i, (images, labels) in pbar:  # progress bar
            ni = i + nb * epoch  # number integrated batches (since train start)
            images, labels = images.to(device), labels.to(device)

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=False):
                pred = model(images)
                loss = criterion(pred, labels)

            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # computer ACC
            lab.append(labels)
            pre.append(torch.max(pred, 1)[1])
            correct = (torch.cat(lab) == torch.cat(pre)).float()

            # Print
            mloss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            pbar.desc = f"{'%s/%s' % (epoch + 1, epochs):10s}{mem:10s}{mloss / (i + 1):<12.3g}{correct.mean().item():<12.5g}"

        # write Log
        writer.add_scalar('train/ACC', correct.mean().item(), epoch)
        writer.add_scalar('train/Loss', mloss / (i + 1), epoch)
        fitness = correct.mean().item()
        fitness = test(model, valset, device, epoch, epochs, criterion, pbar=pbar, )  # test
        # Scheduler
        scheduler.step()

        # Best fitness
        if fitness > best_fitness:
            best_fitness = fitness

        # Save model
        final_epoch = epoch + 1 == epochs
        if (not opt.nosave) or final_epoch:
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    # 'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'date': datetime.now().isoformat()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fitness:
                torch.save(ckpt, best)
            del ckpt


def test(model, dataloader, device, epoch, epochs, criterion=None, verbose=False, pbar=None):
    model.eval()
    print(f"{'epoch':10s}{'gpu_mem':10s}{'val_loss':12s}{'accuracy':12s}")
    pred, targets, loss = [], [], 0
    with torch.no_grad():

        bar = tqdm(enumerate(dataloader), total=len(dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in bar:
            images, labels = images.to(device), labels.to(device)
            y = model(images)
            pred.append(torch.max(y, 1)[1])
            targets.append(labels)
            if criterion:
                loss += criterion(y, labels)

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            correct = (torch.cat(targets) == torch.cat(pred)).float()
            bar.desc = f"{'%s/%s' % (epoch + 1, epochs):10s}{mem:10s}{(loss / (i + 1)).item():<12.3g}{correct.mean().item():<12.5g}"

        # write Log
        writer.add_scalar('val/ACC', correct.mean().item(), epoch)
        writer.add_scalar('val/Loss', loss / (i + 1), epoch)
    return correct.mean().item()


def main(opt):
    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
        if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            hyp['anchors'] = 3

    save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    opt.save_dir = save_dir

    data = opt.data
    global writer
    writer = SummaryWriter(save_dir)

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Transforms
    trainform = T.Compose([
        T.RandomHorizontalFlip(p=hyp['fliplr']),
        T.RandomVerticalFlip(p=hyp['flipud']),
        T.RandomAffine(degrees=hyp['degrees'], translate=(hyp['translate'], hyp['translate']),
                       scale=(1 / 1.5, 1.5), shear=hyp['shear'], fill=(114, 114, 114)),
        T.ColorJitter(brightness=hyp['hsv_v'], saturation=hyp['hsv_s'], hue=hyp['hsv_h']),
        T.RandomGrayscale(p=0.01),
        T.Resize(size=(opt.imgsz, opt.imgsz)),
        T.ToTensor(),
    ])  # PILImage from [0, 1] to [-1, 1]
    testform = T.Compose([
        T.RandomGrayscale(p=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(20, fill=(144, 144, 144)),
        T.RandomAdjustSharpness(0.5),
        T.Resize(size=(opt.imgsz, opt.imgsz)),
        T.ToTensor(),
    ])  # PILImage from [0, 1] to [-1, 1]
    valform = T.Compose(trainform.transforms[-2:])

    # init dataset
    # train_data = ImageFolder(root=data['train'], transform=trainform)  # train data
    train_data = torchvision.datasets.CIFAR10(root='../datasets/', train=True, download=True, transform=trainform)
    trainset = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size, num_workers=opt.workers)

    # val_data = ImageFolder(root=data['val'], transform=valform)  # val data
    val_data = torchvision.datasets.CIFAR10(root='../datasets/', train=False, download=True, transform=valform)
    valset = DataLoader(val_data, shuffle=True, batch_size=opt.batch_size, num_workers=opt.workers)
    nc = data['nc']
    names = data['names']

    assert nc == len(train_data.classes), 'The nc in the yaml file should be equal to the number of dataset files'

    train(opt, hyp.copy(), trainset, valset, nc, names)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model path(s)')
    parser.add_argument('--cfg', type=str, default='models/classfily.yaml', help='initial weights path')
    parser.add_argument('--data', type=str, default=r'./data/class.yaml',
                        help='cifar10, cifar100, mnist or mnist-fashion')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=96, help='train, test image sizes (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--two-linear-lr', action='store_true', help='two linear LR scheduler')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    return opt


if '__main__' == __name__:
    opt = get_opt()
    main(opt)
