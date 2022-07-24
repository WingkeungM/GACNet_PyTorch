import os
import time
import torch
import logging
import warnings
import datetime
import argparse
import torch.utils.data
import torch.nn.parallel

import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.autograd import Variable

from util import transform
from data.S3DIS.S3DISDataLoader import S3DISDataset
from model.gacnet import GACNet, build_graph_pyramid




warnings.filterwarnings("ignore")

classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

graph_inf = {'stride_list': [1024, 256, 64, 32], #can be seen as the downsampling rate
             'radius_list': [0.1, 0.2, 0.4, 0.8, 1.6], # radius for neighbor points searching
             'maxsample_list': [12, 21, 21, 21, 12] #number of neighbor points for each layer
}

# number of units for each mlp layer
forward_parm = [
                [ [32,32,64], [64] ],
                [ [64,64,128], [128] ],
                [ [128,128,256], [256] ],
                [ [256,256,512], [512] ],
                [ [256,256], [256] ]
]

# for feature interpolation stage
upsample_parm = [
                  [128, 128],
                  [128, 128],
                  [256, 256],
                  [256, 256]
]

# parameters for fully connection layer
fullconect_parm = 128

net_inf = {'forward_parm': forward_parm,
           'upsample_parm': upsample_parm,
           'fullconect_parm': fullconect_parm
}



def parse_args():
    parser = argparse.ArgumentParser('GACNet')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')

    parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training [default: 200]')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers [default: 4]')
    parser.add_argument('--batchSize', type=int, default=2, help='input batch size [default: 24]')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for leakyRelu [default: 0.2]')
    parser.add_argument('--dropout', type=float, default=0, help='dropout [default: 0]')

    parser.add_argument('--optimizer', type=str, default='SGD', help='type of optimizer')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay for Adam')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training [default: 0.001 for Adam, 0.01 for SGD]')

    parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--log_dir', type=str, default='logs_gacnet/',help='decay rate of learning rate')

    parser.add_argument('--datapath', type=str, default='/workspace/VisualPointCloud/testgacnet', help='path of the dataset')
    parser.add_argument('--numpoint', type=int, default=4096, help='number of point for input [default: 4096]')
    parser.add_argument('--numclass', type=int, default=13, help='class number of the dataset [default: 13]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/'+ str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + args.log_dir))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger('GACNet')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_gacnet.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('PARAMETER ...')
    logger.info(args)
    print('Load data...')

    train_transform = transform.Compose([transform.ToTensor()])
    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=args.datapath, num_point=args.numpoint, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    dataloader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batchSize, shuffle=True, num_workers=args.workers,
                                             pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))

    val_transform = transform.Compose([transform.ToTensor()])
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=args.datapath, num_point=args.numpoint, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    testdataloader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = GACNet(args.numclass, graph_inf, net_inf)

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    pretrain = args.pretrain
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0

    def adjust_learning_rate(optimizer, step):
        """Sets the learning rate to the initial LR decayed by 30 every 20000 steps"""
        lr = args.learning_rate * (0.3 ** (step // 20000))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model.cuda()

    history = defaultdict(lambda: list())
    best_acc = 0
    best_meaniou = 0
    step = 0

    for epoch in range(init_epoch,args.epoch):
        for i, data in tqdm(enumerate(dataloader, 0),total=len(dataloader),smoothing=0.9):
            points, target = data
            points, target = Variable(points.float()), Variable(target.long())
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            model = model.train()

            graph_prd, coarse_map = build_graph_pyramid(points[:, :, 0:3], graph_inf)
            pred = model(points[:, :, :6], graph_prd, coarse_map)

            pred = pred.contiguous().view(-1, args.numclass)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target, weight=weights)
            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            step += 1
            adjust_learning_rate(optimizer, step)

        test_metrics, test_hist_acc, cat_mean_iou = test_seg(model, testdataloader, seg_label_to_cat)
        mean_iou = np.mean(cat_mean_iou)

        print('Epoch %d  %s accuracy: %f  meanIOU: %f' % (
                 epoch, blue('test'), test_metrics['accuracy'],mean_iou))
        logger.info('Epoch %d  %s accuracy: %f  meanIOU: %f' % (
                 epoch, 'test', test_metrics['accuracy'],mean_iou))
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            torch.save(model.state_dict(), '%s/GACNet_%.3d_%.4f_%.4f.pth' % (checkpoints_dir, epoch, best_acc, best_meaniou))
            logger.info(cat_mean_iou)
            logger.info('Save model..')
            print('Save model..')
            print(cat_mean_iou)
        if mean_iou > best_meaniou:
            best_meaniou = mean_iou
            torch.save(model.state_dict(), '%s/GACNet_%.3d_%.4f_%.4f.pth' % (checkpoints_dir, epoch, best_acc, best_meaniou))
            logger.info(cat_mean_iou)
            logger.info('Save model..')
            print('Save model..')
            print(cat_mean_iou)
        print('Best accuracy is: %.5f'%best_acc)
        logger.info('Best accuracy is: %.5f'%best_acc)
        print('Best meanIOU is: %.5f'%best_meaniou)
        logger.info('Best meanIOU is: %.5f'%best_meaniou)

def test_seg(model, loader, catdict, num_classes = 13):
    iou_tabel = np.zeros((len(catdict),3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()

        points, target = Variable(points.float()), Variable(target.long())

        points, target = points.cuda(), target.cuda()

        graph_prd, coarse_map = build_graph_pyramid(points[:, :, 0:3], graph_inf)
        pred = model(points[:, :, :6], graph_prd, coarse_map)

        mean_iou, iou_tabel = compute_iou(pred, target, iou_tabel)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
        metrics['iou'].append(mean_iou)
    iou_tabel[:,2] = iou_tabel[:,0] /(iou_tabel[:,1]+0.01)
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [cat_value for cat_value in catdict.values()]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou

def compute_iou(pred,target,iou_tabel=None):
    ious = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]
        batch_target = target[j]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
        for cat in np.unique(batch_target):
            intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            iou = intersection/union
            ious.append(iou)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
    return np.mean(ious), iou_tabel

if __name__ == '__main__':
    args = parse_args()
    main(args)
