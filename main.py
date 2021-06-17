from __future__ import print_function
import os
import torch
from model import WOAD
from video_dataset import Dataset
from test_train import test_train
from test_full import test_full
from train import train
from tensorboard_logger import Logger
import options
import torch.optim as optim
import random
import numpy as np
import math
torch.set_default_tensor_type('torch.cuda.FloatTensor')
def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def filter_wClsLabel(det_res, labels, trainidx):
    vnum = len(det_res)
    res = [[] for i in range(vnum)]
    for i in range(vnum):
        lb = labels[trainidx[i]]
        for j in range(len(det_res[i])):
            if j == 0: # video name
                continue
            if det_res[i][j][0] in lb:
                res[i].append(det_res[i][j])
    return res

def convert_labels(det_res, classlist, durations, fps, num_class, video_name,  trainidx, th=0.0):
    target = dict()
    targetAS = dict()
    for i in range(len(det_res)):
        working_idx = trainidx[i]
        vname = video_name[working_idx].decode('utf-8')
        vlen = math.ceil(durations[working_idx]*fps)
        target[vname]=np.zeros((vlen, num_class+1))
        targetAS[vname]=np.zeros((vlen, 1))
        target[vname][:, 0] = 1
        targetAS[vname][:, 0] = 0
        for j in range(len(det_res[i])):
            sc = det_res[i][j][-1]
            if sc < th:
                continue
            cls = classlist.index(det_res[i][j][0])+1
            st = det_res[i][j][1]
            ed = det_res[i][j][2]
            targetAS[vname][st]=1
            for ind in range(st, min(ed+1, vlen)):
                target[vname][ind, 0]=0
                target[vname][ind, cls]=1
    return target, targetAS

def convert_gt_labels(gt_seg, gt_seg_label, classlist, durations, fps, num_class, video_name,  trainidx):
    target = dict()
    targetAS = dict()
    for i in range(len(trainidx)):
        working_idx = trainidx[i]
        vname = video_name[working_idx].decode('utf-8')
        vlen = math.ceil(durations[working_idx]*fps)
        target[vname]=np.zeros((vlen, num_class+1))
        targetAS[vname]=np.zeros((vlen, 1))
        target[vname][:, 0] = 1
        targetAS[vname][:, 0] = 0
        for j in range(len(gt_seg[working_idx])):
            st = gt_seg[working_idx][j][0]
            ed = gt_seg[working_idx][j][1]
            clsname = gt_seg_label[working_idx][j]
            cls = classlist.index(clsname)+1
            st = int(st*fps)
            ed = int(ed*fps)
            if st >= vlen: ## some annotations is out of video length
                continue
            targetAS[vname][st]=1
            for ind in range(st, min(ed+1, vlen)):
                target[vname][ind, 0]=0
                target[vname][ind, cls]=1
    return target, targetAS

def mix_annotations(targets_noisy, targetsAS_noisy, targets_gt, targetsAS_gt, sup_inds):
    targets = dict()
    targetsAS = dict()
    vnames = []
    for i, vname in enumerate(targets_noisy):
        vnames.append(vname)
        if i in sup_inds:
            targets[vname] = targets_gt[vname]
            targetsAS[vname] = targetsAS_gt[vname]
        else:
            targets[vname] = targets_noisy[vname]
            targetsAS[vname] = targetsAS_noisy[vname]
    return targets, targetsAS 
if __name__ == '__main__':

    args = options.parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    dataset = Dataset(args)
    if not os.path.exists('./ckpt/'):
       os.makedirs('./ckpt/')
    if not os.path.exists('./logs/' + args.model_name):
       os.makedirs('./logs/' + args.model_name)
    logger = Logger('./logs/' + args.model_name)
    model = WOAD(temp_window=args.temp_window, fusion_size=dataset.feature_size, num_classes=dataset.num_class,
                 hidden_size = args.hidden_size, enc_steps=args.enc_steps).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    classlist = [dataset.classlist.tolist()[i].decode('utf-8') for i in range(len(dataset.classlist))]
    targets = None
    targetsAS = None
    perm = torch.randperm(len(dataset.trainidx))
    sup_num = int(len(perm)*args.sup_percent)
    sup_inds = perm[0:sup_num]
    warm_start_itern = args.warm_start
    val_interval = args.eval_intern
    for itr in range(args.max_iter+1):
       train(itr, dataset, args, model, optimizer, logger, device, targets=targets, targetsAS=targetsAS)
       if  itr == args.max_iter:
          torch.save(model.state_dict(), './ckpt/' + args.model_name+'_itr'+str(itr) + '.pkl')
       if itr % val_interval == 0 and itr > warm_start_itern:
          det_res, ap =  test_train(dataset, args, model, device, get_det_res=True) # get pseudo labels on train set
          print("detection ap on train set "+str(ap))

          if len(det_res) == 0: # skip training if no pseudo labels obtained
              targets = None
              targetsAS = None
              continue
          det_res = filter_wClsLabel(det_res, dataset.labels, dataset.trainidx) # filter p-labels using video-level labels

          if len(det_res) == 0: # skip training if no pseudo labels obtained
              targets = None
              targetsAS = None
              continue

          # get per-frame action & action start p-labels
          targets_noisy, targetsAS_noisy = convert_labels(det_res, classlist,dataset.video_duration, args.fps,
                  dataset.num_class, dataset.video_name, dataset.trainidx, th=args.confidence_thred)

          # get per-frame action & action start gt labels
          targets_gt, targetsAS_gt = convert_gt_labels(dataset.segments, dataset.seg_labels, classlist, dataset.video_duration, 
                  args.fps, dataset.num_class, dataset.video_name,  dataset.trainidx)

          if args.supervision == 'video':
              targets = targets_noisy
              targetsAS = targetsAS_noisy
          elif args.supervision == 'segment':

              # use mixed of p-labels and gt labels (semi-supervised)
              targets_mix, targetsAS_mix =  mix_annotations(targets_noisy, targetsAS_noisy, targets_gt, targetsAS_gt, sup_inds)
              perm_inside = torch.randperm(len(dataset.trainidx))
              keep_num = int(len(perm_inside)*args.mix_percent)
              keep_inds = perm_inside[0:keep_num]

              # use some p-labels for regularization
              targets, targetsAS =  mix_annotations(targets_noisy, targetsAS_noisy, targets_mix, targetsAS_mix, keep_inds)
          else:
              print("only video and segment are supported as a supervision")
              raise NotImplementedError
          test_full(itr, dataset, args, model, logger, device, args.fps)
