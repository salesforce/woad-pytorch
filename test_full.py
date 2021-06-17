import os
import torch
import torch.nn as nn
import numpy as np
from utils import getASfromCAS, compute_PAP_result_thumos14
from utils import compute_FAP_result
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def to_device(x, device):
    return x.unsqueeze(0).to(device)
def create_target(gt_seg, gt_seg_label, classnum, idx, vlen, classlist, fps):
    target = np.zeros((vlen, classnum))
    for i, seg in enumerate(gt_seg[idx]):
        st = int(fps*seg[0])
        ed = int(fps*seg[1])
        cls = classlist.index(gt_seg_label[idx][i])+1
        target[st:ed+1, cls] = 1
    bg_inds = np.where(np.sum(target, axis=-1)==0)[0]
    target[bg_inds, 0] = 1        
    return target 
def test_full(itr, dataset, args, model, logger, device, fps):
    softmax = nn.Softmax(dim=-1).to(device)
    scores = []    
    scores_metrics = []    
    target_metrics = []
    videoIds = []
    gt_seg = dataset.segments
    gt_seg_label = dataset.seg_labels
    classlist = [dataset.classlist.tolist()[i].decode('utf-8') for i in range(len(dataset.classlist))]
    classnum=dataset.num_class+1
    done = False
    while not done:
        if dataset.currenttestidx % 100 ==0:
            print('Testing test data point %d of %d' %(dataset.currenttestidx, len(dataset.testidx)))
        working_idx = dataset.testidx[dataset.currenttestidx]
        vname = dataset.video_name[working_idx].decode('utf-8')
        features, labels, done = dataset.load_data(is_training=False)

        vid = vname.split('_')[-1]
        videoIds.extend([int(vid) for i in range(features.shape[0])])

        features = torch.from_numpy(features).float().to(device)
        target = create_target(gt_seg, gt_seg_label, classnum, working_idx, features.shape[0], classlist, fps)
        enc_hx = to_device(torch.zeros(model.hidden_size), device)
        enc_cx = to_device(torch.zeros(model.hidden_size), device)
        temp_feats = torch.zeros((1, model.temp_window, model.hidden_size)).to(device)
        element_logits = []
        for l in range(0, features.shape[0]):
            with torch.no_grad():
                feat_input = to_device(features[l], device)
                enc_score, enc_hx, enc_cx, mask_score, temp_feats = \
                        model.step(feat_input, enc_hx, enc_cx, temp_feats)
            element_logits.append(enc_score.cpu().numpy()[0])
            enc_score = softmax(enc_score).cpu().numpy()[0]
            mask_score = softmax(mask_score).cpu().numpy()[0]
            scores_metrics.append(enc_score) # per frame action score
            target_metrics.append(target[l]) # per frame action label
            start_score = np.zeros_like(enc_score)
            start_score[0] = enc_score[0]*mask_score[0]
            start_score[1:] = enc_score[1:]*mask_score[1]
            scores.append(start_score) # action start score

    frameScores = np.array(scores)
    dist_ths = [1.0, 2.0, 3.0, 4.0 ,5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    GTs = np.load(os.path.join(dataset.path_to_annotations, 'TH14_ASlbs_test.npy')).tolist() # action start label
    scores, times, videoLen = getASfromCAS(frameScores, videoIds, fps) # generate action starts
    for dist_th in dist_ths:
        result_point = compute_PAP_result_thumos14(GTs, videoLen, scores, times, videoIds, dist_th, classnum, ignore=[0])
        print('Test point mAP @ dist_th = ' + str(dist_th), result_point['mAP'])
        logger.log_value('Test point mAP @ dist_th = ' + str(dist_th), result_point['mAP'], itr)

    result_frame = compute_FAP_result(classnum, scores_metrics, target_metrics, ignore_class=[0])
    print('Test frame mAP', result_frame['mAP'])
    logger.log_value('Test frame mAP', result_frame['mAP'], itr)
