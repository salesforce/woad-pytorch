import numpy as np
from collections import OrderedDict
from sklearn.metrics import average_precision_score

def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i].decode('utf-8')][0]

def strlist2indlist(strlist, classlist):
	return [str2ind(s,classlist) for s in strlist]

def strlist2multihot(strlist, classlist):
	return np.sum(np.eye(len(classlist))[strlist2indlist(strlist,classlist)], axis=0)

def idx2multihot(id_list,num_class):
   return np.sum(np.eye(num_class)[id_list], axis=0)

def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   res = feat[r:r+t_max]
   return res, r, r+t_max

def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0,min_len-np.shape(feat)[0]), (0,0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length):
    if len(feat) > length:
        res, st, ed = random_extract(feat, length)
        return res, st, ed
    else:
        return pad(feat, length), 0, length

def compute_FAP_result(num_classes, score_metrics, target_metrics, ignore_class=[], verbose=False):
    result = OrderedDict()
    score_metrics = np.array(score_metrics)
    target_metrics = np.array(target_metrics)
    # Compute AP
    result['AP'] = OrderedDict()
    for cls in range(num_classes):
        if cls not in ignore_class:
            result['AP'][cls] = average_precision_score(
                (target_metrics[:, cls]==1).astype(np.int),
                score_metrics[:, cls])

    # Compute mAP
    result['mAP'] = np.mean(list(result['AP'].values()))
    if verbose:
        print('mAP: {:.5f}'.format(result['mAP']))


    return result

def voc_ap(rec, prec, use_07_metric=True, rec_th=1.0):
    """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., rec_th+rec_th/10.0, rec_th/10.0):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    return ap
def point_average_precision(GTs, videoLen, cls, scores, times, videoIds, dist_th, rec_th =1.0):
    """
    inputs:
    GTs is a dictionary (GTs[videoIds][cls] = [AS1, AS2, AS3])
    Note GTs[videoIds][0] is for ambiguous class which is ignored
    videoLen is a dictionary recording video length in seconds
    class of interest
    CAS for all the classes
    times per-frame times in seconds
    videoIds is the video id of the corresponding confidence and times
    """
    npos = 0
    R = dict()
    for k, v in enumerate(GTs):
        posct = 0
        for ct in range(len(GTs[v][cls])):
            if v == 'video_test_0001292': #ignore videos contain only ambiguous class
                continue
            if GTs[v][cls][ct] <= videoLen[v]:
               posct += 1
        npos += posct
        R[v] = [0 for _ in range(len(GTs[v][cls]))]
    confidence = scores[:,cls]
    sorted_ind = np.argsort(-confidence)

    times = times[sorted_ind]
    videoIds = ['video_test_'+str(int(videoIds[x])).zfill(7) for x in sorted_ind]
    nd = len(videoIds)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        ASs = np.array(GTs[videoIds[d]][cls]).astype(float)
        time = times[d].astype(float)
        dist_min = np.inf
        if len(ASs) > 0:
            # compute absolute distance
            dists = np.abs(time - ASs)
            dist_min = np.min(dists)
            jmin = np.argmin(dists)
        if dist_min <= dist_th:
            if R[videoIds[d]][jmin] == 0:
                tp[d] = 1.
                R[videoIds[d]][jmin] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, True, rec_th)
    return rec, prec, ap
def compute_PAP_result_thumos14(GTs, videoLen, Scores, times, videoIds, dist_th, classnum, ignore=[], rec_th=1.0):
    result = OrderedDict()
    result['pointAP'] = OrderedDict()
    result['mAP'] = OrderedDict()
    for i in range(classnum):
        if not i in ignore:
            rec,prec,result['pointAP'][i] = point_average_precision(GTs, videoLen, i, Scores, times, videoIds, dist_th, rec_th)
    result['mAP'] = np.mean(list(result['pointAP'].values()))
    return result

def getASfromCAS(frameScores, videoIds, fps):
    '''
       inputs: per-frame scores for all classes (N, class_num);
       corresponding per-frame video Ids;
       fps: frames per second

       outputs: action start scores for all classes (N, class_num);
       corresponding per-frame times in second at its videos;
       length of each video;
    '''
    scores = np.zeros(frameScores.shape)
    times = np.zeros(frameScores.shape[0])
    videoLen = dict()
    # get action starts from CAS
    # 1) c_{t-1} neq c_t
    # 2) pred action at t is non-background
    # 3) if 1)&2) hold set action start prob = action prob at t
    # 4) otherwise action prob = 0
    for i in range(0, frameScores.shape[0]):
        if i == 0:
            cprev = 0
        else:
            cprev = np.argmax(frameScores[i-1, :])
        ccurr = np.argmax(frameScores[i, :])
        if cprev != ccurr and ccurr != 0:
            scores[i, ccurr] = frameScores[i, ccurr]
    previd = videoIds[0]
    counter = 0
    for i in range(0, times.shape[0]):
        currid = videoIds[i]
        if currid != previd:
            counter = 0
            previd = currid
            videoLen['video_test_'+str(int(videoIds[i-1])).zfill(7)] = times[i-1]
        times[i] = counter*1.0/fps
        counter += 1
    # add the last one
    videoLen['video_test_'+str(int(videoIds[-1])).zfill(7)] = times[-1]
    return scores, times, videoLen

