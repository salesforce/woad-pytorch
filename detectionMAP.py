import numpy as np

def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]

def smooth(v):
   return v

def filter_segments(segment_predict, videonames, ambilist, factor):
   ind = np.zeros(np.shape(segment_predict)[0])
   for i in range(np.shape(segment_predict)[0]):
      vn = videonames[int(segment_predict[i,0])]
      for a in ambilist:
         if a[0]==vn:
            gt = range(int(round(float(a[2])*factor)), int(round(float(a[3])*factor)))
            pd = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
            IoU = float(len(set(gt).intersection(set(pd))))/float(len(set(gt).union(set(pd))))
            if IoU > 0:
               ind[i] = 1
   s = [segment_predict[i,:] for i in range(np.shape(segment_predict)[0]) if ind[i]==0]
   return np.array(s)

def getLocMAP(predictions, th, dataset, eval_set = 'test', get_det_res=False):

   gtsegments = dataset.segments
   gtlabels = dataset.seg_labels
   videoname = dataset.video_name; videoname = np.array([v.decode('utf-8') for v in videoname])
   subset = dataset.subset; subset = np.array([s.decode('utf-8') for s in subset])
   classlist = dataset.classlist; classlist = np.array([c.decode('utf-8') for c in classlist])
   duration = dataset.video_duration
   ambilist = dataset.path_to_annotations+'/Ambiguous_test.txt'
   factor = 25.0/16.0

   ambilist = list(open(ambilist,'r'))
   ambilist = [a.strip('\n').split(' ') for a in ambilist]
   
   # Keep only the test subset annotations
   gts, gtl, vn, dn = [], [], [], []
   for i, s in enumerate(subset):
      if subset[i]==eval_set:
         gts.append(gtsegments[i])
         gtl.append(gtlabels[i])
         vn.append(videoname[i])
         dn.append(duration[i,0])
   gtsegments = gts
   gtlabels = gtl
   videoname = vn
   duration = dn

   # keep ground truth and predictions for instances with temporal annotations
   gts, gtl, vn, pred, dn = [], [], [], [], []
   for i, s in enumerate(gtsegments):
      if len(s):
         gts.append(gtsegments[i])
         gtl.append(gtlabels[i])
         vn.append(videoname[i])
         pred.append(predictions[i])
         dn.append(duration[i])
   gtsegments = gts
   gtlabels = gtl
   videoname = vn
   predictions = pred

   # which categories have temporal labels ?
   templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

   # the number index for those categories.
   templabelidx = []
   for t in templabelcategories:
      templabelidx.append(str2ind(t,classlist))
             
   # process the predictions such that classes having greater than a certain threshold are detected only
   predictions_mod = []
   c_score = []

   for p in predictions:
      pp = - p; [pp[:,i].sort() for i in range(np.shape(pp)[1])]; pp=-pp
      c_s = np.mean(pp[:int(np.shape(pp)[0]/8),:],axis=0)
      ind = c_s > 0.0
      c_score.append(c_s)
      predictions_mod.append(p*ind)
   predictions = predictions_mod
   detection_results = []
   for i,vn in enumerate(videoname):
      detection_results.append([])
      detection_results[i].append(vn)

   ap = []
   for c in templabelidx:
      segment_predict = []
      # Get list of all predictions for class c
      for i in range(len(predictions)):
         tmp = smooth(predictions[i][:,c])
         threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*0.5
         vid_pred = np.concatenate([np.zeros(1),(tmp>threshold).astype('float32'),np.zeros(1)], axis=0)
         vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
         s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
         e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
         for j in range(len(s)):
            if e[j]-s[j]>=2:               
               segment_predict.append([i,s[j],e[j],np.max(tmp[s[j]:e[j]])+0.7*c_score[i][c]])
               detection_results[i].append([classlist[c], s[j], e[j], np.max(tmp[s[j]:e[j]])+0.7*c_score[i][c]])
      segment_predict = np.array(segment_predict)
      segment_predict = filter_segments(segment_predict, videoname, ambilist, factor)
   
      # Sort the list of predictions for class c based on score
      if len(segment_predict) == 0:
         if get_det_res:
             return [], 0.0
         else:
             return 0.0
      segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]

      # Create gt list 
      segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments)) for j in range(len(gtsegments[i])) if str2ind(gtlabels[i][j],classlist)==c]
      gtpos = len(segment_gt)
      # Compare predictions and gt
      tp, fp = [], []
      for i in range(len(segment_predict)):
         flag = 0.
         for j in range(len(segment_gt)):
            if segment_predict[i][0]==segment_gt[j][0]:
               gt = range(int(round(segment_gt[j][1]*factor)), int(round(segment_gt[j][2]*factor)))
               p = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
               IoU = float(len(set(gt).intersection(set(p))))/float(len(set(gt).union(set(p))))
               if IoU >= th:
                  flag = 1.
                  del segment_gt[j]
                  break
         tp.append(flag)
         fp.append(1.-flag)
      tp_c = np.cumsum(tp)
      fp_c = np.cumsum(fp)
      if sum(tp)==0:
         prc = 0.
      else:
         prc = np.sum((tp_c/(fp_c+tp_c))*tp)/gtpos
      ap.append(prc)
   if get_det_res:
       return detection_results, 100*np.mean(ap)
   else:
       return 100*np.mean(ap)

def getDetectionMAP(predictions, dataset, eval_set='test', get_det_res=False):
   if get_det_res:
       return getLocMAP(predictions, 0.5, dataset, eval_set = eval_set, get_det_res=True)
   iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
   dmap_list = []
   for iou in iou_list:
      print('Testing for IoU %f' %iou)
      dmap_list.append(getLocMAP(predictions, iou, dataset, eval_set = eval_set))
   return dmap_list, iou_list

