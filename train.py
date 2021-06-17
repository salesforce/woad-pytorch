import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:,notice_index]*logsoftmax(input[:,notice_index]), 1)
            return torch.mean(output[target[:,self.ignore_index]!=1])
        else:
            output = torch.sum(-target*logsoftmax(input), 1)
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, weights=None):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        alpha = alpha.view(-1, 1)
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        if not weights is None:
            batch_loss = batch_loss*weights.unsqueeze(1)
        if self.size_average:
            loss = batch_loss.sum()/alpha.sum()
        else:
            loss = batch_loss.sum()
        return loss

def MILL(element_logits, seq_len, batch_size, labels, device):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over, 
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''

    k = np.ceil(seq_len/8).astype('int32')
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CASL(x, element_logits, seq_len, n_similar, labels, device):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    sim_loss = 0.
    n_tmp = 0.
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]], dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), (1 - atn2)/n2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))

        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        n_tmp = n_tmp + torch.sum(Variable(labels[i,:])*Variable(labels[i+1,:]))
    sim_loss = sim_loss / n_tmp
    return sim_loss


def train(itr, dataset, args, model, optimizer, logger, device, targets=None, targetsAS=None):

    features, feat_sts, feat_eds, labels, vname = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    if len(np.where(seq_len==0)[0]) > 0:
        return
    features = features[:,:np.max(seq_len),:]
    feat_eds = feat_sts + np.max(seq_len)
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    scores, st_scores, final_features, element_logits = model(Variable(features))
    milloss = MILL(element_logits, seq_len, args.batch_size, labels, device)
    casloss = CASL(final_features, element_logits, seq_len, args.num_similar, labels, device)
    if targets is None:
        total_loss = args.Lambda * milloss + args.Lambda * casloss
        logger.log_value('milloss', milloss, itr)
        logger.log_value('casloss', casloss, itr)
        logger.log_value('total_loss', total_loss, itr)
        print('Iteration: %d, total Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))
    else:
        target_proc = []
        targetAS_proc = []
        for i, vnm in enumerate(vname):
            ed = feat_eds[i]
            st = feat_sts[i]
            ed_o = targets[vnm].shape[0]
            target = utils.pad(targets[vnm], ed)
            targetAS = utils.pad(targetsAS[vnm], ed)
            target[ed_o:ed, 0] = 1
            target_proc.append(target[st:ed, :])
            targetAS_proc.append(targetAS[st:ed])
        target_proc = np.array(target_proc)
        targetAS_proc = np.array(targetAS_proc)
        batch_size = target_proc.shape[0]
        vlen = target_proc.shape[1]
        lstm_batch = vlen//model.enc_steps
        target_lstm = np.zeros((batch_size, lstm_batch, model.enc_steps, target_proc.shape[-1]))
        targetAS_lstm = np.zeros((batch_size, lstm_batch, model.enc_steps, 1))
        for i in range(0, lstm_batch):
            target_lstm[:, i, :, :] = target_proc[:, i*model.enc_steps:(i+1)*model.enc_steps, :]
            targetAS_lstm[:, i, :, :] = targetAS_proc[:, i*model.enc_steps:(i+1)*model.enc_steps, :]
        target_lstm = torch.from_numpy(target_lstm).to(device)
        targetAS_lstm = torch.from_numpy(targetAS_lstm).to(device)
        target_lstm = target_lstm.view((batch_size*lstm_batch, model.enc_steps, -1))
        targetAS_lstm = targetAS_lstm.view((batch_size*lstm_batch, model.enc_steps, -1))

        target_lstm = target_lstm.transpose(0, 1)
        targetAS_lstm = targetAS_lstm.transpose(0, 1)
        target_lstm = target_lstm.contiguous().view((-1, target_lstm.shape[-1]))
        targetAS_lstm = targetAS_lstm.contiguous().view((-1, 1)).squeeze(1)
        target_lstm = target_lstm.float()
        targetAS_lstm = targetAS_lstm.long()
        #start loss only on pos/neg balanced samples
        pos_inds = (targetAS_lstm.squeeze() != 0).nonzero().squeeze()
        neg_inds = (targetAS_lstm.squeeze() == 0).nonzero().squeeze()
        perm = torch.randperm(len(neg_inds))
        if pos_inds.nelement() == 1:
            pos_inds = torch.tensor([pos_inds])
        if pos_inds.nelement() == 0:
            pos_num = 0
            neg_num = 100
        else:
            pos_num = len(pos_inds)
            neg_num = 3*pos_num
        sample_neg = perm[0:min(neg_num, len(perm))]
        neg_inds = neg_inds[sample_neg]
        train_inds = torch.cat((pos_inds, neg_inds), 0)

        criterion_frame = MultiCrossEntropyLoss().to(device)
        loss_frame = criterion_frame(scores, target_lstm)

        criterion_start = FocalLoss(class_num=2).to(device) # when only on pos/neg balanced samples
        loss_start = criterion_start(st_scores[train_inds, :], targetAS_lstm[train_inds])

        total_loss = loss_frame+loss_start+args.Lambda * milloss + args.Lambda * casloss
        logger.log_value('frameloss', loss_frame, itr)
        logger.log_value('startloss', loss_start, itr)
        logger.log_value('milloss', milloss, itr)
        logger.log_value('casloss', casloss, itr)
        logger.log_value('total_loss', total_loss, itr)
        print('Iteration: %d, total Loss: %.3f, frame loss: %.3f, start loss: %.3f' %(itr, total_loss.data.cpu().numpy(), loss_frame.data.cpu().numpy(), loss_start.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

