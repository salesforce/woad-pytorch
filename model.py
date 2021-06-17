import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data, gain=1)
            else:
                nn.init.normal_(param.data)

def fc_relu(in_features, out_features, inplace=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=inplace),
    )

class Model(torch.nn.Module):
    def __init__(self, n_feature, n_class):
        super(Model, self).__init__()

        self.fc = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, n_class)
        self.dropout = nn.Dropout(0.7)

    def forward(self, inputs, is_training=True):

        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)
        
        return x, self.classifier(x)

class WOAD(nn.Module):
    def __init__(self, enc_steps=64, fusion_size=2048, hidden_size=4096, num_classes=20, temp_window=4):
        super(WOAD, self).__init__()

        self.enc_steps = enc_steps
        self.fusion_size = fusion_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.temp_window = temp_window
        self.enc_drop = nn.Dropout(p=0.1)
        self.enc_lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.classifier_frame = nn.Linear(self.hidden_size, self.num_classes+1)
        self.mask = nn.Linear(self.hidden_size, 2)
        self.model = Model(fusion_size, num_classes)
        self.apply(weights_init)

    def encoder(self, fusion_input, enc_hx, enc_cx, temp_feats):
        enc_hx, enc_cx = \
                self.enc_lstm(self.enc_drop(fusion_input), (enc_hx, enc_cx))
        temp_feats_tmp = torch.zeros_like(temp_feats)
        temp_feats_tmp[:, 0:-1, :] = temp_feats[:,1:,:]
        temp_feats_tmp[:, -1, :] = enc_hx
        pooled_feat = torch.max(temp_feats_tmp, dim=1)[0]
        enc_score = self.classifier_frame(self.enc_drop(enc_hx))
        mask_score = self.mask(self.enc_drop(pooled_feat))
        return enc_score, enc_hx, enc_cx, mask_score, temp_feats_tmp

    def step(self, fusion_input, enc_hx, enc_cx, temp_feats):
        # for evaluation, one frame a time
        features, _ = self.model(fusion_input, is_training=False)
        enc_score, enc_hx, enc_cx, mask_score, temp_feats = \
                self.encoder(features, enc_hx, enc_cx, temp_feats)

        return enc_score, enc_hx, enc_cx, mask_score, temp_feats

    def forward(self, fusion_inputs, device='cuda', is_training=True):
        features, vl_preds = self.model(fusion_inputs, is_training=is_training)
        if is_training:
            batch_size = features.shape[0]
            vlen = features.shape[1]
            feat_dim = features.shape[2]
        else:
            return features, vl_preds
        lstm_batch = vlen // self.enc_steps
        features_lstm = torch.zeros((batch_size, lstm_batch, self.enc_steps, feat_dim)).to(device)
        for i in range(0, lstm_batch):
            features_lstm[:, i, :, :] = features[:, (i*self.enc_steps):((i+1)*self.enc_steps), :]
        features_lstm = features_lstm.view((batch_size*lstm_batch, self.enc_steps, feat_dim))
        features_lstm = features_lstm.transpose(0, 1)        

        enc_hx = torch.zeros((batch_size*lstm_batch, self.hidden_size)).to(device)
        enc_cx = torch.zeros((batch_size*lstm_batch, self.hidden_size)).to(device)
        enc_score_stack = []
        mask_score_stack = []
        temp_feats = torch.zeros((batch_size*lstm_batch, self.temp_window, self.hidden_size)).to(device)
        # Encoder
        for enc_step in range(self.enc_steps):
            enc_score, enc_hx, enc_cx, mask_score, temp_feats = self.encoder(
                features_lstm[enc_step],
                enc_hx, enc_cx, temp_feats,
            )
            enc_score_stack.append(enc_score)
            mask_score_stack.append(mask_score)
        enc_scores = torch.stack(enc_score_stack).view(-1, self.num_classes+1)
        mask_scores = torch.stack(mask_score_stack).view(-1, 2)
        return enc_scores, mask_scores, features, vl_preds

