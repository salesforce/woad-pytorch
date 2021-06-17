from __future__ import print_function
import os
import torch
from model import WOAD
from video_dataset import Dataset
from test_full import test_full
from tensorboard_logger import Logger
import options
torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':

    args = options.parser.parse_args()
    device = torch.device("cuda")
    dataset = Dataset(args)
    if not os.path.exists('./ckpt/'):
       os.makedirs('./ckpt/')
    if not os.path.exists('./logs/' + args.model_name+'_eval'):
       os.makedirs('./logs/' + args.model_name+'_eval')
    logger = Logger('./logs/' + args.model_name+'_eval')
    model = WOAD(temp_window=args.temp_window, fusion_size=dataset.feature_size, num_classes=dataset.num_class,
                 hidden_size = args.hidden_size, enc_steps=args.enc_steps).to(device)

    model.load_state_dict(torch.load(args.pretrained_ckpt), strict = False)
    model.eval()
    test_full(0, dataset, args, model, logger, device, args.fps)
