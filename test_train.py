import torch
from torch.autograd import Variable
from detectionMAP import getDetectionMAP as dmAP
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def test_train(dataset, args, model, device, get_det_res):
    
    done = False
    element_logits_stack = []

    while not done:
        features, labels, done = dataset.load_data(is_training=False, is_testing_on_train=True)
        features = torch.from_numpy(features).float().to(device)

        with torch.no_grad():
          _, element_logits = model(Variable(features), is_training=False)

        element_logits = element_logits.cpu().data.numpy()

        element_logits_stack.append(element_logits)

    if get_det_res:
        return dmAP(element_logits_stack, dataset, eval_set='validation', get_det_res=get_det_res)

