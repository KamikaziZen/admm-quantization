import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm


# class DiceLoss(nn.Module):

#     def __init__(self):
#         super(DiceLoss, self).__init__()
#         self.smooth = 1.0

#     def forward(self, y_pred, y_true):
#         assert y_pred.size() == y_true.size()
#         y_pred = y_pred[:, 0].contiguous().view(-1)
#         y_true = y_true[:, 0].contiguous().view(-1)
#         intersection = (y_pred * y_true).sum()
#         dsc = (2. * intersection + self.smooth) / (
#             y_pred.sum() + y_true.sum() + self.smooth
#         )
#         return 1. - dsc


def accuracy(model, dataset_loader, device='cuda', num_classes=1000):
    def one_hot(x, K):
        return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
    
    # Set BN and Droupout to eval regime
    model.eval()

    total_correct = 0

    for (x, y) in tqdm(dataset_loader):
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), num_classes)
        target_class = np.argmax(y, axis=1)

        with torch.no_grad():
            out = model(x).cpu().detach().numpy()
            predicted_class = np.argmax(out, axis=1)
            total_correct += np.sum(predicted_class == target_class)

    total = len(dataset_loader) * dataset_loader.batch_size
    return total_correct / total


def accuracy_top1top5(model, ds, n_sample=None, ngpu=1, device='cuda'):

    correct1, correct5 = 0, 0
    n_passed = 0
    # Set BN and Droupout to eval regime
    model.eval()
    
    model = torch.nn.DataParallel(model, device_ids=range(ngpu)).to(device)

    for data, target in tqdm(ds):
        n_passed += len(data)
        data = Variable(torch.FloatTensor(data)).to(device)
        indx_target = torch.LongTensor(target)
        output = model(data)
        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()

        if n_sample and n_passed >= n_sample:
            break
    print('samples evaluated:', n_passed)
            
    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5