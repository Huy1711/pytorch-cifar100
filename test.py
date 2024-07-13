#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

import editdistance
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_test_dataloader, decode_cangjie

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

    test_path = "data/etl_952_singlechar_size_64/952_test"
    test_loader = get_test_dataloader(
        path=test_path,
        num_workers=4,
        batch_size=args.b,
        shuffle=False
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    total_edit_distance = 0.0
    total_cangjie_label_length = 0.0

    with torch.no_grad():
        for n_iter, (image, label, _, _, cangjie_raws) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


            cls_out, ctc_out = net(image)
            ctc_preds_size = Variable(torch.IntTensor([ctc_out.size(0)] * ctc_out.size(1)))
            _, pred = cls_out.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()
            #compute top1
            correct_1 += correct[:, :1].sum()


            # ctc preds
            _, ctc_preds = ctc_out.max(2)
            ctc_preds = ctc_preds.transpose(1, 0).contiguous().view(-1)
            sim_ctc_preds = decode_cangjie(ctc_preds.data, ctc_preds_size.data, raw=False)
            for pred, label in zip(sim_ctc_preds, cangjie_raws):
                total_edit_distance += editdistance.eval(pred, label)
                total_cangjie_label_length += len(label)

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
