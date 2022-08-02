from __future__ import print_function
import argparse
from os.path import exists
from os import makedirs

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import UNet
from blindspot import MakeBlindSpot, BlindSpotLoss
from data import get_training_set, get_test_set

import time

if __name__ == '__main__':
    # measure time
    start_time = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--batchSize', type=int, default=30, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--loops', type=int, default=8, help='number of iteration. Default=100')
    parser.add_argument('--scale', type=int, default=2, help='resolution scale. Default=2')
    opt = parser.parse_args()

    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    train_set = get_training_set(opt.scale)
    test_set = get_test_set(opt.scale)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)


    print('===> Building model')
    model = UNet()
    model_out_path = "model/2x_Unet_model_epoch_{}.pth".format(0)
    torch.save(model, model_out_path)
    criterion = nn.MSELoss()
    print(model)
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()


    def train(epoch):
        number_of_loops = opt.loops
        print('===> Training started')
        for i in range(number_of_loops):
            epoch_loss = 0
            for iteration, batch in enumerate(training_data_loader, 1):
                HR_2, HR_2_bs, HR_4_target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

                if cuda:
                    HR_2 = HR_2.cuda()
                    HR_2_bs = HR_2_bs.cuda()
                    HR_4_target = HR_4_target.cuda()

                optimizer.zero_grad()
                HR_2_bs, rand_x_LR, rand_y_LR = MakeBlindSpot(HR_2_bs)
                HR_4_bs = model(HR_2_bs)

                with torch.no_grad():
                    HR_4 = model(HR_2)

                weight2 = 1
                alpha = 1

                # compute blind-spot loss and resize loss for the final level
                loss2_resize = criterion(HR_4, HR_4_target)
                loss2_blind_spot = BlindSpotLoss(HR_4_bs, HR_4_target, rand_x_LR, rand_y_LR, scale=2)
                loss2_constraint = criterion(HR_4_bs, HR_4)

                loss = (loss2_blind_spot + loss2_constraint * alpha) + weight2 * loss2_resize

                epoch_loss += loss.data
                loss.backward()
                optimizer.step()

            print("===> Epoch {}, Loop{}: Avg. Loss: {:.4f}".format(epoch, i, epoch_loss / len(training_data_loader)))


    def checkpoint(epoch):
        if not exists(opt.checkpoint):
            makedirs(opt.checkpoint)
        model_out_path = "model/2x_Unet_model_epoch_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))


    lr = opt.lr
    for epoch in range(1, opt.nEpochs + 1):

        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-5)
        train(epoch)
        if epoch % 10 == 0:
            lr = lr / 2
        checkpoint(epoch)

    print("--- %s seconds ---" % (time.time() - start_time))