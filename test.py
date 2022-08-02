
from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from pylab import rcParams
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
from math import sqrt
import torch.nn as nn
import pytorch_ssim

if __name__ == '__main__':
    rcParams['figure.figsize'] = 40, 24
    rcParams.update({'font.size': 22})

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--test_folder', type=str, default='./dataset/w2s_avg1/test', help='input image to use')
    parser.add_argument('--gt_folder', type=str, default='./dataset/w2s_avg400/test', help='input image to use')
    parser.add_argument('--model', type=str, default='model/pretrained_2x_Unet.pth', help='model file to use')
    parser.add_argument('--save_folder', type=str, default='./results', help='input image to use')
    parser.add_argument('--output_filename', type=str, help='where to save the output image')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
    parser.add_argument('--scale', type=int, default=2, help='resolution scale. Default=2')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')

    opt = parser.parse_args()

    print(opt)
    cuda = opt.cuda


    def test():
        avg_rmse2 = 0
        avg_ssim2 = 0

        sd_rmse2 = 0
        sd_ssim2 = 0

        # Sample count
        k = 0

        for batch in testing_data_loader:
            HR_2, HR_4_target, HR_4_avg400 = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

            if cuda:
                HR_2 = HR_2.cuda()
                HR_4_target = HR_4_target.cuda()
                HR_4_avg400 = HR_4_avg400.cuda()

            HR_4 = model(HR_2)

            mse2 = criterion(HR_4, HR_4_avg400)

            k = k + 1
            old_avg_rmse2 = avg_rmse2
            old_avg_ssim2 = avg_ssim2

            avg_rmse2 += (sqrt(mse2) - avg_rmse2) / k
            avg_ssim2 += (pytorch_ssim.ssim(HR_4, HR_4_avg400) - avg_ssim2) / k

            sd_rmse2 += (sqrt(mse2) - avg_rmse2) * (sqrt(mse2) - old_avg_rmse2)
            sd_ssim2 += (pytorch_ssim.ssim(HR_4, HR_4_avg400) - avg_ssim2) * (pytorch_ssim.ssim(HR_4, HR_4_avg400) - old_avg_ssim2)


        print("===> Avg. RMSE2: {:.4f} dB".format(avg_rmse2), ",===> S.D: {:.4f} dB".format(sd_rmse2 / len(testing_data_loader)))
        print("===> Avg. SSIM2: {:.4f} dB".format(avg_ssim2), ",===> S.D: {:.4f} dB".format(sd_ssim2 / len(testing_data_loader)))

    ##### compute metrics
    test_set = get_test_set(opt.scale)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                         shuffle=False)
    print('the total number of testing images is ', len(test_set))

    criterion = nn.MSELoss()
    model = torch.load(opt.model)
    print("===> Testing stage started")
    test()











