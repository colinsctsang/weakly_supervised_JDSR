import numpy as np
import torch
import torch.nn as nn

criterion = nn.MSELoss()

# blind-spot loss
def BlindSpotLoss(input_patch, input_patch_target, rand_x, rand_y, scale):

    if scale == 1:
        current_batchsize, _, _, _ = input_patch.shape
        total_number_of_blind_spot = len(rand_x[0])
        input_patch_blind_spot = torch.zeros((current_batchsize, total_number_of_blind_spot))
        input_patch_target_blind_spot = torch.zeros((current_batchsize, total_number_of_blind_spot))
        for k in range(current_batchsize):
            for j in range(total_number_of_blind_spot):
                input_patch_blind_spot[k, j] = input_patch[k, 0, rand_x[k][j], rand_y[k][j]]
                input_patch_target_blind_spot[k, j] = input_patch_target[k, 0, rand_x[k][j], rand_y[k][j]]
        loss_blind_spot = criterion(input_patch_blind_spot, input_patch_target_blind_spot)

    else:
        current_batchsize, _, _, _ = input_patch.shape
        total_number_of_blind_spot = len(rand_x[0])
        input_patch_blind_spot = torch.zeros((current_batchsize, scale * scale * total_number_of_blind_spot))
        input_patch_target_blind_spot = torch.zeros((current_batchsize, scale * scale * total_number_of_blind_spot))
        for k in range(current_batchsize):
            for j in range(total_number_of_blind_spot):
                for a in range(0, scale-1):
                    for b in range(0, scale-1):
                        input_patch_blind_spot[k, j] = input_patch[k, 0, scale * rand_x[k][j] + a, scale * rand_y[k][j] + b]
                        input_patch_target_blind_spot[k, j] = input_patch_target[k, 0, scale * rand_x[k][j] + a, scale * rand_y[k][j] + b]
        loss_blind_spot = criterion(input_patch_blind_spot, input_patch_target_blind_spot)

    return loss_blind_spot

# make blind-spots on image
def MakeBlindSpot(input_patch):
    current_batchsize, _, xinput, yinput = input_patch.shape
    total_number_of_blind_spot = max(xinput, yinput)-2
    rand_x_input_patch = [[0] * total_number_of_blind_spot] * current_batchsize
    rand_y_input_patch = [[0] * total_number_of_blind_spot] * current_batchsize
    for k in range(current_batchsize):
        rand_x_input_patch[k] = np.random.randint(low=1, high=xinput - 2, size=total_number_of_blind_spot)
        rand_y_input_patch[k] = np.arange(start=1, stop=yinput-1, step=1)
        for j in range(total_number_of_blind_spot):
            randint = np.random.randint(low=1, high=8)
            if randint == 1:
                rand_a = -1
                rand_b = -1
            elif randint == 2:
                rand_a = -1
                rand_b = 0
            elif randint == 3:
                rand_a = -1
                rand_b = 1
            elif randint == 4:
                rand_a = 0
                rand_b = -1
            elif randint == 5:
                rand_a = 0
                rand_b = 1
            elif randint == 6:
                rand_a = 1
                rand_b = -1
            elif randint == 7:
                rand_a = 1
                rand_b = 0
            elif randint == 8:
                rand_a = 1
                rand_b = 1
            neighbor = input_patch[k, 0, rand_x_input_patch[k][j] + rand_a, rand_y_input_patch[k][j] + rand_b]
            input_patch[k, 0, rand_x_input_patch[k][j], rand_y_input_patch[k][j]] = neighbor


    return input_patch, rand_x_input_patch, rand_y_input_patch
