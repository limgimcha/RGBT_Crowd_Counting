import torch
import os
import argparse
from datasets.crowd import Crowd
from models.fusion import fusion_model
from utils.evaluation import eval_game, eval_relative

from utils.fpn import *
import cv2
import PIL
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default='preprocessed_data',
                        help='training data directory')
parser.add_argument('--save-dir', default='new model/1029-161454/',
                        help='model directory')
parser.add_argument('--model', default='best_model_8.pth'
                    , help='model name')

parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

if __name__ == '__main__':


    datasets = Crowd(os.path.join(args.data_dir, 'test'), method='test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=6, pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    #model = FPN101()
    model = fusion_model()
    model.to(device)
    model_path = os.path.join(args.save_dir, args.model)
    checkpoint = torch.load(model_path, device)
    #model.load_state_dict(checkpoint)
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0

    for inputs, target, name in dataloader:
        if type(inputs) == list:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
        else:
            inputs = inputs.to(device)

        # inputs are images with different sizes
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error

        # 이미지 출력
        
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 3, 1) # rows, cols, index
            ax1.axis("off")
            ax1.text(0,0.5,"name: " + name[0])
            ax1.text(0,0.3,"relative error: {:.5f}".format(relative_error.item()))

            ax2 = fig.add_subplot(2, 3, 4) # rows, cols, index
            rgbImage = inputs[0][0].cpu().permute(1,2,0).numpy()[...,::-1]
            ax2.set_title("RGB Image")
            ax2.imshow(rgbImage)

            ax3 = fig.add_subplot(2, 3, 5) # rows, cols, index
            tImage = inputs[1][0].cpu().permute(1,2,0).numpy()[...,::-1]
            ax3.set_title("T Image")
            ax3.imshow(tImage)

            ax4 = fig.add_subplot(2, 3, 6) # rows, cols, index
            outputImage = outputs[0].cpu().permute(1,2,0).numpy()[...,::-1]
            ax4.set_title("Density Map")
            ax4.imshow(outputImage)
            print("output is*****",outputs.shape) # [1, 1, 60, 80]
            plt.show()
            

    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N

    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
              'MSE {mse:.2f} Re {relative:.4f}, '.\
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error)

    print(log_str)