import torch
import os
import argparse
from datasets.crowd import Crowd
from models.fusion import fusion_model
from utils.evaluation import eval_game, eval_relative
from models.CSRNet_IADM import FusionCSRNet
from utils.census_transform import *

'''
@inproceedings{liu2021cross,
  title={Cross-Modal Collaborative Representation Learning and a Large-Scale RGBT Benchmark for Crowd Counting},
  author={Liu, Lingbo and Chen, Jiaqi and Wu, Hefeng and Li, Guanbin and Li, Chenglong and Lin, Liang},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
'''

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
parser.add_argument('--model', default='best_model_17.pth'
                    , help='model name')

parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

cnt = 0

if __name__ == '__main__':
    for i in range(29):
        cnt+=1
        datasets = Crowd(os.path.join(args.data_dir, 'test'), method='test')
        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                                 num_workers=8, pin_memory=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
        device = torch.device('cuda')

        model = fusion_model()
        # model = FusionCSRNet()
        model.to(device)
        model_path = os.path.join(args.save_dir, args.model+str(cnt)+'.pth')
        checkpoint = torch.load(model_path, device)
        model.load_state_dict(checkpoint)
        model.eval()

        # print('testing...')
        # Iterate over data.
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0

        for inputs, target, name in dataloader:

            for i in range(2):
                census_transform(inputs[i])[0, 0]

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

                # print(target)
                # print("#outputs", outputs[0].size())
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    # if L is 0 and abs_error >= 30.61: 
                    #     print(name, "game[", L, "]", abs_error.detach().cpu().numpy())
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error

        N = len(dataloader)
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N

        log_str = 'Model_{}.pth Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                  'MSE {mse:.2f} Re {relative:.4f}, '.\
            format(cnt, N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error)

        print(log_str)