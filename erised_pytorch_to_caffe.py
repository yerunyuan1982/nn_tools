"""
   Attributter Model Convertor
"""

import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import pytorch_to_caffe
import pytorch_analyser

import argparse
import os

from attributer.model import generate_model

def model_convertor(opt):
    if not os.path.exists(opt.out_path):
        os.makedirs(opt.out_path)
    analysis_path = os.path.join(opt.out_path, '{}_analysis.csv'.format(opt.out_model_name))
    prototxt_path = os.path.join(opt.out_path, '{}.prototxt'.format(opt.out_model_name))
    caffemodel_path = os.path.join(opt.out_path, '{}.caffemodel'.format(opt.out_model_name))

    attr_opt = argparse.Namespace()
    attr_opt.model = opt.model
    attr_opt.conv = opt.conv
    attr_opt.checkpoint = opt.checkpoint
    attr_opt.pretrain = False

    model, parameters, mean, std = generate_model(attr_opt)
    checkpoint = torch.load(attr_opt.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    attributer_model = model.module.cpu()

    input_tensor = torch.ones(1, 3, 224, 224)

    blob_dict, tracked_layers = pytorch_analyser.analyse(attributer_model, input_tensor)

    pytorch_analyser.save_csv(tracked_layers, analysis_path)
    print("Save", analysis_path, "Finish!")

    input_var = Variable(input_tensor)
    pytorch_to_caffe.trans_net(attributer_model, input_var, opt.out_model_name)

    pytorch_to_caffe.save_prototxt(prototxt_path)
    print("Save", prototxt_path, "Finish!")

    pytorch_to_caffe.save_caffemodel(caffemodel_path)
    print("Save", caffemodel_path, "Finish!")

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path',
        default='/root/models/models',
        type=str,
        help='model output path')
    parser.add_argument(
        '--out_model_name',
        default='attributer',
        type=str,
        help='output model name')
    parser.add_argument(
        '--model',
        default='all_in_one',
        type=str,
        help='all_in_one | hyperface')
    parser.add_argument(
        '--conv',
        default='resnet18',
        type=str)
    parser.add_argument(
        '--checkpoint',
        default='',
        type=str,
        help='Save data (.pth) of previous training')

    args = parser.parse_args()

    model_convertor(args)
