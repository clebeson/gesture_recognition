import argparse

import torch
from torch.nn import Parameter


def parse_args(desc = "Tensorflow implementation of action antecipation with context"):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--devices', metavar='N', default=[0], type=int, nargs='+', help='the id of the GPUs to be used')
    parser.add_argument('--alpha', metavar='N', default=[1.,1.], type=float, nargs='+', help='the alpha factors for binary focal loss')

    parser.add_argument('--data_type', metavar='N', default=['m','g','b'], type=str, nargs='+', help='Data to be used (ball = b, gaze = g, movement = m )')
    parser.add_argument('--logfile', type=str, default="BBB.log", help='sale log file')
    parser.add_argument('--epoch', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--n_layers', type=int, default=256, help='RNN number of layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='RNN hidden dimention')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size ')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient Clip threshold ')
    parser.add_argument('--trunc_seq', type=int, default=100, help='point where bigsequances should be truncated or small sequence should be padded')
    parser.add_argument('--seq', type=int, default=16, help='the size of the sequence')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--workers', type=int, default=16, help='number of workers to prefetch data')
    parser.add_argument('--gamma', type=float, default=1.0, help='the gamma factor for binary focal loss')
    parser.add_argument('--tau', type=float, default=0.75, help='The tau factor for MC Dropout regularization')
    parser.add_argument('--dropout', type=float, default=0.2, help='The dropout value for MC Dropout model')
    parser.add_argument('--thres_train', type=float, default=0.95, help='Threshold that controls the training')
    parser.add_argument('--mode', type=str, default='mc', help='architecture mode')
    
    

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--bidirect', dest='bidirectional', action='store_true', help='activate the bidirectional LSTM')
    feature_parser.add_argument('--no-bidirect', dest='bidirectional', action='store_false', help='deactivate the bidirectional LSTM')
    parser.set_defaults(bidirectional=False)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--spotting', dest='spotting', action='store_true', help='activate spotting mode')
    feature_parser.add_argument('--gesture', dest='spotting', action='store_false', help='deactivate segmented gesture mode')
    parser.set_defaults(spotting=True)


    # parser.add_argument('--res_n', type=int, default=18, help='18, 34, 50, 101, 152')
    # parser.add_argument('--dataset', type=str, default='tiny', help='[cifar10, cifar100, mnist, fashion-mnist, tiny')
    # parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',help='Directory name to save the checkpoints')
    # parser.add_argument('--log_dir', type=str, default='logs',help='Directory name to save training logs')

    return parser.parse_args()



def inflate_conv(conv2d,
                time_dim=3,
                time_padding=0,
                time_stride=1,
                time_dilation=1,
                center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = torch.nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d