from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataset import FaultDataset
from fault_detection.models.Inception import InceptionTime
from regression import train_model, load_and_validate_model
import torch
import argparse
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
from models.MLP import MLP
from models.LSTM import LSTMModel
from models.Transformer import TransformerModel
from models.Crossformer import Model as Crossformer
from models.PatchTST import Model as PatchTST
from models.LightTS import Model as LightTS
from models.TCN import TCN
from models.FaultNet import FaultNet
from models.iTransformer import Model as iTransformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaultDetectionConfigs')

    # for training
    parser.add_argument('--model_name', type=str, default='FaultNet', help='loss function')
    parser.add_argument('--is_train', type=int, default=0, help='train or test')
    parser.add_argument('--is_resume', type=int, default=0, help='use resume')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--loss_func', type=str, default='MSE', help='loss function')
    # parser.add_argument('--loss_func', type=str, default='torque_loss', help='loss function')

    # for task
    parser.add_argument('--seq_len', type=int, default=50, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=6, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--num_class', type=int, default=6, help='output sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--task_name', type=str, default="classification", help='expansion factor for Mamba')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--freq', type=str, default='s',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    args = parser.parse_args()

    # 数据加载器
    train_dataset = FaultDataset('dataset/train_data.npz', normalize=args.use_norm)
    test_dataset = FaultDataset('dataset/test_data.npz', normalize=args.use_norm)
    # test_dataset = FaultDataset('dataset/train_data.npz', normalize=args.use_norm)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model_hash = {
        "MLP": MLP(args),
        "LSTM": LSTMModel(args),
        "TCN": TCN(args),
        "Transformer": TransformerModel(args),
        "Crossformer": Crossformer(args),
        "PatchTST": PatchTST(args),
        "iTransformer": iTransformer(args),
        "LightTS": LightTS(args),
        "Inception": InceptionTime(args),
        "FaultNet": FaultNet(args),
    }

    model = model_hash[args.model_name]


    if args.is_resume:
        model.load_state_dict(torch.load("checkpoints/best_model.pth", weights_only=True))

    if args.is_train:
        train_model(model, train_loader, test_loader, epochs=args.epochs, learning_rate=args.lr,
                    loss_func=args.loss_func)
    else:
        load_and_validate_model(model, "checkpoints/" + args.model_name + "/best_model.pth", test_loader)
        # load_and_validate_model(model, "checkpoints" + "/best_model.pth", test_loader)
