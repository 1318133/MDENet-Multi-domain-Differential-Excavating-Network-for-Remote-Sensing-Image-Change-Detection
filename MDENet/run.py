from main import main
import argparse


def get_params():
    parser = argparse.ArgumentParser(description='RSCD_PyTorch')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',  #18 or 4
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0004, metavar='LR',
                        help='learning rate (default: 0.0001)')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    params = vars(get_params())
    main(params)
