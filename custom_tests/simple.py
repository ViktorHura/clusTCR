from clustcr import test_func

import torch

from modelBYOL import SiameseNetworkBYOL as SiameseNetwork, encode_data


def main():
    input_size = (31, 2, 25)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork(input_size).to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    # import torchinfo
    # s = torchinfo.summary(model, [input_size, input_size], batch_dim=0,
    #                   col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose=0)
    # print(s)

    print(test_func())


if __name__ == "__main__":
    main()