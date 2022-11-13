import torch
from datasets.nerf.lib.load_data import load_data


def load_everything(args):
    '''Load images / poses / camera settings / data split.
    '''
    cfg = args.cfg
    data_dict = load_data(cfg.data)

    # # remove useless field
    # kept_keys = {
    #     'hwf', 'HW', 'Ks', 'near', 'far',
    #     'i_train', 'i_val', 'i_test', 'irregular_shape',
    #     'poses', 'render_poses', 'images'}
    # for k in list(data_dict.keys()):
    #     if k not in kept_keys:
    #         data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im) for im in data_dict['images']]
        if data_dict['depths'] is not None:
            data_dict['depths'] = [torch.FloatTensor(im) for im in data_dict['depths']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'])
        if data_dict['depths'] is not None:
            data_dict['depths'] = torch.FloatTensor(data_dict['depths'])
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def load_config_and_data_dict_to_args(params):
    from easydict import EasyDict
    args = EasyDict(params)
    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args)
    args.data_dict = data_dict
    print('data loaded!')
    return args