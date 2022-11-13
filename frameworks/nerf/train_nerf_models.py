import os
import sys
import time
import imageio
import torch
from copy import deepcopy
from easydict import EasyDict

sys.path = [os.getcwd()] + sys.path

from datasets import DataProvider
from datasets.nerf import load_config_and_data_dict_to_args
from frameworks.nerf.modules.utils import compute_bbox_by_cam_frustrm, compute_bbox_by_coarse_geo, \
    compute_bbox_by_cam_frustrm_unbounded
from frameworks.nerf.modules import get_lightning_module, load_nerf
from frameworks import get_params, train_model


def prepare_params(params):
    from utils.tools import parse_params
    params = parse_params(params)

    default_keys = {
        'metric': 'psnr',
        'coarse_model': 'dvgo_coarse',
        'fine_model': 'dvgo_fine',
        'export_test_video': True,
        'save_model': True,
        'save_prefix': '123',
        'pretrained_coarse': "",
        'pretrained_fine': "",
        'skip_fine_train': False,  # whether to skip fine stage training
        'checkpoint_monitor': None,  # None for best checkpoint, 'last' for last checkpoint
    }
    params = {**default_keys, **params}
    return params


def main():
    params = get_params()
    params = prepare_params(params)
    print(params)

    args = load_config_and_data_dict_to_args(deepcopy(params))
    cfg = EasyDict(args.cfg)
    data_dict = args.data_dict
    params.update({'near': data_dict.near, 'far': data_dict.far})
    dataProviderConfig = {
        'batch_size': cfg.coarse_train.N_rand,
        **params['dataset'],
        'cfg_train': cfg.coarse_train,
        'cfg_model': cfg.coarse_model_and_render,
        'cfg_data': cfg.data,
        'data_dict': data_dict,
        'model': None,
    }

    pretrained_coarse = params["pretrained_coarse"]
    pretrained_fine = params["pretrained_fine"]
    if len(pretrained_coarse) == 0:
        dataProvider = DataProvider(dataProviderConfig)
        training_datasets = {
            "train_dataloader": dataProvider.train_dl,
            "val_dataloaders": dataProvider.val_dl,
        }

        if isinstance(cfg.data.xyz_min, (tuple, list)) and isinstance(cfg.data.xyz_max, (tuple, list)):
            xyz_min, xyz_max = torch.tensor(cfg.data.xyz_min), torch.tensor(cfg.data.xyz_max)
        elif cfg.data.unbounded:
            xyz_min, xyz_max = compute_bbox_by_cam_frustrm_unbounded(cfg=cfg, **data_dict)
        else:
            xyz_min, xyz_max = compute_bbox_by_cam_frustrm(cfg=cfg, **data_dict)

        print('compute_bbox: xyz_min', xyz_min, ' xyz_max ', xyz_max)
        params.update({
            'xyz_min': xyz_min,
            'xyz_max': xyz_max,
            'steps_per_epoch': len(dataProvider.train_dl) // params['gpus']
        })

        eps_coarse = time.time()
        coarse_model = get_lightning_module(params['coarse_model'], params)
        coarse_model.setup_dataset(dataProvider.train_dl.dataset)

        if len(cfg.coarse_train.pg_scale) == 0:
            params['num_epochs'] = 1
        else:
            params['num_epochs'] = max(cfg.coarse_train.pg_scale) + 1  # should be consistent with pg_scale
        coarse_model = train_model(coarse_model, params, save_name=params['save_prefix'] + '.' + params['coarse_model'],
                                   mode='max', fit_params=training_datasets,
                                   checkpoint_monitor=params['checkpoint_monitor'])
        print(f'train: coarse training finished in {time.time() - eps_coarse} seconds')
    else:
        coarse_model = load_nerf(pretrained_coarse)

    if params['skip_fine_train']:
        fine_model = coarse_model
    elif len(pretrained_fine) == 0:
        # start fine train
        eps_fine = time.time()
        xyz_min, xyz_max = compute_bbox_by_coarse_geo(coarse_model, thres=cfg.fine_model_and_render.bbox_thres)
        params.update({'xyz_min': xyz_min, 'xyz_max': xyz_max})
        fine_model = get_lightning_module(params['fine_model'], params)
        fine_model.maskCache_from_coarse(coarse_model)

        dataProviderConfig.update({
            'batch_size': cfg.fine_train.N_rand,
            'model': fine_model,  # maskCache is required
            'cfg_train': cfg.fine_train,
            'cfg_model': cfg.fine_model_and_render,
        })
        dataProvider = DataProvider(dataProviderConfig)
        training_datasets = {
            "train_dataloader": dataProvider.train_dl,
            "val_dataloaders": dataProvider.val_dl,
        }
        fine_model.steps_per_epoch = len(dataProvider.train_dl) // params['gpus']
        if len(cfg.fine_train.pg_scale) == 0:
            params['num_epochs'] = 1
        else:
            params['num_epochs'] = max(cfg.fine_train.pg_scale) + 1  # should be consistent with pg_scale
        fine_model = train_model(fine_model, params,
                                 save_name=params['save_prefix'] + '.' + (params['fine_model'][:-len("_fine")]),
                                 mode='max', fit_params=training_datasets,
                                 checkpoint_monitor=params['checkpoint_monitor'])
        print(f'train: fine training finished in {time.time() - eps_fine} seconds')

        if params['save_model']:
            save_name = params['save_prefix'] + '.' + (params['fine_model'][:-len("_fine")])
            test = load_nerf(save_name)  # loading test, may fail
    else:
        fine_model = load_nerf(pretrained_fine)

    if params['export_test_video']:
        from frameworks.nerf.renderers.image_renderer import ImageRenderer
        import utils.backend as backend
        import numpy as np

        save_dir = "test_results"
        os.makedirs(save_dir, exist_ok=True)

        renderer = ImageRenderer(inverse_y=cfg.data.inverse_y,
                                 flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                                 ndc=cfg.data.ndc, img_type=cfg.data.img_type)
        with torch.no_grad():
            rgbs = renderer.renderViews(
                render_poses=data_dict['poses'][data_dict[f'i_test']],
                HW_list=data_dict['HW'][data_dict[f'i_test']],
                K_list=data_dict['Ks'][data_dict[f'i_test']],
                nerf=fine_model.cuda(),
            )
            rgbs = [rgb.cpu() for rgb in rgbs]
            metrics = []
            for pred, gt in zip(rgbs, data_dict['images'][data_dict[f'i_test']]):
                metrics.append(psnr(pred, gt))
            print("==========Test results========")
            print(f"==========PSNR: {torch.tensor(metrics).mean().item()}========")
            backend.log_metric("Test_PSNR", torch.tensor(metrics).mean().item())

        rgb_data = [(255 * np.clip(rgb.numpy(), 0, 1)).astype(np.uint8) for rgb in rgbs]
        try:
            save_path = os.path.join(save_dir, 'video_rgb.mp4')
            imageio.mimwrite(save_path, rgb_data, fps=30, quality=5)
            backend.save_artifact(save_path, "video_rgb")
        except Exception as e:
            print(e)
            from PIL import Image
            for idx, rgb in enumerate(rgb_data):
                save_path = os.path.join(save_dir, f'{idx}.png')
                Image.fromarray(rgb).save(save_path)
                backend.save_artifact(save_path, f'{idx}.png')

    print('Done')


# two metrics from https://github.com/zju3dv/object_nerf/blob/main/utils/metrics.py
def mse(image_pred, image_gt, valid_mask=None, reduction="mean"):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == "mean":
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction="mean"):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


if __name__ == "__main__":
    main()
