import sys, os

sys.path = [os.getcwd()] + sys.path

from utils.tools import submit_jobs, random_params


##  有一个奇怪的bug, 如果 mmcv.Config.fromfile 被调用多次，效果会是多个dict叠加起来的结果
def config_load(path):
    try:
        import mmcv
    except:
        import mmcv
    import copy
    config = mmcv.Config.fromfile(path)
    return file_name_replace(copy.deepcopy(config), path.split('/')[-1].split('.')[0])


def file_name_replace(dic, filename):
    for key in dic:
        if isinstance(dic[key], str):
            dic[key] = dic[key].replace("{{fileBasenameNoExtension}}", filename)
        elif isinstance(dic[key], dict):
            dic[key] = file_name_replace(dic[key], filename)
    return dic


templates = {
    'nerf': {
        'project_name': 'sin_prepare',
        'coarse_model': 'dvgo_coarse',
        'dataset': {'name': 'nerf', 'workers': 1},
        'lr_scheduler': 'StepAutoLR_step',
        'metric': 'psnr',
        'save_model': True,
        'save_prefix': 'model',
    },
}


def test():
    params = templates['nerf']
    config_path = random_params([
        # 'datasets/nerf/configs/toydesk/our_desk_2.py',
        # 'datasets/nerf/configs/nsvf/Palace.py',  # no-pose, pose, dvgo, nwnn
        # 'datasets/nerf/configs/nsvf/Wineholder.py',  # nwnn, cross_scene
        # 'datasets/nerf/configs/nsvf/Steamtrain.py',  # no-pose, pose, dvgo, nwnn
        # 'datasets/nerf/configs/nsvf/Robot.py',  # dvgo
        # 'datasets/nerf/configs/tankstemple/Truck.py',
        # 'datasets/nerf/configs/nerf/chair.py',  # dvgo, nwnn, no-pose, pose
        'datasets/nerf/configs/nerf/ficus.py',  # nwnn, cross_scene. dvgo, nwnn
        # 'datasets/nerf/configs/nerf/lego.py',  # dvgo, nwnn, no-pose, pose
    ])
    name = "_".join(config_path.split("/")[-2:])[:-3]
    seed = 2018011328
    params['cfg'] = config_load(config_path)
    model_type = random_params(['dvgo_fine'])  # 'nwnn_fine', 'dvp_fine', 'dvgo_fine'
    params.update({
        'project_name': 'RealNeRFEdit_Exp',
        'coarse_model': 'dvgo_coarse',
        'fine_model': model_type,
        'save_prefix': f'{name}_{seed}',
        'seed': seed,
        'hint': 'nwnn',
    })
    lr = 1e-1  # better visual but lower psnr
    params['cfg']["fine_train"].update(random_params({
        # 'weight_tv_density': [3e-3, 1e-2, 1e-3],  # special for Barn
        # 'pg_scale': (1, 2, 5),
        'lrate_density': lr,
        'lrate_k0': lr,
        'weight_metric_k0': 0 if model_type != "dvp_fine" else 3e-3,
        'lrate_rgbnet': 1e-3,  # 3e-3 > 1e-3 (0.09dB) 3e-3 > 1e-2 (0.13dB)
    }))
    params['cfg'].fine_model_and_render.update({
        'posbase_pe': random_params([0, 5]),
        # 'num_voxels': 256 ** 3,
    })
    return params


def params_for_Nerf():
    params = test()
    params['cfg'] = dict(params['cfg'])
    return params


if __name__ == "__main__":
    submit_jobs(params_for_Nerf, 'frameworks/nerf/train_nerf_models.py', number_jobs=6, job_directory='.')
