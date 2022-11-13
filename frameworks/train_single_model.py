import os
import sys

from pytorch_lightning.utilities import rank_zero_only

sys.path.append(os.getcwd())


###
# python -m torch.distributed.launch --nproc_per_node=8 ../main.py
# nvidia-smi         发现内存泄露问题，即没有进程时，内存被占用
# kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
###

def get_params():
    from pytorch_lightning import seed_everything
    import utils.backend as backend
    params = backend.load_parameters()
    seed_everything(params['seed'])
    backend.log_params(params)
    return params


def prepare_params(params):
    from utils.tools import parse_params
    params = parse_params(params)
    default_keys = {
        'inference_statics': True,
        'skip_train': False,
        'save_model': True,
        'metric': 'acc',
    }
    return {**default_keys, **params}


def train_model(model, params, save_name='default', checkpoint_monitor=None, mode='max', fit_params=None):
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from utils.tools import get_trainer_params
    import utils.backend as backend

    if fit_params is None:
        fit_params = {}
    save_top_k = 1
    if checkpoint_monitor is None:
        checkpoint_monitor = 'validation/' + params['metric']
    elif checkpoint_monitor == "last":
        checkpoint_monitor = None
        save_top_k = None

    logger = TensorBoardLogger("logs", name=save_name, default_hp_metric=False)
    backend.set_tensorboard_logdir(f'logs/{save_name}')

    checkpoint_callback = ModelCheckpoint(dirpath='saves', save_top_k=save_top_k, monitor=checkpoint_monitor, mode=mode)
    # learningRateMonitor = LearningRateMonitor(logging_interval='step')  # discharge this
    t_params = get_trainer_params(params)
    trainer = Trainer(logger=logger, callbacks=[checkpoint_callback], **t_params)
    trainer.fit(model, **fit_params)
    # trainer.test(model)
    model.eval()
    from utils.tools import get_model_weight_hash
    print(f'model weight hash {get_model_weight_hash(model)}')

    if checkpoint_callback.best_model_path != "" and trainer.is_global_zero:
        import numpy as np
        if params['save_model']:
            if '.' not in save_name:
                save_name = save_name + '.ckpt'
            os.system(f"cp {checkpoint_callback.best_model_path} {save_name}")
            backend.save_artifact(save_name, key='best_model_' + save_name)
        if checkpoint_monitor is not None:
            log_val = checkpoint_callback.best_model_score.item()
            backend.log_metric(checkpoint_monitor.split('/')[-1], float(np.clip(log_val, -1e10, 1e10)))
    else:
        backend.log("Best_model_path not found!")

    backend.log("Training finished")
    return model
