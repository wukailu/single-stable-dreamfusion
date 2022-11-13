from utils.tools import skip_if_excuted
from .dvgo_coarse import DVGO_Coarse
from .dvgo_fine import DVGO_Fine
from .dvgo360_coarse import DVGO360_Coarse
from .dvgo360_fine import DVGO360
from .dvp_fine import DVGO_Plus
from .fastffl_fine import FastFFL
from .ffl_fine import FFL
from .nwnn_fine import NeRFWoNN_Fine
from .osr_fine import OSR_Fine, OSR_Fine_V2, OSR_Fine_V3, OSR_Fine_V4, OSR_Fine_V5, OSR_Fine_RGI, OSR_Fine_V6

model_dict = {
    'dvgo_coarse': DVGO_Coarse,
    'dvgo360_coarse': DVGO360_Coarse,
    'dvgo360_fine': DVGO360,
    'dvgo_fine': DVGO_Fine,
    'nwnn_fine': NeRFWoNN_Fine,
    'ffl_fine': FFL,
    'fastffl_fine': FastFFL,
    'dvp_fine': DVGO_Plus,
    'osr_fine': OSR_Fine,
    'osr_v2_fine': OSR_Fine_V2,
    'osr_v3_fine': OSR_Fine_V3,
    'osr_v4_fine': OSR_Fine_V4,
    'osr_v5_fine': OSR_Fine_V5,
    'osr_v6_fine': OSR_Fine_V6,
    'osr_rgi_fine': OSR_Fine_RGI,
}


def get_lightning_module(module_name: str, hparams):
    assert module_name.endswith('_fine') or module_name.endswith('_coarse')
    return model_dict[module_name](hparams)


@skip_if_excuted
def load_nerf(path: str, strict=False, map_location='cpu'):
    file_type = path.split('.')[-1]
    if file_type.endswith('_coarse'):  # loading a coarse model
        return model_dict[file_type].load_from_checkpoint(path)
    else:
        return model_dict[file_type+'_fine'].load_from_checkpoint(path, strict=strict, map_location=map_location)

