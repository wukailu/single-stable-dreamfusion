from frameworks.nerf.modules import DVGO_Fine, DVGO360_Coarse


class DVGO360(DVGO_Fine, DVGO360_Coarse):
    pass