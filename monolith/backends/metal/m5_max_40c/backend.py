from ..base import MetalBackend
from .scheduling import finalize


class Backend(MetalBackend):
    """Measured 40-core Max fusion policy and explicit draft/decoder recipes."""
    id = "m5_max_40c"
    finalize = staticmethod(finalize)

    def validate_config(self, config):
        from .validation import validate_gdn_config
        validate_gdn_config(config.name, config.gdn_mixer_fusion, config.gpu_cores)
