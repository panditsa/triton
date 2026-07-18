from .._core import builtin
from ._layouts import AMDMFMALayout, AMDWMMALayout
from . import cdna3, cdna4
from . import rdna3, rdna4
from . import gfx1250
from .slice import slice
from .warp_pipeline import warp_pipeline_stage


@builtin
def sched_barrier(_semantic=None):
    """Prevent AMD machine scheduling across this point without emitting code."""
    _semantic.builder.create_sched_barrier()


__all__ = [
    "AMDMFMALayout", "AMDWMMALayout", "cdna3", "cdna4", "rdna3", "rdna4", "gfx1250", "warp_pipeline_stage",
    "sched_barrier", "slice"
]
