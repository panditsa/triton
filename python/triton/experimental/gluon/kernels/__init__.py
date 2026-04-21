"""
Reusable kernels written in Gluon.

The MOE pipeline is split into one module per sublock so each can be
optimised and validated independently:

    moe         GEMM (gate_up + down), the scatter/gather matmul.
    activation  silu_and_mul / gelu_and_mul.
    reduce      moe_sum_reduce across the topk axis.
    topk        top-k softmax of router logits.
    align       moe_align_block_size: sort topk_ids by expert and pad.

Each module exposes a ``@gluon.jit`` kernel and a thin ``invoke_*`` Python
wrapper that handles dispatch, grid sizing, and output-tensor allocation.
"""

from . import activation, align, moe, reduce, topk

# Re-export the most common entry points at the package level for convenience.
from .activation import (  # noqa: F401
    gluon_act_and_mul_kernel,
    invoke_gluon_act_and_mul,
    use_gluon_activation,
)
from .align import (  # noqa: F401
    gluon_moe_align_block_size,
    use_gluon_align,
)
from .moe import (  # noqa: F401
    get_default_gluon_config,
    gluon_available,
    gluon_config_supported,
    gluon_fused_moe_kernel,
    invoke_gluon_fused_moe_kernel,
    use_gluon_moe,
)
from .reduce import (  # noqa: F401
    gluon_moe_sum_reduce_kernel,
    invoke_gluon_moe_sum_reduce,
    use_gluon_reduce,
)
from .topk import (  # noqa: F401
    gluon_topk_softmax_kernel,
    invoke_gluon_topk_softmax,
    use_gluon_topk,
)
