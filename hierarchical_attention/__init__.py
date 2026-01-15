__version__ = "0.1.0"

from .modules import HierarchicalSparseAttention
from .ops import (
    build_tree_topology, 
    generate_span_input_Y,
    build_parent_nodes,
    hierarchical_attention
)

__all__ = [
    "HierarchicalAttention",
    "build_tree_topology",
    "generate_span_input_Y",
    "build_parent_nodes",
    "hierarchical_attention",
    "__version__",
]