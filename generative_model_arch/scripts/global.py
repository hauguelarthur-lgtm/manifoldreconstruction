import torch
import math
from manifoldclustering import construct_whitney_atlas, EmpiricalConfig
from algebraic_engine import apply_algebraic_taylor_regression, AlgebraicWhitneyEvaluator

# 1. Define intrinsic dimension and exact theoretical constraints
d = 3
beta = 4.5
k_degree = math.floor(beta) + 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = EmpiricalConfig(volume_scale=0.2, C_overlap=1.5, beta=beta)

# 2. Extract strictly 1st-Order Topological blueprints via the Global Hub
(global_membership_mask, 
 intrinsic_coords, 
 base_manifold, 
 chart_ambient_indices, 
 fps_centers) = construct_whitney_atlas(data_ambient, intrinsic_dim=d, empirical_config=config)

# 3. Route structural arrays into the Algebraic Engine
augmented_atlas = apply_algebraic_taylor_regression(
    data_ambient=data_ambient,
    chart_ambient_indices=chart_ambient_indices,
    atlas_frames=base_manifold.atlas,
    d=d,
    beta=beta,
    device=device
)

# 4. Instantiate the localized algebraic evaluator
algebraic_manifold = AlgebraicWhitneyEvaluator(
    augmented_atlas=augmented_atlas, 
    k_degree=k_degree, 
    device=device
)

# 5. Evaluate an arbitrary query tensor in ambient space
x_query = torch.randn((100, data_ambient.size(1)), device=device)
x_projected = algebraic_manifold.evaluate_manifold(x_query)