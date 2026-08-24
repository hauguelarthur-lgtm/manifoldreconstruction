import torch

class FeatureMapExtractor:
    """
    Computes the localized position-dependent Jacobians and exact orthogonal projectors 
    (feature maps) for the Kernelized Stochastic Interpolant (KSI) framework.
    """
    def __init__(self, k_degree: int, epsilon: float = 1e-8):
        self.k_degree = k_degree
        self.epsilon = epsilon

    def _compute_analytic_poly_jacobian(self, U: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact analytical Jacobian J_poly(u) of the symmetric polynomial combinations
        with respect to the intrinsic coordinates u, bypassing automatic differentiation.
        
        Args:
            U: Intrinsic coordinates tensor of shape (B, d)
            
        Returns:
            J_poly: Analytical Jacobian tensor of shape (B, P_poly, d)
        """
        B, d = U.shape
        
        if self.k_degree < 2:
            return torch.empty((B, 0, d), device=U.device)
            
        # Base Case (Degree 1): Store (polynomial_term, last_index, gradient_tensor)
        # The gradient of u_i with respect to u is the standard basis vector e_i.
        prev_terms = [
            (
                U[:, i:i+1], 
                i, 
                torch.eye(d, device=U.device)[i:i+1, :].expand(B, -1)
            ) 
            for i in range(d)
        ]
        
        all_jacobian_rows = []
        
        for _ in range(2, self.k_degree + 1):
            curr_terms = []
            for term, last_idx, grad in prev_terms:
                for i in range(last_idx, d):
                    # Evaluate structural combinations strictly to prevent permutations
                    new_term = term * U[:, i:i+1]
                    
                    # Analytical Product Rule: \nabla(P * u_i) = u_i * \nabla P + P * \nabla u_i
                    e_i = torch.zeros((1, d), device=U.device)
                    e_i[0, i] = 1.0
                    
                    new_grad = U[:, i:i+1] * grad + term * e_i
                    
                    curr_terms.append((new_term, i, new_grad))
                    all_jacobian_rows.append(new_grad.unsqueeze(1))
            
            prev_terms = curr_terms
            
        if not all_jacobian_rows:
            return torch.empty((B, 0, d), device=U.device)
            
        return torch.cat(all_jacobian_rows, dim=1)

    def compute_local_feature_map(self, 
                                  X: torch.Tensor, 
                                  mu_i: torch.Tensor, 
                                  Q_i: torch.Tensor, 
                                  W_i: torch.Tensor) -> torch.Tensor:
        """
        Assembles the position-dependent Jacobian and computes its stabilized pseudo-inverse
        to construct the localized orthogonal projector feature map.
        
        Args:
            X: Ambient coordinate tensor (B, p)
            mu_i: Chart center (p,)
            Q_i: 1st-Order PCA Basis (p, d)
            W_i: Multi-order geometric tensor (P_poly, p)
            
        Returns:
            nabla_phi_i: Local KSI feature map tensor of shape (B, p, d)
        """
        B, p = X.shape
        d = Q_i.shape[1]
        
        # 1. Intrinsic Coordinate Projection
        U = torch.matmul(X - mu_i, Q_i)
        
        # 2. Localized Jacobian Assembly
        Q_i_expanded = Q_i.unsqueeze(0).expand(B, -1, -1)
        
        if self.k_degree >= 2 and W_i is not None and W_i.shape[0] > 0:
            # J_poly: (B, P_poly, d)
            J_poly = self._compute_analytic_poly_jacobian(U) 
            
            # W_i^T expanded for batched multiplication: (B, p, P_poly)
            W_i_T = W_i.T.unsqueeze(0).expand(B, -1, -1)
            
            # \nabla g_i(u) = Q_i + W_i^T J_poly(u) -> Evaluates to (B, p, d)
            nabla_g = Q_i_expanded + torch.bmm(W_i_T, J_poly)
        else:
            nabla_g = Q_i_expanded
            
        # 3. Batched Pseudo-Inversion via Regularized Normal Equations
        nabla_g_T = nabla_g.transpose(1, 2)
        
        # Intrinsic Metric Tensor G(u) = \nabla g^T \nabla g -> (B, d, d)
        G = torch.bmm(nabla_g_T, nabla_g)
        
        # Micro-Tikhonov penalty ensures strict local Lipschitz continuity for SDE bounds
        G_reg = G + torch.eye(d, device=X.device).unsqueeze(0) * self.epsilon
        
        try:
            G_inv = torch.linalg.inv(G_reg)
        except (RuntimeError, torch._C._LinAlgError):
            G_inv = torch.linalg.pinv(G_reg, hermitian=True)
        
        # \nabla g^+ = G_inv \nabla g^T -> (B, d, p)
        nabla_g_pinv = torch.bmm(G_inv, nabla_g_T)
        
        # \nabla \phi_i = (\nabla g^+)^T -> (B, p, d)
        nabla_phi_i = nabla_g_pinv.transpose(1, 2)
        
        return nabla_phi_i

    def construct_global_basis(self, 
                               X: torch.Tensor, 
                               atlas: list[dict],
                               partition_of_unity_evaluator) -> torch.Tensor:
        """
        Executes the topological gluing across the atlas to construct the 
        globally C^\infty smooth feature matrix required by the KSI Gram matrix.
        
        Args:
            X: Ambient coordinate tensor (B, p)
            atlas: List of dictionary frames containing mu, Q, r_sq, and W parameters.
            partition_of_unity_evaluator: Function evaluating normalized bump weights.
            
        Returns:
            nabla_phi_global: Concatenated global feature map (B, p, m * d)
        """
        B, p = X.shape
        m = len(atlas)
        d = atlas[0]['Q'].shape[1]
        
        # Evaluate global partition of unity normalized weights -> (B, m)
        normalized_weights = partition_of_unity_evaluator(X)
        
        global_features = []
        
        for i, frame in enumerate(atlas):
            w_i = normalized_weights[:, i].view(B, 1, 1)
            
            # Mask evaluations strictly to non-zero weights for computational efficiency
            active_mask = (normalized_weights[:, i] > 0)
            
            nabla_phi_i = torch.zeros((B, p, d), device=X.device)
            
            if active_mask.any():
                X_active = X[active_mask]
                
                phi_active = self.compute_local_feature_map(
                    X_active, 
                    frame['mu'].to(X.device), 
                    frame['Q'].to(X.device), 
                    frame.get('W', None).to(X.device) if 'W' in frame else None
                )
                
                nabla_phi_i[active_mask] = phi_active
                
            # Apply partition of unity blending
            global_features.append(w_i * nabla_phi_i)
            
        # Concatenate into the macroscopic P-dimensional feature space
        return torch.cat(global_features, dim=2)