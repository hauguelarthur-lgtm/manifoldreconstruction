import torch

class GlobalSubspaceProjector:
    """
    Computes the exact affine subspace of an unknown empirical distribution via SVD.
    Projects R^p -> R^k for integration, and isometrically lifts R^k -> R^p.
    """
    def __init__(self, variance_threshold: float = 0.999):
        self.variance_threshold = variance_threshold
        self.mean = None
        self.V_k = None
        self.k = None

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.mean = X.mean(dim=0, keepdim=True)
        X_centered = X - self.mean
        
        # Execute deterministic SVD (X = U * S * V^T)
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        V = Vh.T
        
        # Identify the spectral gap to determine the effective dimension k
        explained_variance = (S ** 2) / (X.size(0) - 1)
        cumulative_variance = torch.cumsum(explained_variance, dim=0) / torch.sum(explained_variance)
        
        self.k = torch.searchsorted(cumulative_variance, self.variance_threshold).item() + 1
        self.V_k = V[:, :self.k]
        
        print(f"Subspace extraction: Reduced ambient dimension from {X.shape[1]} to {self.k} "
              f"(retaining {self.variance_threshold*100}% variance).")
        
        # Project into the orthogonal R^k basis
        return torch.matmul(X_centered, self.V_k)

    def inverse_transform(self, X_red: torch.Tensor) -> torch.Tensor:
        # Isometrically lift back to the original R^p ambient space
        return torch.matmul(X_red, self.V_k.T) + self.mean