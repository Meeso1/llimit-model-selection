"""Simple feature scaler with state management."""

import numpy as np


class SimpleScaler:
    """
    Simple standard scaler for normalizing features.
    
    Provides get_state_dict/load_state_dict for easy serialization.
    """
    
    def __init__(self) -> None:
        """Initialize the scaler."""
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> "SimpleScaler":  # [n_samples, n_features]
        """
        Fit the scaler to the data.
        
        Args:
            X: Feature array
            
        Returns:
            Self for chaining
        """
        self.mean = np.mean(X, axis=0)  # [n_features]
        self.scale = np.std(X, axis=0)  # [n_features]
        # Avoid division by zero
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
        self._is_fitted = True
        return self

    def fit_unbalanced(self, X: list[np.ndarray]) -> "SimpleScaler":
        """
        Fit the scaler to the data, handling different counts of samples for each feature.
        
        Args:
            X: List of values for each feature
            
        Returns:
            Self for chaining
        """
        self.mean = np.array([np.mean(values) for values in X])  # [n_features]
        self.scale = np.array([np.std(values) for values in X])  # [n_features]
        # Avoid division by zero
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:  # [n_samples, n_features]
        """
        Transform the data using fitted parameters.
        
        Args:
            X: Feature array
            
        Returns:
            Normalized feature array
        """
        assert self._is_fitted, "Scaler must be fitted before transform"
        assert self.mean is not None and self.scale is not None
        
        return (X - self.mean) / self.scale

    def transform_unbalanced(self, X: list[np.ndarray]) -> list[np.ndarray]:
        """
        Transform the data using fitted parameters, handling different counts of samples for each feature.
        
        Args:
            X: List of values for each feature
            
        Returns:
            Normalized feature array
        """
        assert self._is_fitted, "Scaler must be fitted before transform_unbalanced"
        assert self.mean is not None and self.scale is not None
        
        result = []
        for i, values in enumerate(X):
            result.append((values - self.mean[i]) / self.scale[i])

        return result
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:  # [n_samples, n_features]
        """
        Inverse transform the normalized data back to original scale.
        
        Args:
            X: Normalized feature array
            
        Returns:
            Original scale feature array
        """
        assert self._is_fitted, "Scaler must be fitted before inverse_transform"
        assert self.mean is not None and self.scale is not None
        
        return X * self.scale + self.mean
    
    def inverse_transform_unbalanced(self, X: list[np.ndarray]) -> list[np.ndarray]:
        """
        Inverse transform the normalized data back to original scale, handling different counts of samples for each feature.
        
        Args:
            X: List of normalized values for each feature
            
        Returns:
            Original scale feature array
        """
        assert self._is_fitted, "Scaler must be fitted before inverse_transform_unbalanced"
        assert self.mean is not None and self.scale is not None
        
        result = []
        for i, values in enumerate(X):
            result.append(values * self.scale[i] + self.mean[i])
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # [n_samples, n_features]
        """
        Fit and transform in one step.
        
        Args:
            X: Feature array
            
        Returns:
            Normalized feature array
        """
        return self.fit(X).transform(X)
    
    def get_state_dict(self) -> dict:
        """Get state dict for serialization."""
        assert self._is_fitted, "Scaler must be fitted before getting state"
        assert self.mean is not None and self.scale is not None
        
        return {
            'mean': self.mean,
            'scale': self.scale,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict."""
        self.mean = state_dict['mean']
        self.scale = state_dict['scale']
        self._is_fitted = True

    @staticmethod
    def from_state_dict(state_dict: dict) -> "SimpleScaler":
        """Create a scaler from state dict."""
        scaler = SimpleScaler()
        scaler.load_state_dict(state_dict)
        return scaler
