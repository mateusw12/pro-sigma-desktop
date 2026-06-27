"""
Box-Cox transformation utilities.
Ported from weg-statistics-python/app/box_cox/service/box_cox_service.py
"""
import numpy as np


class BoxCoxTransformer:

    def _boxcox_transform(self, y: np.ndarray, lam: float, geom_mean: float) -> np.ndarray:
        if np.isclose(lam, 0.0):
            return geom_mean * np.log(y)
        return (np.power(y, lam) - 1.0) / (lam * (geom_mean ** (lam - 1.0)))

    def _boxcox_inverse(self, y_t: np.ndarray, lam: float, geom_mean: float) -> np.ndarray:
        if np.isclose(lam, 0.0):
            return np.exp(y_t / geom_mean)
        inner = lam * (geom_mean ** (lam - 1.0)) * y_t + 1.0
        inner = np.where(inner <= 0, np.nan, inner)
        return np.power(inner, 1.0 / lam)

    def _safe_float(self, v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def calculate(self, y_array: np.ndarray, lambda_start: float = -2.0, lambda_end: float = 2.0, lambda_step: float = 0.2) -> dict:
        """
        Finds the optimal Box-Cox lambda minimising SSE.

        Returns dict with keys:
            best_lambda, geom_mean, shift, sse_dict (lambda→sse), lambdas, sses
        """
        y = np.asarray(y_array, dtype=float)

        # Shift so all values are strictly positive
        min_y = float(np.min(y))
        shift = abs(min_y) + 1e-8 if min_y <= 0 else 0.0
        y = y + shift

        geom_mean = float(np.exp(np.mean(np.log(y))))
        lambda_values = list(np.round(np.arange(lambda_start, lambda_end + 1e-12, lambda_step), 8))

        sse_dict = {}
        lambdas = []
        sses = []
        best_lambda = None
        best_sse = None

        for lam in lambda_values:
            try:
                y_t = self._boxcox_transform(y, lam, geom_mean)
                # SSE relative to mean (null model baseline)
                residuals = y_t - np.mean(y_t)
                sse = float(np.sum(residuals ** 2))
            except Exception:
                sse = None

            lam_key = float(lam)
            sse_dict[lam_key] = sse
            lambdas.append(lam_key)
            sses.append(sse)

            if sse is not None and (best_sse is None or sse < best_sse):
                best_sse = sse
                best_lambda = lam_key

        return {
            'best_lambda': best_lambda,
            'geom_mean': geom_mean,
            'shift': shift,
            'sse_dict': sse_dict,
            'lambdas': lambdas,
            'sses': sses,
        }

    def transform(self, y_array: np.ndarray, lam: float, geom_mean: float, shift: float) -> np.ndarray:
        """Apply Box-Cox transform with given parameters."""
        y = np.asarray(y_array, dtype=float) + shift
        return self._boxcox_transform(y, lam, geom_mean)
