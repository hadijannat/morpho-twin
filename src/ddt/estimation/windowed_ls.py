from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..interfaces import Estimate, Estimator


@dataclass
class WindowedLeastSquaresEstimator(Estimator):
    """Very small demo estimator.

    Estimates scalar (a,b) for x_{k+1} = a x_k + b u_k from a sliding window.
    Returns theta_hat = [a_hat, b_hat].

    This is *not* MHE; it's here to make the scaffold runnable out of the box.
    Replace with `acados` MHE or a full nonlinear MHE in real use.
    """

    window: int
    _xs: deque[float] = field(default_factory=deque, init=False)
    _us: deque[float] = field(default_factory=deque, init=False)
    _ys: deque[float] = field(default_factory=deque, init=False)
    _theta: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.1], dtype=float), init=False
    )

    def reset(self) -> None:
        self._xs.clear()
        self._us.clear()
        self._ys.clear()
        self._theta = np.array([1.0, 0.1], dtype=float)

    def update(self, y: np.ndarray, u_applied: np.ndarray) -> Estimate:
        y_val = float(np.asarray(y).reshape(()))
        u_val = float(np.asarray(u_applied).reshape(()))

        # store y and u; we treat y as x for this simple demo
        self._ys.append(y_val)
        self._us.append(u_val)
        if len(self._ys) > self.window + 1:
            self._ys.popleft()
        if len(self._us) > self.window + 1:
            self._us.popleft()

        theta_cov = np.eye(2) * 1e3  # very rough placeholder

        if len(self._ys) >= 3:
            # build regression from (x_k, u_k) -> x_{k+1}
            ys = np.array(self._ys, dtype=float)
            us = np.array(self._us, dtype=float)
            xk = ys[:-1]
            xkp1 = ys[1:]
            uk = us[:-1]
            Phi = np.stack([xk, uk], axis=1)  # [N,2]
            # regularized LS
            reg = 1e-6 * np.eye(2)
            theta = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ xkp1)
            # Clamp to reasonable bounds to prevent instability
            # a should be roughly 0.5 to 1.5, b should be positive and small
            theta[0] = np.clip(theta[0], 0.5, 1.5)
            theta[1] = np.clip(theta[1], 0.01, 1.0)
            self._theta = theta
            # covariance approx
            resid = xkp1 - Phi @ theta
            s2 = float(np.mean(resid**2)) if resid.size else 1.0
            theta_cov = s2 * np.linalg.inv(Phi.T @ Phi + reg)

        x_hat = np.array([y_val], dtype=float)
        return Estimate(x_hat=x_hat, theta_hat=self._theta.copy(), theta_cov=theta_cov)
