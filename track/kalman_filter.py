"""Constant-velocity Kalman filter in (cx, cy, aspect, height) space, as used by ByteTrack.

State: [cx, cy, a, h, vcx, vcy, va, vh]. Pure numpy/scipy, Python 3.6+ (runs on Jetson Nano).
"""
import numpy as np
import scipy.linalg


class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        # Uncertainty is scaled relative to box height.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def multi_predict(self, mean, covariance):
        """Vectorised predict. mean: (N, 8), covariance: (N, 8, 8)."""
        h = mean[:, 3]
        ones = np.ones_like(h)
        std_pos = [self._std_weight_position * h, self._std_weight_position * h,
                   1e-2 * ones, self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * h, self._std_weight_velocity * h,
                   1e-5 * ones, self._std_weight_velocity * h]
        sqr = np.square(np.r_[std_pos, std_vel]).T  # (N, 8)
        motion_cov = np.array([np.diag(s) for s in sqr])

        F = self._motion_mat
        mean = mean.dot(F.T)
        covariance = np.matmul(np.matmul(F, covariance), F.T) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        h = mean[3]
        std = [self._std_weight_position * h, self._std_weight_position * h,
               1e-1, self._std_weight_position * h]
        innovation_cov = np.diag(np.square(std))
        H = self._update_mat
        return H.dot(mean), np.linalg.multi_dot((H, covariance, H.T)) + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation.dot(kalman_gain.T)
        new_cov = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_cov
