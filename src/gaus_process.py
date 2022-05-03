import pathlib
from typing import Tuple
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score

class PseudoData:
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.rand = np.random.RandomState(self.random_state)

    @staticmethod
    def model(i):
        return lambda x: np.sin(2 * x) + (i + 1) * 0.02
    
    def generate(self, n_samples) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(-1, 1, n_samples)
        n_features_y = 6
        y = np.vstack([
            self.model(i)(x) + self.rand.normal(scale=0.05, size=n_samples)
            for i in range(n_features_y)
        ]).T
        return x[:, np.newaxis], y

class AutoRBFGPR:
    def __init__(self, rbf_length_scales=None, white_noise_level=1e-2):
        if rbf_length_scales is None:
            rbf_length_scales = np.logspace(-1, 1, 10)
        self.rbf_length_scales = rbf_length_scales
        self.white_noise_level = white_noise_level
    
    @staticmethod
    def _fit_single(X, y, kernel):
        return [
            GaussianProcessRegressor(kernel=kernel).fit(X, y[:, i])
            for i in range(y.shape[1])
        ]
    
    def fit(self, X, y):
        kernels = [
            RBF(length_scale=ls) + WhiteKernel(noise_level=self.white_noise_level)
            for ls in self.rbf_length_scales
        ]

        max_r2 = -np.inf
        for kernel in kernels:
            gprs = self._fit_single(X, y, kernel)
            y_pred = np.vstack([gpr.predict(X) for gpr in gprs]).T
            r2 = r2_score(y, y_pred)
            if r2 > max_r2:
                max_r2 = r2
                self.gprs = gprs
                self.kernel = kernel
        return self
    
    def predict(self, X):
        return np.vstack([gpr.predict(X) for gpr in self.gprs]).T
    
class IndependentGPR(AutoRBFGPR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PCAGPR(AutoRBFGPR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit(self, X, y, n_components=None):
        n_components = n_components or y.shape[1]
        self.pca = PCA(n_components=n_components)
        self.y_pca = self.pca.fit_transform(y)
        super().fit(X, self.y_pca)
        return self
    
    def predict(self, X):
        return self.pca.inverse_transform(
            super().predict(X)
        )


if __name__ == '__main__':
    warnings.simplefilter('ignore', category=ConvergenceWarning)
    mpl.use('Agg') # no display mode

    data = PseudoData(random_state=0)
    X, y = data.generate(41)
    ind_gpr = IndependentGPR().fit(X, y)
    pca_gpr = PCAGPR(white_noise_level=1e0).fit(X, y, n_components=1)

    fig, axes = plt.subplots(ncols=3, nrows=2, dpi=100, figsize=(10, 6), constrained_layout=True)
    x_plt = np.linspace(-1, 1, 200)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(x_plt, PseudoData.model(i)(x_plt), color='dimgray', linewidth=1.2, label='Truth')
        ax.scatter(X, y[:, i], color='black', s=2, zorder=100, label='Data')

        y_ind = ind_gpr.predict(x_plt[:, np.newaxis])
        ax.plot(x_plt, y_ind[:, i], color='red', linestyle=(0, (1, 1)), label='IGP')

        y_pca = pca_gpr.predict(x_plt[:, np.newaxis])
        ax.plot(x_plt, y_pca[:, i], color='blue', linestyle='dashed', label='PCA-GP')

        ax.grid(linestyle='dotted', color='dimgray')
        ax.set_ylabel(f'$y^Te_{i + 1}$')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.5, 1.5)
    axes[0, 0].legend()
    plt.show()
    path = pathlib.Path(__file__).parent / '../figures/gpr_fits.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')

    fig, axes = plt.subplots(ncols=3, nrows=2, dpi=100, figsize=(10, 6), constrained_layout=True)
    x_plt = np.linspace(-1, 1, 500)
    for i, ax in enumerate(axes.flatten()):
        y_true = PseudoData.model(i)(x_plt)
        
        y_ind = ind_gpr.predict(x_plt[:, np.newaxis])
        res = np.sqrt(np.mean((y_ind[:, i] - y_true)**2))
        ax.plot(x_plt, y_ind[:, i] - y_true, color='red', linestyle=(0, (1, 1)), label=f'IGP r.m.s.={res:.2e}')
        
        y_pca = pca_gpr.predict(x_plt[:, np.newaxis])
        res = np.sqrt(np.mean((y_pca[:, i] - y_true)**2))
        ax.plot(x_plt, y_pca[:, i] - y_true, color='blue', linestyle='dashed', label=f'PCA-GP r.m.s.={res:.2e}')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
        ax.set_ylabel(f'$y^Te_{i + 1}' + r'- \hat{f}' + f'_{i + 1}(x)$')
        ax.grid(linestyle='dotted', color='dimgray')
        ax.legend(loc='lower right')
    plt.show()
    path = pathlib.Path(__file__).parent / '../figures/gpr_rms.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
