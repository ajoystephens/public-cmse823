import sqlite3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# See https://github.com/fanurs/dlib-face-study on how to generate the database
with sqlite3.connect('data.db') as conn:
    df = pd.read_sql('SELECT * FROM data', conn)

cols = [f'enc[{i:d}]' for i in range(128)]
X = df[cols].values
x_mean = X.mean(axis=0)

pca = PCA(n_components=3)
pca.fit(X)
print(pca.explained_variance_ratio_)

mpl.use('Agg')
fig, ax = plt.subplots(nrows=2, figsize=(12, 4), sharex=True, constrained_layout=True)
for k, x in enumerate(X):
    kw = dict(color='dimgray', linewidth=0.3)
    name = df.iloc[k]['name']
    if 'Terence' in name:
        kw.update(dict(color='blue', linewidth=0.8, zorder=1000, label='Terence Tao'))
    if 'Simone' in name:
        kw.update(dict(color='red', linewidth=0.8, zorder=1000, label='Simone Biles'))
    ax[0].plot(range(1, 128 + 1), x, **kw)
    ax[1].plot(range(1, 128 + 1), x - x_mean, **kw)
ax[0].legend(loc='lower right', ncol=2)
ax[0].set_xlim(0, 128)
ax[0].set_ylim(-0.7, 0.7)
ax[1].set_ylim(-0.25, 0.25)
ax[0].set_xticks(np.arange(0, 128 + 1, 16))
ax[1].set_xlabel(r'$k$')
ax[0].set_ylabel(r'$y_k$')
ax[1].set_ylabel(r'$y_k - \overline{y}$')
plt.show()
fig.savefig('../figures/dlib.png', dpi=300, bbox_inches='tight')
