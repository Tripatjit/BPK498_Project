from enigmatoolbox.datasets import load_sc, load_fc
from nilearn import plotting
from tvb.simulator.lab import *
import numpy as np
from matplotlib import pyplot as plt
from tvb.simulator.lab import *
from tvb.datatypes import graph
from tvb.datatypes.time_series import TimeSeriesRegion


fc_ctx, fc_ctx_labels, _, _ = load_fc()

print(fc_ctx.shape)
fc_plot = plot_matrix(fc_ctx)
plt.show()