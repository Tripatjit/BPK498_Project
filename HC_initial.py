from tvb.simulator.lab import *
import numpy as np
from matplotlib import pyplot as plt
from tvb.simulator.lab import *
from tvb.datatypes import graph
from tvb.datatypes.time_series import TimeSeriesRegion

conn = connectivity.Connectivity.from_file("connectivity_68.zip")
print(conn.weights.shape)
conn.speed=np.array([4.0])

rww = models.ReducedWongWang(a=np.array([0.27]), w=np.array([1.0]), I_o=np.array([0.3]))
S = np.linspace(0, 1, 50).reshape((1, -1, 1))
C = S * 0.0
dS = rww.dfun(S, C)

plt.figure()
plt.plot(S.flat, dS.flat)

sim = simulator.Simulator(
    model=rww,
    connectivity=conn,
    coupling=coupling.Linear(a=np.array([0.5 / 50.0])),
    integrator=integrators.EulerStochastic(dt=1, noise=noise.Additive(nsig=np.array([1e-5]))), 
    monitors=(monitors.TemporalAverage(period=1.),),
    simulation_length=5e3
).configure()

(time, data), = sim.run()

plt.figure()
plt.plot(time, data[:, 0, :, 0], 'k', alpha=0.1);

tsr = TimeSeriesRegion(connectivity=sim.connectivity,
                       data=data,
                       sample_period=sim.monitors[0].period)
tsr.configure()

def ev(tsr):
    input_shape = tsr.data.shape
    result_shape = (input_shape[2], input_shape[2], input_shape[1], input_shape[3])
    result = np.zeros(result_shape)

    for mode in range(result_shape[3]):
        for var in range(result_shape[2]):
            data = tsr.data[:,var,:, mode].squeeze()
            result[:, :, var, mode] = np.corrcoef(data.T)

    corr_coeff = graph.CorrelationCoefficients(source=tsr, array_data=result)
    return corr_coeff

result = ev(tsr)

FC = result.array_data[..., 0, 0]

print(FC.shape)

plot_tri_matrix(FC,
                cmap=plt.cm.RdYlBu_r, 
                node_labels= sim.connectivity.region_labels,
                size=[10., 10.],
                color_anchor=(-1.0, 1.0));

plt.savefig("sim_FC.png", dpi=300, bbox_inches="tight")
plt.show()



