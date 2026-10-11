from tvb.simulator.lab import connectivity
import numpy as np

conn = connectivity.Connectivity.from_file("connectivity_68.zip")
print(conn.weights.shape)