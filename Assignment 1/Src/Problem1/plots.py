import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

halfLength, width = 0.2, 0.2
hValues = [0.02, 0.01, 0.005]
dumpPath = Path(__file__).resolve().parents[2] / "Dumps" / "Problem1"

fig = plt.figure(figsize=(18, 6))

for plotId, h in enumerate(hValues, 1):
    temperatureValues = np.loadtxt(dumpPath / f"TemperatureSolution_h_{h:.5f}.csv", delimiter=",", skiprows=1, usecols=1)
    xHalf, yCoordinates = np.arange(0, halfLength, h), np.arange(h, width, h)
    T = temperatureValues.reshape(len(yCoordinates), len(xHalf))

    xCentered = np.concatenate((-xHalf[:0:-1], xHalf))
    xFull = xCentered + halfLength
    temperatureFull = np.concatenate((T[:, :0:-1], T), axis=1)
    XFull, YFull = np.meshgrid(xFull, yCoordinates)

    axis = fig.add_subplot(1, 3, plotId, projection="3d")
    axis.plot_surface(XFull, YFull, temperatureFull, cmap="inferno", alpha=0.70, edgecolor="none")
    axis.scatter(XFull, YFull, temperatureFull, c=temperatureFull, cmap="inferno", s=12)

    axis.set_xlabel("x [m]"); axis.set_ylabel("y [m]"); axis.set_zlabel("Temperature [°C]")
    axis.set_title(f"h = {h:.3f} m")
    axis.set_xlim(0, 2 * halfLength); axis.set_ylim(0, width); axis.set_zlim(0, 100)

fig.suptitle("Full Temperature Distribution — Mirrored About x = 0.2 m", fontsize=16)
plt.tight_layout()
plt.show()