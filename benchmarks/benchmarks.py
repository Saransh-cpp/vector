# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
import vector


class TimeCreation:
    def time_awkward_creation(self):
        array = vector.awk(
            [
                [
                    {x: np.random.normal(0, 1) for x in ("px", "py", "pz")}
                    for inner in range(np.random.poisson(1.5))
                ]
                for outer in range(50)
            ]
        )

    def time_numpy_creation(self):
        np.random.normal(0, 1, 150).view([(x, float) for x in ("x", "y", "z")]).view(vector.MomentumNumpy3D).reshape(5, 5, 2)


class MemSuite:
    # def mem_numpy_creation(self):
        # return np.random.normal(0, 1, 150).view([(x, float) for x in ("x", "y", "z")]).view(vector.MomentumNumpy3D).reshape(5, 5, 2)

    def mem_awkward_creation(self):
        return vector.awk(
            [
                [
                    {x: np.random.normal(0, 1) for x in ("px", "py", "pz")}
                    for inner in range(np.random.poisson(1.5))
                ]
                for outer in range(50)
            ]
        )
