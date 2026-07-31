import unittest

import numpy as np
import torch

from tools.pollution_transport_h5 import upwind_step


class PollutionTransportH5Tests(unittest.TestCase):
    def test_upwind_step_preserves_shape_and_finite_values_on_small_grid(self):
        concentration = torch.zeros((3, 4, 5), dtype=torch.float32)
        source = torch.zeros_like(concentration)
        source[1, 2, 2] = 0.1
        velocity = torch.zeros_like(concentration)
        solid = torch.zeros_like(concentration)

        updated = upwind_step(
            concentration,
            velocity,
            velocity,
            velocity,
            solid,
            dt=0.5,
            source=source,
            max_iter=3,
        )

        self.assertEqual(updated.shape, concentration.shape)
        self.assertTrue(np.isfinite(updated.numpy()).all())
        self.assertGreater(float(updated[1, 2, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
