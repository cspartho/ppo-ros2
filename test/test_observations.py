import unittest

import numpy as np

from ppo_ros2.observations import scan_to_observation


class ScanToObservationTests(unittest.TestCase):
    def test_pools_into_fixed_size_normalized_bins(self) -> None:
        result = scan_to_observation([1.0, 2.0, 3.0, 4.0], 0.1, 4.0, 2)

        np.testing.assert_allclose(result, [0.25, 0.75])
        self.assertEqual(result.dtype, np.float32)

    def test_replaces_non_finite_ranges_conservatively(self) -> None:
        result = scan_to_observation([float("nan"), float("inf"), 0.5], 0.1, 5.0, 3)

        np.testing.assert_allclose(result, [0.02, 1.0, 0.1])

    def test_upsamples_short_scans(self) -> None:
        result = scan_to_observation([1.0, 3.0], 0.1, 4.0, 4)

        np.testing.assert_allclose(result, [0.25, 5.0 / 12.0, 7.0 / 12.0, 0.75])

    def test_rejects_empty_scans(self) -> None:
        with self.assertRaises(ValueError):
            scan_to_observation([], 0.1, 5.0, 8)


if __name__ == "__main__":
    unittest.main()
