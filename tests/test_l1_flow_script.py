import unittest
from pathlib import Path

from scripts.run_l1_flow_inference import RunConfig, required_files


class L1FlowScriptTests(unittest.TestCase):
    def test_required_files_are_under_expected_repo_paths(self):
        config = RunConfig(repo_root=Path("/repo"), steps=3)

        files = required_files(config)

        self.assertEqual(files["compression_weight"], Path("/repo/weight/compression.pth"))
        self.assertEqual(files["inference_weight"], Path("/repo/weight/inference.pth"))
        self.assertEqual(
            files["geometry"],
            Path("/repo/l1_regression_based_surrogate_model/geo.npy"),
        )

    def test_default_precision_is_fp32(self):
        config = RunConfig(repo_root=Path("/repo"))

        self.assertEqual(config.precision, "fp32")


if __name__ == "__main__":
    unittest.main()
