import unittest

import numpy as np

from core.analysis import analyze_swv_arrays, swv_trace_integrity


class SwvTraceIntegrityTests(unittest.TestCase):
    def test_uniform_ascending_and_descending_sweeps_pass(self):
        ascending = np.linspace(-0.6, 0.0, 301)
        current = np.sin(np.linspace(0, np.pi, ascending.size))

        self.assertEqual(swv_trace_integrity(ascending, current)["qa_status"], "OK")
        self.assertEqual(
            swv_trace_integrity(ascending[::-1], current[::-1])["qa_status"],
            "OK",
        )

    def test_missing_voltage_block_is_reported(self):
        voltage = np.linspace(-0.6, 0.0, 301)
        keep = ~((voltage > -0.385) & (voltage < -0.300))
        result = swv_trace_integrity(voltage[keep], np.zeros(int(keep.sum())))

        self.assertEqual(result["qa_status"], "FAILED")
        self.assertGreaterEqual(result["qa_estimated_missing_points"], 40)
        self.assertIn("incomplete voltage sweep", result["qa_error"])

    def test_incomplete_sweep_is_rejected_before_peak_analysis(self):
        voltage = np.linspace(-0.6, 0.0, 301)
        current = np.exp(-((voltage + 0.3) / 0.04) ** 2)
        keep = ~((voltage > -0.385) & (voltage < -0.300))

        with self.assertRaisesRegex(ValueError, "incomplete voltage sweep"):
            analyze_swv_arrays(voltage[keep], current[keep], smooth_window=0)


if __name__ == "__main__":
    unittest.main()
