import unittest

import numpy as np

from core.analysis import correct_abrupt_baseline_jumps


class BaselineJumpCorrectionTests(unittest.TestCase):
    def test_removes_persistent_step_without_changing_pre_step_signal(self):
        x = np.linspace(-0.55, 0.0, 301)
        underlying = 0.002 * x + 0.030 * np.exp(-((x + 0.25) / 0.07) ** 2)
        stepped = underlying.copy()
        stepped[x >= -0.36] += 0.010

        corrected, audit = correct_abrupt_baseline_jumps(
            stepped, sensitivity=6.0, minimum_jump_uA=0.002,
        )

        self.assertTrue(audit["jump_correction_applied"])
        self.assertEqual(audit["jump_correction_count"], 1)
        self.assertLess(np.max(np.abs(corrected - underlying)), 5e-4)

    def test_smooth_peak_is_not_treated_as_a_step(self):
        x = np.linspace(-0.55, 0.0, 301)
        smooth_peak = 0.030 * np.exp(-((x + 0.25) / 0.035) ** 2)
        corrected, audit = correct_abrupt_baseline_jumps(
            smooth_peak, sensitivity=6.0, minimum_jump_uA=0.002,
        )
        self.assertFalse(audit["jump_correction_applied"])
        np.testing.assert_allclose(corrected, smooth_peak)
