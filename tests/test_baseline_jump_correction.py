import unittest

import numpy as np

from core.analysis import correct_abrupt_baseline_jumps


class BaselineJumpCorrectionTests(unittest.TestCase):
    def test_short_pulse_is_not_a_persistent_step(self):
        x = np.linspace(-.55, 0., 501)
        y = .03 * np.exp(-((x + .25) / .07) ** 2)
        y[100:103] += .02
        corrected, audit = correct_abrupt_baseline_jumps(y)
        self.assertEqual(audit['jump_correction_count'], 0)
        self.assertGreaterEqual(audit['jump_correction_rejected_candidates'], 2)
        np.testing.assert_array_equal(corrected, y)

    def test_repeated_pulses_are_not_integrated_into_false_baseline(self):
        from core.analysis import analyze_swv_arrays
        x = np.linspace(-.55, 0., 501)
        y = .03 * np.exp(-((x + .25) / .07) ** 2)
        y[100:350:5] += .02
        corrected, audit = correct_abrupt_baseline_jumps(y)
        self.assertTrue(audit['jump_correction_unreliable'])
        np.testing.assert_array_equal(corrected, y)
        with self.assertRaisesRegex(ValueError, 'repeated pulse-like'):
            analyze_swv_arrays(x, y, crop_range=(-.55, 0), correct_baseline_jumps=True)

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
