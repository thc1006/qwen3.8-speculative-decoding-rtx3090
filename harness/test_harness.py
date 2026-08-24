"""Tests for the parts of the harness that have actually been wrong.

Every case here corresponds to a defect this study shipped and later found, so the file is a
record of them as much as a guard. `analyze.py` claimed one of these existed before it did.

Run: python3 harness/test_harness.py
"""
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze as A  # noqa: E402
import bench  # noqa: E402
import cost_model as CM  # noqa: E402
import stats as ST  # noqa: E402
import telemetry as T  # noqa: E402


class TestBootstrapArgumentOrder(unittest.TestCase):
    """(baseline, arm) and (arm, baseline) differ by a sign, and nothing in the types says so."""

    def setUp(self):
        # two classes, two prompts each, the arm uniformly 50 % faster
        self.cls = {f"c{c}_p{p}": f"c{c}" for c in (0, 1) for p in (0, 1)}
        self.base = {t: [10.0, 10.0, 10.0] for t in self.cls}
        self.arm = {t: [15.0, 15.0, 15.0] for t in self.cls}

    def test_correct_order_is_positive(self):
        iv = ST.paired_cluster_bootstrap(self.base, self.arm, self.cls, relative=True)
        self.assertAlmostEqual(iv.point, 50.0, places=6)

    def test_swapping_inverts_the_sign(self):
        """The failure mode: it does not raise, it returns a confident wrong answer."""
        iv = ST.paired_cluster_bootstrap(self.arm, self.base, self.cls, relative=True)
        self.assertLess(iv.point, 0.0)
        self.assertAlmostEqual(iv.point, -100.0 / 3, places=6)

    def test_balanced_returns_arm_first(self):
        """_balanced(arm, baseline) -> (arm, baseline). Reading it as (baseline, arm) is how a
        figure once reported a 60 % win as a 36 % loss."""
        first, second = A._balanced(self.arm, self.base)
        self.assertEqual(first["c0_p0"], [15.0, 15.0, 15.0])
        self.assertEqual(second["c0_p0"], [10.0, 10.0, 10.0])


class TestSingletonClasses(unittest.TestCase):
    """A class holding one prompt contributes no variance, and the interval must say so."""

    def test_flagged(self):
        cls = {"a_p0": "a", "b_p0": "b", "b_p1": "b"}
        base = {t: [10.0] for t in cls}
        arm = {t: [12.0] for t in cls}
        iv = ST.paired_cluster_bootstrap(base, arm, cls, relative=True)
        self.assertIn("a", iv.singleton_classes)
        self.assertTrue(iv.width_understated)

    def test_not_flagged_when_every_class_has_two(self):
        cls = {"a_p0": "a", "a_p1": "a", "b_p0": "b", "b_p1": "b"}
        base = {t: [10.0] for t in cls}
        arm = {t: [12.0] for t in cls}
        iv = ST.paired_cluster_bootstrap(base, arm, cls, relative=True)
        self.assertEqual(iv.singleton_classes, ())
        self.assertFalse(iv.width_understated)


class TestMeanLengthFormula(unittest.TestCase):
    """The first generated token comes from the prompt pass, not a decode forward.

    Leaving it in the forward count inflates that count by one, which moved the fitted marginal
    cost by 0.8 % and survived five repetitions and an r-squared of 0.9998 before being caught
    against the server's own printed figure.
    """

    def test_forward_count_excludes_the_prompt_pass(self):
        pn, accepted = 400, 250
        forwards = pn - accepted - 1
        self.assertEqual(forwards, 149)
        mean_len = (pn - 1) / forwards
        self.assertAlmostEqual(mean_len, 399 / 149, places=9)

    def test_the_wrong_form_is_close_enough_to_have_hidden(self):
        """Why it was not obvious: the two differ by well under a percent."""
        pn, accepted = 400, 250
        wrong = pn / (pn - accepted)
        right = (pn - 1) / (pn - accepted - 1)
        self.assertLess(abs(wrong - right) / right, 0.01)

    def test_collect_uses_the_corrected_form(self):
        """Against the assignment, not the whole function: collect's docstring quotes the
        formula, so checking the source as a block passes on the comment even when the code
        below it is wrong. That is how this test first passed on the reverted copy."""
        lines = [l.strip() for l in inspect.getsource(CM.collect).splitlines()
                 if l.strip().startswith("mean_len =")]
        self.assertEqual(len(lines), 1, f"expected one mean_len assignment, got {lines}")
        self.assertEqual(lines[0], "mean_len = (pn - 1) / forwards", lines[0])


class TestSettleFloorIsPassedThrough(unittest.TestCase):
    """--settle-floor measured the floor, printed it, and never handed it to settle_gpu.

    The flag existed for cards other than the calibrated 3090, so it raised on exactly the
    hardware it was written for, and nothing noticed until an A6000 arrived.
    """

    def test_settle_gpu_needs_one_of_the_two(self):
        with self.assertRaises(ValueError):
            T.settle_gpu(0, target_temp_c=None, idle_floor_c=None, max_wait_s=0)

    def test_floor_plus_margin_is_the_target(self):
        out = T.settle_gpu(0, target_temp_c=None, idle_floor_c=40.0, margin_c=8.0, max_wait_s=0)
        self.assertEqual(out["target_c"], 48.0)
        self.assertEqual(out["idle_floor_c"], 40.0)

    def test_the_call_site_passes_it(self):
        """Shaped as a source check on purpose: the defect was a missing argument at one call
        site, which no behavioural test of settle_gpu itself can see."""
        src = inspect.getsource(bench.run_matrix)
        for needed in ("idle_floor_c=measured_floor", "margin_c=settle_margin_c"):
            # the message carries the missing argument, not the whole function
            self.assertTrue(needed in src, f"run_matrix does not pass {needed} to settle_gpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
