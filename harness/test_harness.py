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


class TestAlgebraicInvariants(unittest.TestCase):
    """The formulas that decide the headline numbers, pinned so a rewrite has to break a test.

    Each of these was wrong in the repo at some point and each looked plausible while it was.
    """

    def test_analyze_mean_len_excludes_the_prompt_pass_token(self):
        import analyze as A
        src = inspect.getsource(A.report)
        self.assertIn("steps = n - da - 1", src,
                      "analyze.py is back to steps = n - da, which counts the prompt-pass token "
                      "as a decode forward")
        self.assertIn("(n - 1) / steps", src)

    def test_analyze_prefers_the_exact_verification_step_counter(self):
        import analyze as A
        src = inspect.getsource(A.report)
        self.assertIn("draft_n_verif_steps", src,
                      "mean_len should use the server's own counter when the response carries it, "
                      "rather than deriving it; llama.cpp #27676 adds that field")

    def test_decode_tok_per_joule_numerator_matches_the_decode_denominator(self):
        import bench as B
        src = inspect.getsource(B.run_matrix) if hasattr(B, "run_matrix") else inspect.getsource(B)
        self.assertIn("max(predicted_n - 1, 0) / decode_energy", src,
                      "decode energy has the prefill calibration subtracted, which removes the "
                      "first token, so the numerator must be N-1 to match")

    def test_cost_model_does_not_fit_across_the_mmvq_dispatch_boundary(self):
        import cost_model as CM
        self.assertEqual(CM.MMVQ_MAX_BATCH_SIZE, 8)
        src = inspect.getsource(CM)
        self.assertIn("w <= mmvq_max", src,
                      "widths past the MMVQ dispatch limit take a different kernel; including "
                      "them in one line drags the MTP coefficient by 24 percent")
        self.assertIn("recorded_mmvq_max(result)", src,
                      "the limit must come from the run's own record where it has one, so a "
                      "future upstream change to MMVQ_MAX_BATCH_SIZE cannot silently make this "
                      "analyser describe a build it never saw")

    def test_the_limit_is_read_from_the_result_when_the_result_records_it(self):
        import cost_model as CM
        import width_groups as WG
        no_record = {"design": {}}
        with_record = {"design": {"kernel_facts": {
            "master": {"mmvq": {"mmvq_max_batch_size": 16}}}}}
        for mod in (CM, WG):
            v, from_rec = mod.recorded_mmvq_max(no_record)
            self.assertEqual((v, from_rec), (mod.MMVQ_MAX_BATCH_SIZE, False),
                             f"{mod.__name__} should fall back and say so")
            v, from_rec = mod.recorded_mmvq_max(with_record)
            self.assertEqual((v, from_rec), (16, True),
                             f"{mod.__name__} should prefer the recorded limit")

    def test_kernel_facts_reads_the_generic_table_rather_than_assuming_it(self):
        import kernel_facts as KF
        import os
        tree = "llamacpp-master"
        if not os.path.isdir(tree):
            self.skipTest("the master tree is not present")
        f = KF.mmvq_facts(tree)
        self.assertEqual(f["mmvq_max_batch_size"], 8)
        self.assertEqual(f["generic_nwarps"]["4"], 4)
        self.assertEqual(f["generic_nwarps"]["5"], 2)
        self.assertNotIn("9", f["generic_nwarps"],
                         "the GENERIC switch has no case 9; a reader that invents one is how "
                         "width 9 got a warp count the table never assigns")

    def test_width_groups_makes_no_warp_prediction_off_the_mmvq_path(self):
        import width_groups as W
        self.assertEqual(W.MMVQ_MAX_BATCH_SIZE, 8)
        self.assertIsNone(W.warps_for(9),
                          "width 9 never reaches MMVQ, so the table predicts nothing for it; "
                          "returning 1 put it in a warp group of its own and let H8 be scored "
                          "against a prediction that was never made")
        self.assertEqual(W.warps_for(4), 4)
        self.assertEqual(W.warps_for(8), 2)

    def test_an_intention_to_treat_series_exists_and_excludes_nothing(self):
        import analyze as A
        recs = [{"arm": "a", "prompt": "p", "pass": 1, "class": "c",
                 "decode_tok_s": 10.0, "hit_cap": True},
                {"arm": "a", "prompt": "q", "pass": 1, "class": "c",
                 "decode_tok_s": 20.0, "hit_cap": False}]
        pp, _, exc, _ = A.build_series({"records": recs})
        itt, _ = A.build_series_itt({"records": recs})
        n_pp = sum(len(v) for arm in pp.values() for v in arm.values())
        n_itt = sum(len(v) for arm in itt.values() for v in arm.values())
        self.assertEqual(n_pp, 1, "the early-stopping record should leave the per-protocol series")
        self.assertEqual(n_itt, 2, "intention-to-treat keeps it; speculation moves where a "
                                   "request stops, so excluding on that selects post-treatment")
        self.assertTrue(any("early stop" in x for v in exc.values() for x in v))

    def test_the_log_cross_check_announces_when_it_skips(self):
        import cost_model as CM
        src = inspect.getsource(CM.cross_check_against_log)
        self.assertIn("skipped.append", src,
                      "a mismatched line count used to `continue` silently, which turns the "
                      "integrity check off without saying so")
        self.assertIn("SKIPPED", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
