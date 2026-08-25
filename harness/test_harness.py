"""Tests for the parts of the harness that have actually been wrong.

Every case here corresponds to a defect this study shipped and later found, so the file is a
record of them as much as a guard. `analyze.py` claimed one of these existed before it did.

Run: python3 harness/test_harness.py
"""
import collections
import inspect
import json
import time
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
        self.assertEqual(lines[0], "mean_len = speclen.mean_len(rec)", lines[0])
        # The formula moved rather than went away, so pin it where it now lives, still against
        # the assignment rather than the block for the reason in this docstring.
        import speclen
        body = [l.strip() for l in inspect.getsource(speclen.mean_len).splitlines()
                if l.strip() and not l.strip().startswith("#")]
        self.assertIn("return 1.0 + accepted / f", body,
                      f"speclen no longer derives it the corrected way: {body}")


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

    def test_mean_len_excludes_the_prompt_pass_token(self):
        """Checked on the value now, not on the source: the formula moved into speclen.py.

        A source assertion could only ever say the right characters are present somewhere. This
        says the number is right, which is what the earlier `steps = n - da` bug got wrong.
        """
        import speclen
        rec = {"predicted_n": 400, "timings": {"t_draft_n_accepted": 305}}
        self.assertEqual(speclen.forwards(rec), 94,
                         "counting the prompt-pass token as a decode forward gives 95")
        self.assertAlmostEqual(speclen.mean_len(rec), 399 / 94)

    def test_mean_len_prefers_the_exact_verification_step_counter(self):
        """llama.cpp #27676 adds draft_n_verif_steps; when it arrives it must win."""
        import speclen
        # a record whose counter disagrees with the derivation, so which one is used is visible
        rec = {"predicted_n": 400,
               "timings": {"t_draft_n_accepted": 305, "draft_n_verif_steps": 93}}
        self.assertEqual(speclen.forwards(rec), 93,
                         "the derivation would say 94; the server's own count must win")
        self.assertAlmostEqual(speclen.mean_len(rec), 1.0 + 305 / 93)
        self.assertTrue(speclen.is_exact(rec))

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

    def test_the_ordering_fixes_are_opt_in(self):
        """Both change the design, so neither may switch itself on mid-study.

        phase_a, phase_r, phase_r2, phase_kv and phase_nmax all ran under the fixed order. A
        default that silently permuted would make phase_l onwards a different experiment from the
        phases it is compared against.
        """
        import bench as B
        sig = inspect.signature(B.run_matrix)
        self.assertIs(sig.parameters["shuffle_prompts"].default, False)
        self.assertIs(sig.parameters["latin_arms"].default, False)
        src = inspect.getsource(B.run_matrix)
        self.assertIn("prompt_order_by_pass", src,
                      "the order actually used has to be in the result, or a later reader cannot "
                      "tell which of the two designs produced a file")
        self.assertIn('"ordinal": _ord', src,
                      "position within the arm-pass has to be recorded, or it cannot be adjusted "
                      "for later")

    def test_latin_arms_closes_the_rotation(self):
        import bench as B
        src = inspect.getsource(B.run_matrix)
        self.assertIn("passes = len(arms)", src,
                      "seven arms over five passes leaves each arm visiting five of seven order "
                      "positions, and a different five; only len(arms) passes closes it")

    def test_both_power_fields_are_sampled(self):
        """power.draw is time-averaged on Ampere; power.draw.instant is not.

        Querying power.draw beside power.draw.average returns the same number on every sample, so
        integrating power.draw alone smears a request over the second around it. Both are sampled
        so a file carries the figure the earlier phases used and the sharper one, and neither set
        of results becomes incomparable.
        """
        import telemetry as T
        self.assertIn("power.draw.instant", T._NVSMI_FIELDS)
        self.assertIn("power.draw", T._NVSMI_FIELDS)
        src = inspect.getsource(T.PowerSampler)
        self.assertIn("energy_j_instant", src)
        self.assertIn("energy_instant_vs_average_pct", src,
                      "the gap between the two integrals is the quantity that says how much the "
                      "averaging mattered, so it belongs in the record rather than being "
                      "recomputable in principle")

    def test_an_incomplete_result_is_announced(self):
        """A table in the README was filled from a file the run was still appending to.

        phase_nmax at 1025 of 1050 gave a DFlash2 coefficient of 0.2479; finished it gives 0.2481.
        Small enough to survive review and wrong, and nothing said the file was short.
        """
        import completeness as CO
        short = {"design": {"passes": 3},
                 "arms": {"a": {}, "b": {}},
                 "records": [{"arm": "a", "prompt": f"p{i}", "pass": 1} for i in range(5)]}
        n, expected, _ = CO.completeness(short)
        self.assertEqual((n, expected), (5, 30))
        self.assertFalse(CO.warn_if_incomplete(short))

        whole = dict(short, records=[{"arm": a, "prompt": f"p{i}", "pass": p}
                                     for a in ("a", "b") for i in range(5) for p in (1, 2, 3)])
        self.assertTrue(CO.warn_if_incomplete(whole))

    def test_the_analysers_call_it(self):
        """Every analyser must consult completeness before it reports.

        analyze_depth reads it per rung and gates the cliff verdict on it rather than printing
        the generic notice, so the assertion is on the module being consulted, not on one
        function name.
        """
        import analyze, analyze_depth, cost_model, width_groups
        for mod in (analyze, analyze_depth, cost_model, width_groups):
            src = inspect.getsource(mod)
            self.assertTrue("warn_if_incomplete" in src or "completeness(" in src,
                            f"{mod.__name__} can be pointed at a half-written file and would say "
                            f"nothing about it")

    def test_the_cache_check_tests_what_the_matrix_declared(self):
        """The detector asserted one invariant and reported it as the other.

        It fired on any t_cache_n > 0 with "despite cache_prompt=False" written into the
        message. phase_l sets CACHE_PROMPT = True on purpose, because every request in an arm
        shares one filler of up to 96 K tokens and re-prefilling it per request would cost more
        than the decode being measured. So every phase_l request was an incident against a
        condition phase_l never claimed: 30 records, 30 incidents, on course for 900 over the
        ladder, which would bury a real one.
        """
        import bench
        src = inspect.getsource(bench)
        self.assertIn("if not cache_prompt and cache_n > 0:", src,
                      "the hit check must be conditional on caching having been asked to be off")
        self.assertIn("elif cache_prompt and cache_n == 0:", src,
                      "with caching on, the failure is a miss: the shared prefix was re-prefilled "
                      "rather than reused, which at 96 K is most of the request")
        self.assertIn("prompt_cache_miss", src)

        import importlib
        import os
        import sys
        mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrices")
        if mdir not in sys.path:
            sys.path.insert(0, mdir)
        m = importlib.import_module("phase_l")
        self.assertTrue(getattr(m, "CACHE_PROMPT", False),
                        "this test is anchored to phase_l actually declaring CACHE_PROMPT; if that "
                        "changes the detector's two branches need revisiting rather than this "
                        "assertion being deleted")
    def test_the_depth_analyser_reads_fields_that_exist(self):
        """analyze_depth.py looked up three record fields by names the records do not use.

        `prompt_tag` raised KeyError right before the bootstrap, so the ladder's primary result -
        speedup against the matching baseline per depth - never printed. `prompt_class` would have
        put every prompt in class "?" and silently discarded the stratification. `mean_len` is not
        a record field at all, so the acceptance-vs-depth table printed "-" in every cell instead
        of saying it could not compute anything. Two of the three fail without a traceback.
        """
        import json
        import os
        import analyze_depth  # noqa: F401
        import speclen

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Comments stripped first. The note explaining this fix quotes the dead lookups, and
        # scanning the raw source matched that note rather than any code - the same way
        # test_collect_uses_the_corrected_form once passed on a docstring while the code under
        # it was wrong.
        code = "\n".join(l for l in inspect.getsource(analyze_depth).splitlines()
                         if not l.strip().startswith("#"))
        for dead in ('r["prompt_tag"]', '"prompt_class"', 'r.get("mean_len")'):
            self.assertNotIn(dead, code, f"{dead} is not a field any record carries")

        # anchored to a real file when one exists, so a rename of the real fields fails here too
        path = os.path.join(here, "results", "phase_c.json")
        if os.path.exists(path):
            with open(path) as fh:
                rec = json.load(fh)["records"][0]
            for live in ("prompt", "class", "predicted_n", "timings"):
                self.assertIn(live, rec, f"records no longer carry {live}")
            self.assertNotIn("mean_len", rec,
                             "if records start carrying mean_len, speclen is no longer the only "
                             "definition and this test should be revisited rather than deleted")

    def test_one_definition_of_mean_length(self):
        """Three files derived it and two had already drifted apart.

        analyze.py's own comment recorded that the first-token correction reached cost_model.py
        and not itself. cost_model.py then never consulted draft_n_verif_steps, so the two would
        have parted company again as soon as llama.cpp #27676 landed and that counter began to
        arrive. The identity below is what keeps the derived and exact paths the same quantity:
        each verification step emits one token of its own plus the drafts it accepted.
        """
        import analyze
        import analyze_depth
        import cost_model
        import speclen

        for mod in (analyze, cost_model, analyze_depth):
            self.assertIn("speclen.", inspect.getsource(mod),
                          f"{mod.__name__} derives mean length itself instead of calling speclen")

        derived = {"predicted_n": 400, "timings": {"t_draft_n": 900, "t_draft_n_accepted": 305}}
        self.assertEqual(speclen.forwards(derived), 400 - 305 - 1)
        self.assertAlmostEqual(speclen.mean_len(derived), 399 / 94)
        self.assertFalse(speclen.is_exact(derived))

        # with the counter present the answer comes from it, and on a consistent record the two
        # paths must agree rather than merely both being plausible
        exact = {"predicted_n": 400,
                 "timings": {"t_draft_n": 900, "t_draft_n_accepted": 305,
                             "draft_n_verif_steps": 94}}
        self.assertEqual(speclen.forwards(exact), 94)
        self.assertAlmostEqual(speclen.mean_len(exact), speclen.mean_len(derived))
        self.assertTrue(speclen.is_exact(exact))

        # a record that never drafted emits one token per forward, which is the figure the
        # speculative arms are measured against, not a missing value
        self.assertAlmostEqual(
            speclen.mean_len({"predicted_n": 400, "timings": {"t_draft_n_accepted": 0}}), 1.0)
        self.assertIsNone(speclen.mean_len({"predicted_n": 1, "timings": {}}))
    def test_the_warp_scorer_knows_every_build_it_is_handed(self):
        """It knew three builds and the v2 run produced four.

        The collector passes forced_down2 third, where the v1 collector passed forced_down, and
        main() read its arguments by position. forced_down and forced_down2 differ at width 1 and
        nowhere else any arm runs - and width 1 is the greedy baseline. Scored against the
        forced_down row, the file takes the branch that says the baseline is part of the
        intervention and cannot also be its control, so the one gate the rebuild exists to make
        applicable is skipped and a paragraph about a different build is printed instead.
        """
        import warp_intervention as W

        self.assertIn("forced_down2", W.TABLES)
        self.assertEqual(W.TABLES["forced_down2"][1], 4,
                         "leaving width 1 at four warps is the whole point of this build: the "
                         "drafter decodes one token at a time and the greedy baseline runs there")
        self.assertEqual(W.TABLES["forced_down2"][3], 2)
        self.assertEqual(W.TABLES["forced_down2"][4], 2)
        self.assertEqual(W.TABLES["control2"], W.TABLES["control"],
                         "control2 is stock; if it were not it could not be the guard")

        # longest match, or the v2 file is labelled v1 and control2 is labelled control
        for name, want in (("results/phase_warp_v2_forced_down2.json", "forced_down2"),
                           ("results/phase_warp_v2_forced_down.json", "forced_down"),
                           ("results/phase_warp_v2_control2.json", "control2"),
                           ("results/phase_warp_v2_control.json", "control"),
                           ("results/phase_warp_v2_forced_up.json", "forced_up")):
            self.assertEqual(W.build_of(name), want, name)
        self.assertIsNone(W.build_of("results/phase_a.json"),
                          "a file that names no build must be refused, not guessed at")

    def test_the_warp_scorer_checks_its_tables_against_what_was_built(self):
        """The docstring promised validate_tables() and no such function existed.

        One mention, zero definitions. A build whose table this file had wrong would have been
        scored against the wrong prediction in silence, which is what happened when the v2 run
        added a fourth build and the table was not updated.
        """
        import os
        import tempfile
        import warp_intervention as W

        self.assertTrue(callable(getattr(W, "validate_tables", None)))

        block = """    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
                return 4;
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if"""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "warp_builds_v2_forced_down2_table.txt"), "w") as fh:
                fh.write(block)
            problems, checked, missing = W.validate_tables(["forced_down2"], dirs=(d,))
            self.assertEqual((problems, checked, missing), ([], ["forced_down2"], []))

            # the same source scored as forced_up must be caught, not accepted
            with open(os.path.join(d, "warp_builds_v2_forced_up_table.txt"), "w") as fh:
                fh.write(block)
            problems, checked, _ = W.validate_tables(["forced_up"], dirs=(d,))
            self.assertTrue(problems, "a table that disagrees with the built source must be "
                                      "reported, not accepted")
            self.assertEqual(checked, [])

        # absent is reported as unchecked rather than treated as agreement
        with tempfile.TemporaryDirectory() as d:
            problems, checked, missing = W.validate_tables(["control"], dirs=(d,))
            self.assertEqual((problems, checked, missing), ([], [], ["control"]))
    def test_the_cliff_verdict_refuses_a_rung_still_being_written(self):
        """The driver calls analyze_depth after every rung, so the deepest file is routinely
        half-written when it runs.

        The three other analysers announce a short file; this one did not, and it is the one that
        decides whether a 25x collapse reproduced. A partial rung is not merely noisier: records
        land one arm-pass at a time and the arm order rotates between passes, so a short file
        holds an unbalanced set of arms and its median is a different estimator.
        """
        import json
        import os
        import subprocess
        import tempfile

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = os.path.join(here, "results", "phase_l_8192.json")
        if not os.path.exists(src):
            self.skipTest("no phase_l rung on disk to build the case from")
        with open(src) as fh:
            whole = json.load(fh)

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "phase_l_8192.json"), "w") as fh:
                json.dump(whole, fh)
            # a deep rung, past the reported threshold, holding half a run
            deep = dict(whole, records=whole["records"][: len(whole["records"]) // 2])
            with open(os.path.join(d, "phase_l_98304.json"), "w") as fh:
                json.dump(deep, fh)
            out = subprocess.run(
                ["python3", os.path.join(here, "harness", "analyze_depth.py"),
                 os.path.join(d, "phase_l_8192.json"), os.path.join(d, "phase_l_98304.json")],
                capture_output=True, text=True, cwd=here).stdout

        self.assertIn("STILL BEING WRITTEN", out,
                      "a short rung must be named before anything is computed from it")
        self.assertIn("VERDICT WITHHELD", out,
                      "the cliff verdict must not be taken from a rung that is half a run")
        self.assertNotIn("REPRODUCES", out)
        self.assertNotIn("DOES NOT REPRODUCE", out)

    def test_retention_says_which_rung_each_method_is_anchored_to(self):
        """Retention is per method against its own shallowest rung.

        A method absent from the shallowest rung is measured against a deeper one and prints
        100 % there, which read beside a column that fell looks like better retention. The
        columns are then not on a common scale, and nothing said so.
        """
        import analyze_depth
        code = "\n".join(l for l in inspect.getsource(analyze_depth).splitlines()
                         if not l.strip().startswith("#"))
        self.assertIn("ANCHORS DIFFER", code)
        self.assertIn("anchors", code)
    def test_the_width_control_checks_they_verified_at_the_same_width(self):
        """n_max is what was asked for. The width verified is one plus what the drafter proposed.

        On phase_nmax the two drafters match to 0.00 columns at widths 3, 5 and 7 and agree on
        25 of 25 prompts at each. At width 9 DFlash2 fills 87 % of its budget and MTP 99 %, so
        they verify at 7.94 and 8.93 columns, one inside MMVQ_MAX_BATCH_SIZE and one past it,
        and they agree on 8 of 25. That was printed as the width account failing. The control
        had never applied there.
        """
        import width_groups as W
        code = "\n".join(l for l in inspect.getsource(W).splitlines()
                         if not l.strip().startswith("#"))
        self.assertIn("NOT A CONTROL", code,
                      "a control between two arms that verified at different widths must say so "
                      "rather than report a disagreement")
        self.assertIn("eff_width", code)
        self.assertIn("speclen.forwards", code,
                      "effective width has to come from the drafts actually proposed per "
                      "verification step, not from the flag")

    def test_the_partition_is_not_reported_as_causal(self):
        """The forced-warp intervention has run and the observational verdict predates it.

        The partition matches calc_nwarps exactly. Forcing that table moves the kernel by up to
        26.68 % of its runtime and moves no output byte in 150 records per direction, with SASS
        showing the edit reached only the kernels at the edited ncols_dst and dispatch showing
        Ampere runs them. A mechanism that cannot change the text cannot change where two texts
        diverge.
        """
        import width_groups as W
        code = inspect.getsource(W)
        self.assertNotIn("H8 SUPPORTED. The partition is exactly the warp-count table.", code,
                         "that sentence asserts a cause the intervention refuted")
        self.assertIn("H8 IS NOT A CAUSAL CLAIM", code)
        self.assertIn("warp_intervention_v2", code,
                      "the verdict should point at the evidence that settled it")

    def test_fork_agreement_separates_family_from_width(self):
        """The caption asks for drafters sharing ONLY their width, and the count included arms
        sharing far more.

        On phase_c the three DFlash2 arms differ only in drafter quantization and agree on 25 of
        25, which alone produces a pooled 20 of 25 and says nothing about width. The count for
        the caption's own criterion is 7 of 20.
        """
        import divergence_report as DR
        code = "\n".join(l for l in inspect.getsource(DR).splitlines()
                         if not l.strip().startswith("#"))
        self.assertIn("NOT MEASURABLE HERE", code,
                      "a file where no width carries two families cannot evaluate the criterion, "
                      "and a bare zero would read as disagreement")
        self.assertTrue(hasattr(DR, "_family") and hasattr(DR, "_width"))
        arms = {"a": {"extra_args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]}}
        self.assertEqual(DR._width(arms, "a"), 5)
        self.assertEqual(DR._family(arms, "a"), "draft-mtp")
        self.assertEqual(DR._width({}, "missing"), 1)
    def test_cross_device_names_the_toolchain_before_blaming_the_cards(self):
        """It had never been run. On its first pair both controls failed.

        Acceptance differed by up to 0.095 and only 29 of 125 fork positions matched, and the
        report said something other than the device is varying. The something was in the file:
        different driver, different kernel, different machine. Two cards of one architecture
        share CUDA kernels only when one toolchain built them, so those controls could not have
        held and their failure says nothing about the cards. Worse, the bandwidth elasticity
        below carried a caveat naming the power budget as the second variable and missed the
        toolchain as a third.
        """
        import analyze_cross_device as X

        self.assertTrue(callable(getattr(X, "_build_env", None)))
        self.assertTrue(callable(getattr(X, "_env_diff", None)))

        a = {"env": {"gpu": "NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02, 8.6",
                     "kernel": "7.0.0-30-generic", "host": "3090"}}
        b = {"env": {"gpu": "NVIDIA RTX A6000, 49140 MiB, 580.95.05, 8.6",
                     "kernel": "6.1.0-39-amd64", "host": "mailer"}}
        fields = {k for k, _, _ in X._env_diff(X._build_env(a), X._build_env(b))}
        self.assertIn("driver", fields)
        self.assertIn("kernel", fields)
        self.assertIn("host", fields)
        self.assertEqual(X._env_diff(X._build_env(a), X._build_env(a)), [],
                         "two files from one installation must report no difference")

        code = "\n".join(l for l in inspect.getsource(X).splitlines()
                         if not l.strip().startswith("#"))
        self.assertIn("THIS IS NOT A BANDWIDTH MEASUREMENT", code,
                      "the elasticity must say so when the step carries the toolchain too")
        self.assertIn("build environment, which decides whether the controls CAN hold", code,
                      "the environment belongs before the controls, not after them")
    def test_relative_has_no_default(self):
        """It switches the unit between a percentage and raw tok/s and both print plausibly.

        A caller that omits it gets an absolute difference where it almost certainly wanted a
        percentage, and nothing says so. Every caller in this repo passes it; the default is
        removed so a new one cannot quietly not.
        """
        import inspect as _i
        import stats as ST
        sig = _i.signature(ST.paired_cluster_bootstrap)
        self.assertIs(sig.parameters["relative"].default, _i.Parameter.empty)
        with self.assertRaises(TypeError):
            ST.paired_cluster_bootstrap({"p": [1.0]}, {"p": [2.0]}, {"p": "c"})

    def test_intervals_that_nearly_touch_zero_are_marked(self):
        """The percentile bootstrap undercovers at this sample size, and one-sidedly.

        Measured on 25 prompts, 800 replications: 90.9 % for a normal draw, 90.6 % uniform,
        88.0 % heavy-tailed, against a nominal 95 %. At 50 prompts it recovers to 92.4 %, and a
        t interval on the same draws reaches 94.1 %. The intervals come out too narrow, so the
        verdicts that move when the coverage is restored are the ones already close to zero.
        Phase A's five arms clear it comfortably; Phase R2's mtp-n7 rows do not.
        """
        import stats as ST

        far = ST.Interval(point=59.77, lo=56.95, hi=62.75, n_clusters=25)
        self.assertAlmostEqual(far.margin_half_widths, 56.95 / ((62.75 - 56.95) / 2), places=6)
        self.assertFalse(far.near_zero)

        near = ST.Interval(point=6.62, lo=1.36, hi=11.95, n_clusters=25)
        self.assertLess(near.margin_half_widths, 1.3)
        self.assertTrue(near.near_zero)

        spanning = ST.Interval(point=0.5, lo=-2.0, hi=3.0, n_clusters=25)
        self.assertEqual(spanning.margin_half_widths, 0.0)
        self.assertFalse(spanning.near_zero, "an interval already spanning zero is not 'near' it")

        import analyze
        self.assertIn("COVERAGE NOTE", inspect.getsource(analyze),
                      "a report carrying a verdict inside the margin must say so")
    def test_elasticity_marks_intervals_the_pin_did_not_hold(self):
        """A pin binds only while the power limit does not.

        Phase R2 pins the SM clock, and it holds at five of seven conditions. At the top two a
        speculative arm draws more at the same clock, reaches the cap first, and lands short: at
        sm1700 the methods sit at 1710, 1698 and 1708 MHz against one request. The report
        explained that in a section below the elasticity tables while the tables themselves
        printed unmarked, and the headline compute comparison, 0.27 against 0.76 and 0.81,
        crosses exactly that condition.
        """
        import elasticity as E
        code = "\n".join(l for l in inspect.getsource(E).splitlines()
                         if not l.strip().startswith("#"))
        self.assertIn("unmatched = set()", code)
        self.assertIn("[unmatched]", code,
                      "each row of an interval that crosses an unmatched condition must say so")
        self.assertIn("did not", code)
    def test_a_text_that_merely_ended_is_not_a_fork(self):
        """compare_outputs stopped its scan at min(len) and called that index the fork.

        When one text is a prefix of the other the scan ends because a text ran out, not because
        a character differed, and first_diff_char is that length. Five consumers read
        `same if identical else first_diff_char`, which has no state for it, so it would have
        plotted as a fork at a position where nothing disagrees. It has not happened here, 0 of
        4673 records, and it can: every arm stops at a token cap and tokens are not a fixed
        number of characters, so two runs of equal token count differ in length.
        """
        import quality as Q

        self.assertFalse(Q.compare_outputs("abc", "abd").prefix_only)
        self.assertEqual(Q.compare_outputs("abc", "abd").first_diff_char, 1 + 1)
        self.assertTrue(Q.compare_outputs("abc", "abcdef").prefix_only)
        self.assertTrue(Q.compare_outputs("abcdef", "abc").prefix_only)
        self.assertFalse(Q.compare_outputs("abc", "abc").prefix_only)

        # the position helper collapses every unusable case to None
        self.assertEqual(Q.fork_position({"identical": False, "first_diff_char": 42,
                                          "len_ref": 100, "len_arm": 120}), 42)
        self.assertIsNone(Q.fork_position({"identical": True}))
        self.assertIsNone(Q.fork_position(None))
        self.assertIsNone(Q.fork_position({"identical": False, "first_diff_char": 100,
                                           "len_ref": 100, "len_arm": 120}),
                          "a record written before prefix_only existed is caught by its lengths")

        # the table helper keeps them apart
        self.assertEqual(Q.fork_cell({"identical": True}), "SAME")
        self.assertEqual(Q.fork_cell(None), "-")
        self.assertEqual(Q.fork_cell({"identical": False, "first_diff_char": 100,
                                      "len_ref": 100, "len_arm": 120}), "PREFIX")
        self.assertEqual(Q.fork_cell({"identical": False, "first_diff_char": 7,
                                      "len_ref": 100, "len_arm": 120}), 7)

        import analyze_cross_device, divergence_report, warp_intervention, width_groups
        for mod in (divergence_report, warp_intervention, width_groups, analyze_cross_device):
            self.assertIn("fork_cell", inspect.getsource(mod),
                          f"{mod.__name__} still decides the three states itself")
    def test_the_depth_ladder_shows_the_clock_it_moved(self):
        """Depth moves the core clock, so part of any decline is not depth.

        A deeper rung moves more memory traffic and takes budget from the core inside one power
        limit. Measured across this ladder the SM clock falls 2.02 % from the 8 K rung to the
        82 K one with the temperature flat, so it is budget and not throttling. Phase R2 puts the
        baseline's compute elasticity near the top of the range at 0.266, which makes that drift
        worth 0.54 % of throughput against a 28.7 % decline. Small, and it belongs in the report
        rather than in a reader's head.
        """
        import analyze_depth
        code = "\n".join(l for l in inspect.getsource(analyze_depth).splitlines()
                         if not l.strip().startswith("#"))
        self.assertIn("CLOCK, WHICH DEPTH MOVES ON ITS OWN", code)
        self.assertIn("0.266", code, "the elasticity that converts drift into throughput")
        self.assertIn("sm_clock_mean_mhz", code)
    def test_the_two_power_integrals_are_compared_per_arm(self):
        """power.draw on Ampere is a rolling average, and averaging is not neutral here.

        A steady load integrates the same either way. A speculative one does not: a drafter pass
        at width 1 and a wide verification are different power states, and the average smooths
        the peak. On the depth ladder the two integrals agree to 0.00 to 0.34 % on the baselines
        and differ by 0.58 to 1.97 % on the speculative arms, always positive. The averaged field
        understates exactly the arms whose energy is compared against a baseline, so the bias is
        one sided and no interval covers it.
        """
        import analyze
        code = "\n".join(l for l in inspect.getsource(analyze).splitlines()
                         if not l.strip().startswith("#"))
        self.assertIn("energy_instant_vs_average_pct", code,
                      "the gap is recorded per record and has to be reported per arm")
        self.assertIn("averaged against instantaneous power", code)
    def test_the_energy_window_records_what_it_covered(self):
        """The integral runs first sample to last, and the record carried only a count.

        The sampler queries nvidia-smi and then waits interval_s, so its period is the sum, not
        interval_s. Measured here the query is about 16 ms against a 100 ms wait, a 118 ms
        period, and 39 samples over a 4.65 s request leave roughly 4 % of the wall outside the
        window. Assuming the nominal 0.10 s instead gives 18 %, which is how a reader without
        this field would get it wrong by four times. Now the span is recorded and the fraction
        is arithmetic rather than an estimate.
        """
        import time
        import telemetry as T

        with T.sampling(0, interval_s=0.10) as s:
            time.sleep(0.6)
        d = s.summary()
        if not d.get("n_power_samples"):
            self.skipTest("no GPU telemetry on this host")
        self.assertIn("sample_span_s", d)
        self.assertIsNotNone(d["sample_span_s"])
        self.assertLess(d["sample_span_s"], 0.6,
                        "the span cannot exceed the window it was sampled in")
        period = d["sample_span_s"] / max(d["n_power_samples"] - 1, 1)
        self.assertGreater(period, 0.10,
                           "the period is the query plus the wait, so it must exceed interval_s")

        import analyze
        self.assertIn("energy window coverage", inspect.getsource(analyze))
    def test_the_port_guard_refuses_a_stale_server(self):
        """The README advertises this guard; it had never been exercised.

        A killed-but-unreaped llama-server keeps answering /health, and a contributor to another
        study published three rows measured against one. The check walks up from whoever owns the
        port looking for our own pid, so the case that matters is a stale owner with no relation
        to the server we think we started.

        The degenerate input is the other half: every process reaches pid 1, so a walk that
        accepts an ancestor match would hold vacuously against it.
        """
        import os
        import socket
        import subprocess
        import sys
        import telemetry as T

        port = 19741
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                self.skipTest(f"port {port} is in use on this host")

        self.assertRaises(RuntimeError, T.assert_port_owned_by, port, 1)
        self.assertRaises(RuntimeError, T.assert_port_owned_by, port, 0)

        hold = (f"import socket,time;s=socket.socket();s.setsockopt(1,2,1);"
                f"s.bind(('127.0.0.1',{port}));s.listen(1);time.sleep(20)")
        stale = subprocess.Popen([sys.executable, "-c", hold])
        other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"])
        try:
            for _ in range(40):
                if T.pid_owning_port(port):
                    break
                time.sleep(0.1)
            if not T.pid_owning_port(port):
                self.skipTest("could not observe the listening socket on this host")
            # the stale holder is not our server, and sharing an ancestor is not ownership
            self.assertRaises(RuntimeError, T.assert_port_owned_by, port, other.pid)
            # our own descendant is
            T.assert_port_owned_by(port, os.getpid())
        finally:
            stale.terminate()
            other.terminate()
            stale.wait()
            other.wait()



class TestDeviceFacts(unittest.TestCase):
    """Three ways devices.py answered a question it had not measured.

    All three were silent: a floor read off a still-cooling card, a card that disappeared because
    one of its ten fields was unsupported, and a stock power limit invented for a card that never
    reported one.
    """

    def test_a_cooling_card_is_not_mistaken_for_a_settled_one(self):
        """nvidia-smi reports whole degrees, so a 1 C/sample fall clears any pairwise tolerance.

        The old test compared each sample with the one before it and called 78, 77, 76, 75 stable,
        returning 75 C. Every arm then met a gate of 75 + margin instantly, which is the no-op the
        function exists to prevent.
        """
        import devices
        real = devices._temp
        try:
            taken = {}
            for series, expect_floor, label in (
                    ([78, 77, 76, 75, 74, 73, 72, 71, 70, 69] + [68] * 6, 68, "cooling then flat"),
                    ([60, 60, 61, 60, 60, 60], 60, "flat throughout")):
                calls = []
                devices._temp = lambda _i, q=series, c=calls: (
                    q[len(c)] if len(c) < len(q) else q[-1], c.append(1))[0]
                floor = devices.idle_floor_c(0, interval_s=0, max_wait_s=5, stable_needed=3,
                                             tol_c=1.0, verbose=False)
                self.assertEqual(floor, expect_floor, f"{label}: wrong floor")
                taken[label] = len(calls)
            self.assertEqual(taken["flat throughout"], 4,
                             "a genuinely flat card should settle in stable_needed + 1 samples")
            self.assertGreater(taken["cooling then flat"], taken["flat throughout"] + 4,
                               "the fall was accepted as stable again: it settled in "
                               f"{taken['cooling then flat']} samples, barely more than the "
                               "4 a flat card takes")
        finally:
            devices._temp = real

    def test_one_unsupported_field_does_not_delete_the_device(self):
        """A card without power-limit control reports [N/A], which used to drop the whole row.

        The caller then saw "no GPU at index 0", which reads as no card installed.
        """
        import subprocess

        import devices
        real = subprocess.check_output
        good = ("0, NVIDIA GeForce RTX 3090, 24576, 8.6, 610.43.02, "
                "420.00, 100.00, 450.00, 9751, 2130")
        try:
            for line in (good,
                         good.replace(", 100.00,", ", [N/A],"),
                         good.replace(", 9751,", ", [N/A],"),
                         good.replace(", 420.00, 100.00, 450.00,", ", [N/A], [N/A], [N/A],")):
                subprocess.check_output = lambda *a, **k: line
                devs = devices.enumerate_devices()
                self.assertEqual(len(devs), 1, f"device vanished on: {line}")
                self.assertEqual(devs[0].model_tag, "rtx3090")
                devs[0].describe()  # must not format None with :.0f
        finally:
            subprocess.check_output = real

    def test_stock_is_refused_rather_than_guessed(self):
        """Restoring the wrong power limit is the defect stock_state_for was added to prevent."""
        import subprocess

        import devices
        real = subprocess.check_output
        try:
            subprocess.check_output = lambda *a, **k: (
                "0, NVIDIA GeForce RTX 3090, 24576, 8.6, 610.43.02, "
                "[N/A], [N/A], [N/A], 9751, 2130")
            # Not merely assertRaises: before enumerate_devices kept such a card, get_device
            # itself raised "no GPU at index 0" and this test passed on the broken code.
            with self.assertRaisesRegex(RuntimeError, "power.default_limit"):
                devices.stock_state_for(devices.get_device(0))
        finally:
            subprocess.check_output = real


class TestForkPositionUnits(unittest.TestCase):
    """The one fork statistic taken across prompts rather than within one.

    Output runs 1.56 characters per token in Chinese against 4.65 in prose on this prompt set, a
    spread of 3.0, so a character index minimised across classes is not a comparable number.
    """

    def test_the_two_columns_can_name_different_records(self):
        """Built so the character minimum and the token minimum fall on different records.

        A Chinese record forks at character 20 and a prose record at character 30. In characters
        the Chinese one is earliest; in tokens it is 13 against the prose record's 6, so the
        character column reports the later fork of the two. That is the whole defect, and a
        report that prints only characters cannot show it.
        """
        import contextlib
        import io as _io

        import divergence_report as DR

        def rec(name, chars, fork):
            return {"arm": "dflash2-n4", "prompt": name, "pass": 0,
                    "predicted_n": 400, "text": "x" * chars,
                    "divergence": {"identical": False, "first_diff_char": fork,
                                   "prefix_only": False, "len_ref": chars, "len_arm": chars,
                                   "common_prefix_frac": fork / chars}}

        res = {"arms": {"dflash2-n4": {"extra_args": ["--spec-draft-n-max", "3"]}},
               "records": [rec("zh_letter", 620, 20), rec("prose_essay", 1860, 30)]}
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            DR.report(res)
        row = [ln for ln in buf.getvalue().splitlines() if ln.startswith("dflash2-n4")][0]
        cols = row.split()
        self.assertEqual(cols[-2], "20", f"character column should be the Chinese fork: {row}")
        self.assertEqual(cols[-1], "6", f"token column should be the prose fork: {row}")

    def test_both_fork_columns_cover_the_same_records(self):
        """A prefix-only record must be absent from both columns, not one."""
        import quality
        prefix_only = {"identical": False, "first_diff_char": 5, "prefix_only": True,
                       "len_ref": 5, "len_arm": 40}
        self.assertIsNone(quality.fork_position(prefix_only))
        src = (Path(__file__).parent / "divergence_report.py").read_text(encoding="utf-8")
        self.assertNotIn('min((d["first_diff_char"] for d in forks)', src,
                         "the character column is back on the unfiltered field, so it and the "
                         "token column no longer describe the same records")


class TestMatchedPairsRunTogether(unittest.TestCase):
    """A matched pair has to meet the card in the same state, in every pass.

    The dense arms were appended to the end of Phase M's list. bench.py rotates arm order by one
    position per pass, so with 21 arms and 3 passes the dense side sat at positions 11-20, 10-19
    and 9-18 and never ran early. Temperature, clock drift and page cache would then have varied
    with the model, which is the axis the phase compares, and this matrix's invocation carries no
    thermal gate to absorb it.
    """

    ROOT = Path(__file__).parent

    def test_a_pair_never_runs_far_apart(self):
        import importlib
        d = self.ROOT / "matrices"
        if not d.exists():
            self.skipTest("no matrices directory")
        sys.path.insert(0, str(d))
        checked = 0
        for f in sorted(d.glob("phase_*.py")):
            try:
                mod = importlib.import_module(f.stem)
            except Exception:
                continue
            arms = [a for a in getattr(mod, "ARMS", None) or [] if hasattr(a, "extra_args")]
            names = {a.name for a in arms}
            pairs = [(a.name, "dense-" + a.name[len("moe-"):]) for a in arms
                     if a.name.startswith("moe-") and "dense-" + a.name[len("moe-"):] in names]
            if not pairs:
                continue
            n = len(arms)
            for p in range(1, 4):          # bench.py: rot = (p_idx - 1) % len(arms)
                rot = (p - 1) % n
                order = [a.name for a in arms[rot:] + arms[:rot]]
                at = {nm: i for i, nm in enumerate(order)}
                for m, dn in pairs:
                    self.assertLessEqual(
                        abs(at[m] - at[dn]), 1,
                        f"{f.stem}: on pass {p}, {m} runs at position {at[m]} and {dn} at "
                        f"{at[dn]}. A pair separated in the run order differs in more than "
                        f"the model.")
                    checked += 1
        if checked == 0:
            self.skipTest("no matched pairs")


class TestSpecDraftBounds(unittest.TestCase):
    """An arm whose n_min exceeds its n_max never speculates, and nothing says so.

    `common/speculative.cpp` clears the draft when `result->size() < params.n_min`, and neither
    arg.cpp nor speculative.cpp checks n_min against n_max. An arm configured that way runs to
    completion, writes a full set of records, and reports roughly no speedup, because every draft
    it produced was thrown away. Adding `--spec-draft-n-min 4` across a ladder that includes
    n_max 2 did exactly that here.
    """

    ROOT = Path(__file__).parent

    @staticmethod
    def _bounds(arm):
        out = {}
        ea = arm.extra_args
        for i, t in enumerate(ea):
            if t in ("--spec-draft-n-min", "--spec-draft-n-max") and i + 1 < len(ea):
                try:
                    out[t] = int(ea[i + 1])
                except ValueError:
                    pass
        return out.get("--spec-draft-n-min"), out.get("--spec-draft-n-max")

    def _all_arms(self):
        import importlib
        d = self.ROOT / "matrices"
        if not d.exists():
            self.skipTest("no matrices directory")
        sys.path.insert(0, str(d))
        for f in sorted(d.glob("phase_*.py")):
            try:
                mod = importlib.import_module(f.stem)
            except Exception:
                continue
            for a in getattr(mod, "ARMS", None) or []:
                # phase_v drives vLLM and declares its arms as dicts, not Arm. These bounds are
                # llama.cpp flags, so that matrix is out of scope rather than being skipped.
                if hasattr(a, "extra_args"):
                    yield f.stem, a

    def test_no_arm_asks_for_more_draft_tokens_than_it_allows(self):
        seen = 0
        for name, arm in self._all_arms():
            lo, hi = self._bounds(arm)
            if lo is None or hi is None:
                continue
            seen += 1
            self.assertLessEqual(
                lo, hi,
                f"{name}: {arm.name} sets n_min={lo} above n_max={hi}. Every draft it makes is "
                f"shorter than n_min, so every draft is cleared and the arm silently measures "
                f"nothing.")
        if seen == 0:
            self.skipTest("no arm sets both bounds")

    def test_matched_arms_differ_only_in_their_model(self):
        """`moe-X` and `dense-X` are a pair; anything else that differs is a confound."""
        import importlib
        d = self.ROOT / "matrices"
        sys.path.insert(0, str(d))
        checked = 0
        for f in sorted(d.glob("phase_*.py")):
            try:
                mod = importlib.import_module(f.stem)
            except Exception:
                continue
            arms = {a.name: a for a in getattr(mod, "ARMS", None) or []
                    if hasattr(a, "extra_args")}
            for nm, arm in arms.items():
                if not nm.startswith("dense-"):
                    continue
                twin = arms.get("moe-" + nm[len("dense-"):])
                if twin is None:
                    continue
                self.assertEqual(
                    arm.extra_args, twin.extra_args,
                    f"{f.stem}: {nm} and {twin.name} are a matched pair but their flags differ. "
                    f"Whatever the comparison then measures, it is not the model alone.")
                checked += 1
        if checked == 0:
            self.skipTest("no matched pairs")


class TestCrossModelMatricesCanFitBothSlopes(unittest.TestCase):
    """A matrix that compares two models has to compare their slopes, not just their levels.

    Phase M was given a dense side and, on the first attempt, a single dense width. H6b compares
    the marginal cost per verified position between the two models; with one point there is no
    slope to fit and c_dense would have had to come from another session, which is the comparison
    the dense arms were added to remove. 1575 records would have answered every hypothesis but
    that one, and the fix is another full run.
    """

    ROOT = Path(__file__).parent
    MMVQ_MAX = 8   # ggml/src/ggml-cuda/mmvq.cu; a wider batch takes a different kernel

    @staticmethod
    def _width(arm):
        ea = arm.extra_args
        for i, t in enumerate(ea):
            if t == "--spec-draft-n-max" and i + 1 < len(ea):
                try:
                    return int(ea[i + 1]) + 1
                except ValueError:
                    return None
        return None

    def test_every_method_on_both_models_has_matched_fittable_widths(self):
        import importlib
        d = self.ROOT / "matrices"
        if not d.exists():
            self.skipTest("no matrices directory")
        sys.path.insert(0, str(d))
        checked = 0
        for f in sorted(d.glob("phase_*.py")):
            try:
                mod = importlib.import_module(f.stem)
            except Exception:
                continue
            arms = getattr(mod, "ARMS", None) or []
            if not any(getattr(a, "model", None) for a in arms):
                continue  # single-model matrix; nothing to match
            for method in ("draft-mtp", "draft-simple", "draft-dflash"):
                sides = {}
                for label, pick in (("override", lambda a: getattr(a, "model", None)),
                                    ("default", lambda a: not getattr(a, "model", None))):
                    ws = sorted({w for a in arms if pick(a) and method in a.extra_args
                                 for w in [self._width(a)] if w and w <= self.MMVQ_MAX})
                    sides[label] = ws
                if not sides["override"] and not sides["default"]:
                    continue
                self.assertEqual(
                    sides["override"], sides["default"],
                    f"{f.stem}: {method} runs on both models with unmatched widths "
                    f"{sides}. A slope fitted on one side has nothing to be compared with.")
                self.assertGreaterEqual(
                    len(sides["default"]), 2,
                    f"{f.stem}: {method} has {sides['default']} inside the MMVQ path, which "
                    f"cannot carry a fit")
                checked += 1
        if checked == 0:
            self.skipTest("no cross-model matrix")


class TestRungDriverGates(unittest.TestCase):
    """A completeness gate that passes on zero records is worse than no gate.

    Both rung drivers compute an expected record count in python and then gate on
    `got >= EXPECTED`. If that python fails for any reason the count is empty, the gate is
    `0 >= 0`, and run_phase_q.sh goes on to delete the rung's weights: 20 to 30 GB removed for a
    run that measured nothing. Found by dry-running the driver from the wrong directory.
    """

    ROOT = Path(__file__).parent.parent

    def test_both_drivers_refuse_an_unusable_expected_count(self):
        checked = 0
        for name in ("run_phase_q.sh", "run_phase_qsmall.sh"):
            f = self.ROOT / name
            if not f.exists():
                continue
            src = f.read_text(encoding="utf-8")
            self.assertIn("EXPECTED", src, f"{name}: no expected count at all")
            self.assertRegex(
                src, r'case\s+"\$\{EXPECTED\}"\s+in',
                f"{name}: the expected record count is used without being checked first. "
                f"An empty or zero value makes every completeness gate pass.")
            self.assertIn("exit 1", src, f"{name}: the guard does not stop the run")
            checked += 1
        if checked == 0:
            self.skipTest("no rung driver present")


class TestMatrixBaselinePairing(unittest.TestCase):
    """An arm has to be scored against a baseline that ran the same model.

    Phase M grew a dense side, and its BASELINE_MAP was a comprehension that sent every arm to
    baseline-moe. A dense arm scored against an MoE baseline does not measure speculation at all;
    it measures the difference between two models, and it would have looked like a clean number.
    """

    ROOT = Path(__file__).parent

    def _matrices(self):
        """Every matrix that imports, and a hard failure for any that does not import benignly.

        A bare `except: continue` here would have swallowed a syntax error in any matrix and
        reported a pass over whatever was left. The only import failure this test accepts is a
        matrix refusing to load because its model file is absent, which several do on purpose.
        """
        import importlib
        d = self.ROOT / "matrices"
        if not d.exists():
            self.skipTest("no matrices directory")
        sys.path.insert(0, str(d))
        for f in sorted(d.glob("phase_*.py")):
            try:
                yield f.stem, importlib.import_module(f.stem)
            except RuntimeError as e:
                if "is missing" in str(e) or "not found" in str(e).lower():
                    continue  # staged elsewhere; the matrix says so itself
                raise AssertionError(f"{f.stem} failed to import: {e}") from e
            except Exception as e:
                raise AssertionError(f"{f.stem} failed to import: {type(e).__name__}: {e}") from e

    def test_every_arm_is_paired_with_a_baseline_on_its_own_model(self):
        checked = 0
        for name, mod in self._matrices():
            arms = getattr(mod, "ARMS", None)
            bmap = getattr(mod, "BASELINE_MAP", None)
            if not arms or not bmap:
                continue
            model_of = {a.name: str(getattr(a, "model", None) or getattr(mod, "MODEL", ""))
                        for a in arms}
            for arm, base in bmap.items():
                if arm == base:
                    continue  # analyze.py skips a self-mapping
                self.assertIn(base, model_of, f"{name}: {arm} maps to unknown baseline {base}")
                self.assertEqual(
                    model_of[arm], model_of[base],
                    f"{name}: {arm} runs {model_of[arm]} but is scored against {base}, "
                    f"which runs {model_of[base]}")
                checked += 1
        if checked == 0:
            self.skipTest("no matrix declares both ARMS and BASELINE_MAP")


class TestReadmeMatchesArtifacts(unittest.TestCase):
    """The README's headline numbers, checked against the files they came from.

    Every drift this repo has shipped had the same shape: a number was copied into the README,
    the run that produced it was superseded, and the prose stayed. It claimed n-max 4 was the
    best DFlash2 setting while `cost_model.py` printed 2; it said the depth ladder had two of
    five rungs in the same sentence that quoted the third. Prose cannot be diffed against a
    result file by eye, so it is diffed here instead.
    """

    ROOT = Path(__file__).parent.parent

    def _readme(self):
        f = self.ROOT / "README.md"
        if not f.exists():
            self.skipTest("no README.md")
        return f.read_text(encoding="utf-8")

    def _prose(self):
        """README plus every docs page, as one string.

        Splitting the README moved the cost-model prose into docs/COST_MODEL.md, and the check
        below went on passing because it was only reading README.md. A claim is a claim wherever
        it is written, so the scan follows the prose rather than the file it started in.
        """
        parts = [self._readme()]
        for f in sorted((self.ROOT / "docs").glob("*.md")):
            parts.append(f.read_text(encoding="utf-8"))
        return "\n".join(parts)

    def _load(self, name):
        f = self.ROOT / "results" / name
        if not f.exists():
            self.skipTest(f"no results/{name}")
        return json.loads(f.read_text())

    def test_best_nmax_in_the_readme_is_the_best_in_the_ladder(self):
        """The 'Which n-max?' cell against the argmax of the completed ladder.

        Convention the cell has to keep: a setting it recommends is in bold, and nothing else in
        that cell is. The row said `**2**` for MTP and `**4**` for DFlash2 while the ladder and
        `cost_model.py` both put DFlash2's best at 2.
        """
        import re
        import statistics
        recs = self._load("phase_nmax.json")["records"]
        rate = collections.defaultdict(list)
        for r in recs:
            v = r.get("decode_tok_s")
            if v:
                rate[r["arm"]].append(v)
        best = set()
        for fam in ("mtp", "dflash2"):
            arms = {a: statistics.median(v) for a, v in rate.items() if a.startswith(fam + "-n")}
            self.assertTrue(arms, f"no {fam} arms in phase_nmax.json")
            best.add(int(re.search(r"n(\d+)$", max(arms, key=arms.get)).group(1)))

        cell = [ln for ln in self._readme().splitlines() if ln.startswith("| **Which n-max?**")]
        self.assertEqual(len(cell), 1, "the 'Which n-max?' row moved or was renamed")
        claimed = {int(x) for x in re.findall(r"\*\*(\d+)\*\*", cell[0])}
        self.assertEqual(claimed, best,
                         f"the ladder's best n-max values are {sorted(best)} but the README cell "
                         f"marks {sorted(claimed)}:\n{cell[0]}")

    def test_the_readme_does_not_claim_c_agrees_when_it_does_not(self):
        """The shared-machinery reading is only available while the two coefficients agree.

        Phase A's two-point fit had them 1.6 % apart and the README inferred that the marginal
        cost sits in the machinery both methods share. On the completed ladder they are 15 %
        apart on a paired interval that clears zero, and the inference does not survive it.
        """
        import cost_model as CM
        res = self._load("phase_nmax.json")
        rows = CM.collect(res)
        by_method = collections.defaultdict(list)
        for r in rows:
            by_method[r["spec_type"]].append(r)
        mmvq = CM.recorded_mmvq_max(res)[0]
        fits = {}
        for m, g in by_method.items():
            on = sorted({r["width"] for r in g if r["width"] <= mmvq})
            if len(on) >= 2:
                fits[m] = (g, on)
        if len(fits) != 2:
            self.skipTest("need two fitted methods")
        (ma, (ga, oa)), (mb, (gb, ob)) = sorted(fits.items())
        d = CM.delta_c_ci(ga, oa, gb, ob, n_boot=400)
        self.assertIsNotNone(d, "the paired delta could not be computed")

        text = self._prose()
        if not d.spans_zero:
            self.assertNotIn("`c` agrees", text,
                             f"the paired interval [{d.lo:+.4f}, {d.hi:+.4f}] clears zero, so the "
                             "README must not say the coefficients agree")

    def test_phase_l_rung_count_matches_the_files_on_disk(self):
        """'N of five rungs complete' against the result files that actually hold 180 records."""
        import re
        complete = 0
        for f in sorted((self.ROOT / "results").glob("phase_l_*.json")):
            if ".partial." in f.name:
                continue
            try:
                if len(json.loads(f.read_text())["records"]) >= 180:
                    complete += 1
            except Exception:
                pass
        if complete == 0:
            self.skipTest("no complete phase_l rungs")
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        text = self._readme()
        stated = set()
        for m in re.finditer(r"\b(one|two|three|four|five|\d+) of five rungs", text):
            g = m.group(1)
            stated.add(words.get(g, int(g) if g.isdigit() else 0))
        self.assertTrue(stated, "the README no longer states a rung count out of five")
        self.assertEqual(stated, {complete},
                         f"{complete} rungs hold 180 records; the README says {sorted(stated)}")

    def test_reproduce_pins_the_commits_the_trees_are_actually_at(self):
        """A reproduce block that clones a moving branch reproduces something else."""
        import re
        import subprocess
        text = self._readme()
        pins = dict(re.findall(r"^(LLAMA_MASTER_COMMIT|DFLASH2_COMMIT)=([0-9a-f]{7,40})",
                               text, re.M))
        self.assertEqual(set(pins), {"LLAMA_MASTER_COMMIT", "DFLASH2_COMMIT"},
                         "the reproduce block does not pin both trees")
        for var, tree in (("LLAMA_MASTER_COMMIT", "llamacpp-master"),
                          ("DFLASH2_COMMIT", "llamacpp-dflash2")):
            d = self.ROOT / tree
            if not (d / ".git").exists():
                continue
            head = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            if head:
                self.assertEqual(pins[var], head,
                                 f"{tree} is at {head} but the README pins {pins[var]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
