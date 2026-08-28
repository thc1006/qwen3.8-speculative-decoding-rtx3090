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
import cross_rung as CR  # noqa: E402
import audit_results as AR  # noqa: E402
import backfill_model_size as BMS  # noqa: E402
import ladder_trend as LT  # noqa: E402
import stats as ST  # noqa: E402
import gpustate as GS  # noqa: E402
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
        # Absolute, from this file. A relative path made the guard depend on the working
        # directory: `python3 harness/test_harness.py` found the tree and `cd harness &&
        # python3 test_harness.py` did not, so the same suite skipped or ran the same check
        # depending on where it was started from, and reported "not present" about a directory
        # that is present.
        tree = str(Path(__file__).parent.parent / "llamacpp-master")
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

        # The bound is the MEASURED wall time of the block, not the nominal 0.6 s. The sampler
        # thread keeps going until __exit__ returns, so its last sample can land after the sleep
        # does, and on a loaded host the gap is real: this assertion failed at 0.60029 against
        # 0.6 while the machine was running a benchmark. Asserting against the requested value
        # rather than the realised one is the same mistake this suite exists to catch elsewhere.
        t0 = time.monotonic()
        with T.sampling(0, interval_s=0.10) as s:
            time.sleep(0.6)
        elapsed = time.monotonic() - t0
        d = s.summary()
        if not d.get("n_power_samples"):
            self.skipTest("no GPU telemetry on this host")
        self.assertIn("sample_span_s", d)
        self.assertIsNotNone(d["sample_span_s"])
        self.assertLessEqual(d["sample_span_s"], elapsed,
                             f"the span cannot exceed the window it was sampled in: "
                             f"{d['sample_span_s']:.5f} s of samples in a {elapsed:.5f} s block")
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


class TestCostModelBaselinePerModel(unittest.TestCase):
    """A speculative arm must be divided by its own model's baseline.

    cost_model.py keyed its baseline lookup on (tree, pass, prompt). Phase M declares a dense
    baseline and an MoE baseline and both carry tree "master", so the second overwrote the first
    on every prompt. Every MoE arm would then have been divided by the dense baseline, 41.6
    against 147.8 tok/s on the live run, inflating its speedup about 3.5x and shrinking the k that
    H6b compares by the same factor. Nothing errors; the number just comes out wrong.
    """

    @staticmethod
    def _rec(arm, prompt, tok_s, drafted=8, accepted=4):
        return {"arm": arm, "pass": 1, "prompt": prompt, "class": "code",
                "decode_tok_s": tok_s, "predicted_n": 400,
                "timings": {"t_draft_n": drafted, "t_draft_n_accepted": accepted}}

    def test_two_baselines_on_one_tree_do_not_collide(self):
        import cost_model as CM
        DENSE, MOE = "/m/dense.gguf", "/m/moe.gguf"
        result = {
            "env": {"model": MOE},
            "arms": {
                "baseline-moe":   {"extra_args": [], "tree": "master", "model": None},
                "baseline-dense": {"extra_args": [], "tree": "master", "model": DENSE},
                "moe-mtp-n2":     {"extra_args": ["--spec-type", "draft-mtp",
                                                  "--spec-draft-n-max", "2"],
                                   "tree": "master", "expects_drafter": True, "model": None},
                "dense-mtp-n2":   {"extra_args": ["--spec-type", "draft-mtp",
                                                  "--spec-draft-n-max", "2"],
                                   "tree": "master", "expects_drafter": True, "model": DENSE},
            },
            "records": [
                self._rec("baseline-moe",   "p1", 148.0),
                self._rec("baseline-dense", "p1",  41.6),
                self._rec("moe-mtp-n2",     "p1", 296.0),   # exactly 2x its own baseline
                self._rec("dense-mtp-n2",   "p1",  83.2),   # exactly 2x its own baseline
            ],
        }
        rows = {r["arm"]: r for r in CM.collect(result)}
        self.assertEqual(set(rows), {"moe-mtp-n2", "dense-mtp-n2"},
                         "both speculative arms should produce a row")
        for arm in ("moe-mtp-n2", "dense-mtp-n2"):
            self.assertAlmostEqual(
                rows[arm]["speedup"], 2.0, places=6,
                msg=f"{arm} was not divided by its own model's baseline; "
                    f"speedup came out {rows[arm]['speedup']:.3f} instead of 2.0")
        self.assertNotEqual(rows["moe-mtp-n2"]["model"], rows["dense-mtp-n2"]["model"],
                            "the two arms were attributed to the same model")


class TestCostModelSeparatesModels(unittest.TestCase):
    """Two targets in one result file must not be fitted as one line.

    cost_model.py grouped by spec_type alone. Phase M runs draft-mtp on a dense target and on an
    MoE target at the same five widths, so every width would have carried k values from both and
    the slope printed would have been neither model's. It would not have errored: the r-squared
    stays high when you average two lines that are close.
    """

    ROOT = Path(__file__).parent.parent

    def test_a_two_model_file_is_fitted_twice(self):
        import subprocess
        import tempfile
        src = self.ROOT / "results" / "phase_nmax.json"
        if not src.exists():
            self.skipTest("no phase_nmax.json")
        d = json.loads(src.read_text())
        # The baseline an arm is scored against has to move with it, or the fixed lookup finds
        # no baseline for that model and the arm drops out entirely. mtp-* is scored against
        # baseline@master, so both go; dflash keeps baseline@pr27342 on the original model.
        for name, meta in d.get("arms", {}).items():
            if name.startswith("mtp-") or name == "baseline@master":
                meta["model"] = "/fake/MODEL_B.gguf"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(d, fh)
            tmp = fh.name
        try:
            out = subprocess.run(
                [sys.executable, str(self.ROOT / "harness" / "cost_model.py"), tmp],
                capture_output=True, text=True).stdout
        finally:
            Path(tmp).unlink(missing_ok=True)

        # `k0=` rather than `MMVQ widths`: the implied-optimum section also names the widths a
        # fit was made over, and it is a reading of the fit rather than a fit, so it carries no
        # model label of its own and would fail the assertion below for the wrong reason.
        fits = [ln for ln in out.splitlines() if "k0=" in ln and "->" in ln]
        self.assertGreaterEqual(len(fits), 2, f"expected a fit per model, got:\n{out[-800:]}")
        self.assertTrue(any("MODEL_B" in ln for ln in fits),
                        "the relabelled arms were not fitted as their own group")
        self.assertTrue(any("MODEL_B" not in ln for ln in fits),
                        "every group was labelled with the same model")
        for ln in fits:
            self.assertIn("@", ln, "a two-model file must name the model on each fit")


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
    `0 >= 0`, and scripts/run_phase_q.sh goes on to delete the rung's weights: 20 to 30 GB removed for a
    run that measured nothing. Found by dry-running the driver from the wrong directory.
    """

    ROOT = Path(__file__).parent.parent

    def test_both_drivers_refuse_an_unusable_expected_count(self):
        checked = 0
        for name in ("scripts/run_phase_q.sh", "scripts/run_phase_qsmall.sh"):
            f = self.ROOT / name
            # Not `if not f.exists(): continue`. A guard test that skips when its subject moves
            # goes green while checking nothing, which is how a rename turns a gate off.
            self.assertTrue(f.exists(), f"{name} is missing; this guard has no subject")
            src = f.read_text(encoding="utf-8")
            # Asserted as a PROPERTY, not as a variable name. scripts/run_phase_q.sh's gate was rewritten
            # to check the shape of a result against the matrix rather than compare one integer,
            # so the count it guards is now N_PROMPTS rather than EXPECTED. What has to survive
            # any such rewrite is that whatever python-derived count the gate rests on is
            # rejected when it comes back empty, non-numeric or zero -- those are the values that
            # make `got >= count` true for a run that measured nothing.
            self.assertRegex(
                src, r"""case\s+"\$\{[A-Z_]+\}"\s+in\s*\n\s*''\|\*\[!0-9\]\*\|0\)""",
                f"{name}: the count the completeness gate rests on is used without being "
                f"checked for empty / non-numeric / zero first. Any of those makes the gate "
                f"pass for a run that measured nothing, and this driver deletes weights on a "
                f"passing gate.")
            self.assertIn("exit 1", src, f"{name}: the guard does not stop the run")
            checked += 1
        self.assertGreater(checked, 0, "no rung driver was checked; the guard has no subject")


    def test_a_rung_checks_free_disk_before_downloading_it(self):
        """Free disk was logged AFTER staging, which is too late to act on.

        The four rungs total about 93 GB and are staged one at a time. On the measurement host
        this would have attempted a 23 GB download into 28 GB of free space, because the 22 GB MoE
        target is held whenever the Phase M anchor does not clear. Filling the root filesystem
        does not merely fail the download: the harness is writing results and server logs to the
        same disk, and _atomic_write_json would fail mid-run.
        """
        for name in ("scripts/run_phase_q.sh", "scripts/run_phase_qsmall.sh"):
            path = self.ROOT / name
            self.assertTrue(path.exists(), f"{name} is missing; this guard has no subject")
            src = path.read_text(encoding="utf-8")
            if "hf download" not in src:
                continue
            self.assertIn("--output=avail", src,
                          f"{name} downloads a rung without checking free disk first")
            dl = src.index("hf download")
            guard = src.index("--output=avail")
            self.assertLess(guard, dl,
                            f"{name} checks free disk after the download rather than before it")


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

    def test_the_prose_matches_the_verdict_the_analysis_actually_reaches(self):
        """The prose about `c` must agree with what cost_model prints, whichever way that goes.

        Two earlier versions of this test each hard-coded an analysis, and each hard-coded a wrong
        one. The first read the paired PROMPT bootstrap as the arbiter and required the prose to
        say the coefficients differ. The second called that the wrong tool, recomputed the bound
        from each fit's width residuals added in quadrature, and required the prose to say the
        question was open. The right answer is the first one, reached a third way: restrict both
        fits to the widths they share, because k(w) is curved and a slope is a chord, then check
        shape on the DIFFERENCE, where the shared curvature cancels. See PREREGISTRATION.md
        Corrections 13 and 14.

        The lesson this test now encodes is that it must not hold its own copy of the analysis. It
        reads cost_model's verdict and checks the prose against that, so the two cannot part
        company again.
        """
        import contextlib
        import cost_model as CM
        import io
        import re
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CM.report(self._load("phase_nmax.json"))
        verdicts = re.findall(r"VERDICT: (.+)", buf.getvalue())
        if not verdicts:
            self.skipTest("phase_nmax produced no c comparison")
        verdict = verdicts[0]

        text = self._prose()
        asserts_difference = ["the marginal cost is not shared",
                              "part of the marginal cost moves with the drafter"]
        says_open = ["not resolved"]
        if verdict.startswith("the marginal costs differ"):
            self.assertTrue(any(a in text.lower() for a in asserts_difference),
                            f"cost_model reports '{verdict}' but the prose does not say the "
                            f"marginal costs differ")
        else:
            self.assertTrue(any(a in text.lower() for a in says_open),
                            f"cost_model reports '{verdict}' but the prose does not say so; "
                            f"silence reads as agreement")
            for a in asserts_difference:
                self.assertNotIn(a, text.lower(),
                                 f"the prose asserts a difference that cost_model does not "
                                 f"reach: '{verdict}'")

    def test_the_comparison_is_made_on_a_matched_width_range(self):
        """A slope is a chord of a curved k(w), so the ranges have to match before comparing."""
        import contextlib
        import cost_model as CM
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CM.report(self._load("phase_nmax.json"))
        out = buf.getvalue()
        self.assertIn("both fitted on the shared widths", out,
                      "the two methods are compared over whatever widths each happened to run")
        self.assertIn("Shape check on the DIFFERENCE", out,
                      "shape is still being checked on each fit separately, which charges shared "
                      "curvature against the comparison twice")

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

    def test_phase_q_rung_count_matches_the_files_on_disk(self):
        """'N rungs of four' against the result files that actually hold 300 records.

        This row has already drifted once: it said one rung of four and gave the reason as disk,
        which was the driver mistaking a VRAM figure for a download size. A count in prose that
        nothing checks is a count that goes stale the next time a rung lands.
        """
        import re
        complete = 0
        for f in sorted((self.ROOT / "results").glob("phase_q_*.json")):
            if ".partial." in f.name:
                continue
            try:
                if len(json.loads(f.read_text())["records"]) >= 300:
                    complete += 1
            except Exception:
                pass
        if complete == 0:
            self.skipTest("no complete phase_q rungs")
        words = {"one": 1, "two": 2, "three": 3, "four": 4}
        # Scoped to the phase Q row. Phase Q-small is a four-rung ladder too, and its prose in
        # the same table would otherwise be collected here and reported as a disagreement.
        text = self._readme()
        row = re.search(r"^\| \*\*Q\*\* \|.*$", text, re.M)
        self.assertIsNotNone(row, "the README no longer has a phase Q row")
        text = row.group(0)
        stated = set()
        # Case-insensitive: the count opens a sentence in the current row, so it is capitalised,
        # and a guard that only reads lowercase reports "the README no longer states a count".
        for m in re.finditer(r"\b(one|two|three|four|\d+) rungs? of four", text, re.I):
            g = m.group(1).lower()
            stated.add(words.get(g, int(g) if g.isdigit() else 0))
        self.assertTrue(stated, "the README no longer states a rung count out of four")
        self.assertEqual(stated, {complete},
                         f"{complete} phase_q rungs hold 300 records; the README says "
                         f"{sorted(stated)}")

    def test_the_lock_file_pins_the_commits_the_trees_are_actually_at(self):
        """A reproduce procedure that clones a moving branch reproduces something else.

        The pins used to be shell variables in the README's reproduce block, and were checked
        with `rev-parse --short`, which compares an abbreviation to the same abbreviation. They
        now live in repro/phase_a.lock.json, are full 40-character objects, and are compared in
        full against what the trees are at.
        """
        import subprocess
        lock = self.ROOT / "repro" / "phase_a.lock.json"
        if not lock.exists():
            self.fail("repro/phase_a.lock.json is missing; the reproduce procedure reads it")
        pins = json.loads(lock.read_text(encoding="utf-8"))
        for key, tree in (("llama_master_commit", "llamacpp-master"),
                          ("dflash2_commit", "llamacpp-dflash2")):
            sha = pins.get(key)
            self.assertIsNotNone(sha, f"the lock file does not pin {key}")
            self.assertRegex(sha, r"^[0-9a-f]{40}$",
                             f"{key} is not a full 40-character object; a prefix can resolve to "
                             f"a different commit as a repository grows")
            d = self.ROOT / tree
            if not (d / ".git").exists():
                continue
            head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            if head:
                self.assertEqual(sha, head, f"{tree} is at {head}, the lock file pins {sha}")

    def test_the_readme_quotes_the_same_commits_the_lock_file_pins(self):
        lock = self.ROOT / "repro" / "phase_a.lock.json"
        if not lock.exists():
            self.skipTest("no lock file")
        pins = json.loads(lock.read_text(encoding="utf-8"))
        text = self._readme()
        for key in ("llama_master_commit", "dflash2_commit"):
            self.assertIn(pins[key], text,
                          f"the README does not quote the {key} the runner will use")


class TestAnchorEstimatorMatchesBand(unittest.TestCase):
    """The Phase M anchor must be judged by the estimator its band was calibrated on.

    The gate first shipped inline in scripts/run_remaining.sh and compared a POOLED MEDIAN against a band
    calibrated on a CLASS-STRATIFIED figure. Its own header names both predecessor numbers,
    "-10.8 % raw, -21.5 % class-stratified", and the band -12 % to -32 % brackets only the second:
    a perfect replication of the raw figure would have failed the gate it was written for. On the
    live 2026-08-26 data the two estimators differ by 6.7 points, which happened not to change the
    verdict because the arm missed the band by 34 points. A borderline arm would have been decided
    by the estimator rather than by the data.

    The discriminating case below is one where the two estimators give OPPOSITE verdicts, so the
    test fails against the old logic and passes against anchor_verdict.py.
    """

    @staticmethod
    def _result(arm_by_class, base_rate=100.0, n_by_class=(("code", 9), ("prose", 3))):
        """A synthetic Phase M shaped so pooled and stratified disagree.

        One class carries most of the prompts and a small penalty; the other carries few and a
        large one. The pooled median is drawn from the crowded class and reports its penalty; the
        stratified mean weights the two classes equally.

        Each prompt is jittered around its class effect. Without that every prompt returns the
        same ratio, the cluster bootstrap finds nothing to resample, and the interval collapses to
        zero width -- which anchor_verdict correctly refuses to call a replication, for a reason
        that has nothing to do with the estimator this fixture exists to test.
        """
        recs = []
        for cls, n in n_by_class:
            for i in range(n):
                tag = f"{cls}_{i}"
                # Symmetric about zero for any n, so each class mean is exactly its target and
                # the assertions below stay exact while the bootstrap still has something to
                # resample.
                eff = arm_by_class[cls] + (i - (n - 1) / 2) * 0.004
                recs.append({"arm": "baseline-moe", "pass": 1, "prompt": tag, "class": cls,
                             "decode_tok_s": base_rate, "predicted_n": 400, "hit_cap": True})
                recs.append({"arm": "moe-draft08b-n8", "pass": 1, "prompt": tag, "class": cls,
                             "decode_tok_s": base_rate * (1 + eff),
                             "predicted_n": 400, "hit_cap": True})
        return {"records": recs}

    def test_stratified_verdict_wins_over_pooled_median(self):
        import anchor_verdict as AV
        # code -5 %, prose -35 %  ->  stratified -20 % (inside the -12/-32 band),
        # pooled median -5 % (outside it). The two verdicts are opposite.
        v = AV.verdict(self._result({"code": -0.05, "prose": -0.35}))
        lo, hi = v["band"]
        self.assertAlmostEqual(v["point"], -20.0, places=6,
                               msg="the primary is not the class-stratified mean")
        # The exact pooled median depends on how the two classes interleave, so assert the
        # property the case exists to create: the two estimators fall on opposite sides of the
        # band, and only then can this test tell which one the verdict used.
        self.assertTrue(lo < v["point"] < hi, "the stratified estimate left the band")
        self.assertFalse(lo < v["pooled_median"] < hi,
                         f"the pooled median ({v['pooled_median']:+.1f} %) is inside the band "
                         f"too, so this case no longer separates the estimators and cannot "
                         f"catch the defect it was written for")
        self.assertTrue(v["holds"],
                        f"the anchor was judged on the pooled estimator: {v['reason']}")

    def test_pooled_median_passing_does_not_rescue_a_failing_stratified_arm(self):
        import anchor_verdict as AV
        # The mirror image: pooled median -20 % (inside the band), stratified -40 % (outside).
        v = AV.verdict(self._result({"code": -0.20, "prose": -0.60}))
        lo, hi = v["band"]
        self.assertAlmostEqual(v["point"], -40.0, places=6)
        self.assertTrue(lo < v["pooled_median"] < hi,
                        f"the pooled median ({v['pooled_median']:+.1f} %) is outside the band "
                        f"too, so this case no longer separates the estimators")
        self.assertFalse(lo < v["point"] < hi, "the stratified estimate entered the band")
        self.assertFalse(v["holds"],
                         "an arm outside the band on the registered estimator was passed "
                         "because a pooled estimator happened to land inside it")

    def test_an_effect_that_could_be_zero_never_holds(self):
        import anchor_verdict as AV
        # Per-prompt effects that straddle zero and average -20 %, squarely inside the band. A
        # point-in-band test alone calls this a reproduction of the penalty; it is not evidence
        # of one. Written out rather than drawn, so the fixture cannot drift with a seed.
        effects = [-0.95, -0.90, -0.85, 0.50, 0.60, 0.40]      # mean -0.20
        recs = []
        for cls in ("code", "prose"):
            for i, eff in enumerate(effects):
                tag = f"{cls}_{i}"
                recs.append({"arm": "baseline-moe", "pass": 1, "prompt": tag, "class": cls,
                             "decode_tok_s": 100.0, "predicted_n": 400, "hit_cap": True})
                recs.append({"arm": "moe-draft08b-n8", "pass": 1, "prompt": tag, "class": cls,
                             "decode_tok_s": 100.0 * (1 + eff),
                             "predicted_n": 400, "hit_cap": True})
        v = AV.verdict({"records": recs})
        lo, hi = v["band"]
        self.assertTrue(lo < v["point"] < hi,
                        "the fixture no longer lands inside the band, so it cannot show that "
                        "being inside the band is not sufficient")
        self.assertTrue(v["spans_zero"], "the fixture stopped spanning zero; rewrite it")
        self.assertFalse(v["holds"],
                         "an interval containing zero was reported as a reproduced penalty")
        self.assertIn("zero", v["reason"])

    def test_band_brackets_the_figure_the_estimator_names(self):
        import anchor_verdict as AV
        lo, hi = AV.ANCHOR["band"]
        self.assertLess(lo, AV.ANCHOR["predecessor_stratified"])
        self.assertLess(AV.ANCHOR["predecessor_stratified"], hi)
        self.assertIn("stratified", AV.ANCHOR["estimator"].lower(),
                      "the band is calibrated on the class-stratified figure, so the estimator "
                      "field must name that quantity")
        # The trap this file closes: the other published figure lies outside the same band, so
        # which estimator is used decides the verdict on a faithful replication.
        self.assertFalse(lo < AV.ANCHOR["predecessor_raw"] < hi,
                         "the two predecessor figures now fall on the same side of the band; "
                         "if that is intended, this test and the module docstring both need "
                         "rewriting, because the estimator would no longer be load-bearing")

    ROOT = Path(__file__).parent

    def test_a_wide_interval_inside_the_band_does_not_count_as_a_replication(self):
        import anchor_verdict as AV
        # Per-prompt effects averaging -20 %, inside the band, with an interval that excludes zero
        # only just. Without a precision rule this "holds" while the data cannot tell a 1 %
        # penalty from a 60 % one.
        effects = [-0.60, -0.55, -0.50, -0.02, -0.01, -0.02]      # mean -0.2833 -> in band
        recs = []
        for cls in ("code", "prose"):
            for i, eff in enumerate(effects):
                tag = f"{cls}_{i}"
                recs.append({"arm": "baseline-moe", "pass": 1, "prompt": tag, "class": cls,
                             "decode_tok_s": 100.0, "predicted_n": 400, "hit_cap": True})
                recs.append({"arm": "moe-draft08b-n8", "pass": 1, "prompt": tag, "class": cls,
                             "decode_tok_s": 100.0 * (1 + eff),
                             "predicted_n": 400, "hit_cap": True})
        v = AV.verdict({"records": recs})
        lo, hi = v["band"]
        self.assertTrue(lo < v["point"] < hi, "the fixture no longer lands inside the band")
        self.assertFalse(v["spans_zero"], "the fixture no longer excludes zero; rewrite it")
        self.assertTrue(v["near_zero"],
                        f"the fixture is no longer imprecise enough to test the rule "
                        f"(margin {v['margin_half_widths']:.2f} half-widths)")
        self.assertFalse(v["holds"],
                         "an effect the data cannot pin down was reported as a reproduction of "
                         "the registered penalty")
        self.assertIn("half-widths", v["reason"])

    def test_the_chain_uses_the_module_and_not_an_inline_copy(self):
        """scripts/run_remaining.sh had its own anchor, and it was a different one.

        The inline block computed a pooled median and compared it against a band calibrated on a
        class-stratified figure. Its own header named both predecessor numbers, -10.8 % raw and
        -21.5 % stratified, and the band brackets only the second, so a faithful replication of the
        raw figure would have failed the gate written for it. On the completed run the two
        estimators differ by 6.7 points.

        A second copy of an analysis is the defect, not the formula it used, so this checks that
        the chain calls the module rather than that the old arithmetic is gone.
        """
        src = (Path(__file__).parent.parent / "scripts" / "run_remaining.sh").read_text(encoding="utf-8")
        self.assertIn("harness/anchor_verdict.py", src,
                      "the chain no longer runs the anchor it gates the MoE deletion on")
        self.assertNotIn("REPLICATION ANCHOR (0.8B", src,
                         "scripts/run_remaining.sh still carries its own copy of the anchor")
        # the deletion gate must still read the marker the module writes
        self.assertIn("results/phase_m_anchor_ok", src)

    def test_a_stale_marker_cannot_gate_a_later_run(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            res = Path(td) / "phase_m.json"
            res.write_text(json.dumps(self._result({"code": -0.20, "prose": -0.60})))
            marker = Path(td) / "anchor_ok"
            marker.write_text("+0.00\n")          # left behind by an earlier, passing run
            p = subprocess.run(
                [sys.executable, str(self.ROOT / "anchor_verdict.py"), str(res),
                 "--marker", str(marker)],
                capture_output=True, text=True)
            self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
            self.assertFalse(marker.exists(),
                             "a failing anchor left the previous run's marker in place, which "
                             "would let scripts/run_remaining.sh delete the MoE target it needs to "
                             "chase the failure")


class TestRejectionSlopeIsIntervalBased(unittest.TestCase):
    """TEST 1's verdict came from a point estimate's sign, with r2 printed but never consulted.

    Three defects in one block, all live on the Phase M data:

    1. `r2` was computed and printed and had no effect on the verdict, so an arm whose fit
       explained 13 % of the variance in `k` was announced as "rejection cost present" in exactly
       the words used for one explaining 99 %.
    2. The summary bound was `max(r_estimates)` -- the maximum of several noisy point estimates,
       biased upward by construction and not a bound on anything. It also gated whether TEST 1's
       conclusion was printed at all, so one arm's noise (r2 = 0.134, clearing zero by 0.10
       half-widths) silenced the finding.
    3. The model treats the draft length as the constant `n_max`. On the 0.8B draft-simple arms
       the realised length is 4.20 against an n_max of 8 and it correlates with acceptance at
       +0.94, because the server reuses a surviving draft tail rather than re-drafting
       (server-context.cpp:2893). The regressor is then inside the response, and the resulting
       bias is positive in the slope, hence negative in r -- which is the conclusion TEST 1
       draws. An omitted variable that pushes toward the paper's own answer.
    """

    @staticmethod
    def _rows(arm, n_max, slope, intercept, widths, n=12):
        """An arm whose k is exactly intercept + slope*acceptance, with given widths."""
        rows = []
        for i in range(n):
            acc = 0.1 + 0.8 * i / (n - 1)
            rows.append({
                "arm": arm, "pass": 1, "prompt": f"p{i}", "class": "code" if i % 2 else "prose",
                "spec_type": "draft-simple", "model": "m", "n_max": n_max, "width": n_max + 1,
                "acceptance": acc, "drafted": 100, "accepted": 50, "forwards": 25,
                "mean_len": 2.0, "speedup": 1.0,
                "draft_per_forward": widths(acc),
                "k": intercept + slope * acc,
            })
        return rows

    def test_r_divides_by_the_realised_width_not_the_requested_n_max(self):
        import cost_model as CM
        # Constant realised width of 6.94 against an n_max of 8: the case dflash2-n8 is in.
        # Dividing by 8 understates r by 13 %, and r is reported as an upper bound.
        rows = self._rows("a", 8, slope=-0.4, intercept=1.0, widths=lambda acc: 6.94)
        iv, r2 = CM.rejection_slope_ci(rows, n_boot=400)
        self.assertAlmostEqual(r2, 1.0, places=6, msg="the fixture is not an exact line")
        self.assertAlmostEqual(iv.point, 0.4 / 6.94, places=6,
                              msg=f"r was divided by n_max rather than the realised width; "
                                  f"got {iv.point:.5f}, expected {0.4/6.94:.5f}")

    def test_the_bound_is_an_upper_limit_over_unconfounded_arms(self):
        import contextlib
        import cost_model as CM
        import io
        DENSE = "/m/dense.gguf"

        def rec(arm, i, tok_s, drafted, accepted, pn=400):
            return {"arm": arm, "pass": 1, "prompt": f"p{i}",
                    "class": "code" if i % 2 else "prose", "decode_tok_s": tok_s,
                    "predicted_n": pn, "hit_cap": True,
                    "timings": {"t_draft_n": drafted, "t_draft_n_accepted": accepted}}

        recs = []
        for i in range(12):
            recs.append(rec("baseline", i, 100.0, 0, 0))
            # pinned: 2 drafted per forward pass whatever the acceptance
            f = 130 + 5 * i
            recs.append(rec("pinned", i, 150.0, 2 * f, 399 - f))
            # varying: drafted per forward pass climbs with acceptance
            f2 = 200 - 8 * i
            recs.append(rec("varying", i, 120.0, int((2 + 0.02 * i) * f2), 399 - f2))
        result = {
            "env": {"model": DENSE},
            "arms": {
                "baseline": {"extra_args": [], "tree": "master", "model": None},
                "pinned": {"extra_args": ["--spec-type", "draft-mtp",
                                          "--spec-draft-n-max", "2"],
                           "tree": "master", "expects_drafter": True, "model": None},
                "varying": {"extra_args": ["--spec-type", "draft-simple",
                                           "--spec-draft-n-max", "8"],
                            "tree": "master", "expects_drafter": True, "model": None},
            },
            "records": recs,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CM.report(result)
        out = buf.getvalue()
        self.assertIn("CONFOUNDED", out,
                      "an arm whose draft length tracks acceptance was not flagged")
        self.assertRegex(out, r"Excluded from the bound as confounded:.*varying",
                         "the confounded arm still contributes to the bound")
        self.assertIn("Largest upper 95 % limit", out,
                      "the bound is still reported as a point estimate")
        self.assertNotIn("Largest r across arms", out,
                         "the old max-of-points summary is still being printed")


class TestRejectionBoundTableMatchesArtifacts(unittest.TestCase):
    """The rejection-bound table in docs/COST_MODEL.md against the files it was read from.

    The section it replaced quoted "|r| <= 0.0028, r^2 between 0.001 and 0.060", which was true of
    the phases that existed when it was written and false by the time Phase M ran. A hand-copied
    number in prose has no way to notice that. This reads both sides.
    """

    ROOT = Path(__file__).parent.parent

    def test_every_row_matches_the_cost_artifact_it_came_from(self):
        import re
        doc = (self.ROOT / "docs" / "COST_MODEL.md").read_text(encoding="utf-8")
        block = re.search(r"\| matrix \| arms fitted \|.*?\n\n", doc, re.S)
        self.assertIsNotNone(block, "the rejection-bound table is gone from docs/COST_MODEL.md")
        rows = re.findall(r"^\| ([A-Z0-9]+) \| (\d+) \| ([+-][\d.]+) \| ([\d.]+) % \|$",
                          block.group(0), re.M)
        self.assertTrue(rows, "the table parsed to no rows; the format changed")
        checked = 0
        for name, arms, r_hi, share in rows:
            art = self.ROOT / "analysis" / f"phase_{name.lower()}_cost.txt"
            if not art.exists():
                continue
            text = art.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Largest upper 95 % limit on r, over the (\d+) arm\(s\) where the "
                          r"model applies: ([+-][\d.]+)", text)
            self.assertIsNotNone(m, f"{art.name} carries no bound line")
            self.assertEqual(arms, m.group(1), f"{name}: arms fitted")
            self.assertEqual(r_hi, m.group(2), f"{name}: upper limit on r")
            ms = re.search(r"accounts for at most \+?([\d.]+) % of the cycle cost", text)
            self.assertIsNotNone(ms, f"{art.name} carries no share line")
            self.assertEqual(share, ms.group(1), f"{name}: share of cycle cost")
            checked += 1
        self.assertGreaterEqual(checked, 3,
                                f"only {checked} of {len(rows)} rows had an artifact to check "
                                f"against; the test would pass on an empty analysis/ directory")

    def test_the_prose_bound_is_not_below_any_row(self):
        import re
        doc = (self.ROOT / "docs" / "COST_MODEL.md").read_text(encoding="utf-8")
        shares = [float(x) for x in re.findall(r"^\| [A-Z0-9]+ \| \d+ \| [+-][\d.]+ \| "
                                               r"([\d.]+) % \|$", doc, re.M)]
        self.assertTrue(shares)
        claim = re.search(r"Nothing reaches ([\d.]+) %", doc)
        self.assertIsNotNone(claim, "the summarising sentence under the table is gone")
        self.assertLess(max(shares), float(claim.group(1)),
                        f"the prose claims nothing reaches {claim.group(1)} % but the table "
                        f"holds {max(shares)} %")


class TestImpliedOptimumReadsTheLadder(unittest.TestCase):
    """The shape of the n-max ladder must come from the data, not from a sentence.

    The section asserted "over the widths measured here it falls monotonically". That was true of
    the dense phases it was written against. Phase M's MTP ladder peaks at n-max 2 on both targets
    -- 1.206x, 1.276x, 1.181x, 1.153x on the MoE -- and the sentence would have gone on being
    printed underneath the numbers contradicting it, in the report the README quotes.
    """

    @staticmethod
    def _result(speedups):
        """A file whose draft-mtp ladder has the given speedup at each n-max."""
        recs, arms = [], {"baseline": {"extra_args": [], "tree": "master", "model": None}}
        for i in range(8):
            recs.append({"arm": "baseline", "pass": 1, "prompt": f"p{i}",
                         "class": "code" if i % 2 else "prose", "decode_tok_s": 100.0,
                         "predicted_n": 400, "hit_cap": True,
                         "timings": {"t_draft_n": 0, "t_draft_n_accepted": 0}})
        for n, sp in sorted(speedups.items()):
            arms[f"mtp-n{n}"] = {"extra_args": ["--spec-type", "draft-mtp",
                                                "--spec-draft-n-max", str(n)],
                                 "tree": "master", "expects_drafter": True, "model": None}
            for i in range(8):
                # accepted chosen so the arm drafts exactly n per forward pass
                f = 200 - 6 * i
                recs.append({"arm": f"mtp-n{n}", "pass": 1, "prompt": f"p{i}",
                             "class": "code" if i % 2 else "prose",
                             "decode_tok_s": 100.0 * sp, "predicted_n": 400, "hit_cap": True,
                             "timings": {"t_draft_n": n * f, "t_draft_n_accepted": 399 - f}})
        return {"env": {"model": "/m/x.gguf"}, "arms": arms, "records": recs}

    def _run(self, speedups):
        import contextlib
        import cost_model as CM
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CM.report(self._result(speedups))
        section = buf.getvalue().split("--- implied optimum ---")[-1]
        # Only the verdict lines. The paragraph above them legitimately contains both "interior
        # maximum" and "falls monotonically" while explaining why the shape is computed, so a
        # substring check against the whole section can neither fail nor pass for the right reason.
        return "\n".join(ln for ln in section.splitlines() if "-> best tested" in ln)

    def test_a_peaked_ladder_is_called_a_peak(self):
        out = self._run({1: 1.20, 2: 1.28, 3: 1.18, 5: 1.15})
        self.assertIn("interior maximum at n-max 2", out,
                      f"a ladder that rises then falls was not reported as peaked:\n{out}")
        self.assertNotIn("falls monotonically", out)

    def test_a_falling_ladder_is_still_called_falling(self):
        out = self._run({1: 1.50, 2: 1.40, 3: 1.20, 5: 1.05})
        self.assertIn("falls monotonically", out,
                      f"a monotone ladder was misreported:\n{out}")
        self.assertNotIn("interior maximum", out)

    def test_the_claim_is_never_printed_without_being_checked(self):
        src = (Path(__file__).parent / "cost_model.py").read_text(encoding="utf-8")
        head, _, tail = src.partition("--- implied optimum ---")
        self.assertTrue(tail, "the implied-optimum section is gone")
        # The words may appear inside the explanatory paragraph, which is allowed. What is not
        # allowed is a verdict string that no comparison produced, so require the shapes to be
        # assigned from a comparison over the measured sequence.
        self.assertIn("zip(seq, seq[1:])", tail,
                      "the ladder's shape is no longer derived from the measured sequence")


class TestFitUncertaintyHasTwoSources(unittest.TestCase):
    """`c` carried a prompt-bootstrap interval and nothing about misfit across widths.

    fit_ci redraws which prompts contribute to each width's mean `k`. It never asks whether a
    straight line is the right shape for those means. On Phase M's MoE arm the w = 4 point misses
    the line by 0.137, 4.7 % of `k`, while the bootstrap reports a half-width of 0.0036.

    The first fix for that was itself wrong and is recorded in PREREGISTRATION.md Correction 14:
    the residual was treated as independent noise on EACH fit and the two were added in
    quadrature, which inflates the uncertainty on a difference by more than an order of magnitude.
    Most of the residual is curvature in k(w), it is shared between arms measured on the same
    card, and it cancels when the difference is taken. So the shape check belongs on the
    difference, and per-fit residuals are reported as lack of fit rather than as a standard error
    on `c`.
    """

    def test_slope_se_matches_the_textbook_value(self):
        import cost_model as CM
        # xs = 1,2,3  ys = 1,3,4 -> slope 1.5, residuals -1/6, +1/3, -1/6, Sxx = 2, dof = 1
        # se = sqrt((1/6) / 1 / 2) = sqrt(1/12)
        self.assertAlmostEqual(CM._slope_se([1, 2, 3], [1, 3, 4]), (1 / 12) ** 0.5, places=10)

    def test_a_perfect_line_has_no_residual_uncertainty(self):
        import cost_model as CM
        self.assertAlmostEqual(CM._slope_se([1, 2, 3, 4], [2, 4, 6, 8]), 0.0, places=12)

    def test_two_points_report_no_se_rather_than_zero(self):
        import cost_model as CM
        # Two points always lie on their own line. Returning 0.0 would say the slope is known
        # exactly, which is the opposite of what two points support.
        self.assertIsNone(CM._slope_se([1, 2], [1, 5]))

    def test_the_report_names_the_wider_figure_when_it_is_wider(self):
        import contextlib
        import cost_model as CM
        import io
        # Widths 2,3,4,6,8 with a deliberate kink, so the width residuals dwarf any prompt spread.
        K = {2: 1.20, 3: 1.50, 4: 2.10, 6: 2.35, 8: 2.95}
        recs, arms = [], {"baseline": {"extra_args": [], "tree": "master", "model": None}}
        for i in range(8):
            recs.append({"arm": "baseline", "pass": 1, "prompt": f"p{i}",
                         "class": "code" if i % 2 else "prose", "decode_tok_s": 100.0,
                         "predicted_n": 400, "hit_cap": True,
                         "timings": {"t_draft_n": 0, "t_draft_n_accepted": 0}})
        for w, k in K.items():
            n = w - 1
            arms[f"mtp-n{n}"] = {"extra_args": ["--spec-type", "draft-mtp",
                                                "--spec-draft-n-max", str(n)],
                                 "tree": "master", "expects_drafter": True, "model": None}
            for i in range(8):
                f = 150
                accepted = 399 - f
                mean_len = 1.0 + accepted / f
                recs.append({"arm": f"mtp-n{n}", "pass": 1, "prompt": f"p{i}",
                             "class": "code" if i % 2 else "prose",
                             "decode_tok_s": 100.0 * mean_len / k, "predicted_n": 400,
                             "hit_cap": True,
                             "timings": {"t_draft_n": n * f, "t_draft_n_accepted": accepted}})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CM.report({"env": {"model": "/m/x.gguf"}, "arms": arms, "records": recs})
        out = buf.getvalue()
        self.assertIn("lack of fit across widths", out,
                      "the fit reports only the prompt bootstrap")
        self.assertIn("k(w=1) extrapolates to", out,
                      "the fit is not checked against the floor a zero-depth cycle must cost")
        # Every prompt in this fixture returns the same k, so the prompt bootstrap collapses to
        # zero width. That is the extreme of the defect, not an exemption from it: a zero-width
        # interval prints as perfect precision and means no precision was estimated.
        self.assertIn("ZERO width", out,
                      "a bootstrap interval that collapsed to a point was reported as if it "
                      "were a precise one")
        # And the residual must not be relabelled back into a standard error on `c`.
        self.assertNotIn("`c` is known to about", out,
                         "the width residual is being reported as sampling uncertainty on c "
                         "again; it is lack of fit, and on a difference it largely cancels")


class TestMatchedAcceptanceContrast(unittest.TestCase):
    """Two methods on one target at the same acceptance must be reported side by side.

    Acceptance is what draft quality buys. Phase M contains a pair where it is matched almost
    exactly across methods on the same target -- the 0.8B draft-simple arm at n-max 4 accepts
    38.7 %, the built-in MTP head at n-max 5 accepts 38.6 % -- and they land 76 points of baseline
    apart, 0.397x against 1.153x. That single pair rules out draft quality as the explanation for
    the sign difference, and it was sitting in the data unremarked because nothing looked for it.
    """

    @staticmethod
    def _result(specs):
        """specs: {arm: (spec_type_args, n_max, drafted_per_req, accepted_per_req, tok_s)}."""
        arms = {"baseline": {"extra_args": [], "tree": "master", "model": None}}
        recs = []
        for i in range(8):
            recs.append({"arm": "baseline", "pass": 1, "prompt": f"p{i}",
                         "class": "code" if i % 2 else "prose", "decode_tok_s": 100.0,
                         "predicted_n": 400, "hit_cap": True,
                         "timings": {"t_draft_n": 0, "t_draft_n_accepted": 0}})
        for arm, (stype, n, drafted, accepted, tok_s) in specs.items():
            arms[arm] = {"extra_args": ["--spec-type", stype, "--spec-draft-n-max", str(n)],
                         "tree": "master", "expects_drafter": True, "model": None}
            for i in range(8):
                recs.append({"arm": arm, "pass": 1, "prompt": f"p{i}",
                             "class": "code" if i % 2 else "prose", "decode_tok_s": tok_s,
                             "predicted_n": 400, "hit_cap": True,
                             "timings": {"t_draft_n": drafted,
                                         "t_draft_n_accepted": accepted}})
        return {"env": {"model": "/m/x.gguf"}, "arms": arms, "records": recs}

    def _run(self, specs):
        import contextlib
        import cost_model as CM
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CM.report(self._result(specs))
        return buf.getvalue()

    def test_a_matched_pair_across_methods_is_found_and_quantified(self):
        out = self._run({
            "simple-n4": ("draft-simple", 4, 1000, 387, 40.0),
            "mtp-n5": ("draft-mtp", 5, 1000, 386, 115.0),
        })
        self.assertIn("matched acceptance", out,
                      f"a pair matched to 0.1 acceptance points was not reported:\n{out[-2500:]}")
        self.assertRegex(out, r"acceptance differs by 0\.1 points, throughput by 7[0-9] points")
        # A pair whose two arms verify at different widths must say which way that pushes.
        # The bare word CONFOUNDED tells a reader to discard the pair; on every Phase M pair the
        # confound runs against the gap, so discarding it would throw away the finding.
        if "CONFOUNDED" in out:
            self.assertRegex(out, r"The confound runs (AGAINST|WITH) the gap",
                             "a pair was flagged as confounded without saying which way")

    def test_two_arms_of_the_SAME_method_are_not_a_contrast(self):
        # Two MTP depths at matched acceptance separate on depth, not on the drafter, so the pair
        # would not isolate anything and must not be presented as if it did.
        out = self._run({
            "mtp-n4": ("draft-mtp", 4, 1000, 387, 140.0),
            "mtp-n5": ("draft-mtp", 5, 1000, 386, 115.0),
        })
        self.assertIn("TEST 2", out, "the report did not run; the assertion below is vacuous")
        self.assertNotIn("matched acceptance", out,
                         "two arms of one method were offered as a cross-method contrast")

    def test_unmatched_acceptance_yields_no_pair(self):
        out = self._run({
            "simple-n4": ("draft-simple", 4, 1000, 200, 40.0),
            "mtp-n5": ("draft-mtp", 5, 1000, 700, 115.0),
        })
        self.assertIn("TEST 2", out, "the report did not run; the assertion below is vacuous")
        self.assertNotIn("matched acceptance", out,
                         "arms 50 acceptance points apart were reported as matched")


class TestBaselinesDoNotAppearAsEffects(unittest.TestCase):
    """A baseline compared to another baseline is not a speculative effect.

    Phase M declares two baselines, one per target, and its BASELINE_MAP covers only the
    speculative arms. `baseline-dense` therefore fell through to analyze.py's `default_baseline`
    -- the first arm present -- and printed in the primary table as "-71.59 % SLOWER" against
    `baseline-moe`, in the same shape as every real row. The number is true and it is the speed
    difference between two models, not an effect of anything.

    The same fallback is the general hazard: any arm a matrix forgets to map is silently compared
    to whichever arm happens to be first.
    """

    @staticmethod
    def _result():
        arms, recs = {}, []
        for name, model, tok_s in (("baseline-a", None, 100.0), ("baseline-b", "/m/b", 40.0),
                                   ("spec-a", None, 130.0), ("orphan", None, 90.0)):
            arms[name] = {"extra_args": ([] if name.startswith("baseline")
                                         else ["--spec-type", "draft-mtp",
                                               "--spec-draft-n-max", "2"]),
                          "tree": "master", "model": model,
                          "expects_drafter": not name.startswith("baseline")}
            for i in range(6):
                recs.append({"arm": name, "pass": 1, "prompt": f"p{i}",
                             "class": "code" if i % 2 else "prose", "decode_tok_s": tok_s,
                             "predicted_n": 400, "hit_cap": True,
                             "timings": {"t_draft_n": 0, "t_draft_n_accepted": 0}})
        return {"arms": arms, "records": recs,
                "design": {"passes": 1, "n_prompts": 6, "prompt_classes": {},
                           "interleaved": True, "fresh_server_per_arm_per_pass": True},
                "baseline_map": {"spec-a": "baseline-a"}}

    def _run(self):
        import analyze as AN
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            AN.report(self._result())
        return buf.getvalue()

    def test_a_baseline_is_not_listed_among_the_effects(self):
        out = self._run()
        # The section header itself ends in "---", so split on a newline-anchored one.
        primary = out.split("--- PRIMARY")[1].split("\n---")[0]
        self.assertNotIn("baseline-b ", primary,
                         f"a baseline appears in the primary effect table:\n{primary}")
        self.assertIn("spec-a", primary, "the real speculative arm went missing")

    def test_the_two_baselines_are_contrasted_separately_and_labelled(self):
        out = self._run()
        self.assertIn("unspeculated contrast", out,
                      "two baselines were not contrasted at all, so the reader loses the "
                      "control the comparison provides")
        contrast = out.split("unspeculated contrast")[1].split("--- PRIMARY")[0]
        self.assertIn("differs in: model", contrast,
                      f"the contrast does not name what differs between the two "
                      f"baselines:\n{contrast}")

    def test_an_unmapped_arm_is_named_rather_than_silently_defaulted(self):
        out = self._run()
        self.assertRegex(out, r"WARNING: orphan is not in the baseline map",
                         "an arm missing from a non-empty baseline map was compared to whichever "
                         "arm happened to be first, without saying so")


class TestEmptyDivergenceSaysWhy(unittest.TestCase):
    """"no divergence records (are the arms greedy, and did the baseline run?)" named neither cause.

    Divergence is attached POST-PASS, because arm order rotates within a pass and the baseline arm
    can run after the arms measured against it, so a result file whose first pass has not closed
    carries none at all. That is the normal state of any run still in progress and it is not a
    fault, but the message guessed at two other causes and sent a reader through bench.py looking
    for a defect that was not there.
    """

    @staticmethod
    def _result(arm_names, temps=0.0, seen=None):
        arms = {"baseline": {"extra_args": [], "tree": "master", "expects_drafter": False}}
        for a in arm_names:
            arms[a] = {"extra_args": ["--spec-type", "draft-mtp"], "tree": "master",
                       "expects_drafter": True}
        recs = []
        for a in (seen if seen is not None else ["baseline"] + list(arm_names)):
            for i in range(3):
                recs.append({"arm": a, "pass": 1, "prompt": f"p{i}", "class": "code",
                             "decode_tok_s": 100.0, "temperature": temps, "predicted_n": 400})
        return {"arms": arms, "records": recs}

    def _run(self, result):
        import contextlib
        import divergence_report as DR
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            DR.report(result)
        return buf.getvalue()

    def test_an_unfinished_pass_is_named_as_such(self):
        out = self._run(self._result(["a", "b", "c"], seen=["baseline", "a"]))
        self.assertIn("no pass has closed yet", out, out[:400])
        self.assertIn("of the 3 arms this matrix declares", out,
                      "the diagnosis does not say how far the pass got")

    def test_a_non_greedy_run_is_named_as_such(self):
        out = self._run(self._result(["a"], temps=0.7))
        self.assertIn("not all greedy", out, out[:400])
        self.assertIn("0.7", out, "the diagnosis does not show the temperatures it found")

    def test_the_old_guess_is_gone(self):
        out = self._run(self._result(["a", "b"], seen=["baseline", "a"]))
        self.assertNotIn("are the arms greedy, and did the baseline run?", out,
                         "the message still guesses instead of diagnosing")


class TestSlopesAreComparedOnSharedWidths(unittest.TestCase):
    """Two `c` values fitted over different width ranges are not the same quantity.

    `k(w)` is curved -- every fit in this study extrapolates below the floor a zero-depth cycle
    must cost -- so a slope is a CHORD, and a chord over widths 3 to 7 is a different number from
    a chord over 2 to 8. phase_nmax fits draft-dflash on {3,5,7} and draft-mtp on {2..8} and the
    reported difference moves from -0.0424 to -0.0473 when the ranges are matched, a sixth of the
    effect. Phase A is worse: {5,8} against {3,4,6}, which share NO width at all, and its
    "the two coefficients agree to 1.7 %" compared chords of disjoint arcs.

    The second half is the shape check. Comparing each fit's width residual separately and adding
    them in quadrature treats shared curvature as independent noise: on phase_nmax the two arms'
    residuals over {3,5,7} are +0.0209/-0.0418/+0.0209 and +0.0210/-0.0420/+0.0210, the same
    number twice, and the difference of the two curves is straight to 2.4e-4. Adding those in
    quadrature inflated the bound twentyfold and produced a retraction that was itself wrong
    (PREREGISTRATION.md Corrections 13 and 14).
    """

    @staticmethod
    def _result(width_k):
        """width_k: {arm: (spec_type, {width: k})} -> a result file that reproduces those k."""
        arms = {"baseline": {"extra_args": [], "tree": "master", "model": None}}
        recs = []
        for i in range(10):
            recs.append({"arm": "baseline", "pass": 1, "prompt": f"p{i}",
                         "class": "code" if i % 2 else "prose", "decode_tok_s": 100.0,
                         "predicted_n": 400, "hit_cap": True,
                         "timings": {"t_draft_n": 0, "t_draft_n_accepted": 0}})
        for arm, (stype, ks) in width_k.items():
            for w, k in ks.items():
                n = w - 1
                name = f"{arm}-n{n}"
                arms[name] = {"extra_args": ["--spec-type", stype,
                                             "--spec-draft-n-max", str(n)],
                              "tree": "master", "expects_drafter": True, "model": None}
                for i in range(10):
                    f = 150
                    accepted = 399 - f
                    mean_len = 1.0 + accepted / f
                    recs.append({"arm": name, "pass": 1, "prompt": f"p{i}",
                                 "class": "code" if i % 2 else "prose",
                                 "decode_tok_s": 100.0 * mean_len / k, "predicted_n": 400,
                                 "hit_cap": True,
                                 "timings": {"t_draft_n": n * f,
                                             "t_draft_n_accepted": accepted}})
        return {"env": {"model": "/m/x.gguf"}, "arms": arms, "records": recs}

    def _run(self, width_k):
        import contextlib
        import cost_model as CM
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CM.report(self._result(width_k))
        return buf.getvalue()

    def test_disjoint_width_ranges_are_refused_not_compared(self):
        out = self._run({
            "a": ("draft-dflash", {5: 2.0, 8: 2.9}),
            "b": ("draft-mtp", {3: 1.5, 4: 1.8, 6: 2.4}),
        })
        self.assertIn("share 0 width(s)", out,
                      f"two fits over disjoint width ranges were compared as if they estimated "
                      f"the same quantity:\n{out[-1500:]}")
        self.assertNotRegex(out, r"c\(.*\) - c\(.*\) = ",
                            "a difference was still printed for disjoint ranges")

    def test_the_comparison_uses_only_the_shared_widths(self):
        # `b` is deliberately curved outside the shared range so that fitting it on its own
        # widths gives a different slope from fitting it on the shared ones.
        out = self._run({
            "a": ("draft-dflash", {3: 1.50, 5: 2.00, 7: 2.50}),
            "b": ("draft-mtp", {2: 1.05, 3: 1.40, 5: 2.10, 7: 2.90, 8: 3.40}),
        })
        self.assertIn("both fitted on the shared widths [3, 5, 7]", out,
                      f"the comparison did not restrict to the shared range:\n{out[-1500:]}")

    def test_shared_curvature_cancels_instead_of_widening_the_bound(self):
        # Both arms carry the SAME curvature and differ by an exact straight line. A shape check
        # that treats each arm's residual as independent noise would call this unresolved; a
        # paired one sees that the difference is straight and resolves it.
        bend = {3: 0.02, 5: -0.04, 7: 0.02}
        a = {w: 1.0 + 0.20 * (w - 1) + bend[w] for w in (3, 5, 7)}
        b = {w: 1.0 + 0.30 * (w - 1) + bend[w] for w in (3, 5, 7)}
        out = self._run({"a": ("draft-dflash", a), "b": ("draft-mtp", b)})
        self.assertIn("shared and cancels", out,
                      f"identical curvature in both arms was charged against the "
                      f"comparison:\n{out[-1800:]}")
        self.assertIn("VERDICT: the marginal costs differ", out,
                      "a difference of exactly 0.10 per position was reported as unresolved")

    def test_unshared_curvature_still_widens_the_bound(self):
        # Only `a` bends. The difference is then not linear and the comparison must say so.
        a = {w: 1.0 + 0.20 * (w - 1) + bend for w, bend in ((3, 0.05), (5, -0.10), (7, 0.05))}
        b = {w: 1.0 + 0.205 * (w - 1) for w in (3, 5, 7)}
        out = self._run({"a": ("draft-dflash", a), "b": ("draft-mtp", b)})
        self.assertIn("curvature does NOT cancel", out,
                      f"curvature present in one arm only was treated as cancelling:\n"
                      f"{out[-1800:]}")
        self.assertIn("VERDICT: not resolved", out)


class TestForwardsIsDerivedInOnePlace(unittest.TestCase):
    """`predicted_n - accepted - 1` may appear in exactly one module.

    speclen.py exists because this study shipped that derivation three times and the copies parted
    company -- analyze.py's own comment records that the second attempt "landed in one file and
    not the other, which left two different mean lengths in the same repo". A fourth copy went
    into analysis/plot_phase_m.py during this session's own review, which is how this test came to
    be written.

    The copies are not merely redundant. `speclen.forwards` returns the EXACT
    `draft_n_verif_steps` counter when a record carries one, and derives only as a fallback. A
    private copy keeps guessing after the llama.cpp patch that exposes that counter lands, and
    nothing errors: two numbers that used to agree quietly stop agreeing.
    """

    ROOT = Path(__file__).parent.parent

    def test_no_module_outside_speclen_recomputes_it(self):
        import ast
        # This study's own code only, recursively. Globbing `*/*.py` from the repo root reached
        # into llamacpp-master/, which holds llama.cpp's own convert scripts -- third-party source
        # that this rule has no business policing -- and would have missed anything nested deeper
        # than two levels in the directories it does own.
        offenders = []
        roots = [self.ROOT / d for d in ("harness", "analysis", "repro")]
        for path in sorted(q for r in roots if r.is_dir() for q in r.rglob("*.py")):
            if path.name in ("speclen.py", "test_harness.py") or ".venv" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            # Expressions only, so the identity written out in a docstring or a comment is fine.
            for node in ast.walk(tree):
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Sub):
                    continue
                try:
                    src = ast.unparse(node)
                except Exception:                                     # noqa: BLE001
                    continue
                low = src.lower()
                if "accept" in low and ("predicted_n" in low or "pn" in low.split()) \
                        and low.rstrip().endswith("- 1"):
                    offenders.append(f"{path.relative_to(self.ROOT)}:{node.lineno}: {src}")
        self.assertFalse(
            offenders,
            "the forward-pass count is derived outside speclen.py, which is how this repo "
            "previously ended up with two different mean lengths, and which stops the exact "
            "draft_n_verif_steps counter from being picked up when it arrives:\n  "
            + "\n  ".join(offenders))


class TestVerificationWidthIsBracketed(unittest.TestCase):
    """`n_max + 1` is what the flags asked for, not what the target verified.

    The fit regressed `k` on the requested width for every arm. For the MTP arms that is right to
    within 1 %. For the 0.8B `draft-simple` arms the drafter delivered 53-77 % of it, and refitting
    on the delivered figure moves `c` from 0.2909 to 0.5851 -- a factor of two hiding behind four
    printed decimals.

    The counters cannot close the bracket, which is why this reports a range rather than swapping
    the regressor. Two mechanisms push the delivered figure below the verified width and neither is
    separable from `t_draft_n`: the drafter can stop short of its budget, and on partial acceptance
    the server replays the accepted prefix instead of drafting -- server-context.cpp:3818 returns
    before `spec_draft` is moved out, so the next cycle takes the reuse branch at :2893 and costs a
    forward pass while generating nothing. llama.cpp counts the real thing at :3859 and does not
    return it.

    The same bracket decides which kernel an arm took, and there the old code and the prose
    disagreed from the same data: `dflash2-n8` brackets [7.94, 9.00] across MMVQ_MAX_BATCH_SIZE of
    8, the fit called it off-path on the 9, and docs/COST_MODEL.md called it on-path on the 7.94.
    """

    @staticmethod
    def _result(arm_widths):
        """arm_widths: {arm: (spec, {n_max: (k, delivered_draft_per_forward)})}."""
        arms = {"baseline": {"extra_args": [], "tree": "master", "model": None}}
        recs = []
        for i in range(8):
            recs.append({"arm": "baseline", "pass": 1, "prompt": f"p{i}",
                         "class": "code" if i % 2 else "prose", "decode_tok_s": 100.0,
                         "predicted_n": 400, "hit_cap": True,
                         "timings": {"t_draft_n": 0, "t_draft_n_accepted": 0}})
        for arm, (stype, ladder) in arm_widths.items():
            for n, (k, dpf) in ladder.items():
                name = f"{arm}-n{n}"
                arms[name] = {"extra_args": ["--spec-type", stype,
                                             "--spec-draft-n-max", str(n)],
                              "tree": "master", "expects_drafter": True, "model": None}
                for i in range(8):
                    f = 150
                    accepted = 399 - f
                    mean_len = 1.0 + accepted / f
                    recs.append({"arm": name, "pass": 1, "prompt": f"p{i}",
                                 "class": "code" if i % 2 else "prose",
                                 "decode_tok_s": 100.0 * mean_len / k, "predicted_n": 400,
                                 "hit_cap": True,
                                 "timings": {"t_draft_n": int(round(dpf * f)),
                                             "t_draft_n_accepted": accepted}})
        return {"env": {"model": "/m/x.gguf"}, "arms": arms, "records": recs}

    def _run(self, arm_widths):
        import contextlib
        import cost_model as CM
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CM.report(self._result(arm_widths))
        return buf.getvalue()

    def test_a_drafter_that_fills_its_budget_gets_no_bracket(self):
        # Delivered == requested, as every MTP arm in this study is. Printing a bracket here would
        # be noise, and the coefficients must not move.
        out = self._run({"mtp": ("draft-mtp", {2: (1.6, 2.0), 4: (2.2, 4.0), 6: (2.8, 6.0)})})
        self.assertNotIn("is bracketed", out,
                         f"a bracket was reported for an arm that delivered its full "
                         f"budget:\n{out[-1200:]}")

    def test_a_drafter_that_falls_short_reports_c_as_a_range(self):
        # Delivered about 60 % of the requested depth, as the 0.8B arms do.
        out = self._run({"simple": ("draft-simple",
                                    {2: (4.4, 1.3), 4: (5.0, 2.3), 6: (5.6, 3.3)})})
        self.assertIn("is bracketed", out,
                      f"c was printed to four decimals for an arm whose verified width is not "
                      f"known to a factor of two:\n{out[-1500:]}")
        self.assertIn("what the FLAGS asked for", out)

    def test_a_width_straddling_the_dispatch_limit_is_named(self):
        # Requested 9, delivered 7.9: the kernel it took is not decided by this data.
        out = self._run({"d": ("draft-dflash",
                               {2: (1.5, 2.0), 4: (2.0, 4.0), 6: (2.5, 6.0), 8: (2.7, 6.9)})})
        self.assertIn("bracket ACROSS the MMVQ limit", out,
                      f"an arm whose width bracket spans the dispatch limit was classified "
                      f"silently:\n{out[-1500:]}")


class TestHostContentionIsRecorded(unittest.TestCase):
    """The card was gated on temperature and clock; the host was gated on nothing.

    On 2026-08-26 a compiler ran on this machine during Phase M pass 2. Finding out afterwards
    meant comparing object-file timestamps against server-log timestamps by hand, and the answer
    was still equivocal (PREREGISTRATION.md Corrections 15 and 16). A field recorded at arm entry
    turns that into a lookup.
    """

    def test_no_descendant_of_this_process_is_counted_as_competition(self):
        """The contract, checked against the live host rather than a name list.

        This asserted that nothing starting with "python" appears in `competing`, which was true
        while python3 was on own_names and became both wrong and flaky when attribution moved to
        descent: another python on the host IS competition now, and whether one happens to be
        alive decides whether the assertion holds. What does hold, always, is that nothing the
        caller started is counted.
        """
        import os
        import subprocess
        import telemetry as T
        load = T.host_load()
        if load.get("note"):
            self.skipTest(load["note"])
        for c in load["competing"]:
            self.assertNotIn("llama-server", c["comm"],
                             "the run's own server is being counted against it")
        ps = subprocess.run(["ps", "-eo", "pid,ppid", "--no-headers"],
                            capture_output=True, text=True, timeout=30)
        ppid_of = {}
        for line in ps.stdout.splitlines():
            f = line.split()
            if len(f) == 2:
                try:
                    ppid_of[int(f[0])] = int(f[1])
                except ValueError:
                    pass
        mine = T._descendants_of(os.getpid(), ppid_of)
        for c in load["competing"]:
            self.assertNotIn(c.get("pid"), mine,
                             f"a process this one started is being counted against it: {c}")

    # A ps table with a compiler two levels below pid 100, an rsync belonging to nobody, and
    # two processes under the 5 % floor.
    PS = ("  100     1  0.5 bash\n"
          "  200   100  0.3 python3\n"
          "  300   200 310.0 cc1plus\n"
          "  400     1 290.0 rsync\n"
          "  500     1  2.0 sshd")

    def test_a_busy_host_is_flagged(self):
        """Fed a table, not measured on the machine running the suite.

        This used to pass own_names=() so the harness's own python counted against it, which is
        how the threshold was exercised. Descent attribution makes that impossible: the harness
        is always its own descendant. Burning CPU to make a real busy host would put load on a
        machine that may be measuring, which is the one thing this module exists to prevent.
        """
        import telemetry as T
        load = T.host_load(own_names=(), _ps_output=self.PS, _self_pid=999)
        self.assertAlmostEqual(load["competing_pct"], 600.0)
        self.assertEqual({c["comm"] for c in load["competing"]}, {"cc1plus", "rsync"})
        self.assertTrue(load["contended"])
        self.assertEqual(load["contended"], load["competing_pct"] >= 25.0)

    def test_a_descendant_however_deep_is_not_competition(self):
        import telemetry as T
        load = T.host_load(own_names=(), _ps_output=self.PS, _self_pid=100)
        self.assertAlmostEqual(load["competing_pct"], 290.0,
                               msg="cc1plus is a grandchild of 100 and must not be counted")
        self.assertEqual({c["comm"] for c in load["competing"]}, {"rsync"})

    def test_a_quiet_host_is_not_flagged_and_the_floor_holds(self):
        import telemetry as T
        quiet = "  100     1  0.5 bash\n  500     1  2.0 sshd\n  600     1  4.9 cron"
        load = T.host_load(own_names=(), _ps_output=quiet, _self_pid=999)
        self.assertEqual(load["competing_pct"], 0.0,
                         "everything here is under the 5 % floor")
        self.assertFalse(load["contended"])

    def test_the_gate_is_wired_into_arm_entry(self):
        # Shaped as a source check on purpose: the defect this guards against is the call being
        # absent, which no unit test of host_load() can see.
        src = (Path(__file__).parent / "bench.py").read_text(encoding="utf-8")
        self.assertIn("T.host_load()", src, "arm entry does not sample host load")
        self.assertIn("arm_pass_host_load", src, "the sample is not recorded per arm-pass")
        self.assertIn("host_contended", src, "a contended host raises no incident")
        settle = src.index("settle['entry_sm_clock_mhz']")
        self.assertLess(settle, src.index("T.host_load()"),
                        "host load is sampled before the thermal settle, so it measures the wait "
                        "rather than the conditions the arm actually ran under")


class TestPassStabilityFindsTheNoisyArmPass(unittest.TestCase):
    """The unit is the arm-pass against the same prompts in its own other passes.

    Corrections 15, 16 and 17 are three readings of one question, done by hand, and the first two
    were wrong. The first compared a spread computed over one number per arm; the second used the
    per-prompt spread and found no group effect; the third compared each arm-pass against its own
    repeats and found that the suspect arm was equally noisy in a pass that predates the event
    entirely. Only the third measure could reach that.
    """

    @staticmethod
    def _result(noisy_arm=None, host_load=None):
        import math
        recs = []
        for arm in ("a", "b", "c"):
            for p in (1, 2):
                for i in range(12):
                    base = 100.0 + 3.0 * math.sin(i)          # prompt-to-prompt, cancels when paired
                    wob = 0.0
                    if arm == noisy_arm and p == 2:
                        wob = 9.0 * math.cos(i * 2.0)          # this arm-pass only
                    recs.append({"arm": arm, "pass": p, "prompt": f"p{i}",
                                 "class": "code" if i % 2 else "prose",
                                 "decode_tok_s": base + wob, "predicted_n": 400, "hit_cap": True})
        out = {"records": recs}
        if host_load is not None:
            out["arm_pass_host_load"] = host_load
        return out

    def _run(self, result):
        import contextlib
        import io
        import pass_stability as PS
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            PS.report(result)
        return buf.getvalue()

    def test_a_quiet_matrix_flags_nothing(self):
        out = self._run(self._result())
        self.assertIn("No arm-pass exceeds the threshold", out,
                      f"a matrix with no disturbed arm-pass produced a flag:\n{out}")

    def test_the_disturbed_arm_pass_is_named(self):
        out = self._run(self._result(noisy_arm="b"))
        self.assertIn("OUTLIER", out, f"the disturbed arm-pass was not flagged:\n{out}")
        line = [l for l in out.splitlines() if "OUTLIER" in l]
        self.assertTrue(any(l.strip().startswith("b ") for l in line),
                        f"the wrong arm-pass was flagged:\n{out}")

    def test_a_recorded_contended_host_is_reported_as_a_cause(self):
        out = self._run(self._result(noisy_arm="b", host_load={
            "pass02_b": {"contended": True, "competing_pct": 310.0,
                         "competing": [{"comm": "cc1plus", "pcpu": 290.0}]}}))
        self.assertIn("host was contended", out)
        self.assertIn("cc1plus", out, "the competing process is not named")
        self.assertIn("recorded cause, not an inference", out)

    def test_an_uncontended_host_says_the_scatter_is_the_arms_own(self):
        out = self._run(self._result(noisy_arm="b", host_load={
            "pass02_b": {"contended": False, "competing_pct": 0.0, "competing": []}}))
        self.assertIn("the arm's own", out,
                      "a flagged arm-pass on a quiet host was left ambiguous, which is the "
                      "question Correction 17 had to answer by hand")

    def test_scatter_that_tracks_power_at_constant_clock_is_named(self):
        import contextlib
        import io
        import math
        import pass_stability as PS
        # Identical work, same clock, and power moving with throughput: the GPU idling less. That
        # is the signature every outlier in Phase M carries, and the thermal gate cannot see it.
        recs = []
        for p in (1, 2):
            for i in range(12):
                wob = 9.0 * math.cos(i * 2.0) if p == 2 else 0.0
                recs.append({"arm": "a", "pass": p, "prompt": f"p{i}",
                             "class": "code" if i % 2 else "prose",
                             "decode_tok_s": 100.0 + wob, "predicted_n": 400, "hit_cap": True,
                             "power": {"power_mean_w": 200.0 + 2.0 * wob,   # moves with it
                                       "sm_clock_mean_mhz": 1920.0}})       # does not
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            PS.report({"records": recs})
        out = buf.getvalue()
        self.assertIn("tracks POWER", out, f"the power signature was not reported:\n{out}")
        self.assertIn("idling less", out)

    def test_a_file_without_the_field_says_so_rather_than_guessing(self):
        out = self._run(self._result(noisy_arm="b"))
        self.assertIn("predates arm_pass_host_load", out,
                      "a result with no host record implied the cause was known")


class TestCrossRungRefusesWhatItCannotCompare(unittest.TestCase):
    """The first run of cross_rung.py compared a finished rung against a one-pass rung.

    It printed a paired interval, then took the ONE rung that had three passes, called its
    pass-to-pass spread the within-rung drift for both, and concluded the difference was "several
    times the within-rung drift". The half-finished rung had entered that maximum as a spread of
    0.0000, because one value has a range of zero -- which reads as "no drift" when it means "no
    estimate". A completeness guard had been in the design and was left out of the code.

    Every case below is that failure or one of its neighbours.
    """

    ROOT = Path(__file__).parent.parent
    FIXTURE = "results/phase_q_UD-Q4_K_XL.json"

    def _load(self, mutate=None):
        p = self.ROOT / self.FIXTURE
        if not p.exists():
            self.skipTest(f"{self.FIXTURE} not present")
        d = json.loads(p.read_text(encoding="utf-8"))
        if mutate is not None:
            mutate(d)
        return CR.rung_view(d, str(p))

    def _other(self):
        """A second view that differs only in the fields the guards key on."""
        p = self.ROOT / self.FIXTURE
        d = json.loads(p.read_text(encoding="utf-8"))
        d["env"]["model"] = "/somewhere/else/other.gguf"
        d["env"]["model_sha256"] = "ffffffffffff0000"
        return CR.rung_view(d, str(p).replace("Q4", "Q9"))

    def test_a_rung_that_stopped_after_one_pass_is_refused(self):
        def truncate(d):
            d["records"] = [r for r in d["records"] if r["pass"] == 1]
        short = self._load(truncate)
        full = self._other()
        why = " | ".join(CR.guards(short, full))
        self.assertIn("not finished", why,
                      "a rung missing whole arm-passes was accepted for comparison")
        self.assertIn("no within-rung estimate of session drift", why,
                      "a one-pass rung was accepted, so the drift confound has nothing to bound it")

    def test_one_pass_yields_no_drift_estimate_rather_than_a_spread_of_zero(self):
        def truncate(d):
            d["records"] = [r for r in d["records"] if r["pass"] == 1]
        short = self._load(truncate)
        per_pass = CR.per_pass_c(short, short["on_path"])
        self.assertEqual(len(per_pass), 1,
                         "one pass should produce one fit, not a fabricated second one")
        src = inspect.getsource(CR.report)
        self.assertIn("len(spreads) >= 2", src,
                      "the drift verdict must require a spread from BOTH rungs; with one rung's "
                      "spread standing in for both, a rung with no drift estimate silently "
                      "borrows the other's")
        self.assertNotIn("(not spreads) or", src,
                         "an empty spread set must not satisfy the drift check")

    def test_a_ragged_arm_pass_is_refused(self):
        def drop_one(d):
            for i, r in enumerate(d["records"]):
                if r["pass"] == 2 and "mtp-n3" in r["arm"]:
                    del d["records"][i]
                    return
            raise AssertionError("fixture has no mtp-n3 pass-2 record to drop")
        ragged = self._load(drop_one)
        why = " | ".join(CR.guards(ragged, self._other()))
        self.assertIn("wrong prompt count", why,
                      "an arm-pass short by a prompt was accepted; the total record count alone "
                      "cannot see it")

    def test_rungs_measured_with_different_binaries_are_refused(self):
        a = self._load()
        b = self._other()
        b["binaries"] = {"master": {"llama-server": "0000000000000000"}}
        why = " | ".join(CR.guards(a, b))
        self.assertIn("different binaries", why,
                      "two rungs built from different binaries are not a quantization contrast")

    def test_the_same_weights_compared_against_themselves_are_refused(self):
        a = self._load()
        b = self._load()
        why = " | ".join(CR.guards(a, b))
        self.assertTrue("same weights" in why or "same model file" in why,
                        f"comparing a rung with itself was allowed: {why}")

    def test_expected_shape_comes_from_intent_not_from_what_survived(self):
        """Inferring the pass count from the passes present makes every truncation look complete."""
        src = inspect.getsource(CR.rung_view)
        self.assertIn('design.get("passes")', src,
                      "the expected pass count must come from the design block bench.py wrote "
                      "before the run, not from the records that happen to exist")
        self.assertIn('design.get("n_prompts")', src)


class TestRungDriverDeletesOnPathNotOnProvenance(unittest.TestCase):
    """Four defects in run_phase_q.sh, all of which had already fired or were one retry away.

    The rung driver deletes 20 GB of weights on a passing gate, so each of these is a case where
    the wrong thing gets deleted, the right thing gets kept forever, or a rung is skipped that
    could have run.
    """

    ROOT = Path(__file__).parent.parent

    def _src(self):
        # Not a skip. These four guards exist because each defect had already fired once, and a
        # guard that skips when its subject is renamed is a guard that turns itself off: moving
        # the drivers into scripts/ silently disabled all five of them, green, on 2026-08-27.
        f = self.ROOT / "scripts" / "run_phase_q.sh"
        self.assertTrue(f.exists(), "scripts/run_phase_q.sh is missing; this guard has no subject")
        return f.read_text(encoding="utf-8")

    def test_deletion_is_decided_by_path_not_by_whether_this_run_downloaded_it(self):
        """DOWNLOADED=0 on exactly the retry path, which is where cleanup matters most.

        A rung that dies mid-run keeps its weights on purpose. The next invocation finds them
        already staged, takes the reuse branch, sets DOWNLOADED=0 -- and then, having succeeded,
        declines to delete them. 20 GB stays on a disk that had to be cleared by hand to start
        the rung in the first place.
        """
        src = self._src()
        delete = [ln for ln in src.splitlines() if "rm -f" in ln and "SRC" in ln]
        self.assertTrue(delete, "no deletion of the staged file at all")
        guard = [ln for ln in src.splitlines()
                 if 'KEEP' in ln and '$SRC' in ln and ln.strip().startswith("if")]
        self.assertTrue(guard, "the deletion is not guarded by KEEP and a path test")
        self.assertNotIn("DOWNLOADED", " ".join(guard),
                         "deletion still depends on whether THIS invocation downloaded the file; "
                         "that is false on the retry path, where the staged copy is exactly the "
                         "one that should be cleaned up")

    def test_disk_and_vram_are_separate_tables(self):
        """One table used for both is how UD-Q5_K_XL was skipped with enough disk to run it.

        23.1 is a VRAM figure in decimal GB. The guard added 10 and compared it against `df -BG`,
        which answers in GiB, so a 19.44 GiB file was refused into 28 GiB of free space and a
        night of measurement was lost.
        """
        src = self._src()
        self.assertIn("NEED_VRAM_GB", src, "no VRAM table")
        self.assertIn("NEED_DISK_GIB", src, "no separate on-disk-size table")
        dl = src.index("hf download")
        disk_use = src.index("NEED_DISK_GIB[$RUNG]", src.index("free_gb="))
        self.assertLess(disk_use, dl, "the disk guard does not consult the on-disk table")
        head = src[:src.index("hf download")]
        self.assertNotIn("NEED_VRAM_GB[$RUNG]} + 10", head,
                         "a VRAM figure is still being used as a disk requirement")

    def test_the_completeness_gate_reads_the_arm_list_from_the_matrix(self):
        """A hardcoded arm count silently truncates a matrix that grows a width.

        With N_ARMS=4 baked into the driver, adding a fifth width to phase_q.py makes a complete
        run 375 records; the driver calls it done at 300 and deletes the weights.
        """
        src = self._src()
        self.assertIn("matrices.phase_q", src,
                      "the gate does not consult the matrix for the arm list")
        self.assertNotIn("N_ARMS=", src,
                         "an arm count is still hardcoded in the driver")
        self.assertIn("arm_source", src,
                      "the gate binds itself to the matrix with no fallback; the matrix refuses "
                      "to import once the rung's weights are gone, which is the state every "
                      "skip decision runs in")

    def test_the_gate_survives_a_matrix_that_cannot_be_imported(self):
        """A finished rung failed its own gate and started re-downloading 20 GB.

        harness/matrices/phase_q.py raises at import time when its target gguf is absent, and
        run_phase_q.sh deletes that gguf as soon as the rung is verified complete. So every
        later invocation asks the gate about a rung whose matrix will not import. Binding the
        gate to the matrix without a fallback turned "already complete, skip" into "partial,
        archive it and re-stage 20 GB". Observed 2026-08-26; the old driver had a comment
        warning about exactly this and the rewrite reintroduced it anyway.

        QWEN_Q_TARGET is set to a value the matrix rejects outright, so the import failure is
        guaranteed regardless of which ggufs happen to be on disk when this runs.
        """
        import os, re, subprocess, tempfile
        f = self.ROOT / "scripts" / "run_phase_q.sh"
        self.assertTrue(f.exists(), "scripts/run_phase_q.sh is missing; this guard has no subject")
        src = f.read_text(encoding="utf-8")
        m = re.search(r"^gate\(\) \{.*?^\}", src, re.S | re.M)
        self.assertIsNotNone(m, "no gate() function to test")
        result = self.ROOT / "results/phase_q_UD-Q5_K_XL.json"
        if not result.exists():
            self.skipTest("no complete rung result to gate")
        # Built as a line list: embedding newlines in the literals is what broke this once.
        lines = ["set -u", "cd " + str(self.ROOT), "N_PROMPTS=25", "PASSES=3",
                 m.group(0), 'gate "' + str(result) + '" NOT_A_REAL_RUNG']
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as th:
            th.write("\n".join(lines) + "\n")
            tmp = th.name
        self.addCleanup(lambda: os.unlink(tmp))
        out = subprocess.run(["bash", tmp], capture_output=True, text=True, timeout=120).stdout
        self.assertTrue(out.startswith("OK"),
                        f"a complete rung was rejected because the matrix would not import: {out!r}")

    def test_deleting_a_rung_also_removes_its_staging_sidecars(self):
        """`rm -f $SRC` leaves a .sha256 naming a file that no longer exists."""
        src = self._src()
        line = [ln for ln in src.splitlines() if "rm -f" in ln and "SRC" in ln]
        self.assertTrue(any(".sha256" in ln for ln in line),
                        "the sha256 sidecar outlives the weights it describes")
        self.assertIn(".cache/huggingface/download", src,
                      "the downloader's per-file cache entry is never cleaned up")


class TestDriverTablesMatchTheirMatrix(unittest.TestCase):
    """A driver that names its files differently from the matrix downloads the wrong thing.

    scripts/run_phase_qsmall.sh listed `Qwen3.5-9B-MTP-Q4_K_M.gguf` for every rung. The repository is
    named `unsloth/Qwen3.5-9B-MTP-GGUF` but the files inside it are `Qwen3.5-9B-Q4_K_M.gguf`, so
    each rung would have 404'd -- and had it not, the matrix, which had the names right, would
    then have failed to import against the file that landed. Two tables for one fact, and nothing
    comparing them.

    Parsed rather than imported: a matrix raises at import time when its target gguf is absent,
    which is the state a checkout is normally in.
    """

    ROOT = Path(__file__).parent.parent
    PAIRS = [("scripts/run_phase_q.sh", "harness/matrices/phase_q.py"),
             ("scripts/run_phase_qsmall.sh", "harness/matrices/phase_qsmall.py")]

    @staticmethod
    def _matrix_files(path):
        """{rung: filename} from the matrix's RUNGS literal, without executing the module."""
        import ast
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(getattr(t, "id", None) == "RUNGS" for t in node.targets):
                continue
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                name = v.elts[0].value if isinstance(v, ast.Tuple) else None
                if name is not None:
                    out[k.value] = name
            return out
        return {}

    @staticmethod
    def _driver_files(path):
        """{rung: filename} from the driver's `declare -A FILE=(...)` block."""
        import re
        src = path.read_text(encoding="utf-8")
        m = re.search(r"declare -A FILE=\((.*?)\)", src, re.S)
        if not m:
            return {}
        return dict(re.findall(r"\[([^\]]+)\]=(\S+)", m.group(1)))

    def test_every_driver_names_the_files_its_matrix_names(self):
        checked = 0
        for driver, matrix in self.PAIRS:
            dp, mp = self.ROOT / driver, self.ROOT / matrix
            if not (dp.exists() and mp.exists()):
                continue
            d, m = self._driver_files(dp), self._matrix_files(mp)
            self.assertTrue(d, f"{driver}: no FILE table found")
            self.assertTrue(m, f"{matrix}: no RUNGS table found")
            for rung, fname in d.items():
                self.assertIn(rung, m, f"{driver} defines rung {rung}, {matrix} does not")
                self.assertEqual(
                    fname, m[rung],
                    f"{driver} would download {fname!r} for rung {rung} but {matrix} looks for "
                    f"{m[rung]!r}; one of them is wrong and the download is the expensive half")
            checked += 1
        self.assertGreater(checked, 0, "no driver/matrix pair was checked; the guard has no "
                                        "subject")

    def test_no_driver_hardcodes_how_many_arms_its_matrix_defines(self):
        """N_ARMS=4 against a matrix defining five is a gate that passes on a truncated run.

        phase_qsmall defines baseline plus n-max 2, 3, 5 and 6. With four written into the
        driver, a run that lost every n-max 6 arm-pass lands on exactly the expected count and
        has its weights deleted -- and n-max 6 is the arm that matches llama.cpp #26750.
        """
        for driver, matrix in self.PAIRS:
            dp = self.ROOT / driver
            if not dp.exists():
                continue
            src = dp.read_text(encoding="utf-8")
            # Comments only, stripped: these drivers explain in prose why the old hardcoded count
            # was wrong, and that prose necessarily contains the token being banned. assertFalse
            # rather than assertNotIn so a failure prints the reason and not the whole script.
            code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
            self.assertFalse("N_ARMS=" in code,
                             f"{driver} hardcodes an arm count instead of asking the matrix")
            self.assertTrue(Path(matrix).stem in code,
                            f"{driver} never consults {matrix} for the arm list")
            self.assertTrue("arm_source" in code,
                            f"{driver} binds to the matrix with no fallback; the matrix refuses "
                            f"to import once a rung's weights are staged out, which is the state "
                            f"every skip decision runs in")


class TestModelSizeSurvivesTheWeightsBeingDeleted(unittest.TestCase):
    """The ladders delete their weights, and a label is not a number.

    phase_q and phase_qsmall stage a rung, measure it, verify it and delete it. A plot of `c`
    against quantization has to be against measured bits per weight -- file size over parameter
    count -- because UD-Q5_K_XL is a name, not a quantity. env recorded the path and the hash,
    which say WHICH file ran, and nothing that says how big it was. After deletion the figure was
    recoverable only from a download log or by re-fetching 20 GB.
    """

    def test_environment_snapshot_records_the_model_size(self):
        from pathlib import Path as P
        e = bench.environment_snapshot({}, P(__file__))
        self.assertIn("model_size_bytes", e,
                      "env does not record the model's size, so bits per weight cannot be "
                      "computed once the weights are staged out")
        self.assertEqual(e["model_size_bytes"], P(__file__).stat().st_size)

    def test_the_size_probe_never_raises(self):
        """A snapshot that dies takes the whole run with it, before a single record is written."""
        from pathlib import Path as P
        self.assertIsNone(bench._size(P("/nonexistent/never/here.gguf")))
        self.assertIsNone(bench._size(P("/")))  # a directory: stat succeeds, but not a file size
        e = bench.environment_snapshot({}, P("/nonexistent/never/here.gguf"))
        self.assertIsNone(e["model_size_bytes"])


class TestLadderTrendIdentifiesItsRungs(unittest.TestCase):
    """A ladder plotted against a label is not plotted against anything measurable.

    ladder_trend took the rung's name from the filename, as stem.split("_")[-1]. That gives "XL"
    for phase_q_UD-Q4_K_XL and "M" for phase_qsmall_Q4_K_M, so two rungs of the same ladder
    collapse onto one label -- and the guard that catches duplicate rungs keys on hashes, not on
    labels, so the collision passes it and shows up only in the printed table. The name now comes
    from the arms, which the matrix builds from the rung it was told to run.
    """

    ROOT = Path(__file__).parent.parent
    FIXTURE = "results/phase_q_UD-Q4_K_XL.json"

    def _load(self, mutate=None):
        p = self.ROOT / self.FIXTURE
        if not p.exists():
            self.skipTest(f"{self.FIXTURE} not present")
        d = json.loads(p.read_text(encoding="utf-8"))
        if mutate is not None:
            mutate(d)
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as th:
            json.dump(d, th)
            tmp = th.name
        self.addCleanup(lambda: os.unlink(tmp))
        return LT.load_rung(tmp)

    def test_the_label_comes_from_the_arms_not_the_filename(self):
        v = self._load()
        self.assertEqual(v["label"], "UD-Q4_K_XL",
                         "the rung label is not derived from the arm names; a filename split "
                         "collapses UD-Q4_K_XL and UD-Q5_K_XL onto the same string")

    def test_both_ladder_tools_name_rungs_the_same_way(self):
        """cross_rung had the same defect in a different disguise.

        It stripped the literal prefix "phase_q_", which is right for phase_q and silently wrong
        for phase_qsmall, whose prefix is longer. Both rungs of a phase_qsmall comparison then
        printed as "phase_qsma" and the two columns became indistinguishable -- in a report whose
        entire content is a difference between those two columns.
        """
        import json as _json
        p = self.ROOT / self.FIXTURE
        if not p.exists():
            self.skipTest(f"{self.FIXTURE} not present")
        d = _json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(CR.rung_view(d, str(p))["label"], LT.load_rung(str(p))["label"],
                         "cross_rung and ladder_trend disagree about what to call a rung")
        qs = self.ROOT / "results/phase_qsmall_Q4_K_M.json"
        if qs.exists():
            lbl = CR.rung_view(_json.loads(qs.read_text(encoding="utf-8")), str(qs))["label"]
            self.assertEqual(lbl, "Q4_K_M",
                             f"a phase_qsmall rung is labelled {lbl!r}; a prefix-strip that only "
                             f"knows 'phase_q_' leaves the phase name in the label")

    def test_a_rung_with_no_recorded_size_has_no_x_value(self):
        """env.model_size_bytes is the x axis. Without it the rung cannot be placed."""
        a = self._load()
        a["size_bytes"] = None
        b = self._load(lambda d: d["env"].update(model_size_bytes=999, model_sha256="b" * 64))
        c = self._load(lambda d: d["env"].update(model_size_bytes=1998, model_sha256="c" * 64))
        why = " | ".join(LT.guards([a, b, c]))
        self.assertIn("model_size_bytes absent", why,
                      "a rung with no recorded file size was accepted onto a size axis")

    def test_two_rungs_with_the_same_weights_are_refused(self):
        a = self._load(lambda d: d["env"].update(model_size_bytes=100, model_sha256="a" * 64))
        b = self._load(lambda d: d["env"].update(model_size_bytes=200, model_sha256="a" * 64))
        c = self._load(lambda d: d["env"].update(model_size_bytes=300, model_sha256="c" * 64))
        why = " | ".join(LT.guards([a, b, c]))
        self.assertIn("same model hash", why,
                      "two rungs carrying identical weights were accepted as distinct points")

    def test_bits_per_weight_needs_a_bf16_rung_and_says_so_when_absent(self):
        a = self._load(lambda d: d["env"].update(model_size_bytes=100, model_sha256="a" * 64))
        bpw, how = LT.bits_per_weight([a])
        self.assertIsNone(bpw, "bits per weight was computed with no bf16 rung to scale it")
        self.assertIn("file size alone", how,
                      "the fallback does not say the axis became file size")

    def test_a_trend_refuses_fewer_than_three_rungs(self):
        """Two rungs are a difference, not a trend, and cross_rung.py already does differences."""
        src = inspect.getsource(LT.main)
        self.assertIn("len(paths) < 3", src)
        self.assertIn("cross_rung", src, "the refusal does not point at the pairwise tool")


class TestSizeBackfillCannotInventANumber(unittest.TestCase):
    """This tool writes a number into a measurement file, so its sourcing has to be exact.

    The first version attached a size comment to the next hash line anywhere below it. A prose
    block naming two sizes therefore gave the second one to a hash that owned the first:
    UD-Q5_K_XL's hash came back with UD-Q4_K_XL's 17559178144 bytes, and that would have been
    written into results/phase_q_UD-Q5_K_XL.json as the size of the weights that produced it.
    Both results lack the field, so this was not hypothetical.
    """

    def _table(self, text):
        import tempfile, os
        from pathlib import Path as P
        with tempfile.NamedTemporaryFile("w", suffix=".SUMS", delete=False) as th:
            th.write(text)
            tmp = th.name
        self.addCleanup(lambda: os.unlink(tmp))
        old = BMS.SUMS
        BMS.SUMS = P(tmp)
        self.addCleanup(lambda: setattr(BMS, "SUMS", old))
        return BMS.sizes_by_hash()

    def test_a_size_two_lines_above_a_hash_is_not_attached_to_it(self):
        t = self._table(
            "# 111111111 bytes\n"
            "# some prose about something else entirely\n"
            + "a" * 64 + "  models/a.gguf\n")
        self.assertEqual(t, {},
                         "a size separated from its hash by another comment was still attached")

    def test_the_last_of_several_sizes_does_not_capture_a_later_hash(self):
        t = self._table(
            "# 111111111 bytes for one file\n"
            "# 222222222 bytes for a different file\n"
            + "a" * 64 + "  models/a.gguf\n")
        self.assertEqual(t[("a" * 64)][0], 222222222,
                         "adjacency should take the immediately preceding line")
        self.assertEqual(len(t), 1)

    def test_a_size_adjacent_to_its_hash_is_attached(self):
        t = self._table("# 333333333 bytes (0.31 GiB)\n" + "b" * 64 + "  models/b.gguf\n")
        self.assertEqual(t, {"b" * 64: (333333333, "models/b.gguf")})

    def test_the_real_sums_file_gives_each_hash_its_own_size(self):
        """Every size in the committed file belongs to the hash it sits above."""
        table = BMS.sizes_by_hash()
        if not table:
            self.skipTest("no sizes recorded yet")
        for h, (size, path) in table.items():
            stem = Path(path).name
            self.assertGreater(size, 10**8, f"{stem}: implausible size {size}")
            # the ladders' own tables are the cross-check available without the files
            if "UD-Q4_K_XL" in stem:
                self.assertEqual(size, 17559178144, "UD-Q4_K_XL took another rung's size")
            if "UD-Q5_K_XL" in stem:
                self.assertEqual(size, 20876938144, "UD-Q5_K_XL took another rung's size")

    def test_one_hash_with_two_sizes_is_refused_rather_than_resolved(self):
        """The same weights cannot have two sizes; keeping the last one writes an unknown number.

        The parser built a dict keyed on hash, so a second entry overwrote the first in silence.
        Whichever came last would then be written into a measurement file as the size of the
        weights that produced it -- and both cannot be right.
        """
        text = ("# 111111111 bytes\n" + "a" * 64 + "  models/x.gguf\n"
                "# 222222222 bytes\n" + "a" * 64 + "  models/y.gguf\n")
        with self.assertRaises(ValueError) as cm:
            self._table(text)
        self.assertIn("two different sizes", str(cm.exception))

    def test_the_same_hash_twice_with_the_same_size_is_fine(self):
        """A repeated line is not a contradiction, only a duplicate."""
        text = ("# 111111111 bytes\n" + "b" * 64 + "  models/x.gguf\n"
                "# 111111111 bytes\n" + "b" * 64 + "  models/x.gguf\n")
        t = self._table(text)
        self.assertEqual(t[("b" * 64)][0], 111111111)

    def test_a_result_whose_hash_is_not_in_the_table_is_refused(self):
        src = inspect.getsource(BMS.main)
        self.assertIn("refusing to invent one", src)
        self.assertIn("already has model_size_bytes", src,
                      "the tool must not overwrite a size recorded from the file itself")


class TestPhaseVFlagsExistInTheInstalledVllm(unittest.TestCase):
    """A matrix written before its engine was installed is a list of guesses.

    phase_v.py was authored months before vLLM went on this box. One of its six flags,
    `--disable-log-requests`, does not exist in vLLM 0.27.1 -- it was renamed to the
    `--enable-log-requests` / `--no-enable-log-requests` pair. An unknown flag stops the server
    during argument parsing, so the entire phase would have failed at startup with an argparse
    message, after 18 GiB of weights and a CUDA-graph compile, and the failure would have said
    nothing about speculation.

    Skips when vLLM is not installed, so a checkout without it still runs green.
    """

    ROOT = Path(__file__).parent.parent

    def test_every_common_arg_is_a_flag_this_vllm_accepts(self):
        import subprocess
        vllm = self.ROOT / ".venv-vllm/bin/vllm"
        if not vllm.exists():
            self.skipTest("vLLM not installed in .venv-vllm")
        sys.path.insert(0, str(self.ROOT / "harness"))
        from matrices import phase_v
        try:
            # `--help` alone prints only config-group names; the flags are behind --help=all.
            out = subprocess.run([str(vllm), "serve", "--help=all"],
                                 capture_output=True, text=True, timeout=600).stdout
        except Exception as e:
            self.skipTest(f"could not query vllm: {e.__class__.__name__}")
        self.assertGreater(len(out), 10000, "vllm --help=all returned too little to check against")
        missing = [a for a in phase_v.COMMON_ARGS if a.startswith("--") and a not in out]
        self.assertFalse(
            missing,
            f"phase_v.COMMON_ARGS names flags this vLLM does not have: {missing}. An unknown "
            f"flag stops the server during argument parsing, so the phase fails at startup and "
            f"the message is about argparse rather than about the measurement.")


class TestAuditCannotBeSilencedByAnEmptyArmList(unittest.TestCase):
    """A result that declares no arms was passing every shape check.

    audit_results compares the (arm, pass) grid against `result["arms"]`, which bench.py writes
    from the matrix -- intent, not outcome. With that list empty the comparison has nothing to
    compare and the file was reported ok with a note. Any permutation of records would have
    passed: one arm repeated, half the passes missing, anything. completeness() has the same
    blind spot, since it falls back to counting distinct arms in the records, which is the file
    certifying itself. This is the hole the rung drivers' gates were rewritten to close, left
    open in the tool that audits their output.
    """

    ROOT = Path(__file__).parent.parent
    FIXTURE = "results/phase_q_UD-Q4_K_XL.json"

    def _audit_with(self, mutate):
        import tempfile, os
        p = self.ROOT / self.FIXTURE
        if not p.exists():
            self.skipTest(f"{self.FIXTURE} not present")
        d = json.loads(p.read_text(encoding="utf-8"))
        mutate(d)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as th:
            json.dump(d, th)
            tmp = th.name
        self.addCleanup(lambda: os.unlink(tmp))
        return AR.audit(Path(tmp))

    def test_records_with_no_declared_arms_is_a_fail(self):
        r = self._audit_with(lambda d: d.update(arms={}))
        self.assertTrue(r["fails"],
                        "a result with records but no declared arms passed the audit; the shape "
                        "check has nothing to compare against and any permutation would pass")
        self.assertIn("declares no arms", " ".join(r["fails"]))

    def test_the_unmodified_fixture_still_passes(self):
        """The guard must not fire on a healthy file."""
        r = self._audit_with(lambda d: None)
        self.assertFalse(r["fails"], f"a clean result was failed: {r['fails']}")

    def test_a_repeated_arm_reaching_the_right_total_is_still_caught(self):
        """The shape check's original purpose, re-asserted against this fixture."""
        def one_arm(d):
            first = d["records"][0]["arm"]
            keep = [r for r in d["records"] if r["arm"] == first and r["pass"] == 1]
            d["records"] = [dict(r, **{"pass": p}) for p in range(1, 13) for r in keep]
        r = self._audit_with(one_arm)
        self.assertTrue(r["fails"], "one arm repeated to the right record count passed the audit")


class TestBothLadderToolsReportDrift(unittest.TestCase):
    """cross_rung refused to score two rungs without a drift estimate; ladder_trend had none.

    The paired bootstrap covers prompt sampling only, and pairing makes it very tight -- on
    phase_qsmall the slope's half-width lands an order of magnitude below any single rung's
    interval, which is the cancellation working as designed. But each rung is one session and the
    rungs ran hours apart, so that interval is not the binding uncertainty on a cross-rung claim.
    cross_rung says this and prints the per-pass spread; ladder_trend printed the interval alone,
    and Correction 22 scored H11 from it.
    """

    ROOT = Path(__file__).parent.parent

    def test_ladder_trend_computes_a_per_pass_spread(self):
        self.assertTrue(hasattr(LT, "per_pass_c"),
                        "ladder_trend has no per-pass refit, so a cross-rung slope is reported "
                        "with prompt-sampling uncertainty as if it were the only kind")
        src = inspect.getsource(LT.report)
        self.assertIn("DRIFT YARDSTICK", src)
        self.assertIn("necessary and not sufficient", src,
                      "the yardstick must say that clearing it is not sufficient; it bounds "
                      "within-run drift and the rungs are separated by more than that")

    def test_the_two_tools_agree_on_what_a_per_pass_refit_is(self):
        """Same estimand in both, or the two reports disagree about the same ladder."""
        a = inspect.getsource(LT.per_pass_c)
        b = inspect.getsource(CR.per_pass_c)
        for token in ("_fit_prompts", "_fit_on", 'r["pass"]'):
            self.assertIn(token, a, f"ladder_trend.per_pass_c does not use {token}")
            self.assertIn(token, b, f"cross_rung.per_pass_c does not use {token}")

    def test_per_pass_c_returns_one_fit_per_pass(self):
        f = self.ROOT / "results/phase_qsmall_BF16.json"
        if not f.exists():
            self.skipTest("phase_qsmall_BF16 not present")
        v = LT.load_rung(str(f))
        per = LT.per_pass_c(v, v["on_path"])
        self.assertEqual(sorted(per), sorted({r["pass"] for r in v["rows"]}))
        self.assertTrue(all(x > 0 for x in per.values()), f"non-positive c per pass: {per}")


class TestVllmMetricsReaderSurvivesLabels(unittest.TestCase):
    """Every vLLM metric is labelled, and two readers here looked one up as if it were not.

    `vllm/v1/metrics/loggers.py:468` sets `labelnames = ["model_name", "engine"]` on every
    counter and histogram, and `vllm/v1/spec_decode/metrics.py:253` adds `position` to the
    per-draft-position one. So `/metrics` never publishes a line named
    `vllm:request_decode_time_seconds_sum`; it publishes
    `vllm:request_decode_time_seconds_sum{engine="0",model_name="..."}`.

    `decode_rate` looked up the bare name and got 0.0 at both ends, which made every field zero
    and left `decode_tok_s` unset -- the reading "this request generated nothing" rather than
    "this reader cannot find the counter". `spec_delta` scanned for substrings, so "accepted"
    and "token" also matched `spec_decode_num_accepted_tokens_per_pos`, and which one it
    returned depended on the order the server emitted them in.

    The body below was generated by prometheus_client 0.24 -- the version in `.venv-vllm` -- from
    counters and histograms declared with vLLM 0.27.1's own names and labels, then pinned here so
    this test does not need that venv. The per-position series is placed FIRST and given absurd
    values, which is the ordering the substring scan could not survive.
    """

    BODY = """\
# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position.
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4",position="0"} 9999.0
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4",position="0"} 1.7877583851787028e+09
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4",position="1"} 8888.0
vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4",position="1"} 1.7877583851787028e+09
vllm:spec_decode_num_drafts_total{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 100.0
vllm:spec_decode_num_drafts_created{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 1.7877583851787028e+09
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 300.0
vllm:spec_decode_num_draft_tokens_created{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 1.7877583851787028e+09
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 210.0
vllm:spec_decode_num_accepted_tokens_created{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 1.7877583851787028e+09
vllm:generation_tokens_total{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 160.0
vllm:generation_tokens_created{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 1.7877583851787028e+09
vllm:request_decode_time_seconds_bucket{engine="0",le="1.0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 0.0
vllm:request_decode_time_seconds_bucket{engine="0",le="+Inf",model_name="RedHatAI/Qwen3.8-27B-INT4"} 1.0
vllm:request_decode_time_seconds_count{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 1.0
vllm:request_decode_time_seconds_sum{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 4.0
vllm:request_prefill_time_seconds_count{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 1.0
vllm:request_prefill_time_seconds_sum{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 1.0
vllm:request_queue_time_seconds_count{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 1.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="RedHatAI/Qwen3.8-27B-INT4"} 0.25
"""

    def _parsed(self):
        import vllm_server as V
        spec = V._parse_metrics(self.BODY, lambda b: bool(V._SPEC_PATTERN.match(b)))
        want = (V.DECODE_TIME, V.PREFILL_TIME, V.QUEUE_TIME, V.GEN_TOKENS)
        tim = V._parse_metrics(self.BODY, lambda b: b.startswith(want))
        return V, spec, tim

    def test_decode_rate_finds_a_labelled_counter(self):
        V, _, tim = self._parsed()
        rate = V.decode_rate({k: 0.0 for k in tim}, tim)
        # 160 tokens over 4.0 s of decode, with prefill and queueing excluded
        self.assertAlmostEqual(rate["decode_tok_s"], 40.0)
        self.assertAlmostEqual(rate["decode_s"], 4.0)
        self.assertAlmostEqual(rate["prefill_s"], 1.0)
        self.assertAlmostEqual(rate["queue_s"], 0.25)
        self.assertAlmostEqual(rate["prefill_share_of_inference"], 0.2)

    def test_a_bare_name_lookup_would_have_returned_zero(self):
        """The defect this class exists for, stated as an assertion rather than as prose."""
        V, _, tim = self._parsed()
        self.assertEqual(tim.get(V.DECODE_TIME + "_sum", 0.0), 0.0)
        self.assertEqual(tim.get(V.GEN_TOKENS, 0.0), 0.0)
        self.assertTrue(any(k.startswith(V.DECODE_TIME + "_sum{") for k in tim))

    def test_acceptance_comes_from_the_total_not_a_draft_position(self):
        V, spec, _ = self._parsed()
        d = V.spec_delta({k: 0.0 for k in spec}, spec)
        self.assertEqual(d["drafted"], 300.0)
        self.assertEqual(d["accepted"], 210.0)
        self.assertEqual(d["drafts"], 100.0)
        self.assertAlmostEqual(d["accept_rate"], 0.7)
        self.assertAlmostEqual(d["mean_len"], 3.1)

    def test_created_timestamps_are_not_summed_into_a_token_count(self):
        """A _created series is a Unix time. Added to a token count it still looks like one."""
        V, spec, _ = self._parsed()
        self.assertEqual(V.series_sum(spec, "vllm:spec_decode_num_draft_tokens"), 300.0)
        self.assertTrue(any("_created" in k for k in spec))

    def test_a_build_without_the_counters_reads_as_absent_not_as_zero(self):
        V, _, _ = self._parsed()
        d = V.spec_delta({}, {})
        self.assertIsNone(d["drafted"])
        with self.assertRaises(V.VllmError):
            V.assert_speculation_observed(d, "mtp-k1")


class TestAResultCanBeTiedBackToWhatProducedIt(unittest.TestCase):
    """Two ways a run could not be traced afterwards, both of them silent.

    Server logs: every phase writes into results/server_logs/ under `pass{NN}_{arm}.log`, with
    no phase in the name. Forty-one filenames in this repository were written by more than one
    phase, one of them by seven, and no result references its own log, so the overwrite is
    invisible -- a log with the right name sits beside the right result and belongs to another
    run.

    Matrix knobs: phase_q, phase_qsmall, phase_l and phase_warp read QWEN_* variables at import
    time, so the same matrix file produces different arm sets on different runs. Nothing recorded
    which value was in effect. The parameter could be read back out of the arm names -- until two
    configurations produce the same names, which is what phase_qsmall's rungs do apart from a
    suffix.
    """

    def test_two_phases_sharing_an_arm_name_write_to_different_logs(self):
        import bench
        log_dir = Path("/nonexistent/server_logs")
        a = bench.server_log_path(log_dir, Path("results/phase_a.json"), "pass01_baseline@master")
        b = bench.server_log_path(log_dir, Path("results/phase_b.json"), "pass01_baseline@master")
        self.assertNotEqual(a, b)
        self.assertIn("phase_a", a.name)
        self.assertIn("phase_b", b.name)

    def test_provenance_records_the_matrix_file_and_the_knobs_in_effect(self):
        import bench, hashlib, types, os
        mfile = Path(bench.__file__).parent / "matrices" / "phase_q.py"
        mod = types.SimpleNamespace(__file__=str(mfile))
        saved = {k: v for k, v in os.environ.items() if k.startswith("QWEN_")}
        try:
            for k in saved:
                del os.environ[k]
            os.environ["QWEN_Q_TARGET"] = "UD-Q5_K_XL"
            os.environ["NOT_A_MATRIX_KNOB"] = "ignored"
            prov = bench.matrix_provenance_snapshot(mod, "phase_q", ["--matrix", "phase_q"])
        finally:
            os.environ.pop("QWEN_Q_TARGET", None)
            os.environ.pop("NOT_A_MATRIX_KNOB", None)
            os.environ.update(saved)
        self.assertEqual(prov["module"], "phase_q")
        self.assertEqual(prov["file"], "phase_q.py")
        self.assertEqual(prov["file_sha256"],
                         hashlib.sha256(mfile.read_bytes()).hexdigest())
        self.assertEqual(prov["knobs"], {"QWEN_Q_TARGET": "UD-Q5_K_XL"})
        self.assertEqual(prov["argv"], ["--matrix", "phase_q"])

    def test_the_hash_changes_when_the_matrix_changes(self):
        """A knob value alone does not identify a run: which knobs a matrix reads is a property
        of the file, and the file is editable between runs."""
        import bench, types, tempfile
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.py"
            f.write_text("ARMS = []\n")
            first = bench.matrix_provenance_snapshot(types.SimpleNamespace(__file__=str(f)), "m", [])
            f.write_text("ARMS = []  # one comment later\n")
            second = bench.matrix_provenance_snapshot(types.SimpleNamespace(__file__=str(f)), "m", [])
        self.assertNotEqual(first["file_sha256"], second["file_sha256"])


class TestPhaseVDriverRefusesBeforeItLoadsAnything(unittest.TestCase):
    """The gates phase_v declared, now that something reads them.

    `REQUIRES_VRAM_GB` and the DFlash2 arms' `requires_vram_gb` sat in matrices/phase_v.py with
    no consumer: the first is read by bench.py, which that matrix's own docstring says does not
    run it, and the second was read by nothing at all. A declaration no code enforces is a
    comment, and the comment in question is the one that keeps a 3.58 GiB speculator from being
    loaded beside an 18.1 GiB target on a 24 GiB card.
    """

    class _Card:
        def __init__(self, gb):
            self.index, self.name, self._gb = 0, "RTX 3090", gb

        @property
        def vram_gb(self):
            return self._gb

    @staticmethod
    def _phase_v():
        import importlib
        import os
        import sys
        mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrices")
        if mdir not in sys.path:
            sys.path.insert(0, mdir)
        return importlib.import_module("phase_v")

    def test_a_24_gib_card_takes_the_matrix_and_refuses_the_a6000_arms(self):
        import vllm_bench as VB
        phase_v = self._phase_v()
        card = self._Card(23.56)
        VB.assert_arms_fit(card, list(phase_v.ARMS), phase_v.REQUIRES_VRAM_GB, "phase_v")
        with self.assertRaises(RuntimeError) as cm:
            VB.assert_arms_fit(card, list(phase_v.ARMS) + list(phase_v.A6000_ONLY_ARMS),
                               phase_v.REQUIRES_VRAM_GB, "phase_v")
        self.assertIn("dflash2-k4", str(cm.exception))

    def test_a_48_gib_card_takes_them_all(self):
        import vllm_bench as VB
        phase_v = self._phase_v()
        VB.assert_arms_fit(self._Card(47.5),
                           list(phase_v.ARMS) + list(phase_v.A6000_ONLY_ARMS),
                           phase_v.REQUIRES_VRAM_GB, "phase_v")

    def test_arm_order_rotates_without_dropping_an_arm(self):
        import vllm_bench as VB
        arms = [{"name": n} for n in ("a", "b", "c")]
        seen = [[a["name"] for a in VB.arm_order_for_pass(arms, p)] for p in (1, 2, 3, 4)]
        self.assertEqual(seen[0], ["a", "b", "c"])
        self.assertEqual(seen[1], ["b", "c", "a"])
        self.assertEqual(seen[2], ["c", "a", "b"])
        self.assertEqual(seen[3], seen[0])
        for order in seen:
            self.assertEqual(sorted(order), ["a", "b", "c"])
        # Every arm leads exactly once over one full cycle, which is the point: a fixed order
        # gives the coldest card to the same arm every pass.
        self.assertEqual(sorted(o[0] for o in seen[:3]), ["a", "b", "c"])

    def test_vllms_own_processes_are_not_counted_as_competition(self):
        """setproctitle reaches /proc/pid/comm truncated to 15 characters.

        "VLLM::EngineCore" arrives in `ps -eo comm` as "VLLM::EngineCor". Measured on
        2026-08-26 against the setproctitle in .venv-vllm. Without a match here every arm-pass
        would be recorded as contended by the very server it was measuring.
        """
        import vllm_bench as VB
        for comm in ("VLLM::EngineCor", "VLLM::APIServer", "VLLM::Worker_TP0"):
            self.assertTrue(any(o in comm for o in VB.OWN_PROCESS_NAMES), comm)
        for comm in ("cc1plus", "sha256sum", "gh", "rsync"):
            self.assertFalse(any(o in comm for o in VB.OWN_PROCESS_NAMES), comm)


class TestContentionIsAttributedByDescentNotByName(unittest.TestCase):
    """Matching processes by name was wrong in both directions, and both directions bit.

    False positive: the run polls nvidia-smi continuously to integrate power into joules, and
    `nvidia-smi` was not in own_names, so a poll alive when `ps` ran was recorded as outside
    competition. Phase B pass 2 carries exactly that incident, "50% of CPU is not this run:
    nvidia-smi 50%", and it was this run's own sampler.

    False negative, which is worse: `python3` WAS in own_names, so every python on the host was
    invisible -- including the analysis scripts this study runs while a phase is measuring. A
    clean incident log was only ever evidence about processes that are not python.
    """

    def test_a_descendant_is_not_competition_however_deep(self):
        import telemetry
        # 100 -> 200 -> 300 -> 400, and 500 unrelated
        ppid = {100: 1, 200: 100, 300: 200, 400: 300, 500: 1}
        mine = telemetry._descendants_of(100, ppid)
        self.assertEqual(mine, {100, 200, 300, 400})
        self.assertNotIn(500, mine)

    def test_a_cycle_in_the_table_does_not_hang(self):
        """ps output is a snapshot; pids are reused and a self-parent can appear in one."""
        import telemetry
        self.assertEqual(telemetry._descendants_of(7, {7: 7}), {7})
        self.assertEqual(telemetry._descendants_of(1, {1: 2, 2: 1}), {1, 2})

    def test_someone_elses_python_is_no_longer_exempt(self):
        import telemetry
        import inspect
        names = inspect.signature(telemetry.host_load).parameters["own_names"].default
        for n in ("python3", "python", "bench.py"):
            self.assertNotIn(n, names,
                             f"{n!r} back in own_names would make every other {n} on the host "
                             f"invisible, which is the hole descent was added to close")
        # llama-server stays: it is started with start_new_session and can outlive its parent,
        # and a reparented server would otherwise be charged to the run as competition.
        self.assertIn("llama-server", names)

    def test_the_vllm_driver_keeps_only_the_reparenting_fallback(self):
        import vllm_bench as VB
        self.assertEqual(VB.OWN_PROCESS_NAMES, ("VLLM::",))
        self.assertTrue(any(o in "VLLM::EngineCor" for o in VB.OWN_PROCESS_NAMES))


class TestMechanismBRecoversAKnownTruth(unittest.TestCase):
    """An analysis tool that has never been shown a known answer is an assertion, not a measurement.

    Two synthetic worlds are built from Phase B's own arm design -- n_max in {3,7} crossed with
    p_min in {0,.5,.75}, which is what makes drafted and rejected counts move by different
    amounts. In one world the cost is exactly per drafted token; in the other, exactly per
    rejected token. The tool has to pick the right one in each.
    """

    ARMS = [(3, 2.985, 1.320), (3, 2.067, 0.601), (3, 1.343, 0.174),
            (7, 6.930, 4.689), (7, 3.567, 1.581), (7, 1.911, 0.422)]

    def _result(self, ms_per_drafted, ms_per_rejected, ms_per_step=0.0):
        import random
        rng = random.Random(11)
        classes = ("code", "prose", "reason", "chat", "zh")
        prompts = [(f"p{i:02d}", classes[i % len(classes)]) for i in range(25)]
        arms_meta = {"baseline": {"expects_drafter": False, "extra_args": []}}
        recs = []
        tau0 = 20.0
        for tag, cls in prompts:
            f0 = 200
            recs.append({"arm": "baseline", "pass": 1, "prompt": tag, "class": cls,
                         "predicted_n": f0 + 1,
                         "timings": {"t_predicted_ms": f0 * tau0, "t_draft_n": 0,
                                     "t_draft_n_accepted": 0}})
        for i, (nmax, d, r) in enumerate(self.ARMS):
            name = f"mtp-n{nmax}-{i}"
            arms_meta[name] = {"expects_drafter": True,
                               "extra_args": ["--spec-draft-n-max", str(nmax)]}
            for tag, cls in prompts:
                f = 200
                drafted = int(round(d * f))
                rejected = int(round(r * f))
                accepted = drafted - rejected
                ms = (f * tau0 + ms_per_step * f
                      + ms_per_drafted * drafted + ms_per_rejected * rejected)
                ms *= 1.0 + rng.gauss(0, 0.01)      # 1 % per-request noise
                recs.append({"arm": name, "pass": 1, "prompt": tag, "class": cls,
                             "predicted_n": accepted + f + 1,
                             "timings": {"t_predicted_ms": ms, "t_draft_n": drafted,
                                         "t_draft_n_accepted": accepted}})
        return {"arms": arms_meta, "records": recs}

    def _fits(self, result):
        import mechanism_b as M
        rs = M.rows(result)
        self.assertTrue(rs)
        bd = M._fit1([(x["excess_ms"], x["drafted"]) for x in rs])
        br = M._fit1([(x["excess_ms"], x["rejected"]) for x in rs])
        rss_d = sum((x["excess_ms"] - bd * x["drafted"]) ** 2 for x in rs)
        rss_r = sum((x["excess_ms"] - br * x["rejected"]) ** 2 for x in rs)
        return bd, br, rss_d, rss_r

    def test_a_world_where_cost_is_per_drafted_token_is_read_that_way(self):
        bd, br, rss_d, rss_r = self._fits(self._result(ms_per_drafted=7.0, ms_per_rejected=0.0))
        self.assertLess(rss_d, rss_r, "the drafted model must fit a drafted world better")
        self.assertAlmostEqual(bd, 7.0, delta=0.35)

    def test_a_world_where_cost_is_per_rejected_token_is_read_that_way(self):
        bd, br, rss_d, rss_r = self._fits(self._result(ms_per_drafted=0.0, ms_per_rejected=7.0))
        self.assertLess(rss_r, rss_d, "the rejected model must fit a rejected world better")
        self.assertAlmostEqual(br, 7.0, delta=0.35)

    def test_a_per_step_cost_is_not_charged_to_either_token_count(self):
        """The drafter runs one forward per step whatever the gate does.

        Without a step term that cost has to be absorbed by whichever token count happens to
        correlate with step count, which would make a purely per-step world look like evidence
        for one of the two hypotheses.
        """
        import mechanism_b as M
        rs = M.rows(self._result(ms_per_drafted=0.0, ms_per_rejected=0.0, ms_per_step=5.0))
        fit = M._fit2([(x["excess_ms"], float(x["forwards"]), float(x["drafted"])) for x in rs])
        self.assertIsNotNone(fit)
        per_step, per_token = fit
        self.assertAlmostEqual(per_step, 5.0, delta=0.4)
        self.assertAlmostEqual(per_token, 0.0, delta=0.25)

    def test_a_baseline_only_result_yields_no_rows_rather_than_a_fit(self):
        import mechanism_b as M
        r = self._result(7.0, 0.0)
        r["records"] = [x for x in r["records"] if x["arm"] == "baseline"]
        self.assertEqual(M.rows(r), [])


class TestEveryDocumentLinkPointsAtSomethingAClonWouldHave(unittest.TestCase):
    """A link that resolves on this disk and not in the repository is a broken link.

    Twice now. `analysis/plot_phase_m.png` was untracked on purpose while the run it drew was
    incomplete, and the README kept the <img> tag. `models/SHA256SUMS.phase_a` was written,
    linked from the Reproduce section, and silently refused by `models/*` in .gitignore. Both
    render correctly for the author and 404 for everyone else, which is the failure mode a check
    against the filesystem cannot see.

    Checked against `git ls-files`, not against `Path.exists()`.
    """

    ROOT = Path(__file__).parent.parent
    DOCS = ("README.md", "TODO.md", "PREREGISTRATION.md")

    def _tracked(self):
        import subprocess
        out = subprocess.run(["git", "-C", str(self.ROOT), "ls-files"],
                             capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            self.skipTest("not a git checkout")
        return set(out.stdout.split("\n")) - {""}

    def test_no_document_links_at_an_untracked_path(self):
        import re
        tracked = self._tracked()
        dirs = {d for t in tracked for d in _parents_of(t)}
        broken = []
        for name in self.DOCS:
            f = self.ROOT / name
            if not f.exists():
                continue
            text = f.read_text(encoding="utf-8")
            targets = [t for _, t in re.findall(r"\[([^\]]+)\]\(([^)#][^)]*)\)", text)]
            targets += re.findall(r'src="([^"]+)"', text)
            targets += re.findall(r'srcset="([^"]+)"', text)
            for t in targets:
                if t.startswith(("http://", "https://", "mailto:")):
                    continue
                path = t.split("#")[0].rstrip("/")
                if not path or path in tracked or path in dirs:
                    continue
                broken.append(f"{name} -> {path}")
        self.assertFalse(broken,
                         "these paths are linked from a committed document and are not in the "
                         "repository, so they resolve for the author and 404 for a reader: "
                         + ", ".join(broken))


def _parents_of(path):
    parts = path.split("/")
    return {"/".join(parts[:i]) for i in range(1, len(parts))}


class TestReadmeSaysWhatTheArtifactsSay(unittest.TestCase):
    """The README drifts because the same conclusion is copied into six places by hand.

    Correction 25 established that phase_c's ngram-mod emitting no drafts is the method working
    as designed; the README kept "its flag was accepted and did nothing" for a further day.
    Phase M's anchor artifact has said ANCHOR DOES NOT HOLD since the run finished; the opening
    paragraph kept "the sign belongs to the drafting method, not the architecture".

    These are not style checks. Each one binds a sentence in the README to the artifact that
    decides whether that sentence is true, so a retraction in one place cannot leave the other
    standing. They fail closed: a missing artifact is a skip, a present artifact that contradicts
    the README is a failure.
    """

    ROOT = Path(__file__).parent.parent

    def setUp(self):
        f = self.ROOT / "README.md"
        if not f.exists():
            self.skipTest("no README.md")
        self.readme = f.read_text(encoding="utf-8")

    def _artifact(self, name):
        f = self.ROOT / "analysis" / name
        if not f.exists():
            self.skipTest(f"no analysis/{name}")
        return f.read_text(encoding="utf-8")

    def _result(self, name):
        f = self.ROOT / "results" / name
        if not f.exists():
            self.skipTest(f"no results/{name}")
        return json.loads(f.read_text(encoding="utf-8"))

    def test_record_counts_quoted_in_the_readme_match_the_result_files(self):
        import re
        for phase, pattern in (("phase_a.json", r"Phase A \((\d+) request records"),
                               ("phase_m.json", r"Phase M, (\d+) records")):
            m = re.search(pattern, self.readme)
            if not m:
                continue
            self.assertEqual(int(m.group(1)), len(self._result(phase)["records"]),
                             f"the README's count for {phase} is not the file's")

    def test_no_architecture_claim_while_the_phase_m_anchor_fails(self):
        anchor = self._artifact("phase_m_anchor.txt")
        if "ANCHOR DOES NOT HOLD" not in anchor:
            self.skipTest("the anchor holds; this guard is for when it does not")
        for claim in ("not the architecture",
                      "sign belongs to the drafting method",
                      "rules out a large architecture effect"):
            self.assertNotIn(claim, self.readme,
                             f"the README asserts {claim!r} while phase_m_anchor.txt says the "
                             f"anchor does not hold and that nothing in Phase M may then be read "
                             f"as a statement about the predecessor or the architecture")

    def test_no_phase_m_cost_numbers_while_its_mean_len_check_fails(self):
        cost = self._artifact("phase_m_cost.txt")
        if "The derivation is wrong" not in cost:
            self.skipTest("Phase M's mean_len check passes; this guard is for when it does not")
        row = [l for l in self.readme.splitlines() if l.startswith("| **M** |")]
        self.assertTrue(row, "no Phase M row in the later-phases table to check")
        self.assertTrue(any(w in row[0] for w in ("withheld", "withdrawn")),
                        "Phase M's mean_len derivation fails its own integrity check, so the "
                        "README's Phase M row has to say its cost interpretation is withheld")

    def test_a_zero_draft_arm_is_not_described_as_an_ignored_flag(self):
        res = self._result("phase_c.json")
        drafted = {}
        for r in res["records"]:
            n = (r.get("timings") or {}).get("t_draft_n") or 0
            drafted[r["arm"]] = drafted.get(r["arm"], 0) + n
        if not any(v == 0 for a, v in drafted.items() if "ngram" in a):
            self.skipTest("no zero-draft n-gram arm in phase_c")
        for claim in ("flag was accepted and did nothing", "flag was ignored",
                      "accepted and silently ignored"):
            self.assertNotIn(claim, self.readme,
                             f"the README says {claim!r}; Correction 25 established that zero "
                             f"drafts is ngram-mod's designed behaviour at its default n_min=48")

    def test_no_committed_document_still_carries_a_withdrawn_claim(self):
        """The same sentence lives in six files, and a retraction lands in one of them.

        PREREGISTRATION.md is exempt: a Correction has to quote the wording it withdraws. So is
        any line that marks itself as a withdrawal, for the same reason.
        """
        import subprocess
        out = subprocess.run(["git", "-C", str(self.ROOT), "ls-files", "*.md"],
                             capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            self.skipTest("not a git checkout")
        claims = ("not the architecture", "sign belongs to the drafting method",
                  "rules out a large architecture effect", "flag was accepted and did nothing",
                  "flag was ignored", "no quantization anywhere", "the cost is linear",
                  "costs a fixed c", "rules out warp count")
        marks = ("withdraw", "Withdraw", "used to read", "an earlier version", "earlier draft",
                 "no longer", "retract")
        found = []
        for name in out.stdout.split("\n"):
            if not name or "PREREGISTRATION" in name or name.startswith(("upstream/", "llamacpp")):
                continue
            f = self.ROOT / name
            if not f.exists():
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if any(m in line for m in marks):
                    continue
                for c in claims:
                    if c in line:
                        found.append(f"{name}:{i} {c!r}")
        self.assertFalse(found,
                         "these lines assert something this study has withdrawn, and do not mark "
                         "themselves as quoting it: " + "; ".join(found))

    def test_the_capped_window_is_not_called_byte_identical(self):
        """Every request stops at the token cap, so a match inside it is right-censored."""
        self.assertNotIn("Byte-identical output against each rung's own baseline", self.readme)
        self.assertNotIn("no quantization anywhere in the target", self.readme)


class TestCompletenessIsNotDerivedFromWhatIsThere(unittest.TestCase):
    """An expected count computed from the records always equals the records.

    `completeness()` multiplied arms by the number of DISTINCT PROMPTS SEEN by passes. A run that
    died inside its first pass after ten prompts has ten distinct prompts and one pass, so
    1 x 10 x 1 is exactly what the file holds and it reported complete -- the check that exists
    to catch a truncated file passing it. bench.py writes `design.n_prompts` from the matrix,
    which is what the run set out to do rather than what it managed.
    """

    def test_a_run_that_died_in_its_first_pass_is_not_complete(self):
        import completeness as C
        recs = [{"arm": "a", "prompt": f"p{i}", "pass": 1} for i in range(10)]
        n, expected, _ = C.completeness(
            {"records": recs, "arms": {"a": {}}, "design": {"passes": 3, "n_prompts": 25}})
        self.assertEqual(n, 10)
        self.assertEqual(expected, 75)
        self.assertLess(n, expected)

    def test_a_file_without_the_declared_count_says_so(self):
        import completeness as C
        recs = [{"arm": "a", "prompt": f"p{i}", "pass": 1} for i in range(10)]
        _, _, note = C.completeness({"records": recs, "arms": {"a": {}}, "design": {"passes": 3}})
        self.assertIn("prompt count not recorded", note)

    def test_a_finished_phase_still_reads_as_finished(self):
        import completeness as C
        f = Path(__file__).parent.parent / "results" / "phase_a.json"
        if not f.exists():
            self.skipTest("no results/phase_a.json")
        n, expected, _ = C.completeness(json.loads(f.read_text(encoding="utf-8")))
        self.assertEqual(n, expected)


class TestEveryTestInThisFileActuallyRuns(unittest.TestCase):
    """A class appended after the `__main__` guard is defined too late to be collected.

    That is how TestAnchorEstimatorMatchesBand first landed: five cases appended to the end of
    the file, `Ran 61 tests` unchanged from before the edit, and a green suite that had executed
    none of them. The guard is the last thing in the file, so "append to the end" is the obvious
    edit and it silently does nothing -- the worst shape a test defect can take, because the
    signal it produces is the same one a passing suite produces.

    Comparing what the source declares against what the loader collected catches it, and catches
    any later variant of the same mistake.
    """

    def test_the_loader_collects_every_test_the_source_declares(self):
        import ast
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        declared = {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not any("TestCase" in ast.dump(b) for b in node.bases):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and item.name.startswith("test"):
                    declared[f"{node.name}.{item.name}"] = item.lineno

        module = sys.modules[type(self).__module__]
        collected = set()
        for suite in unittest.TestLoader().loadTestsFromModule(module):
            for case in suite:
                collected.add(f"{type(case).__name__}.{case._testMethodName}")

        missing = sorted(set(declared) - collected)
        self.assertFalse(
            missing,
            "declared in the source but never collected -- most likely defined after the "
            "__main__ guard at the end of the file, where the class is created only after "
            "unittest.main() has already chosen what to run: "
            + ", ".join(f"{m} (line {declared[m]})" for m in missing))
        # The reverse direction would mean a test exists that the source does not declare at
        # top level, which is not a defect this file can produce, so it is not asserted.


class TheTwoTreesAreStillCheckedAgainstEachOther(unittest.TestCase):
    """The dual-tree fix mapped every baseline to itself, and the cross-tree control vanished.

    At a 400-token cap `baseline@pr27342` carried a divergence against `baseline@master` on 125 of
    125 records, all showing no divergence inside the window: the evidence that the PR branch
    matches master with speculation off. At a 400-token cap none of them reached EOS, so it is
    agreement through the measured span rather than through a whole answer. `divergence_baseline_map` then gave each arm the baseline of its own tree,
    which is right for a treatment arm and makes each baseline its own reference, and
    `_attach_baseline_comparisons` skips every baseline. The 1600-token re-run carries no
    comparison for `baseline@pr27342` at all. The comment introducing the map says the next pair
    of trees need not agree, which is the argument for measuring it rather than for stopping.
    """

    def _run(self, arms_in_order, texts, pass_order=False):
        """Called the way the old signature allowed unless `pass_order`, so a failure here is the
        missing comparison rather than the added parameter."""
        import bench
        names = set(arms_in_order)
        recs = [{"arm": a, "prompt": "p1", "pass": 1, "temperature": 0.0, "text": t}
                for a, t in texts.items()]
        recs.append({"arm": "mtp-n2", "prompt": "p1", "pass": 1, "temperature": 0.0,
                     "text": "a wholly different answer"})
        result = {"records": recs}
        btext = {(a, "p1", 1): t for a, t in texts.items() if a in names}
        bmap = {a: a for a in names}
        bmap["mtp-n2"] = arms_in_order[0]
        extra = [arms_in_order] if pass_order else []
        bench._attach_baseline_comparisons(result, btext, bmap, names, 1, *extra)
        return {r["arm"]: r for r in recs}

    def test_the_second_baseline_is_compared_against_the_first(self):
        by = self._run(["baseline@master", "baseline@pr"],
                       {"baseline@master": "the same answer", "baseline@pr": "the same answer"})
        pr = by["baseline@pr"]
        self.assertIn("tree_divergence", pr,
                      "the branch's own no-speculation arm is the only thing that says the branch "
                      "matches master inside the window; nothing else in the design checks it")
        self.assertEqual(pr["tree_compared_against"], "baseline@master")
        self.assertTrue(pr["tree_divergence"]["identical"])

    def test_a_difference_between_the_trees_is_visible(self):
        by = self._run(["baseline@master", "baseline@pr"],
                       {"baseline@master": "the same answer", "baseline@pr": "the same answEr"})
        self.assertFalse(by["baseline@pr"]["tree_divergence"]["identical"],
                         "a control that cannot come back negative is not a control")

    def test_the_control_never_enters_the_method_effect_field(self):
        by = self._run(["baseline@master", "baseline@pr"],
                       {"baseline@master": "the same answer", "baseline@pr": "the same answEr"})
        self.assertNotIn("divergence", by["baseline@pr"],
                         "a baseline compared against another baseline is a control, and reading "
                         "it as a method effect is how the arm with no fork position was printed "
                         "as a group of a fork-position partition")
        self.assertNotIn("tree_divergence", by["baseline@master"],
                         "the reference is not compared against itself")

    def test_a_single_tree_run_produces_no_cross_comparison(self):
        by = self._run(["baseline@master"], {"baseline@master": "the same answer"})
        self.assertNotIn("tree_divergence", by["baseline@master"])

    def test_the_reference_is_the_arm_order_and_not_the_alphabet(self):
        """Falling back to sorted names picks a reference by spelling, which is only ever right
        by luck. The caller passes the order the matrix declares."""
        by = self._run(["zzz-baseline", "aaa-baseline"],
                       {"zzz-baseline": "one answer", "aaa-baseline": "another answer"},
                       pass_order=True)
        self.assertEqual(by["aaa-baseline"]["tree_compared_against"], "zzz-baseline")
        self.assertNotIn("tree_divergence", by["zzz-baseline"])


class CensoredCellsDoNotPartition(unittest.TestCase):
    """A cell that ran out of budget holds no fork position, so it cannot separate two widths.

    At a 400-token cap widths 3 and 4 share a fork vector; at 1600 one cell of code_sql_report
    comes back censored for width 4 while width 3 forks at char 5423, and comparing those two
    values split the widths and printed "H8 NOT SUPPORTED" with a note that the mechanism offered
    in llama.cpp #25618 needed withdrawing. Nothing about width 4 had changed: its fork was
    simply not reached inside the window. A partition that moves when the cap moves is a
    property of the cap.
    """

    ARMS = {"mtp-n2": {"extra_args": ["--spec-type", "mtp", "--spec-draft-n-max", "2"]},
            "mtp-n3": {"extra_args": ["--spec-type", "mtp", "--spec-draft-n-max", "3"]}}

    def _run(self, third_cell_for_w4):
        import contextlib, io, json, tempfile, os
        import width_groups as W

        def rec(arm, prompt, char):
            capped = char is None
            div = ({"identical": True} if capped else
                   {"identical": False, "first_diff_char": char, "prefix_only": False,
                    "len_ref": char + 500, "len_arm": char + 500})
            return {"arm": arm, "prompt": prompt, "pass": 1, "divergence": div,
                    "hit_cap": capped, "finish_reason": "length" if capped else "stop",
                    "predicted_n": 1600 if capped else 900, "text_len": 3400}

        recs = []
        for prompt, c in (("p1", 100), ("p2", 200)):
            recs.append(rec("mtp-n2", prompt, c))
            recs.append(rec("mtp-n3", prompt, c))
        recs.append(rec("mtp-n2", "p3", 5423))
        recs.append(rec("mtp-n3", "p3", third_cell_for_w4))

        data = {"design": {"max_tokens": 1600, "n_prompts": 3, "passes": 1},
                "arms": self.ARMS, "records": recs}
        fh = tempfile.NamedTemporaryFile("w", prefix="wg_", suffix=".json", delete=False)
        json.dump(data, fh)
        fh.close()
        argv = sys.argv
        buf = io.StringIO()
        try:
            sys.argv = ["width_groups.py", fh.name]
            with contextlib.redirect_stdout(buf):
                W.main()
        finally:
            sys.argv = argv
            os.unlink(fh.name)
        out = buf.getvalue()
        return out[out.index("--- observed groups"):].split("--- H8")[0]

    def test_a_censored_cell_leaves_the_widths_grouped(self):
        groups = self._run(None)
        self.assertIn("{3, 4}", groups,
                      "width 4 did not diverge inside the window on p3; that is an unobserved "
                      "fork, not a fork elsewhere, and it must not split the group:\n" + groups)
        self.assertIn("grouped on 2 of 3 prompts", groups,
                      "the reader has to be told how many prompts actually determined the "
                      "grouping, or 2-of-3 reads as 3-of-3:\n" + groups)

    def test_a_genuinely_different_fork_still_splits_them(self):
        groups = self._run(4111)
        self.assertNotIn("{3, 4}", groups,
                         "both widths diverged on p3 and at different characters, which is real "
                         "evidence of different fork positions:\n" + groups)


class TheExpectedRecordCountIsComputedOnce(unittest.TestCase):
    """`completeness` and `audit_results` each subtracted the may_fail arm-passes.

    Phase V has three arms over three passes against 25 prompts, so 225 records were designed and
    six arm-passes of two `may_fail` vLLM arms recorded a startup failure instead of running: 150
    records that were never going to exist, and 75 that should. `completeness` subtracts them and
    `audit_results` subtracted them again, giving an expectation of **-75** and reporting the 75
    real records as "more than the design". The second subtraction also counted every failed
    arm-pass rather than only those of a may_fail arm, so the two were not even the same quantity.
    """

    def _file(self):
        import tempfile, os
        arms = {"ok": {}, "mtp-k1": {"may_fail": True}, "mtp-k2": {"may_fail": True}}
        failed = {f"pass{q}_{a}": "vllm exited with 1 during startup"
                  for q in (1, 2, 3) for a in ("mtp-k1", "mtp-k2")}
        recs = [{"arm": "ok", "prompt": f"p{i}", "pass": q, "decode_tok_s": 40.0}
                for q in (1, 2, 3) for i in range(25)]
        d = {"design": {"n_prompts": 25, "passes": 3}, "arms": arms,
             "arm_pass_failed": failed, "records": recs}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as th:
            json.dump(d, th)
            tmp = th.name
        self.addCleanup(lambda: os.unlink(tmp))
        return d, tmp

    def test_the_two_readers_cannot_disagree(self):
        import completeness as CP
        d, tmp = self._file()
        got, expected, _ = CP.completeness(d)
        self.assertEqual((got, expected), (75, 75),
                         "225 designed less the six recorded may_fail arm-passes is 75")
        self.assertEqual(AR.audit(Path(tmp))["expected"], expected,
                         "the audit must not subtract the may_fail arm-passes a second time")

    def test_a_negative_expectation_is_never_produced(self):
        d, tmp = self._file()
        self.assertGreater(AR.audit(Path(tmp))["expected"], 0,
                           "a run cannot be designed to produce a negative number of records")

    def test_the_count_is_not_reported_as_over_the_design(self):
        d, tmp = self._file()
        over = [f for f in AR.audit(Path(tmp))["fails"] if "more than the design" in f]
        self.assertFalse(over, "75 of an expected 75 is not more than the design: " + str(over))


class ArmsWithoutAForkAreNotAForkGroup(unittest.TestCase):
    """`divergence_report` partitioned arms by fork position and let the no-position states in.

    `quality.fork_cell` returns a character index for a fork and a string for each state that has
    none. Feeding those into the position map put every censored arm on a prompt into one bucket,
    which reads as a shared fork position. On Phase A it printed `baseline@pr27342` -- the arm that
    never diverges, 125 of 125 identical -- as a group of a fork-position partition.
    """

    def _partition(self, records, arms):
        import contextlib, io
        import divergence_report as DR
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            DR.group_stability({"records": records, "arms": arms})
        out = buf.getvalue()
        return [l for l in out.splitlines() if "modal partition" in l]

    def _recs(self):
        def rec(arm, prompt, q, char):
            div = ({"identical": True} if char is None else
                   {"identical": False, "first_diff_char": char, "prefix_only": False,
                    "len_ref": char + 500, "len_arm": char + 500})
            return {"arm": arm, "prompt": prompt, "pass": q, "divergence": div,
                    "hit_cap": char is None, "finish_reason": "length" if char is None else "stop",
                    "predicted_n": 400, "text_len": 1400}
        recs = []
        for q in (1, 2):
            for prompt in ("p1", "p2", "p3"):
                recs.append(rec("armA", prompt, q, 100))
                recs.append(rec("armB", prompt, q, 100))
                recs.append(rec("never", prompt, q, None))   # censored on every prompt
        return recs

    def test_an_arm_that_never_diverges_is_not_a_group(self):
        lines = self._partition(self._recs(), {})
        self.assertTrue(lines, "the partition did not print")
        for l in lines:
            self.assertNotIn("never", l,
                             "an arm with no fork position cannot be a group in a partition of "
                             "fork positions: " + l)
            self.assertIn("{armA,armB}", l, l)

    def test_two_censored_arms_are_not_reported_as_agreeing(self):
        recs = self._recs()
        # armB is censored on two of the three prompts, so "both ran out of budget" is the modal
        # shape. A single such prompt loses the tie-break in `most_common` and never reaches the
        # printed line, which is why the first version of this test passed against the defect.
        for r in recs:
            if r["arm"] == "armB" and r["prompt"] in ("p2", "p3"):
                r["divergence"] = {"identical": True}
                r["hit_cap"], r["finish_reason"] = True, "length"
        lines = self._partition(recs, {})
        for l in lines:
            self.assertNotIn("{armB,never}", l,
                             "both ran out of budget on p2; neither has a position, so they did "
                             "not land anywhere together: " + l)


class TheWindowIsTheCapNotTheOutputLength(unittest.TestCase):
    """`predicted_n` is what a record produced; it equals the cap only for records that hit it.

    While every record hit the 400-token cap the two were the same number and reading the window
    off the outputs was accidentally right. At 1600 half the records stop on their own, the
    inferred window fans out into 41 distinct values, `censored_prompts` reports "no single
    window" and returns None, and width_groups divided by it: TypeError, float / NoneType.
    """

    def test_the_design_states_the_cap(self):
        import truncation_audit as TA
        data = {"design": {"max_tokens": 1600},
                "records": [{"predicted_n": 634, "finish_reason": "stop"},
                            {"predicted_n": 1600, "hit_cap": True, "finish_reason": "length"},
                            {"predicted_n": 640, "finish_reason": "stop"}]}
        self.assertEqual(TA.budget(data), 1600)
        self.assertEqual(TA.censored_prompts(data)[1], 1600,
                         "a run whose records stop at different lengths still has one cap")

    def test_an_older_file_falls_back_to_the_records_that_hit_the_cap(self):
        import truncation_audit as TA
        data = {"records": [{"predicted_n": 634, "finish_reason": "stop"},
                            {"predicted_n": 400, "hit_cap": True, "finish_reason": "length"}]}
        self.assertEqual(TA.budget(data), 400,
                         "the records that ran out of budget are the ones whose length is the cap")
        self.assertIsNone(TA.budget({"records": [{"predicted_n": 12, "finish_reason": "stop"}]}),
                          "nothing was censored, so no cap is evidenced")

    def test_reaching_eos_is_read_from_either_field(self):
        import truncation_audit as TA
        self.assertTrue(TA.reached_eos({"finish_reason": "stop"}))
        self.assertFalse(TA.reached_eos({"hit_cap": True}))
        self.assertFalse(TA.reached_eos({"finish_reason": "length"}))


class AProvenanceCheckThatResolvesNothing(unittest.TestCase):
    """Three of `compare_reproduction`'s provenance paths did not exist in any result file.

    `design.kernel_facts.master.commit`, `env.llama_commit` and `env.driver`. `dig` returned None
    for both sides, None equals None, and the table printed three reassuring rows for fields
    nobody was comparing. A provenance check that resolves nothing is worse than none, because it
    looks like one.
    """

    def _files(self, mutate_candidate=None):
        import json, tempfile, os
        base = {
            "arms": {"base@m": {"tree": "master", "extra_args": []},
                     "spec": {"tree": "master", "extra_args": ["--spec-draft-n-max", "2"]}},
            "records": [{"arm": a, "prompt": p, "class": c, "pass": q, "decode_tok_s": v,
                         "hit_cap": True, "finish_reason": "length"}
                        for p, c in (("p1", "code"), ("p2", "prose"))
                        for q in (1, 2)
                        for a, v in (("base@m", 40.0), ("spec", 64.0))],
            "incidents": [],
            "env": {"llama_cpp_revisions": {"master": "abc1234"}, "model_sha256": "aa",
                    "model_size_bytes": 1, "model": "/m.gguf", "gpu": "RTX 3090, 610.43.02",
                    "overclock_state": {"power_limit_w": 420.0}, "kernel": "k",
                    "python": "3.13.5", "host": "h"},
            "design": {"max_tokens": 400, "passes": 2, "n_prompts": 2,
                       "device": {"name": "RTX 3090"}},
            "matrix": {"file_sha256": "m1"},
        }
        out = []
        for i in range(2):
            d = json.loads(json.dumps(base))
            if i == 1 and mutate_candidate:
                mutate_candidate(d)
            fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            json.dump(d, fh)
            fh.close()
            self.addCleanup(lambda n=fh.name: os.unlink(n))
            out.append(fh.name)
        return out

    def _run(self, mutate=None, *flags):
        import subprocess
        a, b = self._files(mutate)
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "compare_reproduction.py"),
                            a, b, *flags], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def test_every_declared_provenance_path_resolves(self):
        import compare_reproduction as CR
        d = json.loads(Path(__file__).parent.parent.joinpath(
            "results/phase_a.json").read_text()) if False else None
        # Against the fixture rather than a committed result, so the test does not depend on a
        # 60 MB file being present, but the point is the same: every path must reach a value.
        a, _ = self._files()
        d = json.loads(Path(a).read_text())
        dead = [label for label, path in CR.PROVENANCE if CR.dig(d, path) is None]
        self.assertFalse(dead, f"these provenance paths resolve to nothing: {dead}")

    def test_two_identical_runs_compare_clean(self):
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        self.assertIn("consistent with a reproduction", out)

    def test_a_provenance_difference_is_a_problem(self):
        rc, out = self._run(lambda d: d["env"].update(model_sha256="bb"))
        self.assertEqual(rc, 1)
        self.assertIn("provenance differs", out)

    def test_an_incident_fails_by_default_and_can_be_allowed(self):
        inc = lambda d: d["incidents"].append({"kind": "host_contended", "arm": "base@m",
                                               "pass": 1, "detail": "python3 99%"})
        self.assertEqual(self._run(inc)[0], 1)
        self.assertEqual(self._run(inc, "--allow-incidents")[0], 0)

    def test_a_different_effect_fails(self):
        def slower(d):
            for r in d["records"]:
                if r["arm"] == "spec":
                    r["decode_tok_s"] = 90.0
        rc, out = self._run(slower)
        self.assertEqual(rc, 1)
        self.assertIn("do not overlap", out)

    def test_overlap_is_not_reported_as_agreement(self):
        _, out = self._run()
        self.assertIn("failure to exclude", out,
                      "Correction 26: two intervals that overlap have failed to exclude each "
                      "other, which is weaker than agreement and must not be printed as it")


class TheRegistryDeclaresAVocabularyAndMustEnforceIt(unittest.TestCase):
    """`inference` is documented as a controlled vocabulary and nothing checked it.

    `INFERENCE.get(value, value)` fell back to printing whatever was there, so a typo would have
    reached the README verbatim with nothing raised.
    """

    def _reg(self, mutate=None):
        import copy, json
        import render_evidence as RE
        reg = json.loads(RE.REGISTRY.read_text())
        if mutate:
            mutate(reg)
        return reg

    def test_the_committed_registry_is_valid(self):
        import render_evidence as RE
        RE.validate(self._reg())

    def test_an_unknown_inference_value_is_refused(self):
        import render_evidence as RE
        with self.assertRaises(SystemExit) as cm:
            RE.validate(self._reg(lambda r: r["phases"][0].update(inference="reportd")))
        self.assertIn("reportd", str(cm.exception))

    def test_a_duplicate_phase_id_is_refused(self):
        import copy
        import render_evidence as RE
        with self.assertRaises(SystemExit) as cm:
            RE.validate(self._reg(lambda r: r["phases"].append(copy.deepcopy(r["phases"][0]))))
        self.assertIn("duplicate", str(cm.exception))

    def test_a_phase_with_no_results_is_refused(self):
        import render_evidence as RE
        with self.assertRaises(SystemExit):
            RE.validate(self._reg(lambda r: r["phases"][0].update(results=[])))

    def test_every_result_file_is_claimed_by_exactly_one_entry(self):
        import glob, json, pathlib
        import render_evidence as RE
        reg = self._reg()
        skip = reg.get("skip_patterns") or []
        root = Path(__file__).parent.parent
        claimed = {}
        for phase in reg["phases"]:
            for pat in phase["results"]:
                for p in glob.glob(str(root / pat)):
                    if any(s in pathlib.Path(p).name for s in skip):
                        continue
                    claimed.setdefault(p, []).append(phase["id"])
        on_disk = [p for p in glob.glob(str(root / "results/phase_*.json"))
                   if not any(s in pathlib.Path(p).name for s in skip)]
        unclaimed = sorted(pathlib.Path(p).name for p in on_disk if p not in claimed)
        self.assertFalse(unclaimed, f"no registry entry claims: {unclaimed}")
        twice = {pathlib.Path(p).name: v for p, v in claimed.items() if len(v) > 1}
        self.assertFalse(twice, f"claimed more than once, records would double-count: {twice}")


class ThePrefixOnlyStateStillHasNotHappened(unittest.TestCase):
    """`quality.fork_position` treats a prefix-only scan as no fork, and says it never happens.

    The docstring carried the count in prose -- "0 of 4673 records" -- and by the time anyone
    looked the repository held 13900. The conclusion survived the drift; the denominator did not,
    and a claim whose denominator is three times smaller than reality understates its own evidence.
    The number lives here now, where it is recomputed instead of remembered.
    """

    def test_prefix_only_has_still_never_happened(self):
        import glob
        root = Path(__file__).parent.parent
        total = prefix = files = 0
        for p in sorted(glob.glob(str(root / "results" / "*.json"))):
            name = Path(p).name
            if ".partial." in name or name.startswith("dryrun") or ".pre_repair." in name:
                continue
            files += 1
            for rec in json.loads(Path(p).read_text())["records"]:
                total += 1
                div = rec.get("divergence")
                if div and div.get("prefix_only"):
                    prefix += 1
        self.assertGreater(files, 20, "the sweep found almost no result files; check the glob")
        self.assertEqual(prefix, 0,
                         f"prefix_only has now happened: {prefix} of {total} records over {files} "
                         f"files. quality.fork_position folds it into 'no fork', and every "
                         f"consumer of a fork position needs to be re-read before that stands.")


class TheCitationFileMayNotNameAThingThatDoesNotExist(unittest.TestCase):
    """`CITATION.cff` is metadata a reader is invited to trust without opening the repository.

    GitHub renders it behind a "Cite this repository" button, and Zenodo and the CFF tooling read
    it mechanically, so anything it asserts travels further than the README does and is checked by
    fewer people. That makes it the easiest place in the tree to publish an unbacked claim: a
    `version:` nobody tagged, a `doi:` nobody minted, a `date-released:` for a release that never
    happened. Those three keys are absent on purpose. This test is what keeps them absent until
    the artifact they name is real, and checks the ORCID here is the one the README shows -- two
    copies of an identifier is two chances to publish a wrong one.
    """

    @staticmethod
    def _load():
        root = Path(__file__).parent.parent
        text = (root / "CITATION.cff").read_text()
        try:
            import yaml
        except ImportError:
            return None, text
        return yaml.safe_load(text), text

    def test_it_parses_and_carries_what_cff_1_2_0_requires(self):
        data, text = self._load()
        if data is None:
            self.skipTest("PyYAML is not installed; the structural check needs a parser")
        for key in ("cff-version", "message", "title", "authors", "type"):
            self.assertIn(key, data, f"CITATION.cff has no {key!r}; CFF 1.2.0 requires it")
        self.assertEqual(data["cff-version"], "1.2.0")
        self.assertTrue(data["authors"], "CITATION.cff lists no authors")
        for a in data["authors"]:
            self.assertIn("family-names", a)
            self.assertIn("given-names", a)

    def test_no_version_doi_or_release_date_without_the_artifact_behind_it(self):
        import re
        import subprocess
        data, text = self._load()
        root = Path(__file__).parent.parent
        if data is None:
            declared = {k for k in ("version", "doi", "date-released")
                        if re.search(rf"(?m)^{k}:", text)}
        else:
            declared = {k for k in ("version", "doi", "date-released") if k in data}
        if not declared:
            return
        # Each of the three has to be backed by something outside this file.
        if "version" in declared:
            v = data["version"] if data else ""
            tags = subprocess.run(["git", "-C", str(root), "tag", "--list"],
                                  capture_output=True, text=True).stdout.split()
            self.assertIn(str(v), tags,
                          f"CITATION.cff declares version {v!r} and no such git tag exists. "
                          f"Tag the commit first, or take the key back out.")
        if "doi" in declared:
            self.fail("CITATION.cff declares a DOI. Remove this branch of the test only when the "
                      "deposit exists and its identifier resolves -- and check that it does.")
        if "date-released" in declared and "version" not in declared:
            self.fail("CITATION.cff carries a release date with no version to attach it to")

    def test_the_orcid_matches_the_one_the_readme_prints(self):
        import re
        data, text = self._load()
        root = Path(__file__).parent.parent
        readme = (root / "README.md").read_text()
        ids = set(re.findall(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", text))
        in_readme = set(re.findall(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", readme))
        if not ids:
            self.skipTest("CITATION.cff names no ORCID")
        self.assertTrue(in_readme,
                        "CITATION.cff carries an ORCID the README never shows; a reader who "
                        "never opens the .cff has no way to check it against anything")
        self.assertEqual(ids, in_readme,
                         f"the ORCIDs disagree: CITATION.cff {sorted(ids)} vs README {sorted(in_readme)}")



class ACoolerCardIsAnInterventionAndMustBeRecorded(unittest.TestCase):
    """The stock gate covered clocks and power and said nothing at all about cooling.

    `overclock_state` exists because a card was found carrying +400 MHz of memory while the
    write-up said "stock", and that run was discarded. Cooling reaches the same place by another
    road: hold the card cooler and it holds a higher sustained clock. Phase B's own telemetry has
    every arm shedding 8-12 % of its clock inside one arm-pass, and `arm_pass_gpu` shows the
    memory-bound baseline ending at 76.7 C / 1775 MHz against 81.6 C / 1751 MHz for the arms it
    is compared to. Until 2026-08-29 nothing anywhere in the harness read a fan, so a changed
    curve would have shown up only as temperatures that were quietly lower.
    """

    REAL = (
        "\n  Attribute 'GPUFanControlState' (host:0[gpu:0]): 0.\n"
        "  Attribute 'GPUTargetFanSpeed' (host:0[fan:0]): 30.\n"
        "  Attribute 'GPUCurrentFanSpeed' (host:0[fan:0]): 0.\n"
        "  Attribute 'GPUTargetFanSpeed' (host:0[fan:1]): 30.\n"
        "  Attribute 'GPUCurrentFanSpeed' (host:0[fan:1]): 0.\n"
    )

    def test_a_partial_answer_from_a_nonzero_exit_is_still_parsed(self):
        """This card has two fans, so a four-fan query exits 1 with five good values on stdout."""
        got = GS.parse_fan_query(self.REAL)
        self.assertEqual(got["fan_control_state"], 0)
        self.assertEqual(got["fan_control"], "auto")
        self.assertEqual(got["fan_count"], 2)
        self.assertEqual(got["fan_targets_pct"], {"0": 30.0, "1": 30.0})
        self.assertEqual(got["fan_current_pct"], {"0": 0.0, "1": 0.0})

    def test_target_and_current_are_kept_apart(self):
        """At 41 C this card reports target 30 % and current 0 %: a 3090 stops its fans below a
        threshold. Collapsing the two would make 'the curve asks for nothing' and 'the fans are
        dead' the same reading, and it is target-against-temperature that a changed curve moves."""
        got = GS.parse_fan_query(self.REAL)
        self.assertNotEqual(got["fan_targets_pct"]["0"], got["fan_current_pct"]["0"])

    def test_a_missing_control_state_is_an_error_not_a_default(self):
        only_fans = "  Attribute 'GPUTargetFanSpeed' (host:0[fan:0]): 30.\n"
        got = GS.parse_fan_query(only_fans)
        self.assertIn("fan_error", got)
        self.assertEqual(got["fan_control"], "unknown")

    def test_manual_fan_control_is_not_stock(self):
        base = {"mem_transfer_rate_offset": 0.0, "graphics_clock_offset": 0.0,
                "power_limit_w": 420.0, "power_default_limit_w": 420.0}
        self.assertTrue(T.is_stock({**base, "fan_control": "auto"}))
        self.assertFalse(T.is_stock({**base, "fan_control": "manual"}),
                         "a card whose fans a human is driving is not at stock settings")

    def test_an_unreadable_fan_is_not_stock_either(self):
        """Unknown fails, the same way an unreadable clock offset fails. A host that cannot answer
        the question has not answered it 'no'."""
        base = {"mem_transfer_rate_offset": 0.0, "graphics_clock_offset": 0.0,
                "power_limit_w": 420.0, "power_default_limit_w": 420.0}
        self.assertFalse(T.is_stock({**base, "fan_control": "unknown"}))
        self.assertFalse(T.is_stock(base), "fan_control absent entirely must not pass")

    def test_the_clock_and_power_clauses_still_hold(self):
        """Regression: extracting this rule from overclock_state must not have loosened it."""
        ok = {"mem_transfer_rate_offset": 0.0, "graphics_clock_offset": 0.0,
              "power_limit_w": 420.0, "power_default_limit_w": 420.0, "fan_control": "auto"}
        self.assertTrue(T.is_stock(ok))
        self.assertFalse(T.is_stock({**ok, "mem_transfer_rate_offset": 800.0}))
        self.assertFalse(T.is_stock({**ok, "graphics_clock_offset": 100.0}))
        self.assertFalse(T.is_stock({**ok, "power_limit_w": 450.0}))
        self.assertFalse(T.is_stock({**ok, "mem_transfer_rate_offset": "unverifiable"}),
                         "an offset that could not be read leaves only one numeric, and one is "
                         "not two")


if __name__ == "__main__":
    unittest.main(verbosity=2)
