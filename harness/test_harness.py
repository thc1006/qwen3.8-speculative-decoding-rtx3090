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

if __name__ == "__main__":
    unittest.main(verbosity=2)
