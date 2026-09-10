"""Every test in the suite has to assert something when it runs.

A test that passes without executing a single assertion is counted among the suite's tests,
reported as a pass, and checks nothing. This repository has shipped three of them.

  * `test_record_counts_quoted_in_the_readme_match_the_result_files` searched the prose for
    `Phase A (875 request records`, a form the documents have never used, and its loop turned the
    miss into a pass with `if not m: continue`. It asserted nothing from the commit that added it,
    in v1.0.0 and v1.0.4, both deposited. Correction 59.
  * `test_no_descendant_of_this_process_is_counted_as_competition` put both its assertions inside
    `for c in load["competing"]`, which is empty on a quiet host -- the state a measurement runs
    in. Correction 60.
  * Correction 45's four guards could not fail for other reasons, and were found by reading rather
    than by a check.

Reading the source cannot find these. The first needed a corpus to match against, the second
needed a live process table; both looked like ordinary tests and only the run knows. So this
counts assertions as they execute, by wrapping every `assert*` and `fail*` on `TestCase` before
the suite runs, and fails on any test that reached the end having called none of them.

A test whose failure mode is an exception out of a validator rather than an assertion is
legitimate -- `RE.validate(reg)` fails by raising, and "must not raise" is the assertion. Those
are named in ALLOWED with the call that carries the check. The list is verified in both
directions: an entry naming a test that no longer exists fails, and so does an entry for a test
that has since started asserting, because then the exemption is describing something else.

Usage:  python3 harness/assert_coverage.py
"""
from __future__ import annotations

import collections
import io
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# test id -> the call that does the checking, since it is not an assertion
ALLOWED = {
    "test_harness.TheRegistryDeclaresAVocabularyAndMustEnforceIt.test_the_committed_registry_is_valid":
        "render_evidence.validate() raises on an invalid registry; 'must not raise' is the check",
    "test_harness.TestPhaseVDriverRefusesBeforeItLoadsAnything.test_a_48_gib_card_takes_them_all":
        "vllm_bench.assert_arms_fit() raises when an arm does not fit; the same shape",
}


def _instrument():
    """Count every assertion each test executes. Returns the counter and the current-test slot."""
    counts: collections.Counter[str] = collections.Counter()
    current: list[str | None] = [None]
    import inspect
    for name in dir(unittest.TestCase):
        if not name.startswith(("assert", "fail")):
            continue
        attr = getattr(unittest.TestCase, name)
        # `callable` is not the test: `failureException` is AssertionError, a class, and wrapping
        # it in a function makes unittest's own `issubclass(exc_info[0], test.failureException)`
        # raise TypeError the first time any test errors -- which is the moment this check is
        # most needed. Only plain methods get counted.
        if not inspect.isfunction(attr):
            continue

        def make(orig):
            def wrapper(self, *a, **k):
                if current[0] is not None:
                    counts[current[0]] += 1
                return orig(self, *a, **k)
            return wrapper

        setattr(unittest.TestCase, name, make(attr))
    return counts, current


def main() -> int:
    counts, current = _instrument()
    import test_harness  # noqa: E402  -- after the wrapping, so nothing is missed

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            current[0] = test.id()
            counts.setdefault(test.id(), 0)
            super().startTest(test)

        def stopTest(self, test):
            current[0] = None
            super().stopTest(test)

    suite = unittest.TestLoader().loadTestsFromModule(test_harness)
    res = unittest.TextTestRunner(stream=io.StringIO(), resultclass=Result, verbosity=0).run(suite)

    if not counts:
        print("FAIL: no test ran, so nothing was measured")
        return 1

    skipped = {t.id() for t, _ in res.skipped}
    silent = sorted(t for t, c in counts.items() if c == 0 and t not in skipped)
    unlisted = [t for t in silent if t not in ALLOWED]
    stale = sorted(t for t in ALLOWED if t not in counts)
    outgrown = sorted(t for t in ALLOWED if counts.get(t, 0) > 0)

    print(f"{len(counts)} tests, {sum(counts.values())} assertions executed, "
          f"{len(skipped)} skipped, {len(ALLOWED)} exempt")

    bad = False
    for t in unlisted:
        print(f"FAIL: {t} ran and asserted nothing")
        bad = True
    for t in stale:
        print(f"FAIL: exempt test no longer exists: {t}")
        bad = True
    for t in outgrown:
        print(f"FAIL: {t} is exempt but now executes {counts[t]} assertion(s); take it off "
              f"the list rather than leaving it described as something it is not")
        bad = True
    if res.failures or res.errors:
        print(f"FAIL: the suite itself is red here ({len(res.failures)} failures, "
              f"{len(res.errors)} errors); section 1 reports it")
        bad = True
    if not bad:
        print("every test that ran asserted something")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
