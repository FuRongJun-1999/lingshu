"""Issue #146: declared verification methods and references survive SQL tags."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.provenance import VERIFY_METHODS, from_legacy, new_provenance


class VerificationRoundTripTests(unittest.TestCase):
    def test_each_declared_method_and_status_survives_sql_round_trip(self):
        for status in ("partial", "verified", "anchored"):
            for method in sorted(VERIFY_METHODS - {"none"}):
                with self.subTest(status=status, method=method):
                    original = new_provenance(
                        "测试台", "bench_fixture", "fixture",
                        verify_status=status, verify_methods=[method],
                        verify_ref="tests/test_example.py::test_case:seed=7",
                        observed_at=1700000000.0,
                        existence_constraint="synthetic verification fixture")
                    self.assertEqual(original.validate(), [])
                    cs, tags = original.to_sql()
                    restored = from_legacy(tags, condition_space=cs)
                    self.assertEqual(restored.verify_methods, [method])
                    self.assertEqual(restored.verify_ref, original.verify_ref)
                    self.assertEqual(restored.verify_status, status)
                    self.assertEqual(restored.strength(), original.strength())
                    self.assertEqual(restored.validate(), [])

    def test_mixed_methods_preserve_order_without_duplicate_legacy_alias(self):
        original = new_provenance(
            "测试台", "bench_fixture", "fixture", verify_status="verified",
            verify_methods=["cross_channel", "whitebox_code", "action_world"],
            verify_ref="python -m unittest tests.test_example:seed=7",
            observed_at=1700000000.0, existence_constraint="synthetic fixture")
        restored = from_legacy(
            original.to_tags(["白箱校验", "action_world"]),
            condition_space=original.to_condition_space())
        # Extra tags precede declared tags; method order follows first occurrence.
        self.assertEqual(restored.verify_methods,
                         ["whitebox_code", "action_world", "cross_channel"])
        self.assertEqual(restored.verify_ref, original.verify_ref)
        self.assertEqual(restored.strength(), "strong")

    def test_legacy_chinese_and_llm_tags_keep_existing_interpretation(self):
        restored = from_legacy(["白箱校验", "llm_verify", "partial"])
        self.assertEqual(restored.verify_methods, ["whitebox_code", "llm_verify"])
        self.assertEqual(restored.verify_ref, "")
        self.assertEqual(restored.strength(), "strong")

    def test_unknown_tags_do_not_create_verification_evidence(self):
        restored = from_legacy(["verified", "unregistered_method"])
        self.assertEqual(restored.verify_methods, ["none"])
        self.assertEqual(restored.verify_ref, "")
        self.assertEqual(restored.strength(), "weak")

    def test_reference_alone_does_not_upgrade_status_or_strength(self):
        restored = from_legacy(["vref:tests/test_example.py::test_case"])
        self.assertEqual(restored.verify_ref, "tests/test_example.py::test_case")
        self.assertEqual(restored.verify_status, "unverified")
        self.assertEqual(restored.strength(), "none")


if __name__ == "__main__":
    unittest.main()
