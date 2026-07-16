from shunt.verifiers.tier1 import RegexVerifier


class TestRegexVerifier:
    def setup_method(self) -> None:
        self.v = RegexVerifier()

    def test_success_tests_passed(self) -> None:
        result = self.v.verify("all tests passed successfully")
        assert result.outcome == "success"
        assert result.confidence == 0.4
        assert not result.is_infra_failure

    def test_success_all_tests_succeed(self) -> None:
        result = self.v.verify("All tests succeed on this implementation")
        assert result.outcome == "success"
        assert result.confidence == 0.4

    def test_success_checkmark(self) -> None:
        result = self.v.verify("output: \u2713 all good")
        assert result.outcome == "success"
        assert result.confidence == 0.4

    def test_failure_error(self) -> None:
        result = self.v.verify("Error: something went wrong")
        assert result.outcome == "failure"
        assert result.confidence == 0.3
        assert not result.is_infra_failure

    def test_failure_traceback(self) -> None:
        result = self.v.verify('Traceback (most recent call last):\n  File "test.py"')
        assert result.outcome == "failure"
        assert result.confidence == 0.3

    def test_failure_failed(self) -> None:
        result = self.v.verify("2 tests failed out of 10")
        assert result.outcome == "failure"
        assert result.confidence == 0.3

    def test_infra_module_not_found(self) -> None:
        result = self.v.verify("ModuleNotFoundError: No module named 'xyz'")
        assert result.outcome == "failure"
        assert result.confidence == 0.3
        assert not result.is_infra_failure

    def test_infra_import_error(self) -> None:
        result = self.v.verify("ImportError: cannot import name 'foo'")
        assert result.outcome == "failure"
        assert result.confidence == 0.3
        assert not result.is_infra_failure

    def test_infra_no_module(self) -> None:
        result = self.v.verify("No module named 'something'")
        assert result.outcome == "infra_failure"
        assert result.is_infra_failure

    def test_infra_module_not_found_only(self) -> None:
        result = self.v.verify("ModuleNotFound: no module named 'xyz'")
        assert result.outcome == "infra_failure"
        assert result.confidence == 0.2
        assert result.is_infra_failure

    def test_weak_success_implementation_complete(self) -> None:
        result = self.v.verify("The implementation is complete")
        assert result.outcome == "weak_success"
        assert result.confidence == 0.2
        assert not result.is_infra_failure

    def test_weak_success_works_correctly(self) -> None:
        result = self.v.verify("This works correctly now")
        assert result.outcome == "weak_success"
        assert result.confidence == 0.2

    def test_unknown_no_signal(self) -> None:
        result = self.v.verify("Here is some random text with no relevant patterns.")
        assert result.outcome == "unknown"
        assert result.confidence == 0.0
        assert result.matched_pattern is None

    def test_highest_confidence_wins(self) -> None:
        text = "Error: something failed. But all tests passed."
        result = self.v.verify(text)
        assert result.outcome == "success"
        assert result.confidence == 0.4

    def test_matched_pattern_stored(self) -> None:
        result = self.v.verify("tests passed")
        assert result.matched_pattern is not None
        assert "tests passed" in result.matched_pattern

    def test_case_insensitive(self) -> None:
        result = self.v.verify("ALL TESTS SUCCEED")
        assert result.outcome == "success"
        assert result.confidence == 0.4
