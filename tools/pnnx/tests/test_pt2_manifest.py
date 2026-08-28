# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import re
import sys

from pt2_expectations import PASS
from pt2_expectations import PT2_EXPECTED_FAILURES
from pt2_expectations import UNCLASSIFIED


TEST_DIR = Path(__file__).resolve().parent

INFRASTRUCTURE_TESTS = {
    "test_exported_program",
    "test_exported_program_roundtrip",
    "test_pt2_manifest",
    "test_pt2_version_compatibility",
    "test_pnnx_test_utils",
}


def _top_level_test_names():
    return {path.stem for path in TEST_DIR.glob("test_*.py")}


def _pt2_registration_names():
    cmake_text = (TEST_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    names = re.findall(
        r"^\s*pnnx_add_(?:pt2_)?test\(([A-Za-z0-9_]+)\)\s*$",
        cmake_text,
        re.MULTILINE,
    )
    return ["test_" + name for name in names]


def _format_names(names):
    return "\n".join("  " + name for name in sorted(names))


def main():
    errors = []

    all_tests = _top_level_test_names()
    missing_infrastructure = INFRASTRUCTURE_TESTS - all_tests
    if missing_infrastructure:
        errors.append("infrastructure manifest names missing files:\n" + _format_names(missing_infrastructure))

    operator_tests = all_tests - INFRASTRUCTURE_TESTS
    expected_failure_names = set(PT2_EXPECTED_FAILURES)
    registration_list = _pt2_registration_names()
    registration_names = set(registration_list)

    duplicate_registrations = {
        name for name in registration_names if registration_list.count(name) != 1
    }
    if duplicate_registrations:
        errors.append("duplicate pt2 CMake registrations:\n" + _format_names(duplicate_registrations))

    extra_expected_failures = expected_failure_names - operator_tests
    if extra_expected_failures:
        errors.append("pt2 expected failures without operator tests:\n" + _format_names(extra_expected_failures))

    missing_registrations = operator_tests - registration_names
    if missing_registrations:
        errors.append("operator tests missing pt2 CMake registrations:\n" + _format_names(missing_registrations))

    extra_registrations = registration_names - operator_tests
    if extra_registrations:
        errors.append("pt2 CMake registrations without operator tests:\n" + _format_names(extra_registrations))

    unclassified = {
        name for name, (status, _) in PT2_EXPECTED_FAILURES.items() if status == UNCLASSIFIED
    }
    if unclassified:
        errors.append("unclassified pt2 expectations:\n" + _format_names(unclassified))

    invalid_diagnostics = set()
    for name, (status, diagnostic) in PT2_EXPECTED_FAILURES.items():
        if status == PASS and diagnostic:
            invalid_diagnostics.add(name)
        if status != PASS and not diagnostic:
            invalid_diagnostics.add(name)
    if invalid_diagnostics:
        errors.append("invalid pt2 expectation diagnostics:\n" + _format_names(invalid_diagnostics))

    if errors:
        sys.stderr.write("\n\n".join(errors) + "\n")
        return 1

    print(
        "pt2 manifest: %d declared operator tests, %d expected failures, %d infrastructure tests"
        % (len(operator_tests), len(expected_failure_names), len(INFRASTRUCTURE_TESTS))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
