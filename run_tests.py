#!/usr/bin/env python3
"""
Test runner script for Point Cloud Classification System.
"""
import sys
import subprocess
import argparse


def run_pytest(test_paths=None, markers=None, coverage=False, verbose=False):
    """Run pytest with given arguments."""
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    if markers:
        cmd.extend(["-m", markers])

    if coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing", "--cov-report=html"])

    if test_paths:
        cmd.extend(test_paths)
    else:
        cmd.append("tests/")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests for Point Cloud Classification System")
    parser.add_argument("test_paths", nargs="*", help="Specific test files or directories to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-c", "--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("-m", "--markers", help="Run tests with specific markers")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")

    args = parser.parse_args()

    # Determine which tests to run
    test_paths = args.test_paths
    markers = args.markers

    if args.unit and not markers:
        markers = "not integration and not gpu and not slow"
    elif args.integration and not markers:
        markers = "integration"

    # Run tests
    return_code = run_pytest(
        test_paths=test_paths,
        markers=markers,
        coverage=args.coverage,
        verbose=args.verbose
    )

    sys.exit(return_code)


if __name__ == "__main__":
    main()