"""Single command-line entry point for pre-treatment client classification."""

import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in {"chunk", "solve", "compare-assignments"}:
        from patient_assignment.cli import main as command_main
    else:
        from clinical_target_extraction.src.cli import main as command_main
    return command_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
