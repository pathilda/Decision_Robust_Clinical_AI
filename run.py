"""Single command-line entry point for the clinical target extraction pipeline."""

from clinical_target_extraction.src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
