"""Run the FlowMOP command line with `python -m flowmop`."""

from flowmop.cli import main

# The guard stops the spawned worker processes from starting a new run.
if __name__ == "__main__":
    raise SystemExit(main())
