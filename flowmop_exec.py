"""Old entry point for FlowMOP, kept for existing commands and scripts.

New use: install FlowMOP (pip install .) and run `flowmop`. This file still works
from any folder, with no install:

    python /path/to/FlowMOP/flowmop_exec.py sample.fcs --output-dir out
"""

# Python puts the folder of this script on sys.path, so the flowmop package
# next to it is found with no install.
from flowmop.cli import main
from flowmop.process import (  # noqa: F401  (names for old imports)
    filter_numerical_columns,
    load_data,
    process_file,
    write_filtered_fcs_files,
)

# The guard stops the spawned worker processes from starting a new run.
if __name__ == "__main__":
    raise SystemExit(main())
