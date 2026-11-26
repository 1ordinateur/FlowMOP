"""FlowMOP: automated time, debris and doublet gating for flow cytometry data.

Command line:  flowmop INPUT... --output-dir OUT
Python:        from flowmop import process_file
"""

__version__ = "0.3.0"

__all__ = ["__version__", "process_file", "FlowMOP"]


def __getattr__(name):
    # Lazy imports keep `flowmop --help` and `flowmop --version` fast.
    if name == "process_file":
        from flowmop.process import process_file

        return process_file
    if name == "FlowMOP":
        from flowmop.pipeline import FlowMOP

        return FlowMOP
    raise AttributeError(f"module 'flowmop' has no attribute {name!r}")
