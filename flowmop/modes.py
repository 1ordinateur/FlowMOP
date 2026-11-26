"""Names of the fluorescence modes of the FlowMOP time gate.

- positives: the mode of the FlowMOP paper. Global thresholds find the positive
  events; the time gate follows their geometric mean in each time bin.
- peaks: finds the positive peaks again in each time bin.
- geomean: the geometric mean of all events in each time bin.
- both: peaks and geomean together; an event is removed only when two or more
  checks reject it.
"""

FLUOR_MODES = ("positives", "peaks", "geomean", "both")

# Old names that are still accepted. Up to version 0.2.0, the paper mode had the
# name "positive_geomeans", and "positives" was the name of the peaks mode.
FLUOR_MODE_ALIASES = {"positive_geomeans": "positives"}


def canonical_fluor_mode(mode: str) -> str:
    """Return the current name of a fluorescence mode; an old name gives its new name."""
    return FLUOR_MODE_ALIASES.get(mode, mode)
