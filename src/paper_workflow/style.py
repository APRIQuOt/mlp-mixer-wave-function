"""Architecture identities and plotting conventions shared by paper assets."""
from typing import NamedTuple
import numpy as np
import matplotlib as mpl

# Marker styles shared by the paper plots. The architecture
# plots below use a separate marker mapping so that marker shape identifies the
# execution mode while colour identifies the backbone family.
markers = ("*", "^", "h", "+", "o", ">", "<", "H", "1", "2", "3", "4", "D", "8", "P")


MODE_MARKERS = {
    "site": "o",
    "patch": "^",
    "hierarchical": "s",
    "hiermixer": "D",
    "hierrnn": "P",
}


def _colours_for_family(cmap_name: str, count: int) -> tuple[np.ndarray, ...]:
    """Return clearly separated shades from one family colour range."""
    family_cmap = mpl.colormaps[cmap_name]
    return tuple(
        np.asarray(family_cmap(value))
        for value in np.linspace(0.45, 0.85, count)
    )


ARCHITECTURE_SPECS = (
    ("mixer_patch", "M", "mixer", "patch"),
    ("mixer_hierarchical", "HM", "mixer", "hierarchical"),
    ("rnn_site", "RNN", "rnn", "site"),
    ("rnn_patch", "PRNN", "rnn", "patch"),
    ("rnn_hierarchical", "HRNN", "rnn", "hierarchical"),
    ("transformer_site", "TF", "transformer", "site"),
    ("transformer_patch", "PTF", "transformer", "patch"),
    ("transformer_hiermixer", "HTF+M", "transformer", "hiermixer"),
    ("transformer_hierrnn", "HTF+R", "transformer", "hierrnn"),
)

_FAMILY_COLOURS = {
    "mixer": _colours_for_family("Oranges", 2),
    "rnn": _colours_for_family("Greens", 3),
    "transformer": _colours_for_family("Blues", 4),
}

# Retain the old public name for callers that use the helper interactively, but
# make the palette family-aware instead of spreading all classes over viridis.
colours = tuple(
    colour
    for family in ("mixer", "rnn", "transformer")
    for colour in _FAMILY_COLOURS[family]
)


class Alias(NamedTuple):
    identifier: str
    marker: str
    colour: np.ndarray


ARCH_ALIAS = {}
_family_indices = {family: 0 for family in _FAMILY_COLOURS}
for key, identifier, family, mode in ARCHITECTURE_SPECS:
    colour = _FAMILY_COLOURS[family][_family_indices[family]]
    _family_indices[family] += 1
    ARCH_ALIAS[key] = Alias(
        identifier=identifier,
        marker=MODE_MARKERS[mode],
        colour=colour,
    )


ARCH_FAMILY = {
    key: family
    for key, _, family, _ in ARCHITECTURE_SPECS
}



def flop_tick_positions(lower: float, upper: float, count: int = 5) -> np.ndarray:
    """Choose readable tick coordinates without changing the logarithmic scale."""
    from matplotlib.ticker import MaxNLocator

    lower = max(float(lower), np.finfo(float).tiny)
    upper = max(float(upper), lower * (1.0 + 1e-6))
    count = max(3, count)
    exponent_low = int(np.floor(np.log10(lower)))
    exponent_high = int(np.ceil(np.log10(upper)))
    candidates = np.asarray(
        [factor * 10.0**exponent
         for exponent in range(exponent_low, exponent_high + 1)
         for factor in (1.0, 2.0, 5.0)],
        dtype=float,
    )
    candidates = candidates[(candidates >= lower) & (candidates <= upper)]
    if len(candidates) >= 3:
        if len(candidates) > count:
            indices = np.unique(np.linspace(0, len(candidates) - 1, count).round().astype(int))
            candidates = candidates[indices]
        return candidates

    # A narrow range may not contain three 1/2/5 ticks. Use rounded numeric
    # positions (e.g. 1.2, 1.3, 1.4 M), rather than arbitrary geometric values.
    for bins in range(count - 1, 2 * count + 1):
        candidates = MaxNLocator(nbins=bins, steps=[1, 2, 2.5, 5, 10]).tick_values(lower, upper)
        candidates = candidates[(candidates >= lower) & (candidates <= upper)]
        if len(candidates) >= 3:
            return candidates
    raise ValueError(f"Cannot choose three readable FLOP ticks within {lower:g}..{upper:g}")


