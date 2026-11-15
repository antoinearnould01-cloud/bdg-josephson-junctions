"""Search for the critical current I_max and the critical phase phi_max.

The CPR I(phi) is evaluated by a full self-consistent solve per point, so the
maximum is located with a derivative-free hill climb:

1. evaluate I at phi - step, phi, phi + step;
2. move towards the larger neighbour while the current increases;
3. once phi is a local maximum, divide the step by `shrink`;
4. stop when step < min_step.

An optional coarse scan adds extra points to the history only to check, at the
end, that the maximum found is the largest current seen (otherwise the result
is flagged as unreliable).
"""

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass
class CriticalPoint:
    phase: float      # phase of the maximum (the variable that was optimised)
    current: float    # I_max
    reliable: bool    # True if I_max >= every current evaluated during the search


def find_critical_point(
    current: Callable[[float], float],
    phase_guess: float = np.pi / 2,
    step: float = 1.0,
    shrink: float = 2.0,
    min_step: float = 1e-6,
    coarse_phases: Iterable[float] | None = None,
    max_moves: int = 1000,
    nan_safe: bool = False,
) -> CriticalPoint:
    """Hill-climb on I(phi). See module docstring.

    nan_safe : `current` may return NaN (failed solve); the search then stops
               and only the finite currents are used for the final check.
    """
    phase = phase_guess
    c_mid = current(phase)
    c_left = current(phase - step)
    c_right = current(phase + step)
    seen = [c_left, c_mid, c_right]
    if coarse_phases is not None:
        seen += [current(p) for p in coarse_phases]

    for _ in range(max_moves):
        if nan_safe and np.isnan([c_left, c_mid, c_right]).any():
            break
        if (c_left > c_mid and c_right < c_mid) or (c_left > c_mid and c_right == c_mid):
            phase -= step                                   # move left
            c_right, c_mid = c_mid, c_left
            c_left = current(phase - step)
            seen.append(c_left)
        elif (c_right > c_mid and c_mid > c_left) or (c_right > c_mid and c_mid == c_left):
            phase += step                                   # move right
            c_left, c_mid = c_mid, c_right
            c_right = current(phase + step)
            seen.append(c_right)
        elif (c_right < c_mid and c_mid > c_left) or (c_right == c_mid and c_mid > c_left) \
                or (c_right < c_mid and c_mid == c_left):
            step /= shrink                                  # refine around the maximum
            if step < min_step:
                break
            c_left = current(phase - step)
            c_right = current(phase + step)
            seen += [c_left, c_right]
        elif c_right == c_mid == c_left:
            break
        elif c_right > c_mid and c_mid < c_left:
            break                                           # local minimum

    if nan_safe:
        finite = [c for c in seen if not np.isnan(c)]
        reliable = bool(finite) and c_mid >= max(finite)
    else:
        reliable = bool(np.all(c_mid >= np.array(seen)))
    return CriticalPoint(phase, c_mid, reliable)
