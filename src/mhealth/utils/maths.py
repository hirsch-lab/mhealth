import numpy as np
import pandas as pd
import numpy.typing as npt

from typing import Any, Collection, Union

Scalar = Union[int, float, bool, complex, np.number]
def split_contiguous(arr: Collection[Scalar],
                     tol: Any,
                     inclusive: bool=False,
                     indices: bool=False) -> Collection[npt.ArrayLike]:
    """
    Split contiguous chunks from the input array. The function assumes a
    sorted array.

    Arguments:
        arr:        Array-like or None
        tol:        Tolerance that defines contiguity. If two consecutive
                    values differ by more than tol, they will belong to
                    different chunks
        inclusive:  If True, x[i+1]-x[i] >= tol (larger equal) is evaluated,
                    otherwise x[i+1]-x[i] > tol (strictly larger)
        indices:    If True, return a list of start and stop index for the
                    chunks of contiguous data: [(i0, i1), (i1, i2), ...].
                    Note that the stop index points one position beyond the
                    actual position, so that slices work: arr[i0:i1].
                    If False, return the actual data chunks (as views on the
                    original input array.
    """
    if len(arr)==0:
        return []
    op = np.greater_equal if inclusive else np.greater
    idx = np.where(op(np.diff(arr), tol))[0]+1  # type: ignore
    if indices:
        chunks = list(zip([0]+idx.tolist(), idx.tolist()+[len(arr)]))
    else:
        # np.split preserves pandas objects!
        chunks = np.split(arr, idx)
    return chunks
