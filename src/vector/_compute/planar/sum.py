# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Planar.sum(self, axis=..., dtype=..., initial=...)
"""

import numpy

from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    _aztype,
    _flavor_of,
    _from_signature,
    _handler_of,
    _lib_of,
)


def ret_sum(lib, axis, az1, az2):
    if axis is None:
        sum_val = lib.sum(az1) + lib.sum(az2)
        return sum_val
    elif axis == 0:
        sum_val = lib.array(
            [
                numpy.sum(az1),
                numpy.sum(az2),
            ]
        )
    elif axis == 1:
        sum_val = az1 + az2
    else:
        raise ValueError("axis must be 0, 1, or None")

    return sum_val


def xy(lib, axis, initial, x, y):
    return ret_sum(lib, axis, x, y) + initial


def rhophi(lib, axis, initial, rho, phi):
    return ret_sum(lib, axis, rho, phi) + initial


dispatch_map = {
    (AzimuthalXY,): (xy, float, numpy.ndarray),
    (AzimuthalRhoPhi,): (rhophi, float, numpy.ndarray),
}


def dispatch(
    axis: typing.Optional[int],
    initial: typing.Any,
    v: typing.Any,
) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (_aztype(v),),
    )
    with numpy.errstate(all="ignore"):
        return _handler_of(v)._wrap_result(
            _flavor_of(v),
            function(
                _lib_of(v),
                axis,
                initial,
                *v.azimuthal.elements,
            ),
            returns,
            1,
        )
