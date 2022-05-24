# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Spatial.isclose(self, other, rtol=..., atol=..., equal_nan=...)
"""

import numpy

from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    _aztype,
    _flavor_of,
    _from_signature,
    _ltype,
)


def ret_sum(lib, axis, az1, az2, l1):

    if axis is None:
        sum_val = lib.sum(az1) + lib.sum(az2) + lib.sum(l1)
    elif axis == 0:
        sum_val = lib.array([lib.sum(az1), lib.sum(az2), lib.sum(l1)])
    elif axis == 1:
        sum_val = az1 + az2 + l1

    return sum_val


def xy_z(lib, axis, initial, x, y, z):
    return ret_sum(lib, axis, x, y, z) + initial


def xy_theta(lib, axis, initial, x, y, theta):
    return ret_sum(lib, axis, x, y, theta) + initial


def xy_eta(lib, axis, initial, x, y, eta):
    return ret_sum(lib, axis, x, y, eta) + initial


def rhophi_z(lib, axis, initial, rho, phi, z):
    return ret_sum(lib, axis, rho, phi, z) + initial


def rhophi_theta(lib, axis, initial, rho, phi, theta):
    return ret_sum(lib, axis, rho, phi, theta) + initial


def rhophi_eta(lib, axis, initial, rho, phi, eta):
    return ret_sum(lib, axis, rho, phi, eta) + initial


# same types
dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (xy_z, float, numpy.ndarray),
    (AzimuthalXY, LongitudinalTheta): (xy_theta, float, numpy.ndarray),
    (AzimuthalXY, LongitudinalEta): (xy_eta, float, numpy.ndarray),
    (AzimuthalRhoPhi, LongitudinalZ): (rhophi_z, float, numpy.ndarray),
    (AzimuthalRhoPhi, LongitudinalTheta): (rhophi_theta, float, numpy.ndarray),
    (AzimuthalRhoPhi, LongitudinalEta): (rhophi_eta, float, numpy.ndarray),
}


def dispatch(
    axis: typing.Optional[int],
    initial: typing.Any,
    v: typing.Any,
) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v),
            _ltype(v),
        ),
    )
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            function(v.lib, *v.azimuthal.elements, *v.longitudinal.elements),
            returns,
            1,
        )
