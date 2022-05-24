# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    @property
    Lorentz.sum(self, axis=..., dtype=..., initial=...)
"""

import numpy

from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    TemporalT,
    TemporalTau,
    _aztype,
    _flavor_of,
    _from_signature,
    _ltype,
    _ttype,
)


def ret_sum(lib, axis, az1, az2, l1, t1):
    if axis is None:
        sum_val = lib.sum(az1) + lib.sum(az2) + lib.sum(l1) + lib.sum(t1)
    elif axis == 0:
        sum_val = lib.array([lib.sum(az1), lib.sum(az2), lib.sum(l1), lib.sum(t1)])
    elif axis == 1:
        sum_val = az1 + az2 + l1 + t1
    else:
        raise ValueError("axis must be 0, 1, or None")

    return sum_val


def xy_z_t(lib, axis, initial, x, y, z, t):
    return ret_sum(lib, axis, x, y, z, t) + initial


def xy_z_tau(lib, axis, initial, x, y, z, tau):
    return ret_sum(lib, axis, x, y, z, tau) + initial


def xy_theta_t(lib, axis, initial, x, y, theta, t):
    return ret_sum(lib, axis, x, y, theta, t) + initial


def xy_theta_tau(lib, axis, initial, x, y, theta, tau):
    return ret_sum(lib, axis, x, y, theta, tau) + initial


def xy_eta_t(lib, axis, initial, x, y, eta, t):
    return ret_sum(lib, axis, x, y, eta, t) + initial


def xy_eta_tau(lib, axis, initial, x, y, eta, tau):
    return ret_sum(lib, axis, x, y, eta, tau) + initial


def rhophi_z_t(lib, axis, initial, rho, phi, z, t):
    return ret_sum(lib, axis, rho, phi, z, t) + initial


def rhophi_z_tau(lib, axis, initial, rho, phi, z, tau):
    return ret_sum(lib, axis, rho, phi, z, tau) + initial


def rhophi_theta_t(lib, axis, initial, rho, phi, theta, t):
    return ret_sum(lib, axis, rho, phi, theta, t) + initial


def rhophi_theta_tau(lib, axis, initial, rho, phi, theta, tau):
    return ret_sum(lib, axis, rho, phi, theta, tau) + initial


def rhophi_eta_t(lib, axis, initial, rho, phi, eta, t):
    return ret_sum(lib, axis, rho, phi, eta, t) + initial


def rhophi_eta_tau(lib, axis, initial, rho, phi, eta, tau):
    return ret_sum(lib, axis, rho, phi, eta, tau) + initial


dispatch_map = {
    (AzimuthalXY, LongitudinalZ, TemporalT): (xy_z_t, float, numpy.ndarray),
    (AzimuthalXY, LongitudinalZ, TemporalTau): (xy_z_tau, float, numpy.ndarray),
    (AzimuthalXY, LongitudinalTheta, TemporalT): (xy_theta_t, float, numpy.ndarray),
    (AzimuthalXY, LongitudinalTheta, TemporalTau): (xy_theta_tau, float, numpy.ndarray),
    (AzimuthalXY, LongitudinalEta, TemporalT): (xy_eta_t, float, numpy.ndarray),
    (AzimuthalXY, LongitudinalEta, TemporalTau): (xy_eta_tau, float, numpy.ndarray),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalT): (rhophi_z_t, float, numpy.ndarray),
    (AzimuthalRhoPhi, LongitudinalZ, TemporalTau): (rhophi_z_tau, float, numpy.ndarray),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalT): (
        rhophi_theta_t,
        float,
        numpy.ndarray,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta, TemporalTau): (
        rhophi_theta_tau,
        float,
        numpy.ndarray,
    ),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalT): (rhophi_eta_t, float, numpy.ndarray),
    (AzimuthalRhoPhi, LongitudinalEta, TemporalTau): (
        rhophi_eta_tau,
        float,
        numpy.ndarray,
    ),
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
            _ttype(v),
        ),
    )
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v),
            function(
                v.lib,
                axis,
                initial,
                *v.azimuthal.elements,
                *v.longitudinal.elements,
                *v.temporal.elements
            ),
            returns,
            1,
        )
