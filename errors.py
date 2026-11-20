"""Bearing intersection and nonlinear pointing-error propagation."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from geometry import wrap

SEED = 20261120


def _phi(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2. * np.pi)


def probability(sigma, window_deg=5.):
    """Integrate the projected-normal bearing density against gimbal error."""
    w = np.deg2rad(window_deg)
    tau = np.deg2rad(np.hypot(0.8, 0.6))
    if sigma == 0.:
        return float(2. * ndtr(w / tau) - 1.)
    m = 40. / sigma

    def density(beta):
        parallel = m * np.cos(beta)
        f = _phi(m * np.sin(beta)) * (_phi(parallel) + parallel * ndtr(parallel))
        shifts = 2. * np.pi * np.arange(-1, 2)
        conditional = np.sum(ndtr((shifts + w - beta) / tau) - ndtr((shifts - w - beta) / tau))
        return f * conditional

    points = sorted({0., w, min(1. / m, np.pi), min(5. / m, np.pi), np.pi / 2})
    return float(2. * quad(density, 0., np.pi, epsabs=1e-10, epsrel=1e-10, points=points)[0])


def pointing_error_results():
    linear_sigma = 40. * np.deg2rad(np.sqrt((5. / ndtri(0.975)) ** 2 - 0.8 ** 2 - 0.6 ** 2))
    sigma = brentq(lambda x: probability(x) - .95, 0.3, 3., xtol=1e-12)
    s_max = (sigma - .3) / .02
    variances = np.array([(np.rad2deg(sigma / 40.)) ** 2, .8 ** 2, .6 ** 2])
    rng = np.random.default_rng(SEED)
    samples = 600_000
    x, y, zero, control = rng.standard_normal((4, samples))
    trials = []
    for distance in (0., s_max - 1., s_max, s_max + 1., 80.):
        sd = .3 + .02 * distance
        angles = wrap(np.rad2deg(np.arctan2(-sd * y, 40. - sd * x)) + .8 * zero + .6 * control)
        p = float(np.mean(np.abs(angles) <= 5.))
        z = ndtri(.975)
        den = 1. + z * z / samples
        center = (p + z * z / (2. * samples)) / den
        margin = z * np.sqrt(p * (1. - p) / samples + z * z / (4. * samples ** 2)) / den
        interval = [float(center - margin), min(1., float(center + margin))]
        trials.append(dict(distance_m=distance, exact=probability(sd), monte_carlo=p,
                           mc_95_interval=interval, mc_95_halfwidth=float(margin)))
    return dict(sigma_limit_m=sigma, max_distance_m=s_max,
                linear_sigma_limit_m=linear_sigma, linear_max_distance_m=(linear_sigma - .3) / .02,
                a_limit_at_b_0_m=sigma, b_limit_at_a_0=sigma / 80., b_limit_at_a_0_3=(sigma - .3) / 80.,
                variance_shares=dict(zip(("position", "zero_offset", "control"), (variances / variances.sum()).tolist())),
                samples=samples, trials=trials)


def validate_intersection(p, q, target):
    points = np.asarray([p, q])
    direction = target.xy - points
    alpha = np.arctan2(direction[:, 1], direction[:, 0])
    ranges = np.linalg.norm(direction, axis=1)
    jac = np.column_stack((-np.sin(alpha), np.cos(alpha))) / ranges[:, None]
    inv = np.linalg.inv(jac)
    covariance = np.deg2rad(1.) ** 2 * inv @ inv.T
    rng = np.random.default_rng(SEED + 1)
    noise = rng.normal(0., np.deg2rad(1.), size=(300_000, 2))
    measured = alpha + noise
    u, v = (np.stack((np.cos(a), np.sin(a)), axis=-1) for a in measured.T)
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    baseline = q - p
    # Exclude intersecting angles within 10 degrees of the singular orientations.
    valid = np.abs(cross) >= np.sin(np.deg2rad(10.))
    t = (baseline[0] * v[valid, 1] - baseline[1] * v[valid, 0]) / cross[valid]
    radii = np.linalg.norm(p + t[:, None] * u[valid] - target.xy, axis=1)
    return dict(covariance_m2=covariance.tolist(), cutoff_deg=10., samples=len(noise),
                rejected=int((~valid).sum()), monte_carlo_conditional_rmse_m=float(np.sqrt(np.mean(radii ** 2))),
                monte_carlo_conditional_p95_m=float(np.quantile(radii, .95)))
