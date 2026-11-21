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


def intersection_sweep(samples=300_000, cutoff_deg=10.):
    """Check local DRMS against exact line intersections at varied geometries."""
    if samples < 1 or not 0. < cutoff_deg < 90.:
        raise ValueError("Use a positive sample count and a rejection angle between 0 and 90 degrees.")
    sigma = np.deg2rad(1.)
    cutoff = np.deg2rad(cutoff_deg)
    noise = np.random.default_rng(SEED + 2).normal(0., sigma, (2, samples))
    angles = (1, 5, 10, 15, 30, 45, 60, 90, 120, 150, 165, 170, 175, 179)
    ranges = ((10., 10.), (20., 20.), (40., 40.), (10., 20.), (10., 40.))
    rows = []
    for angle in angles:
        gamma = np.deg2rad(angle)
        nominal = np.array([0., gamma])
        measured = nominal[:, None] + noise
        u, v = (np.column_stack((np.cos(a), np.sin(a))) for a in measured)
        cross = np.sin(measured[1] - measured[0])
        kept = np.abs(cross) >= np.sin(cutoff)
        accepted = int(kept.sum())
        for r, s in ranges:
            p = np.array([-r, 0.])
            q = -s * np.array([np.cos(gamma), np.sin(gamma)])
            direction = q - p
            numerator = direction[0] * v[:, 1] - direction[1] * v[:, 0]
            norm = np.linalg.norm(cross[:, None] * p + numerator[:, None] * u, axis=1)
            error = np.divide(norm, np.abs(cross), out=np.full(samples, np.inf), where=cross != 0.)
            theory = sigma * np.hypot(r, s) / np.sin(gamma)
            y = r * noise[0]
            linear_error = np.hypot((y * np.cos(gamma) - s * noise[1]) / np.sin(gamma), y)
            conditional = error[kept]
            rmse = float(np.sqrt(np.mean(conditional ** 2))) if accepted else None
            filtered_linear = float(np.sqrt(np.mean(linear_error[kept] ** 2))) if accepted else None
            rows.append(dict(angle_deg=angle, range_1_m=r, range_2_m=s, bearing_sd_deg=1.,
                             reject_within_deg=cutoff_deg, samples=samples, accepted=accepted,
                             rejection_rate=1. - accepted / samples, theoretical_drms_m=float(theory),
                             linear_p95_m=float(np.quantile(linear_error, .95)),
                             unfiltered_p95_m=float(np.quantile(error, .95)),
                             conditional_rmse_m=rmse, conditional_linear_rmse_m=filtered_linear,
                             conditional_p95_m=float(np.quantile(conditional, .95)) if accepted else None,
                             conditional_to_linear_ratio=rmse / filtered_linear if filtered_linear else None))
    return rows


def pointing_variance_curve():
    """Shares of first-order horizontal bearing variance at the specified 40 m range."""
    s = np.arange(0., 101.)
    variances = np.column_stack((np.rad2deg((.3 + .02 * s) / 40.) ** 2,
                                  np.full(len(s), .8 ** 2), np.full(len(s), .6 ** 2)))
    shares = variances / variances.sum(axis=1, keepdims=True)
    return [dict(distance_m=float(d), position=float(p), zero_offset=float(z), control=float(c))
            for d, (p, z, c) in zip(s, shares)]
