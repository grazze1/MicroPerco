# SPDX-License-Identifier: Apache-2.0
"""Exact non-negative distances between supported convex particles."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from ..exceptions import GeometryError
from ..numerics import DEFAULT_NUMERICAL_POLICY, MACHINE_EPSILON, NumericalPolicy
from ..particles import (
    Cylinder,
    Particle,
    Sphere,
    _orthogonal_plane_basis,
    _radial_aabb_factors,
)

FloatArray = NDArray[np.float64]
SupportFunction = Callable[[FloatArray], FloatArray]
DistanceFallback = Callable[[FloatArray], float]
RationalVector = tuple[Fraction, Fraction, Fraction]


def _norm3(vector: ArrayLike) -> float:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("vector must have three components")
    return math.hypot(*(float(component) for component in array))


def _dot3(first: ArrayLike, second: ArrayLike) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    try:
        result = math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    except OverflowError as exc:
        raise GeometryError("vector projection is not representable") from exc
    if not math.isfinite(result):
        raise GeometryError("vector projection is not representable")
    return result


def sphere_sphere_distance(first: Sphere, second: Sphere) -> float:
    """Return the surface gap between two closed spheres."""

    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.asarray(second.center) - np.asarray(first.center)
    value = _norm3(delta)
    if not math.isfinite(value):
        raise GeometryError("sphere center separation is not representable")
    return max(0.0, value - first.radius - second.radius)


def point_cylinder_distance(point: ArrayLike, cylinder: Cylinder) -> float:
    """Return the distance from a point to a closed flat-ended cylinder."""

    position = np.asarray(point, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("point must be a finite three-vector")
    with np.errstate(over="ignore", invalid="ignore"):
        relative = position - np.asarray(cylinder.center)
    if not np.all(np.isfinite(relative)):
        raise GeometryError("point-cylinder separation is not representable")
    axial_coordinate = _dot3(relative, cylinder.axis)
    first, second = _orthogonal_plane_basis(cylinder.axis)
    radial_distance = math.hypot(_dot3(relative, first), _dot3(relative, second))
    axial_gap = max(abs(axial_coordinate) - cylinder.half_length, 0.0)
    radial_gap = max(radial_distance - cylinder.radius, 0.0)
    return math.hypot(axial_gap, radial_gap)


def sphere_cylinder_distance(sphere: Sphere, cylinder: Cylinder) -> float:
    """Return the surface gap between a sphere and a flat cylinder."""

    return max(0.0, point_cylinder_distance(sphere.center, cylinder) - sphere.radius)


def _cylinder_support_offset(
    cylinder: Cylinder,
    half_length: float,
    direction: FloatArray,
) -> FloatArray:
    """Return a stable support offset for a coaxial cylinder slice."""

    norm = _norm3(direction)
    if norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    unit = direction / norm
    axial_projection = _dot3(unit, cylinder.axis)
    if axial_projection > 0.0:
        axial_offset = half_length * cylinder.axis
    elif axial_projection < 0.0:
        axial_offset = -half_length * cylinder.axis
    else:
        axial_offset = np.zeros(3, dtype=np.float64)

    return np.asarray(
        axial_offset + _cylinder_radial_support_offset(cylinder, unit),
        dtype=np.float64,
    )


def _cylinder_radial_support_offset(cylinder: Cylinder, unit: FloatArray) -> FloatArray:
    """Return the radial part of a cylinder support point for a unit direction."""

    first_basis, second_basis = _orthogonal_plane_basis(cylinder.axis)
    first_component = _dot3(unit, first_basis)
    second_component = _dot3(unit, second_basis)
    perpendicular_norm = math.hypot(first_component, second_component)
    if perpendicular_norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(
        cylinder.radius
        * (first_component * first_basis + second_component * second_basis)
        / perpendicular_norm,
        dtype=np.float64,
    )


def _clipped_axis_interval(
    projection_center: float,
    projection_radius: float,
    half_length: float,
) -> tuple[float, float]:
    """Return center and half-width after clipping a projection interval."""

    low = min(half_length, max(-half_length, projection_center - projection_radius))
    high = min(half_length, max(-half_length, projection_center + projection_radius))
    return 0.5 * low + 0.5 * high, 0.5 * (high - low)


def _closest_free_axis_parameters(
    base: FloatArray,
    first_axis: FloatArray,
    first_half_length: float,
    second_axis: FloatArray,
    second_half_length: float,
) -> tuple[float, float]:
    """Minimize ``|base + s*u - t*v|`` over two bounded axis intervals."""

    candidates: list[tuple[float, float]] = []

    def clipped(value: float, half_length: float) -> float:
        return min(half_length, max(-half_length, value))

    for first_parameter in (-first_half_length, first_half_length):
        second_parameter = clipped(
            _dot3(base + first_parameter * first_axis, second_axis),
            second_half_length,
        )
        candidates.append((first_parameter, second_parameter))
    for second_parameter in (-second_half_length, second_half_length):
        first_parameter = clipped(
            -_dot3(base - second_parameter * second_axis, first_axis),
            first_half_length,
        )
        candidates.append((first_parameter, second_parameter))

    axis_dot = _dot3(first_axis, second_axis)
    cross_squared = _norm3(np.cross(first_axis, second_axis)) ** 2
    if cross_squared > 0.0:
        first_projection = _dot3(base, first_axis)
        second_projection = _dot3(base, second_axis)
        first_parameter = (axis_dot * second_projection - first_projection) / cross_squared
        second_parameter = (second_projection - axis_dot * first_projection) / cross_squared
        if (
            -first_half_length <= first_parameter <= first_half_length
            and -second_half_length <= second_parameter <= second_half_length
        ):
            candidates.append((first_parameter, second_parameter))
    return min(
        candidates,
        key=lambda pair: (
            _norm3(base + pair[0] * first_axis - pair[1] * second_axis),
            pair,
        ),
    )


def _project_to_local_cylinder(
    point: FloatArray,
    center: FloatArray,
    cylinder: Cylinder,
) -> FloatArray:
    """Project a local-coordinate point onto a closed finite cylinder."""

    relative = point - center
    axial = min(
        cylinder.half_length,
        max(-cylinder.half_length, _dot3(relative, cylinder.axis)),
    )
    first_basis, second_basis = _orthogonal_plane_basis(cylinder.axis)
    first_component = _dot3(relative, first_basis)
    second_component = _dot3(relative, second_basis)
    radial_norm = math.hypot(first_component, second_component)
    radial_scale = 1.0 if radial_norm <= cylinder.radius else cylinder.radius / radial_norm
    return np.asarray(
        center
        + axial * cylinder.axis
        + radial_scale * (first_component * first_basis + second_component * second_basis),
        dtype=np.float64,
    )


def _cylinders_intersect_by_projection(
    first: Cylinder,
    second: Cylinder,
    center_delta: FloatArray,
    tolerance: float,
) -> bool:
    """Find an overlap witness with deterministic Dykstra projections."""

    origin = np.zeros(3, dtype=np.float64)
    starts = (
        origin,
        center_delta,
        0.5 * center_delta,
        first.half_length * first.axis,
        -first.half_length * first.axis,
        center_delta + second.half_length * second.axis,
        center_delta - second.half_length * second.axis,
    )
    roundoff = (
        64.0
        * MACHINE_EPSILON
        * max(
            first.length,
            first.radius,
            second.length,
            second.radius,
            _norm3(center_delta),
            1.0,
        )
    )
    for start in starts:
        current = np.asarray(start, dtype=np.float64).copy()
        first_correction = np.zeros(3, dtype=np.float64)
        second_correction = np.zeros(3, dtype=np.float64)
        previous_gap = math.inf
        stalled = 0
        for _ in range(512):
            argument = current + first_correction
            on_first = _project_to_local_cylinder(argument, origin, first)
            first_correction = argument - on_first
            argument = on_first + second_correction
            on_second = _project_to_local_cylinder(argument, center_delta, second)
            second_correction = argument - on_second
            gap = _norm3(on_first - on_second)
            if gap <= tolerance:
                return True
            if abs(previous_gap - gap) <= roundoff:
                stalled += 1
                if stalled >= 12:
                    break
            else:
                stalled = 0
            previous_gap = gap
            current = on_second
    return False


def _segment_closest_data(
    start_a: ArrayLike,
    end_a: ArrayLike,
    start_b: ArrayLike,
    end_b: ArrayLike,
) -> tuple[float, float, float]:
    """Return distance and normalized closest parameters for two segments."""

    p0, p1, q0, q1 = (
        np.asarray(value, dtype=np.float64) for value in (start_a, end_a, start_b, end_b)
    )
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (p0, p1, q0, q1)):
        raise ValueError("segment endpoints must be finite three-vectors")

    def converted(vector: FloatArray) -> RationalVector:
        return (
            Fraction.from_float(float(vector[0])),
            Fraction.from_float(float(vector[1])),
            Fraction.from_float(float(vector[2])),
        )

    def subtract(left: RationalVector, right: RationalVector) -> RationalVector:
        return (
            left[0] - right[0],
            left[1] - right[1],
            left[2] - right[2],
        )

    def add_scaled(
        point: RationalVector,
        vector: RationalVector,
        scale: Fraction,
    ) -> RationalVector:
        return (
            point[0] + scale * vector[0],
            point[1] + scale * vector[1],
            point[2] + scale * vector[2],
        )

    def dot(left: RationalVector, right: RationalVector) -> Fraction:
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]

    def cross(left: RationalVector, right: RationalVector) -> RationalVector:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    rational_p0, rational_p1, rational_q0, rational_q1 = (
        converted(value) for value in (p0, p1, q0, q1)
    )
    u = subtract(rational_p1, rational_p0)
    v = subtract(rational_q1, rational_q0)

    def point_segment(
        point: RationalVector,
        start: RationalVector,
        vector: RationalVector,
    ) -> tuple[Fraction, Fraction]:
        relative = subtract(point, start)
        squared_length = dot(vector, vector)
        if squared_length == 0:
            return dot(relative, relative), Fraction(0)
        parameter = min(Fraction(1), max(Fraction(0), dot(relative, vector) / squared_length))
        residual = subtract(point, add_scaled(start, vector, parameter))
        return dot(residual, residual), parameter

    p0_distance, p0_on_b = point_segment(rational_p0, rational_q0, v)
    p1_distance, p1_on_b = point_segment(rational_p1, rational_q0, v)
    q0_distance, q0_on_a = point_segment(rational_q0, rational_p0, u)
    q1_distance, q1_on_a = point_segment(rational_q1, rational_p0, u)
    zero = Fraction(0)
    one = Fraction(1)
    candidates = [
        (p0_distance, zero, p0_on_b),
        (p1_distance, one, p1_on_b),
        (q0_distance, q0_on_a, zero),
        (q1_distance, q1_on_a, one),
    ]
    normal = cross(u, v)
    normal_squared = dot(normal, normal)
    if normal_squared != 0:
        delta = subtract(rational_q0, rational_p0)
        parameter_u = dot(cross(delta, v), normal) / normal_squared
        parameter_v = dot(cross(delta, u), normal) / normal_squared
        if zero <= parameter_u <= one and zero <= parameter_v <= one:
            first_point = add_scaled(rational_p0, u, parameter_u)
            second_point = add_scaled(rational_q0, v, parameter_v)
            residual = subtract(first_point, second_point)
            candidates.append((dot(residual, residual), parameter_u, parameter_v))
    squared_distance, rational_first, rational_second = min(candidates)
    with localcontext() as context:
        context.prec = 80
        decimal_squared = Decimal(squared_distance.numerator) / Decimal(
            squared_distance.denominator
        )
        result = (
            float(decimal_squared.sqrt()),
            float(rational_first),
            float(rational_second),
        )
    if not all(math.isfinite(value) for value in result):
        raise GeometryError("segment separation is not representable")
    return result


def segment_segment_distance(
    start_a: ArrayLike,
    end_a: ArrayLike,
    start_b: ArrayLike,
    end_b: ArrayLike,
) -> float:
    """Return the distance between two closed segments."""

    return _segment_closest_data(start_a, end_a, start_b, end_b)[0]


def _segment_intersects_aabb(
    start: ArrayLike,
    end: ArrayLike,
    low: ArrayLike,
    high: ArrayLike,
) -> bool:
    """Return whether a segment intersects a possibly degenerate AABB."""

    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (start, end, low, high))
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in arrays):
        raise GeometryError("segment-box coordinates must be finite three-vectors")
    segment_start, segment_end, box_low, box_high = arrays
    parameter_low = Fraction(0)
    parameter_high = Fraction(1)
    for index in range(3):
        coordinate = Fraction.from_float(float(segment_start[index]))
        direction = Fraction.from_float(float(segment_end[index])) - coordinate
        lower = Fraction.from_float(float(box_low[index]))
        upper = Fraction.from_float(float(box_high[index]))
        if direction == 0:
            if coordinate < lower or coordinate > upper:
                return False
            continue
        first_parameter = (lower - coordinate) / direction
        second_parameter = (upper - coordinate) / direction
        if first_parameter > second_parameter:
            first_parameter, second_parameter = second_parameter, first_parameter
        parameter_low = max(parameter_low, first_parameter)
        parameter_high = min(parameter_high, second_parameter)
        if parameter_low > parameter_high:
            return False
    return True


def _closest_simplex_point(
    points: Sequence[FloatArray], *, feasibility_tolerance: float
) -> tuple[FloatArray, list[FloatArray]]:
    """Project the origin onto a simplex by enumerating active faces."""

    best_norm_squared = math.inf
    best_point: FloatArray | None = None
    best_face: list[FloatArray] | None = None
    for face_size in range(1, len(points) + 1):
        for indices in combinations(range(len(points)), face_size):
            matrix = np.column_stack([points[index] for index in indices])
            if face_size == 1:
                weights = np.ones(1, dtype=np.float64)
            else:
                # Express the affine hull as p0 + D*mu. Solving directly with
                # D avoids both squaring its condition number through D.T@D
                # and mixing tiny Gram entries with a unit KKT constraint.
                # Those two effects otherwise lose genuine intersections for
                # valid high-aspect-ratio convex bodies.
                base = matrix[:, 0]
                differences = matrix[:, 1:] - base[:, np.newaxis]
                try:
                    coordinates = np.linalg.lstsq(differences, -base, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
                weights = np.concatenate(
                    (np.array((1.0 - float(np.sum(coordinates)),)), coordinates)
                )
            if float(np.min(weights)) < -feasibility_tolerance:
                continue
            weights = np.maximum(weights, 0.0)
            total = float(np.sum(weights))
            if total <= feasibility_tolerance:
                continue
            weights /= total
            candidate = matrix @ weights
            # Matrix-vector cancellation can leave a component of order eps
            # times its summands even when the affine projection is exactly
            # zero on that coordinate. Such a residue can point a support
            # query at the remote end of a very long body. Remove only values
            # covered by a componentwise forward-error bound; a genuinely
            # small coordinate assembled from small summands is retained.
            component_roundoff = (
                64.0
                * MACHINE_EPSILON
                * np.sum(
                    np.abs(matrix) * np.abs(weights)[np.newaxis, :],
                    axis=1,
                )
            )
            candidate = np.where(np.abs(candidate) <= component_roundoff, 0.0, candidate)
            norm_squared = float(np.dot(candidate, candidate))
            if norm_squared < best_norm_squared:
                active = [
                    index for index, weight in enumerate(weights) if weight > feasibility_tolerance
                ]
                if not active:
                    active = [int(np.argmax(weights))]
                best_norm_squared = norm_squared
                best_point = candidate
                best_face = [points[indices[index]] for index in active]
    if best_point is None or best_face is None:
        index = int(np.argmin([float(np.dot(point, point)) for point in points]))
        return points[index], [points[index]]
    return best_point, best_face


def _project_normalized_cylinder_coordinates(values: ArrayLike) -> FloatArray:
    """Project six axial/disk coordinates onto their convex feasible set."""

    coordinates = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
    for first, second in ((1, 2), (4, 5)):
        radial_norm = math.hypot(
            float(coordinates[first]),
            float(coordinates[second]),
        )
        if radial_norm > 1.0:
            coordinates[first] /= radial_norm
            coordinates[second] /= radial_norm
    return np.asarray(coordinates, dtype=np.float64)


def _certified_cylinder_distance_fallback(
    first: Cylinder,
    second: Cylinder,
    effective_delta: FloatArray,
    first_half_length: float,
    second_half_length: float,
    scale: float,
    gjk_closest: FloatArray,
    policy: NumericalPolicy,
) -> float:
    """Close a stalled GJK bracket with convex primal and dual programs.

    The primal selects one point from each normalized cylinder.  The dual
    minimizes their Minkowski support function over the unit ball.  Both
    programs are convex, and every feasible pair gives a global distance
    bracket.  Optimizer success alone is therefore never treated as a
    certificate: the returned value must also close that bracket to the
    configured geometry tolerance.
    """

    gjk_point = np.asarray(gjk_closest, dtype=np.float64)
    if gjk_point.shape != (3,) or not np.all(np.isfinite(gjk_point)):
        raise GeometryError("GJK supplied an invalid closest point")
    tolerance = policy.tolerance(scale) / scale
    first_basis = _orthogonal_plane_basis(first.axis)
    second_basis = _orthogonal_plane_basis(second.axis)
    center = -effective_delta / scale
    coefficients = (
        np.column_stack(
            (
                first_half_length * first.axis,
                first.radius * first_basis[0],
                first.radius * first_basis[1],
                -second_half_length * second.axis,
                -second.radius * second_basis[0],
                -second.radius * second_basis[1],
            )
        )
        / scale
    )

    def primal_vector(values: ArrayLike) -> FloatArray:
        return np.asarray(
            center + coefficients @ np.asarray(values, dtype=np.float64),
            dtype=np.float64,
        )

    def primal_objective(values: FloatArray) -> float:
        vector = primal_vector(values)
        return 0.5 * _dot3(vector, vector)

    def primal_gradient(values: FloatArray) -> FloatArray:
        return np.asarray(coefficients.T @ primal_vector(values), dtype=np.float64)

    def first_disk_constraint(values: FloatArray) -> float:
        return 1.0 - float(values[1]) ** 2 - float(values[2]) ** 2

    def first_disk_jacobian(values: FloatArray) -> FloatArray:
        return np.asarray(
            (0.0, -2.0 * values[1], -2.0 * values[2], 0.0, 0.0, 0.0),
            dtype=np.float64,
        )

    def second_disk_constraint(values: FloatArray) -> float:
        return 1.0 - float(values[4]) ** 2 - float(values[5]) ** 2

    def second_disk_jacobian(values: FloatArray) -> FloatArray:
        return np.asarray(
            (0.0, 0.0, 0.0, 0.0, -2.0 * values[4], -2.0 * values[5]),
            dtype=np.float64,
        )

    try:
        least_squares_start = np.linalg.lstsq(coefficients, -center, rcond=None)[0]
    except np.linalg.LinAlgError:
        least_squares_start = np.zeros(6, dtype=np.float64)
    primal_starts = (
        _project_normalized_cylinder_coordinates(least_squares_start),
        np.zeros(6, dtype=np.float64),
    )
    primal_options = {
        "maxiter": policy.gjk_max_iterations,
        "ftol": max(64.0 * MACHINE_EPSILON, 0.01 * tolerance * tolerance),
        "disp": False,
    }
    constraints = (
        {"type": "ineq", "fun": first_disk_constraint, "jac": first_disk_jacobian},
        {"type": "ineq", "fun": second_disk_constraint, "jac": second_disk_jacobian},
    )
    normalized_magnitude = max(
        1.0,
        _norm3(center),
        float(np.max(np.abs(coefficients))),
    )
    best_upper = _norm3(gjk_point) + (1024.0 * MACHINE_EPSILON * normalized_magnitude)
    best_primal_vector = np.array(gjk_point, copy=True)
    for start in primal_starts:
        result = minimize(
            primal_objective,
            start,
            jac=primal_gradient,
            method="SLSQP",
            bounds=((-1.0, 1.0),) * 6,
            constraints=constraints,
            options=primal_options,
        )
        if (
            not result.success
            or not np.all(np.isfinite(result.x))
            or not math.isfinite(float(result.fun))
        ):
            continue
        raw = np.asarray(result.x, dtype=np.float64)
        violation = max(
            0.0,
            -first_disk_constraint(raw),
            -second_disk_constraint(raw),
        )
        if violation > max(math.sqrt(tolerance), 256.0 * MACHINE_EPSILON):
            continue
        feasible = _project_normalized_cylinder_coordinates(raw)
        vector = primal_vector(feasible)
        component_magnitude = np.abs(center) + np.abs(coefficients) @ np.abs(feasible)
        evaluation_roundoff = (
            256.0
            * MACHINE_EPSILON
            * max(
                1.0,
                float(np.sum(component_magnitude)),
            )
        )
        upper = _norm3(vector) + evaluation_roundoff
        if math.isfinite(upper) and upper < best_upper:
            best_upper = upper
            best_primal_vector = vector
    if best_upper <= tolerance:
        return 0.0

    def support_value_and_subgradient(direction: FloatArray) -> tuple[float, FloatArray]:
        projections = np.asarray(coefficients.T @ direction, dtype=np.float64)
        first_radial = math.hypot(float(projections[1]), float(projections[2]))
        second_radial = math.hypot(float(projections[4]), float(projections[5]))
        value = math.fsum(
            (
                _dot3(center, direction),
                abs(float(projections[0])),
                first_radial,
                abs(float(projections[3])),
                second_radial,
            )
        )
        subgradient = np.array(center, copy=True)
        if projections[0] != 0.0:
            subgradient += math.copysign(1.0, float(projections[0])) * coefficients[:, 0]
        if first_radial > 0.0:
            subgradient += (
                projections[1] * coefficients[:, 1] + projections[2] * coefficients[:, 2]
            ) / first_radial
        if projections[3] != 0.0:
            subgradient += math.copysign(1.0, float(projections[3])) * coefficients[:, 3]
        if second_radial > 0.0:
            subgradient += (
                projections[4] * coefficients[:, 4] + projections[5] * coefficients[:, 5]
            ) / second_radial
        return value, np.asarray(subgradient, dtype=np.float64)

    def dual_objective(direction: FloatArray) -> float:
        return support_value_and_subgradient(direction)[0]

    def dual_gradient(direction: FloatArray) -> FloatArray:
        return support_value_and_subgradient(direction)[1]

    def unit_ball_constraint(direction: FloatArray) -> float:
        return 1.0 - _dot3(direction, direction)

    def unit_ball_jacobian(direction: FloatArray) -> FloatArray:
        return np.asarray(-2.0 * direction, dtype=np.float64)

    vector_norm = _norm3(best_primal_vector)
    dual_starts = [
        -best_primal_vector / vector_norm,
        np.zeros(3, dtype=np.float64),
    ]
    center_norm = _norm3(center)
    if center_norm > 0.0:
        dual_starts.append(-center / center_norm)
    dual_starts.extend(
        sign * np.eye(3, dtype=np.float64)[axis] for axis in range(3) for sign in (-1.0, 1.0)
    )
    dual_options = {
        "maxiter": policy.gjk_max_iterations,
        "ftol": max(64.0 * MACHINE_EPSILON, 0.01 * tolerance),
        "disp": False,
    }
    best_lower = 0.0
    dual_converged = False
    for start in dual_starts:
        result = minimize(
            dual_objective,
            start,
            jac=dual_gradient,
            method="SLSQP",
            bounds=((-1.0, 1.0),) * 3,
            constraints={
                "type": "ineq",
                "fun": unit_ball_constraint,
                "jac": unit_ball_jacobian,
            },
            options=dual_options,
        )
        if (
            not result.success
            or not np.all(np.isfinite(result.x))
            or not math.isfinite(float(result.fun))
        ):
            continue
        direction = np.asarray(result.x, dtype=np.float64)
        direction_norm = _norm3(direction)
        if direction_norm > 1.0:
            direction /= direction_norm
        value, support_point = support_value_and_subgradient(direction)
        if not math.isfinite(value):
            continue
        projections = np.asarray(coefficients.T @ direction, dtype=np.float64)
        magnitude = math.fsum(
            [1.0, abs(_dot3(center, direction))]
            + [abs(float(component)) for component in projections]
        )
        evaluation_roundoff = 256.0 * MACHINE_EPSILON * magnitude
        support_upper = _norm3(support_point) + evaluation_roundoff
        if math.isfinite(support_upper):
            best_upper = min(best_upper, support_upper)
        best_lower = max(best_lower, max(0.0, -value - evaluation_roundoff))
        dual_converged = True
    if not dual_converged:
        raise GeometryError("cylinder distance fallback dual optimization did not converge")

    certificate_roundoff = (
        512.0
        * MACHINE_EPSILON
        * max(
            1.0,
            best_upper,
            best_lower,
            _norm3(center),
            float(np.max(np.abs(coefficients))),
        )
    )
    if best_lower > best_upper + certificate_roundoff:
        raise GeometryError("cylinder distance fallback produced an invalid primal-dual bracket")
    if best_upper - best_lower > tolerance + certificate_roundoff:
        raise GeometryError(
            "cylinder distance fallback did not certify the requested geometry tolerance"
        )
    return max(0.0, best_upper)


def _certified_support_map_distance_fallback(
    minkowski_support: SupportFunction,
    scale: float,
    gjk_closest: FloatArray,
    policy: NumericalPolicy,
    shape_label: str,
) -> float:
    """Certify a stalled support-map query with a convex dual bracket.

    For a normalized Minkowski body ``K``, minimizing its support function
    ``h_K(u)`` over the unit ball is the convex dual of distance to ``K``.
    Every feasible direction gives the lower bound ``-h_K(u)``; the support
    point returned at that direction is itself in ``K`` and therefore gives
    an upper bound.  Consequently optimizer success is accepted only when
    those independently valid bounds close to the requested tolerance.
    """

    closest = np.asarray(gjk_closest, dtype=np.float64)
    if closest.shape != (3,) or not np.all(np.isfinite(closest)):
        raise GeometryError("GJK supplied an invalid closest point")
    tolerance = policy.tolerance(scale) / scale
    closest_norm = _norm3(closest)
    best_upper = closest_norm + 1024.0 * MACHINE_EPSILON * max(1.0, closest_norm)
    if best_upper <= tolerance:
        return 0.0

    def support_value_and_point(direction: FloatArray) -> tuple[float, FloatArray]:
        point = np.asarray(minkowski_support(direction), dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise GeometryError(f"{shape_label} support point is not finite")
        return _dot3(direction, point), point

    def dual_objective(direction: FloatArray) -> float:
        return support_value_and_point(direction)[0]

    def dual_gradient(direction: FloatArray) -> FloatArray:
        return support_value_and_point(direction)[1]

    def unit_ball_constraint(direction: FloatArray) -> float:
        return 1.0 - _dot3(direction, direction)

    def unit_ball_jacobian(direction: FloatArray) -> FloatArray:
        return np.asarray(-2.0 * direction, dtype=np.float64)

    dual_starts = [
        -closest / closest_norm,
        np.zeros(3, dtype=np.float64),
    ]
    center_point = np.asarray(
        minkowski_support(np.zeros(3, dtype=np.float64)),
        dtype=np.float64,
    )
    center_norm = _norm3(center_point)
    if center_norm > 0.0:
        dual_starts.append(-center_point / center_norm)
    dual_starts.extend(
        sign * np.eye(3, dtype=np.float64)[axis] for axis in range(3) for sign in (-1.0, 1.0)
    )
    options = {
        "maxiter": policy.gjk_max_iterations,
        "ftol": max(64.0 * MACHINE_EPSILON, 0.01 * tolerance),
        "disp": False,
    }
    best_lower = 0.0
    converged = False
    for start in dual_starts:
        result = minimize(
            dual_objective,
            start,
            jac=dual_gradient,
            method="SLSQP",
            bounds=((-1.0, 1.0),) * 3,
            constraints={
                "type": "ineq",
                "fun": unit_ball_constraint,
                "jac": unit_ball_jacobian,
            },
            options=options,
        )
        if (
            not result.success
            or not np.all(np.isfinite(result.x))
            or not math.isfinite(float(result.fun))
        ):
            continue
        direction = np.asarray(result.x, dtype=np.float64)
        direction_norm = _norm3(direction)
        if direction_norm > 1.0:
            direction /= direction_norm
        value, support_point = support_value_and_point(direction)
        support_norm = _norm3(support_point)
        evaluation_roundoff = (
            512.0
            * MACHINE_EPSILON
            * max(
                1.0,
                support_norm,
                abs(value),
            )
        )
        best_upper = min(best_upper, support_norm + evaluation_roundoff)
        best_lower = max(best_lower, max(0.0, -value - evaluation_roundoff))
        converged = True
    if not converged:
        raise GeometryError(f"{shape_label} distance fallback dual did not converge")

    certificate_roundoff = (
        1024.0
        * MACHINE_EPSILON
        * max(
            1.0,
            best_upper,
            best_lower,
        )
    )
    if best_lower > best_upper + certificate_roundoff:
        raise GeometryError(f"{shape_label} distance fallback produced an invalid bracket")
    if best_upper - best_lower > tolerance + certificate_roundoff:
        raise GeometryError(
            f"{shape_label} distance fallback did not certify the requested tolerance"
        )
    return max(0.0, best_upper)


def _gjk_distance_normalized(
    minkowski_support: SupportFunction,
    initial_direction: FloatArray,
    scale: float,
    policy: NumericalPolicy,
    *,
    fallback: DistanceFallback | None = None,
) -> float:
    """Solve a scaled support-map GJK distance query."""

    tolerance = policy.tolerance(scale) / scale
    feasibility_tolerance = max(32.0 * np.finfo(np.float64).eps, tolerance * 0.01)
    if _norm3(initial_direction) <= tolerance:
        initial_direction = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    simplex = [minkowski_support(initial_direction)]
    closest = simplex[0]
    for _ in range(policy.gjk_max_iterations):
        upper = _norm3(closest)
        if upper <= tolerance:
            return 0.0
        support = minkowski_support(-closest)
        lower = float(np.dot(closest, support)) / upper
        if upper - max(0.0, lower) <= tolerance:
            return max(0.0, upper * scale)
        if any(_norm3(support - vertex) <= tolerance for vertex in simplex):
            if fallback is not None and policy.gjk_max_iterations > 1:
                return fallback(closest) * scale
            raise GeometryError(
                "GJK stalled before its distance bracket reached the requested tolerance"
            )
        simplex.append(support)
        closest, simplex = _closest_simplex_point(
            simplex, feasibility_tolerance=feasibility_tolerance
        )
        new_upper = _norm3(closest)
        if new_upper <= tolerance:
            return 0.0
    if fallback is not None and policy.gjk_max_iterations > 1:
        return fallback(closest) * scale
    raise GeometryError("GJK did not converge within the configured maximum iteration count")


def cylinder_cylinder_distance(
    first: Cylinder,
    second: Cylinder,
    *,
    policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY,
) -> float:
    """Return the GJK gap between finite flat-ended cylinders."""

    with np.errstate(over="ignore", invalid="ignore"):
        center_delta = np.asarray(second.center) - np.asarray(first.center)
    separation = _norm3(center_delta)
    if not math.isfinite(separation):
        raise GeometryError("cylinder center separation is not representable")
    if (
        point_cylinder_distance(first.center, second) == 0.0
        or point_cylinder_distance(second.center, first) == 0.0
    ):
        return 0.0
    if np.array_equal(first.axis, second.axis) or np.array_equal(first.axis, -second.axis):
        # Parallel cylinders are a Cartesian product of an axial interval and
        # a radial disk. This closed form is exact and avoids asking GJK to
        # retain tiny end/radius features beside arbitrarily long axes.
        axial_separation = abs(_dot3(center_delta, first.axis))
        first_basis, second_basis = _orthogonal_plane_basis(first.axis)
        radial_separation = math.hypot(
            _dot3(center_delta, first_basis),
            _dot3(center_delta, second_basis),
        )
        axial_gap = max(
            axial_separation - first.half_length - second.half_length,
            0.0,
        )
        radial_gap = max(radial_separation - first.radius - second.radius, 0.0)
        return math.hypot(axial_gap, radial_gap)

    feature_scale = max(first.length, first.radius, second.length, second.radius)
    minimum_feature = min(first.length, first.radius, second.length, second.radius)
    if feature_scale / minimum_feature >= 1.0e5:
        axis_distance, first_parameter, second_parameter = _segment_closest_data(
            *first.endpoints,
            *second.endpoints,
        )
        # At an interior closest pair, the axis-to-axis displacement is
        # perpendicular to both axes and therefore lies in both radial disk
        # planes.  Disk overlap is then an exact sufficient intersection
        # witness, including the nearly parallel regime where GJK can lose
        # the tiny radial feature beside a very long axial feature.
        if (
            0.0 < first_parameter < 1.0
            and 0.0 < second_parameter < 1.0
            and axis_distance <= first.radius + second.radius
        ):
            return 0.0
        if _cylinders_intersect_by_projection(
            first,
            second,
            center_delta,
            policy.tolerance(max(feature_scale, separation, 1.0)),
        ):
            return 0.0

    axis_dot = _dot3(first.axis, second.axis)
    axis_cross = _norm3(np.cross(first.axis, second.axis))
    first_projection_radius = second.half_length * abs(axis_dot) + second.radius * axis_cross
    second_projection_radius = first.half_length * abs(axis_dot) + first.radius * axis_cross
    if not math.isfinite(first_projection_radius) or not math.isfinite(second_projection_radius):
        raise GeometryError("cylinder projection is not representable")
    first_axis_center, first_half_length = _clipped_axis_interval(
        _dot3(center_delta, first.axis),
        first_projection_radius,
        first.half_length,
    )
    second_axis_center, second_half_length = _clipped_axis_interval(
        -_dot3(center_delta, second.axis),
        second_projection_radius,
        second.half_length,
    )
    try:
        effective_delta = np.asarray(
            [
                math.fsum(
                    (
                        float(center_delta[index]),
                        second_axis_center * float(second.axis[index]),
                        -first_axis_center * float(first.axis[index]),
                    )
                )
                for index in range(3)
            ],
            dtype=np.float64,
        )
    except OverflowError as exc:
        raise GeometryError("cylinder separation is not representable") from exc
    effective_separation = _norm3(effective_delta)
    if not math.isfinite(effective_separation):
        raise GeometryError("cylinder separation is not representable")
    scale = max(
        2.0 * first_half_length,
        2.0 * second_half_length,
        first.radius,
        second.radius,
        effective_separation,
        1.0,
    )

    def minkowski_support(direction: FloatArray) -> FloatArray:
        direction_norm = _norm3(direction)
        if direction_norm == 0.0:
            return -effective_delta / scale
        unit = direction / direction_norm
        radial_first = _cylinder_radial_support_offset(first, unit)
        radial_second = _cylinder_radial_support_offset(second, -unit)
        first_radial_tied = _norm3(radial_first) == 0.0
        second_radial_tied = _norm3(radial_second) == 0.0
        first_projection = _dot3(unit, first.axis)
        second_projection = -_dot3(unit, second.axis)
        first_tied = first_projection == 0.0
        second_tied = second_projection == 0.0

        def choose_axis_parameters(base: FloatArray) -> tuple[float, float]:
            first_parameter = (
                0.0 if first_tied else math.copysign(first_half_length, first_projection)
            )
            second_parameter = (
                0.0 if second_tied else math.copysign(second_half_length, second_projection)
            )
            if first_tied and second_tied:
                return _closest_free_axis_parameters(
                    base,
                    first.axis,
                    first_half_length,
                    second.axis,
                    second_half_length,
                )
            if first_tied:
                first_parameter = min(
                    first_half_length,
                    max(
                        -first_half_length,
                        -_dot3(
                            base - second_parameter * second.axis,
                            first.axis,
                        ),
                    ),
                )
            elif second_tied:
                second_parameter = min(
                    second_half_length,
                    max(
                        -second_half_length,
                        _dot3(
                            base + first_parameter * first.axis,
                            second.axis,
                        ),
                    ),
                )
            return first_parameter, second_parameter

        def free_radial_offset(
            residual: FloatArray,
            axis: FloatArray,
            radius: float,
            sign: float,
        ) -> FloatArray:
            projected = residual - _dot3(residual, axis) * axis
            target = sign * projected
            target_norm = _norm3(target)
            if target_norm <= radius:
                return np.asarray(target, dtype=np.float64)
            return np.asarray(radius * target / target_norm, dtype=np.float64)

        base = -effective_delta + radial_first - radial_second
        first_parameter, second_parameter = choose_axis_parameters(base)
        for _ in range(2):
            axial = first_parameter * first.axis - second_parameter * second.axis
            if first_radial_tied:
                residual_without_first = -effective_delta - radial_second + axial
                radial_first = free_radial_offset(
                    residual_without_first,
                    first.axis,
                    first.radius,
                    -1.0,
                )
            if second_radial_tied:
                residual_without_second = -effective_delta + radial_first + axial
                radial_second = free_radial_offset(
                    residual_without_second,
                    second.axis,
                    second.radius,
                    1.0,
                )
            base = -effective_delta + radial_first - radial_second
            first_parameter, second_parameter = choose_axis_parameters(base)
        return (base + first_parameter * first.axis - second_parameter * second.axis) / scale

    return _gjk_distance_normalized(
        minkowski_support,
        effective_delta / scale,
        scale,
        policy,
        fallback=lambda closest: _certified_cylinder_distance_fallback(
            first,
            second,
            effective_delta,
            first_half_length,
            second_half_length,
            scale,
            closest,
            policy,
        ),
    )


def point_rectangle_distance(
    point: ArrayLike, rectangle_center: ArrayLike, half_extents: ArrayLike
) -> float:
    """Return exact distance from a point to an axis-aligned rectangle."""

    position = np.asarray(point, dtype=np.float64)
    center = np.asarray(rectangle_center, dtype=np.float64)
    extents = np.asarray(half_extents, dtype=np.float64)
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (position, center)):
        raise ValueError("point and rectangle_center must be finite three-vectors")
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents < 0.0):
        raise ValueError("half_extents must be a finite non-negative three-vector")
    with np.errstate(over="ignore", invalid="ignore"):
        outside = np.maximum(np.abs(position - center) - extents, 0.0)
    result = _norm3(outside)
    if not math.isfinite(result):
        raise GeometryError("point-rectangle separation is not representable")
    return result


def sphere_rectangle_distance(
    sphere: Sphere, rectangle_center: ArrayLike, half_extents: ArrayLike
) -> float:
    """Return exact sphere-to-rectangle surface gap."""

    return max(
        0.0,
        point_rectangle_distance(sphere.center, rectangle_center, half_extents) - sphere.radius,
    )


def cylinder_rectangle_distance(
    cylinder: Cylinder,
    rectangle_center: ArrayLike,
    half_extents: ArrayLike,
    *,
    policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY,
) -> float:
    """Return GJK gap from a flat cylinder to an axis-aligned rectangle."""

    center = np.asarray(rectangle_center, dtype=np.float64)
    extents = np.asarray(half_extents, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("rectangle_center must be a finite three-vector")
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents < 0.0):
        raise ValueError("half_extents must be a finite non-negative three-vector")
    with np.errstate(over="ignore", invalid="ignore"):
        rectangle_low = center - extents
        rectangle_high = center + extents
    if not np.all(np.isfinite((rectangle_low, rectangle_high))):
        raise GeometryError("rectangle bounds must be representable as finite floats")
    if _segment_intersects_aabb(
        *cylinder.endpoints,
        rectangle_low,
        rectangle_high,
    ):
        return 0.0
    if point_cylinder_distance(center, cylinder) == 0.0:
        return 0.0
    if point_rectangle_distance(cylinder.center, center, extents) == 0.0:
        return 0.0

    # For a fixed rectangle point, the closest point on a right cylinder has
    # axial coordinate equal to the point's projection clamped to the finite
    # axis segment. The rectangle's projections form one interval, so clipping
    # the cylinder to that interval preserves the global closest pair while
    # removing irrelevant extreme length from the GJK conditioning.
    with np.errstate(over="ignore", invalid="ignore"):
        rectangle_delta = center - np.asarray(cylinder.center)
    if not np.all(np.isfinite(rectangle_delta)):
        raise GeometryError("cylinder-rectangle separation is not representable")
    projection_center = _dot3(rectangle_delta, cylinder.axis)
    try:
        projection_radius = math.fsum(
            float(extent) * abs(float(component))
            for extent, component in zip(extents, cylinder.axis, strict=True)
        )
    except OverflowError as exc:
        raise GeometryError("rectangle projection is not representable") from exc
    if not math.isfinite(projection_radius):
        raise GeometryError("rectangle projection is not representable")
    axial_center, axial_half_length = _clipped_axis_interval(
        projection_center,
        projection_radius,
        cylinder.half_length,
    )
    effective_center = np.asarray(cylinder.center) + axial_center * cylinder.axis
    effective_extent = axial_half_length * np.abs(
        cylinder.axis
    ) + cylinder.radius * _radial_aabb_factors(cylinder.axis)
    zero_axes = np.flatnonzero(extents == 0.0)
    if len(zero_axes) == 1:
        normal_axis = int(zero_axes[0])
        transverse = [index for index in range(3) if index != normal_axis]
        particle_low = effective_center - effective_extent
        particle_high = effective_center + effective_extent
        if all(
            particle_low[index] >= rectangle_low[index]
            and particle_high[index] <= rectangle_high[index]
            for index in transverse
        ):
            return max(
                0.0,
                abs(float(effective_center[normal_axis]) - float(center[normal_axis]))
                - float(effective_extent[normal_axis]),
            )
    cylinder_low = effective_center - effective_extent
    cylinder_high = effective_center + effective_extent
    clipped_low = np.maximum(rectangle_low, np.minimum(cylinder_low, rectangle_high))
    clipped_high = np.minimum(rectangle_high, np.maximum(cylinder_high, rectangle_low))
    clipped_center = clipped_low + 0.5 * (clipped_high - clipped_low)
    clipped_extents = 0.5 * (clipped_high - clipped_low)
    with np.errstate(over="ignore", invalid="ignore"):
        center_delta = clipped_center - effective_center
    separation = _norm3(center_delta)
    if not math.isfinite(separation):
        raise GeometryError("cylinder-rectangle separation is not representable")
    scale = max(
        2.0 * axial_half_length,
        cylinder.radius,
        separation,
        float(np.max(clipped_extents)),
        1.0,
    )

    def minkowski_support(direction: FloatArray) -> FloatArray:
        direction_norm = _norm3(direction)
        if direction_norm == 0.0:
            return -center_delta / scale
        unit = direction / direction_norm
        axial_projection = _dot3(unit, cylinder.axis)
        axial_tied = axial_projection == 0.0
        radial_offset = _cylinder_radial_support_offset(cylinder, unit)
        rectangle_offset = np.zeros(3, dtype=np.float64)
        rectangle_tied = unit == 0.0
        for index in range(3):
            if not rectangle_tied[index]:
                rectangle_offset[index] = math.copysign(
                    float(clipped_extents[index]),
                    -float(unit[index]),
                )
        rectangle_point = center_delta + rectangle_offset
        if axial_tied:
            axial_parameter = min(
                axial_half_length,
                max(
                    -axial_half_length,
                    _dot3(rectangle_point - radial_offset, cylinder.axis),
                ),
            )
        else:
            axial_parameter = math.copysign(axial_half_length, axial_projection)

        radial_projection = math.hypot(
            _dot3(unit, _orthogonal_plane_basis(cylinder.axis)[0]),
            _dot3(unit, _orthogonal_plane_basis(cylinder.axis)[1]),
        )
        if radial_projection == 0.0:
            desired = rectangle_point - axial_parameter * cylinder.axis
            desired -= _dot3(desired, cylinder.axis) * cylinder.axis
            desired_norm = _norm3(desired)
            if desired_norm <= cylinder.radius:
                radial_offset = desired
            elif desired_norm > 0.0:
                radial_offset = cylinder.radius * desired / desired_norm

        cylinder_point = radial_offset + axial_parameter * cylinder.axis
        for index in range(3):
            if rectangle_tied[index]:
                rectangle_offset[index] = min(
                    float(clipped_extents[index]),
                    max(
                        -float(clipped_extents[index]),
                        float(cylinder_point[index] - center_delta[index]),
                    ),
                )
        rectangle_point = center_delta + rectangle_offset
        if axial_tied:
            axial_parameter = min(
                axial_half_length,
                max(
                    -axial_half_length,
                    _dot3(rectangle_point - radial_offset, cylinder.axis),
                ),
            )
            cylinder_point = radial_offset + axial_parameter * cylinder.axis
        return (cylinder_point - rectangle_point) / scale

    return _gjk_distance_normalized(
        minkowski_support,
        center_delta / scale,
        scale,
        policy,
        fallback=lambda closest: _certified_support_map_distance_fallback(
            minkowski_support,
            scale,
            closest,
            policy,
            "cylinder-rectangle",
        ),
    )


def distance(
    first: Particle,
    second: Particle,
    *,
    policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY,
) -> float:
    """Return unified non-negative surface gap for supported particles."""

    if isinstance(first, Sphere) and isinstance(second, Sphere):
        return sphere_sphere_distance(first, second)
    if isinstance(first, Sphere) and isinstance(second, Cylinder):
        return sphere_cylinder_distance(first, second)
    if isinstance(first, Cylinder) and isinstance(second, Sphere):
        return sphere_cylinder_distance(second, first)
    if isinstance(first, Cylinder) and isinstance(second, Cylinder):
        return cylinder_cylinder_distance(first, second, policy=policy)
    raise TypeError("distance supports Sphere and Cylinder particles only")


sphere_sphere_gap = sphere_sphere_distance
cylinder_distance_gjk = cylinder_cylinder_distance


def cylinder_sphere_gap(cylinder: Cylinder, sphere: Sphere) -> float:
    return sphere_cylinder_distance(sphere, cylinder)
