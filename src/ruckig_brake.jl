# Brake Profiles for Ruckig
# Handles initial states outside kinematic limits by pre-pending a braking trajectory
# Ported from C++ reference: ruckig/src/ruckig/brake.cpp

#=============================================================================
 Constants
=============================================================================#

const BRAKE_EPS = 2.2e-14  # Matching C++ BrakeProfile::eps

#=============================================================================
 Helper Functions
=============================================================================#

"""Velocity at time t given initial conditions and constant jerk."""
@inline v_at_t(v0, a0, j, t) = v0 + t * (a0 + j * t / 2)

"""Velocity when acceleration reaches zero (a0 + j*t = 0 => t = -a0/j)."""
@inline v_at_a_zero(v0, a0, j) = v0 + (a0 * a0) / (2 * j)

"""Integrate kinematic state over time t with constant jerk j."""
@inline function integrate_brake(t, p0, v0, a0, j)
    p = p0 + t * (v0 + t * (a0 / 2 + t * j / 6))
    v = v0 + t * (a0 + t * j / 2)
    a = a0 + t * j
    return (p, v, a)
end

#=============================================================================
 BrakeProfile Struct
=============================================================================#

"""
    BrakeProfile{T}

Two-phase braking trajectory to bring initial state within kinematic limits.
Used when a0 > aMax, a0 < aMin, v0 > vMax, or v0 < vMin.
"""
mutable struct BrakeProfile{T}
    duration::T
    t::NTuple{2,T}    # Phase times
    j::NTuple{2,T}    # Jerk values
    a::NTuple{2,T}    # Acceleration at phase start
    v::NTuple{2,T}    # Velocity at phase start
    p::NTuple{2,T}    # Position at phase start
end

function BrakeProfile{T}() where T
    z = zero(T)
    BrakeProfile{T}(z, (z, z), (z, z), (z, z), (z, z), (z, z))
end

#=============================================================================
 Brake Computation Functions
=============================================================================#

"""
Acceleration brake: called when a0 > aMax (or a0 < aMin with swapped limits).
Applies negative jerk to reduce acceleration to within limits.
"""
function acceleration_brake!(bp::BrakeProfile{T}, v0, a0, vMax, vMin, aMax, aMin, jMax) where T
    j0 = -jMax

    t_to_a_max = (a0 - aMax) / jMax
    t_to_a_zero = a0 / jMax

    v_at_a_max = v_at_t(v0, a0, -jMax, t_to_a_max)
    v_at_a_zero_val = v_at_t(v0, a0, -jMax, t_to_a_zero)

    if (v_at_a_zero_val > vMax && jMax > 0) || (v_at_a_zero_val < vMax && jMax < 0)
        # Velocity limit would be violated - use velocity brake instead
        velocity_brake!(bp, v0, a0, vMax, vMin, aMax, aMin, jMax)

    elseif (v_at_a_max < vMin && jMax > 0) || (v_at_a_max > vMin && jMax < 0)
        # Need second phase to handle velocity limit
        t_to_v_min = -(v_at_a_max - vMin) / aMax
        t_to_v_max = -aMax / (2 * jMax) - (v_at_a_max - vMax) / aMax

        t0 = t_to_a_max + BRAKE_EPS
        t1 = max(min(t_to_v_min, t_to_v_max - BRAKE_EPS), zero(T))
        bp.t = (t0, t1)
        bp.j = (j0, zero(T))
    else
        # Single phase is sufficient
        t0 = t_to_a_max + BRAKE_EPS
        bp.t = (t0, zero(T))
        bp.j = (j0, zero(T))
    end
end

"""
Velocity brake: called when v0 > vMax (or v0 < vMin with swapped limits).
Applies jerk to reduce velocity while respecting acceleration limits.
"""
function velocity_brake!(bp::BrakeProfile{T}, v0, a0, vMax, vMin, aMax, aMin, jMax) where T
    j0 = -jMax

    t_to_a_min = (a0 - aMin) / jMax

    # Time to reach vMax or vMin with constant jerk
    disc_vmax = a0^2 + 2 * jMax * (v0 - vMax)
    disc_vmin = a0^2 / 2 + jMax * (v0 - vMin)

    t_to_v_max = disc_vmax >= 0 ? a0/jMax + sqrt(disc_vmax) / abs(jMax) : T(Inf)
    t_to_v_min = disc_vmin >= 0 ? a0/jMax + sqrt(disc_vmin) / abs(jMax) : T(Inf)
    t_min_to_v_max = min(t_to_v_max, t_to_v_min)

    if t_to_a_min < t_min_to_v_max
        # Reach aMin before velocity limit - need second phase with constant acceleration
        v_at_a_min = v_at_t(v0, a0, -jMax, t_to_a_min)
        t_to_v_max_with_constant = -(v_at_a_min - vMax) / aMin
        t_to_v_min_with_constant = aMin / (2 * jMax) - (v_at_a_min - vMin) / aMin

        t0 = max(t_to_a_min - BRAKE_EPS, zero(T))
        t1 = max(min(t_to_v_max_with_constant, t_to_v_min_with_constant), zero(T))
        bp.t = (t0, t1)
        bp.j = (j0, zero(T))
    else
        # Single jerk phase sufficient
        t0 = max(t_min_to_v_max - BRAKE_EPS, zero(T))
        bp.t = (t0, zero(T))
        bp.j = (j0, zero(T))
    end
end

"""
    get_position_brake_trajectory!(bp, v0, a0, vMax, vMin, aMax, aMin, jMax)

Main entry point for computing third-order position brake trajectory.
Determines which type of braking is needed based on initial state and limits.
"""
function get_position_brake_trajectory!(bp::BrakeProfile{T}, v0, a0, vMax, vMin, aMax, aMin, jMax) where T
    # Reset to zero
    bp.t = (zero(T), zero(T))
    bp.j = (zero(T), zero(T))

    # Ignore braking for zero-limits
    if jMax == 0 || aMax == 0 || aMin == 0
        return
    end

    if a0 > aMax
        # Acceleration too high - need to reduce
        acceleration_brake!(bp, v0, a0, vMax, vMin, aMax, aMin, jMax)

    elseif a0 < aMin
        # Acceleration too low - need to increase (swap limits, negate jMax)
        acceleration_brake!(bp, v0, a0, vMin, vMax, aMin, aMax, -jMax)

    elseif (v0 > vMax && v_at_a_zero(v0, a0, -jMax) > vMin) || (a0 > 0 && v_at_a_zero(v0, a0, jMax) > vMax)
        # Velocity too high - need to reduce
        velocity_brake!(bp, v0, a0, vMax, vMin, aMax, aMin, jMax)

    elseif (v0 < vMin && v_at_a_zero(v0, a0, jMax) < vMax) || (a0 < 0 && v_at_a_zero(v0, a0, -jMax) < vMin)
        # Velocity too low - need to increase (swap limits, negate jMax)
        velocity_brake!(bp, v0, a0, vMin, vMax, aMin, aMax, -jMax)
    end
end

"""
    get_second_order_position_brake_trajectory!(bp, v0, vMax, vMin, aMax, aMin)

Compute second-order (acceleration-limited, no jerk limit) position brake trajectory.
"""
function get_second_order_position_brake_trajectory!(bp::BrakeProfile{T}, v0, vMax, vMin, aMax, aMin) where T
    # Reset to zero
    bp.t = (zero(T), zero(T))
    bp.j = (zero(T), zero(T))
    bp.a = (zero(T), zero(T))

    # Ignore braking for zero-limits
    if aMax == 0 || aMin == 0
        return
    end

    if v0 > vMax
        bp.a = (aMin, zero(T))
        t0 = (vMax - v0) / aMin + BRAKE_EPS
        bp.t = (t0, zero(T))

    elseif v0 < vMin
        bp.a = (aMax, zero(T))
        t0 = (vMin - v0) / aMax + BRAKE_EPS
        bp.t = (t0, zero(T))
    end
end

"""
    get_velocity_brake_trajectory!(bp, a0, aMax, aMin, jMax)

Compute third-order velocity (acceleration-only target) brake trajectory.
"""
function get_velocity_brake_trajectory!(bp::BrakeProfile{T}, a0, aMax, aMin, jMax) where T
    # Reset to zero
    bp.t = (zero(T), zero(T))
    bp.j = (zero(T), zero(T))

    # Ignore braking for zero-limits
    if jMax == 0
        return
    end

    if a0 > aMax
        bp.j = (-jMax, zero(T))
        t0 = (a0 - aMax) / jMax + BRAKE_EPS
        bp.t = (t0, zero(T))

    elseif a0 < aMin
        bp.j = (jMax, zero(T))
        t0 = -(a0 - aMin) / jMax + BRAKE_EPS
        bp.t = (t0, zero(T))
    end
end

"""
    get_second_order_velocity_brake_trajectory!(bp)

Compute second-order velocity brake trajectory (no-op for this interface).
"""
function get_second_order_velocity_brake_trajectory!(bp::BrakeProfile{T}) where T
    bp.t = (zero(T), zero(T))
    bp.j = (zero(T), zero(T))
end

#=============================================================================
 Finalization Functions
=============================================================================#

"""
    finalize_brake!(bp, ps, vs, as) -> (new_ps, new_vs, new_as)

Finalize third-order braking by integrating along kinematic state.
Returns the state after braking, updates bp with intermediate states.
"""
function finalize_brake!(bp::BrakeProfile{T}, ps, vs, as) where T
    if bp.t[1] <= 0 && bp.t[2] <= 0
        bp.duration = zero(T)
        return (ps, vs, as)
    end

    bp.duration = bp.t[1]
    bp.p = (ps, zero(T))
    bp.v = (vs, zero(T))
    bp.a = (as, zero(T))

    ps, vs, as = integrate_brake(bp.t[1], ps, vs, as, bp.j[1])

    if bp.t[2] > 0
        bp.duration += bp.t[2]
        bp.p = (bp.p[1], ps)
        bp.v = (bp.v[1], vs)
        bp.a = (bp.a[1], as)

        ps, vs, as = integrate_brake(bp.t[2], ps, vs, as, bp.j[2])
    end

    return (ps, vs, as)
end

"""
    finalize_second_order_brake!(bp, ps, vs, as) -> (new_ps, new_vs, new_as)

Finalize second-order braking by integrating along kinematic state.
"""
function finalize_second_order_brake!(bp::BrakeProfile{T}, ps, vs, as) where T
    if bp.t[1] <= 0
        bp.duration = zero(T)
        return (ps, vs, as)
    end

    bp.duration = bp.t[1]
    bp.p = (ps, zero(T))
    bp.v = (vs, zero(T))

    # Second order: jerk is zero, acceleration is constant
    ps, vs, as = integrate_brake(bp.t[1], ps, vs, bp.a[1], zero(T))

    return (ps, vs, as)
end
