# Velocity Control Interface for Ruckig
# Targets final velocity (ignoring position) with optional time synchronization
# Ported from C++ reference: ruckig/src/ruckig/velocity_third_step1.cpp, velocity_third_step2.cpp

#=============================================================================
 Third-Order Velocity Control (Jerk-Limited)
=============================================================================#

"""
    check_for_velocity!(buf, jf, aMax, aMin, limits, control_signs) -> Bool

Validate and finalize a velocity control profile. Sets jerk pattern and integrates
forward to compute velocity/acceleration at phase boundaries.

Unlike position control, velocity control only checks acceleration limits (no position target).
"""
function check_for_velocity!(buf::ProfileBuffer{T}, v0, a0, vf, af, jf, aMax, aMin, limits::ReachedLimits, control_signs::ControlSigns) where T
    # Check all times are non-negative
    for i in 1:7
        buf.t[i] < 0 && return false
    end

    # For ACC0 limit, t[2] (coast at aMax) must be positive
    if limits == LIMIT_ACC0
        buf.t[2] < EPS && return false
    end

    # Compute cumulative times
    buf.t_sum[1] = buf.t[1]
    for i in 2:7
        buf.t_sum[i] = buf.t_sum[i-1] + buf.t[i]
    end

    # Set jerk pattern based on control signs
    if control_signs == UDDU
        buf.j[1] = buf.t[1] > 0 ? jf : zero(T)
        buf.j[2] = zero(T)
        buf.j[3] = buf.t[3] > 0 ? -jf : zero(T)
        buf.j[4] = zero(T)
        buf.j[5] = buf.t[5] > 0 ? -jf : zero(T)
        buf.j[6] = zero(T)
        buf.j[7] = buf.t[7] > 0 ? jf : zero(T)
    else  # UDUD
        buf.j[1] = buf.t[1] > 0 ? jf : zero(T)
        buf.j[2] = zero(T)
        buf.j[3] = buf.t[3] > 0 ? -jf : zero(T)
        buf.j[4] = zero(T)
        buf.j[5] = buf.t[5] > 0 ? jf : zero(T)
        buf.j[6] = zero(T)
        buf.j[7] = buf.t[7] > 0 ? -jf : zero(T)
    end

    # Set initial state
    buf.a[1] = a0
    buf.v[1] = v0
    buf.p[1] = zero(T)  # Position not tracked for velocity control

    # Integrate forward
    for i in 1:7
        ti = buf.t[i]
        ji = buf.j[i]
        ai = buf.a[i]
        vi = buf.v[i]
        pi = buf.p[i]

        buf.a[i+1] = ai + ti * ji
        buf.v[i+1] = vi + ti * (ai + ti * ji / 2)
        buf.p[i+1] = pi + ti * (vi + ti * (ai / 2 + ti * ji / 6))
    end

    # Check acceleration limits at boundaries (velocity control doesn't check position)
    # Note: aMax/aMin may be swapped for DOWN direction, so use max/min to handle both orderings
    aUppLim = max(aMax, aMin) + A_PRECISION
    aLowLim = min(aMax, aMin) - A_PRECISION

    # Check acceleration at phase boundaries 1, 3, 5 (after jerk phases)
    for i in (2, 4, 6)
        (buf.a[i] > aUppLim || buf.a[i] < aLowLim) && return false
    end

    # Check final velocity
    abs(buf.v[8] - vf) > V_PRECISION && return false

    # Check final acceleration
    abs(buf.a[8] - af) > A_PRECISION && return false

    buf.limits = limits
    buf.control_signs = control_signs
    return true
end

"""
    time_acc0_velocity!(buf, v0, a0, vf, af, aMax, aMin, jMax) -> Bool

ACC0 velocity profile: accelerate to aMax, coast at constant acceleration, decelerate to af.

Profile structure: [jMax, 0, -jMax, 0, 0, 0, 0]
- Phase 0: jerk to reach aMax
- Phase 1: coast at aMax (constant acceleration)
- Phase 2: jerk to reach af
"""
function time_acc0_velocity!(buf::ProfileBuffer{T}, v0, a0, vf, af, aMax, aMin, jMax) where T
    vd = vf - v0

    buf.t[1] = (-a0 + aMax) / jMax
    buf.t[2] = (a0^2 + af^2) / (2 * aMax * jMax) - aMax / jMax + vd / aMax
    buf.t[3] = (-af + aMax) / jMax
    buf.t[4] = zero(T)
    buf.t[5] = zero(T)
    buf.t[6] = zero(T)
    buf.t[7] = zero(T)

    return check_for_velocity!(buf, v0, a0, vf, af, jMax, aMax, aMin, LIMIT_ACC0, UDDU)
end

"""
    time_none_velocity!(buf, v0, a0, vf, af, aMax, aMin, jMax) -> Bool

NONE velocity profile: no acceleration limit reached. Two solutions possible.

Profile structure: [jMax, 0, -jMax, 0, 0, 0, 0]
"""
function time_none_velocity!(buf::ProfileBuffer{T}, v0, a0, vf, af, aMax, aMin, jMax) where T
    vd = vf - v0

    h1 = (a0^2 + af^2) / 2 + jMax * vd
    h1 < 0 && return false
    h1 = sqrt(h1)

    # Solution 1: negative root
    buf.t[1] = -(a0 + h1) / jMax
    buf.t[2] = zero(T)
    buf.t[3] = -(af + h1) / jMax
    buf.t[4] = zero(T)
    buf.t[5] = zero(T)
    buf.t[6] = zero(T)
    buf.t[7] = zero(T)

    if check_for_velocity!(buf, v0, a0, vf, af, jMax, aMax, aMin, LIMIT_NONE, UDDU)
        return true
    end

    # Solution 2: positive root
    buf.t[1] = (-a0 + h1) / jMax
    buf.t[2] = zero(T)
    buf.t[3] = (-af + h1) / jMax
    buf.t[4] = zero(T)
    buf.t[5] = zero(T)
    buf.t[6] = zero(T)
    buf.t[7] = zero(T)

    return check_for_velocity!(buf, v0, a0, vf, af, jMax, aMax, aMin, LIMIT_NONE, UDDU)
end

"""
    time_all_single_step_velocity!(buf, v0, a0, vf, af, aMax, aMin) -> Bool

Special case for velocity control when jMax == 0 (infinite jerk / second-order dynamics).
Only valid when af ≈ a0 (no acceleration change possible without jerk).
"""
function time_all_single_step_velocity!(buf::ProfileBuffer{T}, v0, a0, vf, af, aMax, aMin) where T
    # Without jerk, acceleration cannot change
    abs(af - a0) > EPS && return false

    vd = vf - v0

    for i in 1:7
        buf.t[i] = zero(T)
    end

    if abs(a0) > EPS
        # Constant acceleration: vd = a0 * t
        buf.t[4] = vd / a0
        return check_for_velocity!(buf, v0, a0, vf, af, zero(T), aMax, aMin, LIMIT_NONE, UDDU)
    elseif abs(vd) < EPS
        # Already at target velocity with zero acceleration
        return check_for_velocity!(buf, v0, a0, vf, af, zero(T), aMax, aMin, LIMIT_NONE, UDDU)
    end

    return false
end

#=============================================================================
 Third-Order Velocity Control Step 2 (Time Synchronization)
=============================================================================#

"""
    time_acc0_velocity_step2!(buf, tf, v0, a0, vf, af, aMax, aMin, jMax) -> Bool

Step 2 ACC0 velocity profile: find profile that reaches (vf, af) in exactly time tf.
Multiple solution strategies are tried.
"""
function time_acc0_velocity_step2!(buf::ProfileBuffer{T}, tf, v0, a0, vf, af, aMax, aMin, jMax) where T
    vd = vf - v0
    ad = af - a0

    # UD Solution 1/2: symmetric jerk phases
    disc = (-ad^2 + 2*jMax*((a0 + af)*tf - 2*vd)) / (jMax^2) + tf^2
    if disc >= 0
        h1 = sqrt(disc)

        buf.t[1] = ad/(2*jMax) + (tf - h1)/2
        buf.t[2] = h1
        buf.t[3] = tf - (buf.t[1] + h1)
        buf.t[4] = zero(T)
        buf.t[5] = zero(T)
        buf.t[6] = zero(T)
        buf.t[7] = zero(T)

        if check_for_velocity!(buf, v0, a0, vf, af, jMax, aMax, aMin, LIMIT_ACC0, UDDU)
            return true
        end
    end

    # UU Solution: two jerk phases of same sign
    h1 = -ad + jMax*tf
    if abs(h1) > EPS
        buf.t[1] = -ad^2/(2*jMax*h1) + (vd - a0*tf)/h1
        buf.t[2] = -ad/jMax + tf
        buf.t[3] = zero(T)
        buf.t[4] = zero(T)
        buf.t[5] = zero(T)
        buf.t[6] = zero(T)
        buf.t[7] = tf - (buf.t[1] + buf.t[2])

        if check_for_velocity!(buf, v0, a0, vf, af, jMax, aMax, aMin, LIMIT_ACC0, UDDU)
            return true
        end
    end

    # UU Solution - 2 step: coast phase only
    buf.t[1] = zero(T)
    buf.t[2] = -ad/jMax + tf
    buf.t[3] = zero(T)
    buf.t[4] = zero(T)
    buf.t[5] = zero(T)
    buf.t[6] = zero(T)
    buf.t[7] = ad/jMax

    return check_for_velocity!(buf, v0, a0, vf, af, jMax, aMax, aMin, LIMIT_ACC0, UDDU)
end

"""
    time_none_velocity_step2!(buf, tf, v0, a0, vf, af, aMax, aMin, jMax) -> Bool

Step 2 NONE velocity profile: find profile with no acceleration limit in exactly time tf.
"""
function time_none_velocity_step2!(buf::ProfileBuffer{T}, tf, v0, a0, vf, af, aMax, aMin, jMax) where T
    vd = vf - v0
    ad = af - a0

    # Special case: already at target
    if abs(a0) < EPS && abs(af) < EPS && abs(vd) < EPS
        buf.t[1] = zero(T)
        buf.t[2] = tf
        buf.t[3] = zero(T)
        buf.t[4] = zero(T)
        buf.t[5] = zero(T)
        buf.t[6] = zero(T)
        buf.t[7] = zero(T)

        if check_for_velocity!(buf, v0, a0, vf, af, jMax, aMax, aMin, LIMIT_NONE, UDDU)
            return true
        end
    end

    # UD Solution with reduced jerk
    h1 = 2*(af*tf - vd)
    if abs(h1) > EPS && abs(ad) > EPS
        buf.t[1] = h1/ad
        buf.t[2] = tf - buf.t[1]
        buf.t[3] = zero(T)
        buf.t[4] = zero(T)
        buf.t[5] = zero(T)
        buf.t[6] = zero(T)
        buf.t[7] = zero(T)

        jf = ad^2/h1

        # Check that required jerk is within limits
        if abs(jf) < abs(jMax) + EPS
            if check_for_velocity!(buf, v0, a0, vf, af, jf, aMax, aMin, LIMIT_NONE, UDDU)
                return true
            end
        end
    end

    return false
end

#=============================================================================
 Second-Order Velocity Control (Acceleration-Limited, No Jerk)
=============================================================================#

"""
    time_velocity_second_order_step1!(buf, v0, vf, aMax, aMin) -> Bool

Second-order velocity Step 1: constant acceleration profile.
"""
function time_velocity_second_order_step1!(buf::ProfileBuffer{T}, v0, vf, aMax, aMin) where T
    vd = vf - v0

    # Choose acceleration direction
    af = vd > 0 ? aMax : aMin

    for i in 1:7
        buf.t[i] = zero(T)
    end

    buf.t[2] = vd / af

    # Check non-negative time
    buf.t[2] < 0 && return false

    # Compute cumulative times
    buf.t_sum[1] = buf.t[1]
    for i in 2:7
        buf.t_sum[i] = buf.t_sum[i-1] + buf.t[i]
    end

    # Set jerk to zero (second-order)
    for i in 1:7
        buf.j[i] = zero(T)
    end

    # Set acceleration pattern
    buf.a[1] = zero(T)
    buf.a[2] = buf.t[2] > 0 ? af : zero(T)
    for i in 3:8
        buf.a[i] = zero(T)
    end

    # Set velocity
    buf.v[1] = v0
    for i in 1:7
        buf.v[i+1] = buf.v[i] + buf.t[i] * buf.a[i]
    end

    # Set position (not tracked for velocity control, but fill for consistency)
    buf.p[1] = zero(T)
    for i in 1:7
        buf.p[i+1] = buf.p[i] + buf.t[i] * (buf.v[i] + buf.t[i] * buf.a[i] / 2)
    end

    buf.limits = LIMIT_ACC0
    buf.control_signs = UDDU
    return true
end

"""
    time_velocity_second_order_step2!(buf, tf, v0, vf, aMax, aMin) -> Bool

Second-order velocity Step 2: constant acceleration profile for specified duration.
"""
function time_velocity_second_order_step2!(buf::ProfileBuffer{T}, tf, v0, vf, aMax, aMin) where T
    vd = vf - v0
    af = vd / tf

    # Check acceleration limits
    (af > aMax + A_PRECISION || af < aMin - A_PRECISION) && return false

    for i in 1:7
        buf.t[i] = zero(T)
    end

    buf.t[2] = tf

    # Compute cumulative times
    buf.t_sum[1] = buf.t[1]
    for i in 2:7
        buf.t_sum[i] = buf.t_sum[i-1] + buf.t[i]
    end

    # Set jerk to zero (second-order)
    for i in 1:7
        buf.j[i] = zero(T)
    end

    # Set acceleration pattern
    buf.a[1] = zero(T)
    buf.a[2] = af
    for i in 3:8
        buf.a[i] = zero(T)
    end

    # Set velocity
    buf.v[1] = v0
    for i in 1:7
        buf.v[i+1] = buf.v[i] + buf.t[i] * buf.a[i]
    end

    # Set position
    buf.p[1] = zero(T)
    for i in 1:7
        buf.p[i+1] = buf.p[i] + buf.t[i] * (buf.v[i] + buf.t[i] * buf.a[i] / 2)
    end

    buf.limits = LIMIT_NONE
    buf.control_signs = UDDU
    return true
end

#=============================================================================
 API Functions
=============================================================================#

"""
    calculate_velocity_trajectory(lim::JerkLimiter; vf, v0=0, a0=0, af=0, tf=nothing)

Calculate velocity trajectory from (v0, a0) to (vf, af).
Uses jerk-limited third-order dynamics.

If `tf` is not specified, computes the minimum-time trajectory.
If `tf` is specified, computes a trajectory with exactly that duration.

This is the "velocity interface" - it targets a final velocity and acceleration
rather than a final position.

# Returns
`RuckigProfile` representing the velocity trajectory. Position in the profile
represents the displacement that would occur during the trajectory.
"""
function calculate_velocity_trajectory(lim::JerkLimiter{T}; vf, v0=zero(T), a0=zero(T), af=zero(T), tf=nothing) where T
    (; vmax, vmin, amax, amin, jmax, buffer, brake) = lim
    buf = buffer
    clear!(buf)

    # Compute velocity brake profile if initial acceleration is outside limits
    get_velocity_brake_trajectory!(brake, a0, amax, amin, jmax)
    _, vs, as = finalize_brake!(brake, zero(T), v0, a0)
    brake_duration = brake.duration
    brake_copy = brake_duration > 0 ? deepcopy(brake) : nothing

    v0_eff, a0_eff = vs, as
    vd = vf - v0_eff

    if isnothing(tf)
        # Time-optimal case
        # Zero-limits special case
        if jmax == 0
            if time_all_single_step_velocity!(buf, v0_eff, a0_eff, vf, af, amax, amin)
                return RuckigProfile(buf, zero(T), vf, af; brake_duration, brake=brake_copy)
            end
            error("Velocity trajectory not found for zero-jerk case")
        end

        # Try profiles based on direction
        if af == 0
            # No blocked interval when af==0, return after first found profile
            if vd >= 0
                # Try UP direction first
                if time_none_velocity!(buf, v0_eff, a0_eff, vf, af, amax, amin, jmax) ||
                   time_acc0_velocity!(buf, v0_eff, a0_eff, vf, af, amax, amin, jmax) ||
                   time_none_velocity!(buf, v0_eff, a0_eff, vf, af, amin, amax, -jmax) ||
                   time_acc0_velocity!(buf, v0_eff, a0_eff, vf, af, amin, amax, -jmax)
                    buf.direction = vd >= 0 ? DIR_UP : DIR_DOWN
                    return RuckigProfile(buf, zero(T), vf, af; brake_duration, brake=brake_copy)
                end
            else
                # Try DOWN direction first
                if time_none_velocity!(buf, v0_eff, a0_eff, vf, af, amin, amax, -jmax) ||
                   time_acc0_velocity!(buf, v0_eff, a0_eff, vf, af, amin, amax, -jmax) ||
                   time_none_velocity!(buf, v0_eff, a0_eff, vf, af, amax, amin, jmax) ||
                   time_acc0_velocity!(buf, v0_eff, a0_eff, vf, af, amax, amin, jmax)
                    buf.direction = vd >= 0 ? DIR_UP : DIR_DOWN
                    return RuckigProfile(buf, zero(T), vf, af; brake_duration, brake=brake_copy)
                end
            end
        else
            # af != 0: try all profiles (need to collect for potential blocked intervals)
            # For simplicity in min-time case, just return first valid
            if time_none_velocity!(buf, v0_eff, a0_eff, vf, af, amax, amin, jmax) ||
               time_none_velocity!(buf, v0_eff, a0_eff, vf, af, amin, amax, -jmax) ||
               time_acc0_velocity!(buf, v0_eff, a0_eff, vf, af, amax, amin, jmax) ||
               time_acc0_velocity!(buf, v0_eff, a0_eff, vf, af, amin, amax, -jmax)
                buf.direction = vd >= 0 ? DIR_UP : DIR_DOWN
                return RuckigProfile(buf, zero(T), vf, af; brake_duration, brake=brake_copy)
            end
        end

        error("Velocity trajectory not found (v0=$v0, a0=$a0, vf=$vf, af=$af)")
    else
        # Time-synchronized case
        tf_eff = tf - brake_duration
        tf_eff < 0 && error("Target duration $tf is less than brake duration $brake_duration")

        # Try profiles based on direction
        if vd >= 0
            # Try UP direction first
            if time_acc0_velocity_step2!(buf, tf_eff, v0_eff, a0_eff, vf, af, amax, amin, jmax) ||
               time_none_velocity_step2!(buf, tf_eff, v0_eff, a0_eff, vf, af, amax, amin, jmax) ||
               time_acc0_velocity_step2!(buf, tf_eff, v0_eff, a0_eff, vf, af, amin, amax, -jmax) ||
               time_none_velocity_step2!(buf, tf_eff, v0_eff, a0_eff, vf, af, amin, amax, -jmax)
                buf.direction = DIR_UP
                return RuckigProfile(buf, zero(T), vf, af; brake_duration, brake=brake_copy)
            end
        else
            # Try DOWN direction first
            if time_acc0_velocity_step2!(buf, tf_eff, v0_eff, a0_eff, vf, af, amin, amax, -jmax) ||
               time_none_velocity_step2!(buf, tf_eff, v0_eff, a0_eff, vf, af, amin, amax, -jmax) ||
               time_acc0_velocity_step2!(buf, tf_eff, v0_eff, a0_eff, vf, af, amax, amin, jmax) ||
               time_none_velocity_step2!(buf, tf_eff, v0_eff, a0_eff, vf, af, amax, amin, jmax)
                buf.direction = DIR_DOWN
                return RuckigProfile(buf, zero(T), vf, af; brake_duration, brake=brake_copy)
            end
        end

        error("Velocity trajectory with duration $tf not found (v0=$v0, a0=$a0, vf=$vf, af=$af)")
    end
end

"""
    calculate_velocity_trajectory(lim::AccelerationLimiter; vf, v0=0, tf=nothing)

Calculate velocity trajectory using second-order dynamics (no jerk limit).

If `tf` is not specified, computes the minimum-time trajectory.
If `tf` is specified, computes a trajectory with exactly that duration.

# Returns
`RuckigProfile` representing the velocity trajectory.
"""
function calculate_velocity_trajectory(lim::AccelerationLimiter{T}; vf, v0=zero(T), tf=nothing) where T
    (; amax, amin, buffer) = lim
    buf = buffer
    clear!(buf)

    if isnothing(tf)
        # Time-optimal case
        if time_velocity_second_order_step1!(buf, v0, vf, amax, amin)
            return RuckigProfile(buf, zero(T), vf, zero(T))
        end
        error("Second-order velocity trajectory not found (v0=$v0, vf=$vf)")
    else
        # Time-synchronized case
        if time_velocity_second_order_step2!(buf, tf, v0, vf, amax, amin)
            return RuckigProfile(buf, zero(T), vf, zero(T))
        end
        error("Second-order velocity trajectory with duration $tf not found (v0=$v0, vf=$vf)")
    end
end

# Export velocity control functions
export calculate_velocity_trajectory
