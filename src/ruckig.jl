# Ruckig: Time-optimal jerk-limited trajectory generation
# Based on: Berscheid & Kröger, "Jerk-limited Real-time Trajectory Generation
# with Arbitrary Target States", 2021
# Reference implementation: https://github.com/pantor/ruckig
# License of reference: MIT License https://github.com/pantor/ruckig/blob/main/LICENSE

using StaticArrays

export JerkLimiter, RuckigProfile
export calculate_trajectory, evaluate_at

#=============================================================================
 Constants (matching reference implementation)
=============================================================================#

const EPS = eps(Float64)
const P_PRECISION = 1e-8
const V_PRECISION = 1e-8
const A_PRECISION = 1e-10
const T_PRECISION = 1e-12

#=============================================================================
 Enums
=============================================================================#

@enum ReachedLimits begin
    LIMIT_ACC0_ACC1_VEL
    LIMIT_ACC0_VEL
    LIMIT_ACC1_VEL
    LIMIT_VEL
    LIMIT_ACC0_ACC1
    LIMIT_ACC0
    LIMIT_ACC1
    LIMIT_NONE
end

@enum ControlSigns begin
    UDDU  # ↑↓↓↑
    UDUD  # ↑↓↑↓
end

#=============================================================================
 Data Structures
=============================================================================#

"""
    RuckigProfile{T}

A 7-phase jerk-limited trajectory profile.
"""
mutable struct RuckigProfile{T}
    t::MVector{7,T}       # Phase durations
    t_sum::MVector{7,T}   # Cumulative times
    j::MVector{7,T}       # Jerk values
    a::MVector{8,T}       # Acceleration at boundaries
    v::MVector{8,T}       # Velocity at boundaries
    p::MVector{8,T}       # Position at boundaries
    pf::T                 # Target position
    vf::T                 # Target velocity
    af::T                 # Target acceleration
    limits::ReachedLimits
    control_signs::ControlSigns

    function RuckigProfile{T}() where T
        new{T}(
            MVector{7,T}(zeros(T, 7)),
            MVector{7,T}(zeros(T, 7)),
            MVector{7,T}(zeros(T, 7)),
            MVector{8,T}(zeros(T, 8)),
            MVector{8,T}(zeros(T, 8)),
            MVector{8,T}(zeros(T, 8)),
            zero(T), zero(T), zero(T),
            LIMIT_NONE, UDDU
        )
    end
end

RuckigProfile(::Type{T}) where T = RuckigProfile{T}()

"""
    JerkLimiter{T}

Jerk-limited trajectory generator with directional limits.
"""
struct JerkLimiter{T}
    vmax::T
    vmin::T
    amax::T
    amin::T
    jmax::T
end

function JerkLimiter(vmax, vmin, amax, amin, jmax)
    args = promote(vmax, vmin, amax, amin, jmax)
    JerkLimiter(args...)
end

JerkLimiter(vmax, amax, jmax) = JerkLimiter(vmax, -vmax, amax, -amax, jmax)

#=============================================================================
 Polynomial Root Finding (matching reference implementation)
=============================================================================#

"""
Solve ax² + bx + c = 0 for real roots.
"""
function solve_quadratic_real(a, b, c)
    if abs(a) < EPS
        abs(b) < EPS && return Float64[]
        return [-c/b]
    end

    disc = b^2 - 4a*c
    disc < 0 && return Float64[]

    if disc < EPS
        return [-b / (2a)]
    end

    sqrt_disc = sqrt(disc)
    return [(-b - sqrt_disc) / (2a), (-b + sqrt_disc) / (2a)]
end

"""
Solve ax³ + bx² + cx + d = 0 for real roots using Cardano's formula.
"""
function solve_cubic_real(a, b, c, d)
    if abs(a) < EPS
        return solve_quadratic_real(b, c, d)
    end

    # Normalize
    p, q, r = b/a, c/a, d/a

    # Depressed cubic: t³ + pt + q = 0 via x = t - p/3
    aa = q - p^2/3
    bb = 2p^3/27 - p*q/3 + r

    disc = bb^2/4 + aa^3/27

    roots = Float64[]

    if disc > EPS
        u = cbrt(-bb/2 + sqrt(disc))
        v = cbrt(-bb/2 - sqrt(disc))
        push!(roots, u + v - p/3)
    elseif disc < -EPS
        m = 2 * sqrt(-aa/3)
        θ = acos(3bb / (aa * m)) / 3
        for k in 0:2
            push!(roots, m * cos(θ - 2π*k/3) - p/3)
        end
    else
        if abs(aa) < EPS
            push!(roots, -p/3)
        else
            push!(roots, 3bb/aa - p/3)
            push!(roots, -3bb/(2aa) - p/3)
        end
    end

    return roots
end

"""
Solve ax⁴ + bx³ + cx² + dx + e = 0 for real roots using Ferrari's method.
"""
function solve_quartic_real(a, b, c, d, e)
    if abs(a) < EPS
        return solve_cubic_real(b, c, d, e)
    end

    # Normalize
    p, q, r, s = b/a, c/a, d/a, e/a

    # Depressed quartic via x = t - p/4
    α = q - 3p^2/8
    β = r - p*q/2 + p^3/8
    γ = s - p*r/4 + p^2*q/16 - 3p^4/256

    roots = Float64[]

    if abs(β) < EPS
        # Biquadratic
        for t2 in solve_quadratic_real(1.0, α, γ)
            if t2 >= 0
                t = sqrt(t2)
                push!(roots, t - p/4)
                push!(roots, -t - p/4)
            end
        end
    else
        # Resolvent cubic
        cubic_roots = solve_cubic_real(1.0, α/2, (α^2 - 4γ)/16, -β^2/64)
        isempty(cubic_roots) && return roots

        y = maximum(abs, cubic_roots)
        # Find the root with correct sign
        for cr in cubic_roots
            if cr >= -EPS
                y = cr
                break
            end
        end
        y < -EPS && return roots
        y = max(y, 0.0)

        w = sqrt(y)
        abs(w) < EPS && return roots

        for sign1 in (-1, 1)
            discriminant = -(α + y + sign1 * β / w)
            if discriminant >= -EPS
                discriminant = max(discriminant, 0.0)
                for sign2 in (-1, 1)
                    t = sign1 * w + sign2 * sqrt(discriminant)
                    push!(roots, t - p/4)
                end
            end
        end
    end

    return unique(sort(roots))
end

#=============================================================================
 Profile Check Function (matching reference implementation exactly)
=============================================================================#

"""
    check!(profile, control_signs, limits, jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af) -> Bool

Validate profile: check times >= 0, integrate, verify limits and final state.
This matches the reference implementation's check() template function.
"""
function check!(profile::RuckigProfile{T}, control_signs::ControlSigns, limits::ReachedLimits,
                jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af=zero(T)) where T

    # Set jerk pattern based on control signs
    if control_signs == UDDU
        profile.j .= (jf, 0, -jf, 0, -jf, 0, jf)
    else  # UDUD
        profile.j .= (jf, 0, -jf, 0, jf, 0, -jf)
    end

    # Check all times non-negative
    @inbounds for i in 1:7
        profile.t[i] < -T_PRECISION && return false
        profile.t[i] = max(profile.t[i], zero(T))
    end

    # Integrate profile (Eq. 2-4 from paper)
    profile.a[1] = a0
    profile.v[1] = v0
    profile.p[1] = p0

    cumtime = zero(T)
    @inbounds for i in 1:7
        ti = profile.t[i]
        ji = profile.j[i]
        ai = profile.a[i]
        vi = profile.v[i]
        pi = profile.p[i]

        profile.a[i+1] = ai + ti * ji
        profile.v[i+1] = vi + ti * (ai + ti * ji / 2)
        profile.p[i+1] = pi + ti * (vi + ti * (ai / 2 + ti * ji / 6))

        cumtime += ti
        profile.t_sum[i] = cumtime
    end

    # Check final state
    abs(profile.p[8] - pf) > P_PRECISION && return false
    abs(profile.v[8] - vf) > V_PRECISION && return false
    abs(profile.a[8] - af) > A_PRECISION && return false

    # Check acceleration limits at critical points (indices 2, 4, 6 in 1-based = boundaries after phases 1, 3, 5)
    @inbounds for i in (2, 4, 6)
        (profile.a[i] > aMax + EPS || profile.a[i] < aMin - EPS) && return false
    end

    # Check velocity limits at critical points (indices 4-7 in 1-based)
    @inbounds for i in 4:7
        (profile.v[i] > vMax + EPS || profile.v[i] < vMin - EPS) && return false
    end

    # Check velocity at acceleration zero-crossings
    @inbounds for i in 3:6
        profile.t[i] < EPS && continue
        ai, ji = profile.a[i], profile.j[i]
        abs(ji) < EPS && continue

        # Time when acceleration crosses zero within this phase
        if ai * profile.a[i+1] < -EPS
            t_zero = -ai / ji
            if 0 < t_zero < profile.t[i]
                v_at_zero = profile.v[i] - ai^2 / (2ji)
                (v_at_zero > vMax + EPS || v_at_zero < vMin - EPS) && return false
            end
        end
    end

    profile.pf = pf
    profile.vf = vf
    profile.af = af
    profile.limits = limits
    profile.control_signs = control_signs

    return true
end

#=============================================================================
 Profile Time Calculations - UDDU (matching reference implementation)
=============================================================================#

"""
Try all velocity-limited profiles (ACC0_ACC1_VEL, ACC1_VEL, ACC0_VEL, VEL).
Returns true if any valid profile is found.
"""
function time_all_vel!(profile::RuckigProfile{T}, p0, v0, a0, pf, vf, af,
                       jMax, vMax, vMin, aMax, aMin) where T
    # Pre-compute common terms
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    af_p3 = af^3
    a0_p4 = a0^4
    af_p4 = af^4
    v0_v0 = v0^2
    vf_vf = vf^2
    pd = pf - p0

    # Strategy 1: ACC0_ACC1_VEL (reach aMax, vMax, aMin)
    begin
        profile.t[1] = (-a0 + aMax) / jMax
        profile.t[2] = (a0_a0/2 - aMax^2 - jMax*(v0 - vMax)) / (aMax * jMax)
        profile.t[3] = aMax / jMax
        profile.t[5] = -aMin / jMax
        profile.t[6] = -(af_af/2 - aMin^2 - jMax*(vf - vMax)) / (aMin * jMax)
        profile.t[7] = profile.t[5] + af / jMax

        # Compute t[4] from position constraint (equation from reference)
        profile.t[4] = (3*(a0_p4*aMin - af_p4*aMax) +
                        8*aMax*aMin*(af_p3 - a0_p3 + 3*jMax*(a0*v0 - af*vf)) +
                        6*a0_a0*aMin*(aMax^2 - 2*jMax*v0) -
                        6*af_af*aMax*(aMin^2 - 2*jMax*vf) -
                        12*jMax*(aMax*aMin*(aMax*(v0 + vMax) - aMin*(vf + vMax) - 2*jMax*pd) +
                                (aMin - aMax)*jMax*vMax^2 +
                                jMax*(aMax*vf_vf - aMin*v0_v0))) / (24*aMax*aMin*jMax_jMax*vMax)

        if check!(profile, UDDU, LIMIT_ACC0_ACC1_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Strategy 2: ACC1_VEL (reach vMax and aMin, not aMax)
    begin
        t_acc0 = sqrt(max(0.0, a0_a0/(2*jMax_jMax) + (vMax - v0)/jMax))
        profile.t[1] = t_acc0 - a0/jMax
        profile.t[2] = 0
        profile.t[3] = t_acc0
        profile.t[5] = -aMin / jMax
        profile.t[6] = -(af_af/2 - aMin^2 - jMax*(vf - vMax)) / (aMin * jMax)
        profile.t[7] = profile.t[5] + af / jMax

        t_acc1 = profile.t[7]
        profile.t[4] = (af_p3 - a0_p3)/(3*jMax_jMax*vMax) +
                       (a0*v0 - af*vf + (af_af*t_acc1 + a0_a0*t_acc0)/2)/(jMax*vMax) -
                       (v0/vMax + 1.0)*t_acc0 - (vf/vMax + 1.0)*t_acc1 + pd/vMax

        if check!(profile, UDDU, LIMIT_ACC1_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Strategy 3: ACC0_VEL (reach aMax and vMax, not aMin)
    begin
        profile.t[1] = (-a0 + aMax) / jMax
        profile.t[2] = (a0_a0/2 - aMax^2 - jMax*(v0 - vMax)) / (aMax * jMax)
        profile.t[3] = aMax / jMax

        t_acc1 = sqrt(max(0.0, af_af/(2*jMax_jMax) + (vMax - vf)/jMax))
        profile.t[5] = t_acc1
        profile.t[6] = 0
        profile.t[7] = t_acc1 + af/jMax

        t_acc0 = profile.t[1]
        profile.t[4] = (af_p3 - a0_p3)/(3*jMax_jMax*vMax) +
                       (a0*v0 - af*vf + (af_af*t_acc1 + a0_a0*t_acc0)/2)/(jMax*vMax) -
                       (v0/vMax + 1.0)*t_acc0 - (vf/vMax + 1.0)*t_acc1 + pd/vMax

        if check!(profile, UDDU, LIMIT_ACC0_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Strategy 4: VEL (reach vMax only, no acceleration limits)
    begin
        t_acc0 = sqrt(max(0.0, a0_a0/(2*jMax_jMax) + (vMax - v0)/jMax))
        t_acc1 = sqrt(max(0.0, af_af/(2*jMax_jMax) + (vMax - vf)/jMax))

        profile.t[1] = t_acc0 - a0/jMax
        profile.t[2] = 0
        profile.t[3] = t_acc0
        profile.t[5] = t_acc1
        profile.t[6] = 0
        profile.t[7] = t_acc1 + af/jMax

        profile.t[4] = (af_p3 - a0_p3)/(3*jMax_jMax*vMax) +
                       (a0*v0 - af*vf + (af_af*t_acc1 + a0_a0*t_acc0)/2)/(jMax*vMax) -
                       (v0/vMax + 1.0)*t_acc0 - (vf/vMax + 1.0)*t_acc1 + pd/vMax

        if check!(profile, UDDU, LIMIT_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
Try ACC0_ACC1 profile (reach both aMax and aMin, but not vMax).
"""
function time_acc0_acc1!(profile::RuckigProfile{T}, p0, v0, a0, pf, vf, af,
                         jMax, vMax, vMin, aMax, aMin) where T
    # Pre-compute common terms
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    af_p3 = af^3
    a0_p4 = a0^4
    af_p4 = af^4
    v0_v0 = v0^2
    vf_vf = vf^2
    pd = pf - p0

    # Compute h1 (from reference implementation)
    h1 = (3*(af_p4*aMax - a0_p4*aMin) +
          aMax*aMin*(8*(a0_p3 - af_p3) + 3*aMax*aMin*(aMax - aMin) + 6*aMin*af_af - 6*aMax*a0_a0) +
          12*jMax*(aMax*aMin*((aMax - 2*a0)*v0 - (aMin - 2*af)*vf) + aMin*a0_a0*v0 - aMax*af_af*vf)) /
         (3*(aMax - aMin)*jMax_jMax) +
         4*(aMax*vf_vf - aMin*v0_v0 - 2*aMin*aMax*pd) / (aMax - aMin)

    h1 < 0 && return false
    h1 = sqrt(h1) / 2

    h2 = a0_a0/(2*aMax*jMax) + (aMin - 2*aMax)/(2*jMax) - v0/aMax
    h3 = -af_af/(2*aMin*jMax) - (aMax - 2*aMin)/(2*jMax) + vf/aMin

    # Try two solutions (from reference implementation)
    # Solution 2: h2 > h1/aMax, h3 > -h1/aMin => t[2] = h2 - h1/aMax, t[6] = h3 + h1/aMin
    # Solution 1: h2 > -h1/aMax, h3 > h1/aMin => t[2] = h2 + h1/aMax, t[6] = h3 - h1/aMin
    for h1_sign in (1, -1)
        t1_cond = h2 > h1_sign * h1 / aMax
        t5_cond = h3 > -h1_sign * h1 / aMin

        t1_cond && t5_cond || continue

        profile.t[1] = (-a0 + aMax) / jMax
        profile.t[2] = h2 - h1_sign * h1 / aMax
        profile.t[3] = aMax / jMax
        profile.t[4] = 0
        profile.t[5] = -aMin / jMax
        profile.t[6] = h3 + h1_sign * h1 / aMin
        profile.t[7] = profile.t[5] + af / jMax

        if check!(profile, UDDU, LIMIT_ACC0_ACC1, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
Try ACC0, ACC1, and NONE profiles (no velocity limit reached).
"""
function time_all_none_acc0_acc1!(profile::RuckigProfile{T}, p0, v0, a0, pf, vf, af,
                                  jMax, vMax, vMin, aMax, aMin) where T
    # Pre-compute common terms
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    af_p3 = af^3
    v0_v0 = v0^2
    vf_vf = vf^2
    pd = pf - p0

    h2_none = af - a0
    h2_none_h2_none = h2_none^2

    # NONE profile: solve quartic
    # polynom_none[0]*t^3 + polynom_none[1]*t^2 + polynom_none[2]*t + polynom_none[3] = 0
    # Note: the reference uses a quartic but with coeff[0]=0, making it cubic in t
    polynom_none_1 = -2*(a0_a0 + af_af - 2*jMax*(v0 + vf)) / jMax_jMax
    polynom_none_2 = 4*(a0_p3 - af_p3 + 3*jMax*(af*vf - a0*v0)) / (3*jMax*jMax_jMax) - 4*pd/jMax
    polynom_none_3 = -h2_none_h2_none / jMax_jMax

    t_min_none = max((a0 - aMax)/jMax, (aMin - af)/jMax, 0.0)
    t_max_none = min((a0 - aMin)/jMax, (aMax - af)/jMax)

    if t_max_none > t_min_none
        # Solve cubic (since leading coeff is 0)
        for t1 in solve_cubic_real(1.0, polynom_none_1, polynom_none_2, polynom_none_3)
            t_min_none - EPS <= t1 <= t_max_none + EPS || continue

            profile.t[1] = t1
            profile.t[2] = 0
            profile.t[3] = (a0 + jMax*t1) / jMax  # Time to reach a=0 from a0+jMax*t1
            profile.t[4] = 0
            profile.t[5] = 0
            profile.t[6] = 0
            profile.t[7] = profile.t[3] - t1 + af/jMax

            # Correction: for NONE, t3 should be such that a goes to 0 then to af
            # From reference: t[3] = (a0 + t[1]*jMax)/jMax, and t[7] involves reaching af
            # Let me recalculate based on the structure

            # UDDU NONE profile: j = (+j, 0, -j, 0, -j, 0, +j) with t2=t4=t5=t6=0
            # After t1: a1 = a0 + j*t1
            # After t3: a3 = a1 - j*t3 = a0 + j*t1 - j*t3
            # After t7: a7 = a3 + j*t7 = a0 + j*t1 - j*t3 + j*t7 = af
            # For NONE, t5=0 means phase 5 is skipped
            # So: a0 + j*(t1 - t3 + t7) = af => t7 = t3 - t1 + (af - a0)/j

            profile.t[3] = (a0 + jMax*t1) / jMax
            profile.t[7] = profile.t[3] - t1 + (af - a0)/jMax

            if check!(profile, UDDU, LIMIT_NONE, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end

    # ACC0 profile: reaches aMax but not aMin or vMax
    # From reference, this involves a quartic polynomial
    h0_acc0 = 3*(a0_p3*a0 - af_p3*af) +
              8*(af_p3 - a0_p3)*aMax +
              24*aMax*jMax*(a0*v0 - af*vf) -
              6*a0_a0*(aMax^2 - 2*jMax*v0) +
              6*af_af*(aMax^2 + 2*jMax*vf) -
              12*jMax*(2*aMax*(jMax*pd - aMax*(v0 + vf)) - jMax*(v0_v0 - vf_vf))

    polynom_acc0_0 = -2*aMax/jMax
    polynom_acc0_1 = (-af_af + aMax^2 + 2*jMax*vf) / jMax_jMax
    polynom_acc0_2 = 0.0
    polynom_acc0_3 = h0_acc0 / (12*jMax_jMax*jMax_jMax)

    t_min_acc0 = max((aMin - af)/jMax, 0.0)
    t_max_acc0 = (aMax - af)/jMax

    if t_max_acc0 > t_min_acc0
        for t7 in solve_quartic_real(1.0, polynom_acc0_0, polynom_acc0_1, polynom_acc0_2, polynom_acc0_3)
            t_min_acc0 - EPS <= t7 <= t_max_acc0 + EPS || continue

            profile.t[1] = (-a0 + aMax) / jMax
            profile.t[2] = (a0_a0/2 - aMax^2 + af_af/2 + jMax*(-v0 + vf + aMax*t7) - af*jMax*t7 + jMax_jMax*t7^2/2) / (aMax*jMax)
            profile.t[3] = aMax / jMax
            profile.t[4] = 0
            profile.t[5] = 0
            profile.t[6] = 0
            profile.t[7] = t7

            if check!(profile, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end

    # ACC1 profile: reaches aMin but not aMax or vMax
    h0_acc1 = -3*(a0_p3*a0 - af_p3*af) +
              8*(a0_p3 - af_p3)*aMin +
              24*aMin*jMax*(af*vf - a0*v0) +
              6*a0_a0*(aMin^2 + 2*jMax*v0) -
              6*af_af*(aMin^2 - 2*jMax*vf) +
              12*jMax*(2*aMin*(jMax*pd + aMin*(v0 + vf)) + jMax*(v0_v0 - vf_vf))

    h2_acc1 = a0_a0 + 2*jMax*v0 + aMin*(aMin - 2*a0)

    polynom_acc1_0 = 2*(2*a0 - aMin)/jMax
    polynom_acc1_1 = (5*a0_a0 + aMin*(aMin - 6*a0) + 2*jMax*v0) / jMax_jMax
    polynom_acc1_2 = 2*(a0 - aMin)*h2_acc1 / (jMax*jMax_jMax)
    polynom_acc1_3 = h0_acc1 / jMax_jMax / jMax_jMax

    t_min_acc1 = max((a0 - aMax)/jMax, 0.0)
    t_max_acc1 = (a0 - aMin)/jMax

    if t_max_acc1 > t_min_acc1
        for t1 in solve_quartic_real(1.0, polynom_acc1_0, polynom_acc1_1, polynom_acc1_2, polynom_acc1_3)
            t_min_acc1 - EPS <= t1 <= t_max_acc1 + EPS || continue

            a1 = a0 + jMax*t1
            profile.t[1] = t1
            profile.t[2] = 0
            profile.t[3] = (a1 - aMin) / jMax
            profile.t[4] = 0
            profile.t[5] = -aMin / jMax
            profile.t[6] = (-af_af/2 + aMin^2 + a1^2/2 - jMax*(vf - v0 - a1*t1 + jMax*t1^2/2)) / (aMin*jMax)
            profile.t[7] = profile.t[5] + af/jMax

            if check!(profile, UDDU, LIMIT_ACC1, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end

    return false
end

#=============================================================================
 Main Profile Search
=============================================================================#

"""
    find_profile(lim, p0, v0, a0, pf, vf, af) -> RuckigProfile or nothing

Find a valid time-optimal profile. Tries profiles in order of complexity.
"""
function find_profile(lim::JerkLimiter{T}, p0, v0, a0, pf, vf, af=zero(T)) where T
    (; vmax, vmin, amax, amin, jmax) = lim

    # Try UP direction first (positive jerk starts the motion)
    profile = RuckigProfile(T)

    # For positive displacement, try UP direction profiles
    if pf >= p0
        # Try velocity-limited profiles first
        if time_all_vel!(profile, p0, v0, a0, pf, vf, af, jmax, vmax, vmin, amax, amin)
            return profile
        end

        # Try ACC0_ACC1 (reaches amax and amin)
        if time_acc0_acc1!(profile, p0, v0, a0, pf, vf, af, jmax, vmax, vmin, amax, amin)
            return profile
        end

        # Try ACC0, ACC1, NONE
        if time_all_none_acc0_acc1!(profile, p0, v0, a0, pf, vf, af, jmax, vmax, vmin, amax, amin)
            return profile
        end
    end

    # Try DOWN direction (flip the problem)
    profile = RuckigProfile(T)
    p0_flip, pf_flip = -p0, -pf
    v0_flip, vf_flip = -v0, -vf
    a0_flip, af_flip = -a0, -af
    vmax_flip, vmin_flip = -vmin, -vmax
    amax_flip, amin_flip = -amin, -amax

    if pf_flip >= p0_flip
        if time_all_vel!(profile, p0_flip, v0_flip, a0_flip, pf_flip, vf_flip, af_flip,
                         jmax, vmax_flip, vmin_flip, amax_flip, amin_flip)
            # Flip back
            profile.p .*= -1
            profile.v .*= -1
            profile.a .*= -1
            profile.j .*= -1
            profile.pf = pf
            profile.vf = vf
            profile.af = af
            return profile
        end

        if time_acc0_acc1!(profile, p0_flip, v0_flip, a0_flip, pf_flip, vf_flip, af_flip,
                           jmax, vmax_flip, vmin_flip, amax_flip, amin_flip)
            profile.p .*= -1
            profile.v .*= -1
            profile.a .*= -1
            profile.j .*= -1
            profile.pf = pf
            profile.vf = vf
            profile.af = af
            return profile
        end

        if time_all_none_acc0_acc1!(profile, p0_flip, v0_flip, a0_flip, pf_flip, vf_flip, af_flip,
                                    jmax, vmax_flip, vmin_flip, amax_flip, amin_flip)
            profile.p .*= -1
            profile.v .*= -1
            profile.a .*= -1
            profile.j .*= -1
            profile.pf = pf
            profile.vf = vf
            profile.af = af
            return profile
        end
    end

    return nothing
end

#=============================================================================
 State Evaluation
=============================================================================#

"""
    evaluate_at(profile::RuckigProfile, t) -> (p, v, a, j)

Evaluate profile at time t.
"""
function evaluate_at(profile::RuckigProfile{T}, t::Real) where T
    T_total = profile.t_sum[7]

    if t <= 0
        return profile.p[1], profile.v[1], profile.a[1], profile.j[1]
    end

    if t >= T_total
        return profile.p[8], profile.v[8], profile.a[8], zero(T)
    end

    # Find phase
    phase = 1
    @inbounds for k in 1:7
        if t <= profile.t_sum[k]
            phase = k
            break
        end
    end

    t_start = phase == 1 ? zero(T) : profile.t_sum[phase-1]
    dt = t - t_start

    pk = profile.p[phase]
    vk = profile.v[phase]
    ak = profile.a[phase]
    jk = profile.j[phase]

    p = pk + dt * (vk + dt * (ak / 2 + dt * jk / 6))
    v = vk + dt * (ak + dt * jk / 2)
    a = ak + dt * jk

    return p, v, a, jk
end

#=============================================================================
 High-Level API
=============================================================================#

"""
    calculate_trajectory(lim::JerkLimiter, p0, v0, a0, pf, vf=0)

Calculate time-optimal trajectory from (p0, v0, a0) to (pf, vf, 0).
"""
function calculate_trajectory(lim::JerkLimiter{T}, p0, v0, a0, pf, vf=zero(T)) where T
    profile = find_profile(lim, p0, v0, a0, pf, vf)

    if profile === nothing
        error("No valid trajectory found from ($p0, $v0, $a0) to ($pf, $vf, 0)")
    end

    return profile
end
