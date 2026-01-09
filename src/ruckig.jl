# Ruckig: Time-optimal jerk-limited trajectory generation
# Based on: Berscheid & Kröger, "Jerk-limited Real-time Trajectory Generation
# with Arbitrary Target States", 2021
# Reference implementation: https://github.com/pantor/ruckig
# License of reference: MIT License https://github.com/pantor/ruckig/blob/main/LICENSE

using StaticArrays

export JerkLimiter, RuckigProfile
export calculate_trajectory, evaluate_at, evaluate_dt

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

# Allow RuckigProfile to broadcast as a scalar
Base.Broadcast.broadcastable(p::RuckigProfile) = Ref(p)

"""
    Roots{T}

Mutable storage for polynomial roots to avoid allocations.
Stores up to 4 roots with a count of valid entries.
"""
mutable struct Roots{T}
    r1::T
    r2::T
    r3::T
    r4::T
    count::Int
end

Roots{T}() where T = Roots{T}(T(NaN), T(NaN), T(NaN), T(NaN), 0)

function clear!(r::Roots{T}) where T
    r.r1 = r.r2 = r.r3 = r.r4 = T(NaN)
    r.count = 0
    r
end

function Base.push!(r::Roots{T}, val) where T
    r.count += 1
    if r.count == 1
        r.r1 = val
    elseif r.count == 2
        r.r2 = val
    elseif r.count == 3
        r.r3 = val
    else
        r.r4 = val
    end
    r
end

Base.length(r::Roots) = r.count

# Iterator interface for for-loop consumption
Base.iterate(r::Roots) = r.count >= 1 ? (r.r1, 2) : nothing
function Base.iterate(r::Roots, i)
    i > r.count && return nothing
    if i == 2
        return (r.r2, 3)
    elseif i == 3
        return (r.r3, 4)
    else
        return (r.r4, 5)
    end
end

"""
    JerkLimiter{T}

Jerk-limited trajectory generator with directional limits.

# Constructor
    JerkLimiter(; vmax, amax, jmax, vmin=-vmax, amin=-amax)

# Arguments
- `vmax`: Maximum velocity
- `amax`: Maximum acceleration
- `jmax`: Maximum jerk
- `vmin`: Minimum velocity (default: `-vmax`)
- `amin`: Minimum acceleration (default: `-amax`)
"""
struct JerkLimiter{T}
    vmax::T
    vmin::T
    amax::T
    amin::T
    jmax::T
    roots::Roots{Float64}  # Always Float64 since polynomial roots are floating-point
end

function JerkLimiter(; vmax, amax, jmax, vmin=-vmax, amin=-amax)
    T = promote_type(typeof(vmax), typeof(vmin), typeof(amax), typeof(amin), typeof(jmax))
    JerkLimiter(T(vmax), T(vmin), T(amax), T(amin), T(jmax), Roots{Float64}())
end

#=============================================================================
 Polynomial Root Finding (matching reference implementation)
=============================================================================#

"""
Solve ax² + bx + c = 0 for real roots, storing results in `roots`.
"""
function solve_quadratic_real!(roots::Roots, a, b, c)
    clear!(roots)
    if abs(a) < EPS
        abs(b) < EPS && return roots
        push!(roots, -c/b)
        return roots
    end

    disc = b^2 - 4a*c
    disc < 0 && return roots

    if disc < EPS
        push!(roots, -b / (2a))
        return roots
    end

    sqrt_disc = sqrt(disc)
    push!(roots, (-b - sqrt_disc) / (2a))
    push!(roots, (-b + sqrt_disc) / (2a))
    return roots
end

"""
Solve ax³ + bx² + cx + d = 0 for real roots using Cardano's formula.
"""
function solve_cubic_real!(roots::Roots, a, b, c, d)
    clear!(roots)
    if abs(a) < EPS
        return solve_quadratic_real!(roots, b, c, d)
    end

    # Normalize
    p, q, r = b/a, c/a, d/a

    # Depressed cubic: t³ + pt + q = 0 via x = t - p/3
    aa = q - p^2/3
    bb = 2p^3/27 - p*q/3 + r

    disc = bb^2/4 + aa^3/27

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
function solve_quartic_real!(roots::Roots, a, b, c, d, e)
    clear!(roots)
    # Handle non-quartic case
    if abs(a) < EPS
        return solve_cubic_real!(roots, b, c, d, e)
    end

    # Normalize to monic quartic: x^4 + px^3 + qx^2 + rx + s = 0
    p, q, r, s = b/a, c/a, d/a, e/a

    # Special cases from reference implementation (roots.hpp lines 201-221)
    if abs(s) < EPS
        if abs(r) < EPS
            # x^4 + px^3 + qx^2 = x^2(x^2 + px + q) = 0
            push!(roots, 0.0)
            D = p^2 - 4*q
            if abs(D) < EPS
                push!(roots, -p/2)
            elseif D > 0
                sqrtD = sqrt(D)
                push!(roots, (-p - sqrtD)/2)
                push!(roots, (-p + sqrtD)/2)
            end
            return roots
        end

        if abs(p) < EPS && abs(q) < EPS
            # x^4 + rx = x(x^3 + r) = 0
            push!(roots, 0.0)
            push!(roots, -cbrt(r))
            return roots
        end
    end

    # General case: Ferrari's method using resolvent cubic
    # Reference implementation (roots.hpp lines 223-280)
    a3 = -q
    b3 = p * r - 4 * s
    c3 = -p^2 * s - r^2 + 4 * q * s

    # Solve resolvent cubic: y^3 + a3*y^2 + b3*y + c3 = 0
    resolvent_roots = solve_cubic_all_real(a3, b3, c3)

    # Choose y with maximal absolute value
    y = resolvent_roots.r1
    if resolvent_roots.count >= 2 && abs(resolvent_roots.r2) > abs(y)
        y = resolvent_roots.r2
    end
    if resolvent_roots.count >= 3 && abs(resolvent_roots.r3) > abs(y)
        y = resolvent_roots.r3
    end

    D = y^2 - 4*s
    if abs(D) < EPS
        q1 = q2 = y / 2
        D2 = p^2 - 4*(q - y)
        if abs(D2) < EPS
            p1 = p2 = p / 2
        else
            sqrtD2 = sqrt(max(D2, 0.0))
            p1 = (p + sqrtD2) / 2
            p2 = (p - sqrtD2) / 2
        end
    else
        sqrtD = sqrt(max(D, 0.0))
        q1 = (y + sqrtD) / 2
        q2 = (y - sqrtD) / 2
        denom = q1 - q2
        if abs(denom) > EPS
            p1 = (p * q1 - r) / denom
            p2 = (r - p * q2) / denom
        else
            p1 = p2 = p / 2
        end
    end

    # Solve two quadratics: x^2 + p1*x + q1 = 0 and x^2 + p2*x + q2 = 0
    eps16 = 16 * EPS

    D1 = p1^2 - 4*q1
    if abs(D1) < eps16
        push!(roots, -p1/2)
    elseif D1 > 0
        sqrtD1 = sqrt(D1)
        push!(roots, (-p1 - sqrtD1)/2)
        push!(roots, (-p1 + sqrtD1)/2)
    end

    D2 = p2^2 - 4*q2
    if abs(D2) < eps16
        push!(roots, -p2/2)
    elseif D2 > 0
        sqrtD2 = sqrt(D2)
        push!(roots, (-p2 - sqrtD2)/2)
        push!(roots, (-p2 + sqrtD2)/2)
    end

    return roots
end

# Solve cubic returning all real roots (for resolvent cubic in quartic solver)
# Marked @inline so compiler can optimize away the Roots allocation
@inline function solve_cubic_all_real(a, b, c)
    roots = Roots{Float64}()
    # Cubic: x^3 + ax^2 + bx + c = 0 (Cardano's formula)
    a_over_3 = a / 3
    a2 = a^2
    q = a2 - b / 3
    r = (a * (2*a2 - b) + c) / 2
    r2 = r^2
    q3 = q^3

    cos120 = -0.5
    sin120 = 0.866025403784438646764

    if r2 < q3
        # Three real roots
        qsqrt = sqrt(q)
        t = clamp(r / (q * qsqrt), -1.0, 1.0)
        qq = -2 * qsqrt
        theta = acos(t) / 3
        ux = cos(theta) * qq
        uyi = sin(theta) * qq
        push!(roots, ux - a_over_3)
        push!(roots, ux * cos120 - uyi * sin120 - a_over_3)
        push!(roots, ux * cos120 + uyi * sin120 - a_over_3)
    else
        # One real root (or two if discriminant is zero)
        A = -cbrt(abs(r) + sqrt(max(r2 - q3, 0.0)))
        if r < 0
            A = -A
        end
        B = (A == 0.0) ? 0.0 : q / A
        x0 = (A + B) - a_over_3
        x1 = -(A + B) / 2 - a_over_3
        x2_imag = sqrt(3.0) * (A - B) / 2

        push!(roots, x0)
        if abs(x2_imag) < EPS
            push!(roots, x1)
            push!(roots, x1)
        end
    end
    return roots
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

    # Check all times non-negative (NaN < x is false, so check explicitly)
    @inbounds for i in 1:7
        (isnan(profile.t[i]) || profile.t[i] < -T_PRECISION) && return false
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
function time_all_none_acc0_acc1!(roots::Roots, profile::RuckigProfile{T}, p0, v0, a0, pf, vf, af,
                                  jMax, vMax, vMin, aMax, aMin) where T
    # Pre-compute common terms
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    af_p3 = af^3
    a0_p4 = a0_a0^2
    af_p4 = af_af^2
    v0_v0 = v0^2
    vf_vf = vf^2
    pd = pf - p0

    # NONE profile: t7 == 0 strategy from reference implementation
    # Solve for t (= t[3] in 1-indexed) using cubic polynomial
    h2_none = (a0_a0 - af_af)/(2*jMax) + (vf - v0)
    h2_h2 = h2_none^2

    t_min_none = (a0 - af)/jMax
    t_max_none = (aMax - aMin)/jMax

    polynom_none_1 = -2*(a0_a0 + af_af - 2*jMax*(v0 + vf)) / jMax_jMax
    polynom_none_2 = 4*(a0_p3 - af_p3 + 3*jMax*(af*vf - a0*v0)) / (3*jMax*jMax_jMax) - 4*pd/jMax
    polynom_none_3 = -h2_h2 / jMax_jMax

    # Reference uses solve_quart_monic with [0, polynom_none_1, polynom_none_2, polynom_none_3]
    # This represents t^4 + 0*t^3 + polynom_none_1*t^2 + polynom_none_2*t + polynom_none_3 = 0
    for t in solve_quartic_real!(roots, 1.0, 0.0, polynom_none_1, polynom_none_2, polynom_none_3)
        (t < t_min_none || t > t_max_none) && continue

        # Single Newton step for refinement (regarding pd)
        if t > EPS
            h1 = jMax*t*t
            orig = -h2_h2/(4*jMax*t) + h2_none*(af/jMax + t) + (4*a0_p3 + 2*af_p3 - 6*a0_a0*(af + 2*jMax*t) + 12*(af - a0)*jMax*v0 + 3*jMax_jMax*(-4*pd + (h1 + 8*v0)*t))/(12*jMax_jMax)
            deriv = h2_none + 2*v0 - a0_a0/jMax + h2_h2/(4*h1) + (3*h1)/4
            t -= orig / deriv
        end

        h0 = h2_none/(2*jMax*t)
        profile.t[1] = h0 + t/2 - a0/jMax
        profile.t[2] = 0
        profile.t[3] = t
        profile.t[4] = 0
        profile.t[5] = 0
        profile.t[6] = 0
        profile.t[7] = -h0 + t/2 + af/jMax

        if check!(profile, UDDU, LIMIT_NONE, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # ACC0 profile: reaches aMax but not aMin or vMax (from reference lines 144-237)
    h3_acc0 = (a0_a0 - af_af)/(2*aMax*jMax) + (vf - v0)/aMax
    t_min_acc0 = (aMax - af)/jMax
    t_max_acc0 = (aMax - aMin)/jMax

    h0_acc0 = 3*(af_p4 - a0_p4) + 8*(a0_p3 - af_p3)*aMax + 24*aMax*jMax*(af*vf - a0*v0) -
              6*a0_a0*(aMax^2 - 2*jMax*v0) + 6*af_af*(aMax^2 - 2*jMax*vf) +
              12*jMax*(jMax*(vf_vf - v0_v0 - 2*aMax*pd) - aMax^2*(vf - v0))
    h2_acc0 = -af_af + aMax^2 + 2*jMax*vf

    polynom_acc0_0 = -2*aMax/jMax
    polynom_acc0_1 = h2_acc0 / jMax_jMax
    polynom_acc0_2 = 0.0
    polynom_acc0_3 = h0_acc0 / (12*jMax_jMax*jMax_jMax)

    for t in solve_quartic_real!(roots, 1.0, polynom_acc0_0, polynom_acc0_1, polynom_acc0_2, polynom_acc0_3)
        (t < t_min_acc0 || t > t_max_acc0) && continue

        # Single Newton step for refinement
        if t > EPS
            h1 = jMax*t
            orig = h0_acc0/(12*jMax_jMax*t) + t*(h2_acc0 + h1*(h1 - 2*aMax))
            deriv = 2*(h2_acc0 + h1*(2*h1 - 3*aMax))
            t -= orig / deriv
        end

        profile.t[1] = (-a0 + aMax)/jMax
        profile.t[2] = h3_acc0 - 2*t + jMax/aMax*t^2
        profile.t[3] = t
        profile.t[4] = 0
        profile.t[5] = 0
        profile.t[6] = 0
        profile.t[7] = (af - aMax)/jMax + t

        if check!(profile, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # ACC1 profile: reaches aMin but not aMax or vMax (from reference lines 159-283)
    h3_acc1 = -(a0_a0 + af_af)/(2*jMax*aMin) + aMin/jMax + (vf - v0)/aMin
    t_min_acc1 = (aMin - a0)/jMax
    t_max_acc1 = (aMax - a0)/jMax

    h0_acc1 = (a0_p4 - af_p4)/4 + 2*(af_p3 - a0_p3)*aMin/3 + (a0_a0 - af_af)*aMin^2/2 +
              jMax*(af_af*vf + a0_a0*v0 + 2*aMin*(jMax*pd - a0*v0 - af*vf) + aMin^2*(v0 + vf) + jMax*(v0_v0 - vf_vf))
    h2_acc1 = a0_a0 - a0*aMin + 2*jMax*v0

    polynom_acc1_0 = 2*(2*a0 - aMin)/jMax
    polynom_acc1_1 = (5*a0_a0 + aMin*(aMin - 6*a0) + 2*jMax*v0) / jMax_jMax
    polynom_acc1_2 = 2*(a0 - aMin)*h2_acc1 / (jMax_jMax*jMax)
    polynom_acc1_3 = h0_acc1 / (jMax_jMax*jMax_jMax)

    for t in solve_quartic_real!(roots, 1.0, polynom_acc1_0, polynom_acc1_1, polynom_acc1_2, polynom_acc1_3)
        (t < t_min_acc1 || t > t_max_acc1) && continue

        # Double Newton step for refinement
        if t > EPS
            h5 = a0_p3 + 2*jMax*a0*v0
            for _ in 1:3
                h1 = jMax*t
                orig = -(h0_acc1/2 + h1*(h5 + a0*(aMin - 2*h1)*(aMin - h1) + a0_a0*(5*h1/2 - 2*aMin) + aMin^2*h1/2 + jMax*(h1/2 - aMin)*(h1*t + 2*v0)))/jMax
                abs(orig) < 1e-9 && break
                deriv = (aMin - a0 - h1)*(h2_acc1 + h1*(4*a0 - aMin + 2*h1))
                t -= min(orig / deriv, t)
            end
        end

        profile.t[1] = t
        profile.t[2] = 0
        profile.t[3] = (a0 - aMin)/jMax + t
        profile.t[4] = 0
        profile.t[5] = 0
        profile.t[6] = h3_acc1 - (2*a0 + jMax*t)*t/aMin
        profile.t[7] = (af - aMin)/jMax

        if check!(profile, UDDU, LIMIT_ACC1, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

#=============================================================================
 Two-Step Profiles (fallback when regular profiles fail)
=============================================================================#

"""
Two-step NONE profile (simplified profile without acceleration limits).
"""
function time_none_two_step!(profile::RuckigProfile{T}, p0, v0, a0, pf, vf, af,
                              jMax, vMax, vMin, aMax, aMin) where T
    a0_a0 = a0^2
    af_af = af^2

    # Two step: compute symmetric acceleration peak
    h0_sq = (a0_a0 + af_af)/2 + jMax*(vf - v0)
    if h0_sq >= 0
        h0 = sqrt(h0_sq) * sign(jMax)
        profile.t[1] = (h0 - a0)/jMax
        profile.t[2] = 0
        profile.t[3] = (h0 - af)/jMax
        profile.t[4] = 0
        profile.t[5] = 0
        profile.t[6] = 0
        profile.t[7] = 0

        if check!(profile, UDDU, LIMIT_NONE, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Single step (only jerk phase)
    profile.t[1] = (af - a0)/jMax
    profile.t[2] = 0
    profile.t[3] = 0
    profile.t[4] = 0
    profile.t[5] = 0
    profile.t[6] = 0
    profile.t[7] = 0

    if check!(profile, UDDU, LIMIT_NONE, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
        return true
    end

    return false
end

"""
Two-step ACC0 profile (simplified profile reaching only aMax).
"""
function time_acc0_two_step!(profile::RuckigProfile{T}, p0, v0, a0, pf, vf, af,
                              jMax, vMax, vMin, aMax, aMin) where T
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    a0_p4 = a0^4
    af_p3 = af^3
    af_p4 = af^4
    pd = pf - p0

    # Strategy 1: Two-step (t[1]=0)
    if abs(a0) > EPS
        profile.t[1] = 0
        profile.t[2] = (af_af - a0_a0 + 2*jMax*(vf - v0))/(2*a0*jMax)
        profile.t[3] = (a0 - af)/jMax
        profile.t[4] = 0
        profile.t[5] = 0
        profile.t[6] = 0
        profile.t[7] = 0

        if check!(profile, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Strategy 2: Three-step reaching aMax
    profile.t[1] = (-a0 + aMax)/jMax
    profile.t[2] = (a0_a0 + af_af - 2*aMax^2 + 2*jMax*(vf - v0))/(2*aMax*jMax)
    profile.t[3] = (-af + aMax)/jMax
    profile.t[4] = 0
    profile.t[5] = 0
    profile.t[6] = 0
    profile.t[7] = 0

    if check!(profile, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
        return true
    end

    # Strategy 3: Three-step with polynomial solution
    h0 = 3*(af_af - a0_a0 + 2*jMax*(v0 + vf))
    if abs(h0) > EPS
        h2 = a0_p3 + 2*af_p3 + 6*jMax_jMax*pd + 6*(af - a0)*jMax*vf - 3*a0*af_af

        # Solve for intermediate acceleration
        # The polynomial is complex; use a simplified approach
        h1_sq = 2*(2*h2^2 + h0*(a0_p4 - 6*a0_a0*(af_af + 2*jMax*vf) +
                8*a0_p3*af + 3*af_p4 - 6*af_af*jMax*vf - 12*jMax_jMax*(vf^2 - pd*(vf - v0))))

        if h1_sq >= 0
            for h1_sign in (1, -1)
                h1 = h1_sign * sqrt(h1_sq)
                a_peak = (a0_a0 + af_af + 2*jMax*(vf - v0) + h1/h0) / 2

                if a_peak > 0
                    a_peak = sqrt(a_peak)

                    profile.t[1] = (a_peak - a0)/jMax
                    profile.t[2] = 0
                    profile.t[3] = (a_peak - af)/jMax
                    profile.t[4] = 0
                    profile.t[5] = 0
                    profile.t[6] = 0
                    profile.t[7] = 0

                    if check!(profile, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                        return true
                    end
                end
            end
        end
    end

    # Strategy 4: Three-step with fixed time constraint (from reference lines 353-369)
    t_fixed = (aMax - aMin)/jMax
    profile.t[1] = (-a0 + aMax)/jMax
    profile.t[2] = (a0_a0 - af_af)/(2*aMax*jMax) + (vf - v0 + jMax*t_fixed^2)/aMax - 2*t_fixed
    profile.t[3] = t_fixed
    profile.t[4] = 0
    profile.t[5] = 0
    profile.t[6] = 0
    profile.t[7] = (af - aMin)/jMax

    if check!(profile, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
        return true
    end

    return false
end

"""
Two-step VEL profile (simplified velocity-limited profile).
"""
function time_vel_two_step!(profile::RuckigProfile{T}, p0, v0, a0, pf, vf, af,
                             jMax, vMax, vMin, aMax, aMin) where T
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    af_p3 = af^3
    pd = pf - p0

    h1_sq = af_af/(2*jMax_jMax) + (vMax - vf)/jMax
    h1_sq < 0 && return false
    h1 = sqrt(h1_sq)

    # Solution 1: t[1] = -a0/jMax (decelerate to zero first)
    profile.t[1] = -a0/jMax
    profile.t[2] = 0
    profile.t[3] = 0
    profile.t[4] = (af_p3 - a0_p3)/(3*jMax_jMax*vMax) +
                   (a0*v0 - af*vf + (af_af*h1)/2)/(jMax*vMax) -
                   (vf/vMax + 1.0)*h1 + pd/vMax
    profile.t[5] = h1
    profile.t[6] = 0
    profile.t[7] = h1 + af/jMax

    if check!(profile, UDDU, LIMIT_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
        return true
    end

    # Solution 2: t[3] = a0/jMax (accelerate through zero)
    profile.t[1] = 0
    profile.t[2] = 0
    profile.t[3] = a0/jMax
    profile.t[4] = (af_p3 - a0_p3)/(3*jMax_jMax*vMax) +
                   (a0*v0 - af*vf + (af_af*h1 + a0_p3/jMax)/2)/(jMax*vMax) -
                   (v0/vMax + 1.0)*a0/jMax -
                   (vf/vMax + 1.0)*h1 + pd/vMax
    profile.t[5] = h1
    profile.t[6] = 0
    profile.t[7] = h1 + af/jMax

    if check!(profile, UDDU, LIMIT_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
        return true
    end

    return false
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

"""
    evaluate_at(profile, ts::AbstractVector)

Evaluate the trajectory at multiple time points.

Returns a tuple of vectors `(positions, velocities, accelerations, jerks)`.
"""
function evaluate_at(profile::RuckigProfile{T}, ts::AbstractVector) where T
    n = length(ts)
    positions = Vector{T}(undef, n)
    velocities = Vector{T}(undef, n)
    accelerations = Vector{T}(undef, n)
    jerks = Vector{T}(undef, n)

    @inbounds for i in eachindex(ts)
        p, v, a, j = evaluate_at(profile, ts[i])
        positions[i] = p
        velocities[i] = v
        accelerations[i] = a
        jerks[i] = j
    end

    return positions, velocities, accelerations, jerks
end

"""
    evaluate_dt(profile, Ts)

Evaluate the trajectory at regular time intervals from 0 to the total duration.

Returns `(positions, velocities, accelerations, jerks, ts)` where `ts` is the time vector.
"""
function evaluate_dt(profile::RuckigProfile, Ts)
    T = profile.t_sum[7]
    ts = 0:Ts:T
    pos, vel, acc, jerk = evaluate_at(profile, ts)
    pos, vel, acc, jerk, ts
end

#=============================================================================
 High-Level API
=============================================================================#

"""
    calculate_trajectory(lim::JerkLimiter; pf, p0=0, v0=0, a0=0, vf=0)

Calculate time-optimal trajectory from (p0, v0, a0) to (pf, vf, 0).

# Arguments
- `lim`: JerkLimiter with velocity, acceleration, and jerk constraints
- `p0`: Initial position (default: 0)
- `v0`: Initial velocity (default: 0)
- `a0`: Initial acceleration (default: 0)
- `pf`: Target position (required)
- `vf`: Target velocity (default: 0)
"""
function calculate_trajectory(lim::JerkLimiter{T}; pf, p0=zero(T), v0=zero(T), a0=zero(T), vf=zero(T), af=zero(T)) where T

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
        if time_all_none_acc0_acc1!(lim.roots, profile, p0, v0, a0, pf, vf, af, jmax, vmax, vmin, amax, amin)
            return profile
        end

        # Try two-step fallback profiles
        if time_none_two_step!(profile, p0, v0, a0, pf, vf, af, jmax, vmax, vmin, amax, amin)
            return profile
        end

        if time_acc0_two_step!(profile, p0, v0, a0, pf, vf, af, jmax, vmax, vmin, amax, amin)
            return profile
        end

        if time_vel_two_step!(profile, p0, v0, a0, pf, vf, af, jmax, vmax, vmin, amax, amin)
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

        if time_all_none_acc0_acc1!(lim.roots, profile, p0_flip, v0_flip, a0_flip, pf_flip, vf_flip, af_flip,
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

        # Try two-step fallback profiles for DOWN direction
        if time_none_two_step!(profile, p0_flip, v0_flip, a0_flip, pf_flip, vf_flip, af_flip,
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

        if time_acc0_two_step!(profile, p0_flip, v0_flip, a0_flip, pf_flip, vf_flip, af_flip,
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

        if time_vel_two_step!(profile, p0_flip, v0_flip, a0_flip, pf_flip, vf_flip, af_flip,
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

    error("No valid trajectory found from ($p0, $v0, $a0) to ($pf, $vf, 0)")

end
