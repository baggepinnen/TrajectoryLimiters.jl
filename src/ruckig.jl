# Ruckig: Time-optimal jerk-limited trajectory generation
# Based on: Berscheid & Kröger, "Jerk-limited Real-time Trajectory Generation
# with Arbitrary Target States", 2021
# Reference implementation: https://github.com/pantor/ruckig
# License of reference: MIT License https://github.com/pantor/ruckig/blob/main/LICENSE

export JerkLimiter, RuckigProfile
export calculate_trajectory, calculate_waypoint_trajectory, evaluate_at, evaluate_dt, duration

#=============================================================================
 Constants (matching reference implementation)
=============================================================================#

const EPS = 1e-12  # Matching reference implementation's v_eps/a_eps/j_eps
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
    ProfileBuffer{T}

Mutable buffer for computing trajectory profiles. Stored in JerkLimiter
to avoid allocations during trajectory calculation.
"""
mutable struct ProfileBuffer{T}
    t::Memory{T}       # Phase durations (length 7)
    t_sum::Memory{T}   # Cumulative times (length 7)
    j::Memory{T}       # Jerk values (length 7)
    a::Memory{T}       # Acceleration at boundaries (length 8)
    v::Memory{T}       # Velocity at boundaries (length 8)
    p::Memory{T}       # Position at boundaries (length 8)
    limits::ReachedLimits
    control_signs::ControlSigns
end

function ProfileBuffer{T}() where T
    ProfileBuffer{T}(
        Memory{T}(undef, 7), Memory{T}(undef, 7), Memory{T}(undef, 7),
        Memory{T}(undef, 8), Memory{T}(undef, 8), Memory{T}(undef, 8),
        LIMIT_NONE, UDDU
    )
end

function clear!(buf::ProfileBuffer{T}) where T
    fill!(buf.t, zero(T))
    fill!(buf.t_sum, zero(T))
    fill!(buf.j, zero(T))
    fill!(buf.a, zero(T))
    fill!(buf.v, zero(T))
    fill!(buf.p, zero(T))
    buf.limits = LIMIT_NONE
    buf.control_signs = UDDU
    buf
end

"""
    RuckigProfile{T}

A 7-phase jerk-limited trajectory profile (immutable result).
"""
struct RuckigProfile{T}
    t::NTuple{7,T}        # Phase durations
    t_sum::NTuple{7,T}    # Cumulative times
    j::NTuple{7,T}        # Jerk values
    a::NTuple{8,T}        # Acceleration at boundaries
    v::NTuple{8,T}        # Velocity at boundaries
    p::NTuple{8,T}        # Position at boundaries
    pf::T                 # Target position
    vf::T                 # Target velocity
    af::T                 # Target acceleration
    limits::ReachedLimits
    control_signs::ControlSigns
end

"""
Create RuckigProfile from ProfileBuffer.
"""
function RuckigProfile(buf::ProfileBuffer{T}, pf, vf, af) where T
    RuckigProfile{T}(
        NTuple{7,T}(buf.t),
        NTuple{7,T}(buf.t_sum),
        NTuple{7,T}(buf.j),
        NTuple{8,T}(buf.a),
        NTuple{8,T}(buf.v),
        NTuple{8,T}(buf.p),
        pf, vf, af,
        buf.limits, buf.control_signs
    )
end

# Allow RuckigProfile to broadcast as a scalar
Base.Broadcast.broadcastable(p::RuckigProfile) = Ref(p)

duration(p::RuckigProfile) = p.t_sum[end]

#=============================================================================
 Block: Stores profile and blocked time intervals for synchronization
=============================================================================#

"""
    BlockInterval{T}

Represents a blocked time interval [left, right) with an associated profile
that becomes valid at the right endpoint.
"""
struct BlockInterval{T}
    left::T
    right::T
    profile::RuckigProfile{T}
end

"""
    Block{T}

Stores the minimum-time profile and any blocked time intervals.
Used to find valid synchronization times across multiple DOFs.

A DOF is "blocked" at time t if:
- t < t_min (faster than minimum time), OR
- t is within interval a: a.left < t < a.right, OR
- t is within interval b: b.left < t < b.right
"""
struct Block{T}
    p_min::RuckigProfile{T}       # Minimum-time profile
    t_min::T                       # Minimum duration
    a::Union{Nothing, BlockInterval{T}}  # First blocked interval (optional)
    b::Union{Nothing, BlockInterval{T}}  # Second blocked interval (optional)
end

"""Create a Block with just the minimum profile (no blocked intervals)."""
function Block(p_min::RuckigProfile{T}) where T
    Block{T}(p_min, duration(p_min), nothing, nothing)
end

"""Check if time t is blocked for this DOF."""
function is_blocked(block::Block, t)
    t < block.t_min && return true
    !isnothing(block.a) && block.a.left < t < block.a.right && return true
    !isnothing(block.b) && block.b.left < t < block.b.right && return true
    return false
end

"""Get the appropriate profile for time t."""
function get_profile(block::Block, t)
    if !isnothing(block.b) && t >= block.b.right
        return block.b.profile
    end
    if !isnothing(block.a) && t >= block.a.right
        return block.a.profile
    end
    return block.p_min
end

#=============================================================================
 Roots: Storage for polynomial roots
=============================================================================#

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
    roots::Roots{Float64}           # Always Float64 since polynomial roots are floating-point
    buffer::ProfileBuffer{Float64}  # Always Float64 for computation
end

function JerkLimiter(; vmax, amax, jmax, vmin=-vmax, amin=-amax)
    T = promote_type(typeof(vmax), typeof(vmin), typeof(amax), typeof(amin), typeof(jmax))
    JerkLimiter(T(vmax), T(vmin), T(amax), T(amin), T(jmax), Roots{Float64}(), ProfileBuffer{Float64}())
end

function Base.show(io::IO, lim::JerkLimiter)
    # Check if vmin/amin are at their default values
    if lim.vmin == -lim.vmax && lim.amin == -lim.amax
        print(io, "JerkLimiter(; vmax=$(lim.vmax), amax=$(lim.amax), jmax=$(lim.jmax))")
    else
        print(io, "JerkLimiter(; vmax=$(lim.vmax), vmin=$(lim.vmin), amax=$(lim.amax), amin=$(lim.amin), jmax=$(lim.jmax))")
    end
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
    # Reference uses a /= 3, then a2 = a*a, so we need (a/3)^2
    a_over_3 = a / 3
    a2 = a_over_3^2  # Must use (a/3)^2, not a^2
    q = a2 - b / 3
    r = (a_over_3 * (2*a2 - b) + c) / 2  # Must use a/3, not a
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
    check!(buf, control_signs, limits, jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af) -> Bool

Validate profile: check times >= 0, integrate, verify limits and final state.
This matches the reference implementation's check() template function.
"""
function check!(buf::ProfileBuffer{T}, control_signs::ControlSigns, limits::ReachedLimits,
                jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af=zero(T)) where T

    # Set jerk pattern based on control signs
    if control_signs == UDDU
        buf.j[1], buf.j[2], buf.j[3], buf.j[4], buf.j[5], buf.j[6], buf.j[7] = jf, 0, -jf, 0, -jf, 0, jf
    else  # UDUD
        buf.j[1], buf.j[2], buf.j[3], buf.j[4], buf.j[5], buf.j[6], buf.j[7] = jf, 0, -jf, 0, jf, 0, -jf
    end

    # Check all times non-negative (NaN < x is false, so check explicitly)
    @inbounds for i in 1:7
        (isnan(buf.t[i]) || buf.t[i] < -T_PRECISION) && return false
        buf.t[i] = max(buf.t[i], zero(T))
    end

    # Integrate profile (Eq. 2-4 from paper)
    buf.a[1] = a0
    buf.v[1] = v0
    buf.p[1] = p0

    cumtime = zero(T)
    @inbounds for i in 1:7
        ti = buf.t[i]
        ji = buf.j[i]
        ai = buf.a[i]
        vi = buf.v[i]
        pi = buf.p[i]

        buf.a[i+1] = ai + ti * ji
        buf.v[i+1] = vi + ti * (ai + ti * ji / 2)
        buf.p[i+1] = pi + ti * (vi + ti * (ai / 2 + ti * ji / 6))

        cumtime += ti
        buf.t_sum[i] = cumtime
    end

    # Check final state
    abs(buf.p[8] - pf) > P_PRECISION && return false
    abs(buf.v[8] - vf) > V_PRECISION && return false
    abs(buf.a[8] - af) > A_PRECISION && return false

    # Determine direction and set limits accordingly (matching reference implementation)
    # When vMax > 0, direction is UP; when vMax <= 0, direction is DOWN (limits swapped)
    if vMax > 0
        vUppLim, vLowLim = vMax + EPS, vMin - EPS
        aUppLim, aLowLim = aMax + EPS, aMin - EPS
    else
        vUppLim, vLowLim = vMin + EPS, vMax - EPS
        aUppLim, aLowLim = aMin + EPS, aMax - EPS
    end

    # Check acceleration limits at critical points (indices 2, 4, 6 in 1-based = boundaries after phases 1, 3, 5)
    @inbounds for i in (2, 4, 6)
        (buf.a[i] > aUppLim || buf.a[i] < aLowLim) && return false
    end

    # Check velocity limits at critical points (indices 4-7 in 1-based)
    @inbounds for i in 4:7
        (buf.v[i] > vUppLim || buf.v[i] < vLowLim) && return false
    end

    # Check velocity at acceleration zero-crossings
    @inbounds for i in 3:6
        buf.t[i] < EPS && continue
        ai, ji = buf.a[i], buf.j[i]
        abs(ji) < EPS && continue

        # Time when acceleration crosses zero within this phase
        if ai * buf.a[i+1] < -EPS
            t_zero = -ai / ji
            if 0 < t_zero < buf.t[i]
                v_at_zero = buf.v[i] - ai^2 / (2ji)
                (v_at_zero > vUppLim || v_at_zero < vLowLim) && return false
            end
        end
    end

    buf.limits = limits
    buf.control_signs = control_signs

    return true
end

#=============================================================================
 Profile Time Calculations - UDDU (matching reference implementation)
=============================================================================#

"""
Try all velocity-limited profiles (ACC0_ACC1_VEL, ACC1_VEL, ACC0_VEL, VEL).
Returns true if any valid profile is found.
"""
function time_all_vel!(buf::ProfileBuffer{T}, p0, v0, a0, pf, vf, af,
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
        buf.t[1] = (-a0 + aMax) / jMax
        buf.t[2] = (a0_a0/2 - aMax^2 - jMax*(v0 - vMax)) / (aMax * jMax)
        buf.t[3] = aMax / jMax
        buf.t[5] = -aMin / jMax
        buf.t[6] = -(af_af/2 - aMin^2 - jMax*(vf - vMax)) / (aMin * jMax)
        buf.t[7] = buf.t[5] + af / jMax

        # Compute t[4] from position constraint (equation from reference)
        buf.t[4] = (3*(a0_p4*aMin - af_p4*aMax) +
                        8*aMax*aMin*(af_p3 - a0_p3 + 3*jMax*(a0*v0 - af*vf)) +
                        6*a0_a0*aMin*(aMax^2 - 2*jMax*v0) -
                        6*af_af*aMax*(aMin^2 - 2*jMax*vf) -
                        12*jMax*(aMax*aMin*(aMax*(v0 + vMax) - aMin*(vf + vMax) - 2*jMax*pd) +
                                (aMin - aMax)*jMax*vMax^2 +
                                jMax*(aMax*vf_vf - aMin*v0_v0))) / (24*aMax*aMin*jMax_jMax*vMax)

        if check!(buf, UDDU, LIMIT_ACC0_ACC1_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Strategy 2: ACC1_VEL (reach vMax and aMin, not aMax)
    # Reference: position_third_step1.cpp lines 40-53
    begin
        t_acc0 = sqrt(max(0.0, a0_a0/(2*jMax_jMax) + (vMax - v0)/jMax))
        buf.t[1] = t_acc0 - a0/jMax
        buf.t[2] = 0
        buf.t[3] = t_acc0
        # Cruise time formula from reference (line 46)
        buf.t[4] = -(3*af_p4 - 8*aMin*(af_p3 - a0_p3) - 24*aMin*jMax*(a0*v0 - af*vf) +
                     6*af_af*(aMin^2 - 2*jMax*vf) -
                     12*jMax*(2*aMin*jMax*pd + aMin^2*(vf + vMax) + jMax*(vMax^2 - vf_vf) +
                              aMin*t_acc0*(a0_a0 - 2*jMax*(v0 + vMax))))/(24*aMin*jMax_jMax*vMax)
        buf.t[5] = -aMin / jMax
        buf.t[6] = -(af_af/2 - aMin^2 - jMax*(vf - vMax)) / (aMin * jMax)
        buf.t[7] = buf.t[5] + af / jMax

        if check!(buf, UDDU, LIMIT_ACC1_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Strategy 3: ACC0_VEL (reach aMax and vMax, not aMin)
    # Reference: position_third_step1.cpp lines 56-67
    begin
        t_acc1 = sqrt(max(0.0, af_af/(2*jMax_jMax) + (vMax - vf)/jMax))

        buf.t[1] = (-a0 + aMax) / jMax
        buf.t[2] = (a0_a0/2 - aMax^2 - jMax*(v0 - vMax)) / (aMax * jMax)
        buf.t[3] = aMax / jMax
        # Cruise time formula from reference (line 62)
        buf.t[4] = (3*a0_p4 + 8*aMax*(af_p3 - a0_p3) + 24*aMax*jMax*(a0*v0 - af*vf) +
                    6*a0_a0*(aMax^2 - 2*jMax*v0) -
                    12*jMax*(-2*aMax*jMax*pd + aMax^2*(v0 + vMax) + jMax*(vMax^2 - v0_v0) +
                             aMax*t_acc1*(-af_af + 2*(vf + vMax)*jMax)))/(24*aMax*jMax_jMax*vMax)
        buf.t[5] = t_acc1
        buf.t[6] = 0
        buf.t[7] = t_acc1 + af/jMax

        if check!(buf, UDDU, LIMIT_ACC0_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Strategy 4: VEL (reach vMax only, no acceleration limits)
    begin
        t_acc0 = sqrt(max(0.0, a0_a0/(2*jMax_jMax) + (vMax - v0)/jMax))
        t_acc1 = sqrt(max(0.0, af_af/(2*jMax_jMax) + (vMax - vf)/jMax))

        buf.t[1] = t_acc0 - a0/jMax
        buf.t[2] = 0
        buf.t[3] = t_acc0
        buf.t[5] = t_acc1
        buf.t[6] = 0
        buf.t[7] = t_acc1 + af/jMax

        buf.t[4] = (af_p3 - a0_p3)/(3*jMax_jMax*vMax) +
                       (a0*v0 - af*vf + (af_af*t_acc1 + a0_a0*t_acc0)/2)/(jMax*vMax) -
                       (v0/vMax + 1.0)*t_acc0 - (vf/vMax + 1.0)*t_acc1 + pd/vMax

        if check!(buf, UDDU, LIMIT_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
Try ACC0_ACC1 profile (reach both aMax and aMin, but not vMax).
"""
function time_acc0_acc1!(buf::ProfileBuffer{T}, p0, v0, a0, pf, vf, af,
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

        buf.t[1] = (-a0 + aMax) / jMax
        buf.t[2] = h2 - h1_sign * h1 / aMax
        buf.t[3] = aMax / jMax
        buf.t[4] = 0
        buf.t[5] = -aMin / jMax
        buf.t[6] = h3 + h1_sign * h1 / aMin
        buf.t[7] = buf.t[5] + af / jMax

        if check!(buf, UDDU, LIMIT_ACC0_ACC1, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
Try ACC0, ACC1, and NONE profiles (no velocity limit reached).
"""
function time_all_none_acc0_acc1!(roots::Roots, buf::ProfileBuffer{T}, p0, v0, a0, pf, vf, af,
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
        buf.t[1] = h0 + t/2 - a0/jMax
        buf.t[2] = 0
        buf.t[3] = t
        buf.t[4] = 0
        buf.t[5] = 0
        buf.t[6] = 0
        buf.t[7] = -h0 + t/2 + af/jMax

        if check!(buf, UDDU, LIMIT_NONE, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
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

        # Single Newton step (regarding pd) - matching reference exactly
        if t > EPS
            h1 = jMax*t
            orig = h0_acc0/(12*jMax_jMax*t) + t*(h2_acc0 + h1*(h1 - 2*aMax))
            deriv = 2*(h2_acc0 + h1*(2*h1 - 3*aMax))
            t -= orig / deriv
        end

        buf.t[1] = (-a0 + aMax)/jMax
        buf.t[2] = h3_acc0 - 2*t + jMax/aMax*t^2
        buf.t[3] = t
        buf.t[4] = 0
        buf.t[5] = 0
        buf.t[6] = 0
        buf.t[7] = (af - aMax)/jMax + t

        if check!(buf, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
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

        # Double Newton step for refinement (matching reference structure exactly)
        if t > EPS
            h5 = a0_p3 + 2*jMax*a0*v0
            h1 = jMax*t
            orig = -(h0_acc1/2 + h1*(h5 + a0*(aMin - 2*h1)*(aMin - h1) + a0_a0*(5*h1/2 - 2*aMin) + aMin^2*h1/2 + jMax*(h1/2 - aMin)*(h1*t + 2*v0)))/jMax
            deriv = (aMin - a0 - h1)*(h2_acc1 + h1*(4*a0 - aMin + 2*h1))
            t -= min(orig / deriv, t)  # First step uses min

            h1 = jMax*t
            orig = -(h0_acc1/2 + h1*(h5 + a0*(aMin - 2*h1)*(aMin - h1) + a0_a0*(5*h1/2 - 2*aMin) + aMin^2*h1/2 + jMax*(h1/2 - aMin)*(h1*t + 2*v0)))/jMax

            if abs(orig) > 1e-9
                deriv = (aMin - a0 - h1)*(h2_acc1 + h1*(4*a0 - aMin + 2*h1))
                t -= orig / deriv  # Second step: no min

                h1 = jMax*t
                orig = -(h0_acc1/2 + h1*(h5 + a0*(aMin - 2*h1)*(aMin - h1) + a0_a0*(5*h1/2 - 2*aMin) + aMin^2*h1/2 + jMax*(h1/2 - aMin)*(h1*t + 2*v0)))/jMax

                if abs(orig) > 1e-9
                    deriv = (aMin - a0 - h1)*(h2_acc1 + h1*(4*a0 - aMin + 2*h1))
                    t -= orig / deriv  # Third step: no min
                end
            end
        end

        buf.t[1] = t
        buf.t[2] = 0
        buf.t[3] = (a0 - aMin)/jMax + t
        buf.t[4] = 0
        buf.t[5] = 0
        buf.t[6] = h3_acc1 - (2*a0 + jMax*t)*t/aMin
        buf.t[7] = (af - aMin)/jMax

        if check!(buf, UDDU, LIMIT_ACC1, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
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
function time_none_two_step!(buf::ProfileBuffer{T}, p0, v0, a0, pf, vf, af,
                              jMax, vMax, vMin, aMax, aMin) where T
    a0_a0 = a0^2
    af_af = af^2

    # Two step: compute symmetric acceleration peak
    h0_sq = (a0_a0 + af_af)/2 + jMax*(vf - v0)
    if h0_sq >= 0
        h0 = sqrt(h0_sq) * sign(jMax)
        buf.t[1] = (h0 - a0)/jMax
        buf.t[2] = 0
        buf.t[3] = (h0 - af)/jMax
        buf.t[4] = 0
        buf.t[5] = 0
        buf.t[6] = 0
        buf.t[7] = 0

        if check!(buf, UDDU, LIMIT_NONE, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Single step (only jerk phase)
    buf.t[1] = (af - a0)/jMax
    buf.t[2] = 0
    buf.t[3] = 0
    buf.t[4] = 0
    buf.t[5] = 0
    buf.t[6] = 0
    buf.t[7] = 0

    if check!(buf, UDDU, LIMIT_NONE, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
        return true
    end

    return false
end

"""
Two-step ACC0 profile (simplified profile reaching only aMax).
"""
function time_acc0_two_step!(buf::ProfileBuffer{T}, p0, v0, a0, pf, vf, af,
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
        buf.t[1] = 0
        buf.t[2] = (af_af - a0_a0 + 2*jMax*(vf - v0))/(2*a0*jMax)
        buf.t[3] = (a0 - af)/jMax
        buf.t[4] = 0
        buf.t[5] = 0
        buf.t[6] = 0
        buf.t[7] = 0

        if check!(buf, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Strategy 2: Three-step reaching aMax
    buf.t[1] = (-a0 + aMax)/jMax
    buf.t[2] = (a0_a0 + af_af - 2*aMax^2 + 2*jMax*(vf - v0))/(2*aMax*jMax)
    buf.t[3] = (-af + aMax)/jMax
    buf.t[4] = 0
    buf.t[5] = 0
    buf.t[6] = 0
    buf.t[7] = 0

    if check!(buf, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
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

                    buf.t[1] = (a_peak - a0)/jMax
                    buf.t[2] = 0
                    buf.t[3] = (a_peak - af)/jMax
                    buf.t[4] = 0
                    buf.t[5] = 0
                    buf.t[6] = 0
                    buf.t[7] = 0

                    if check!(buf, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                        return true
                    end
                end
            end
        end
    end

    # Strategy 4: Three-step with fixed time constraint (from reference lines 353-369)
    t_fixed = (aMax - aMin)/jMax
    buf.t[1] = (-a0 + aMax)/jMax
    buf.t[2] = (a0_a0 - af_af)/(2*aMax*jMax) + (vf - v0 + jMax*t_fixed^2)/aMax - 2*t_fixed
    buf.t[3] = t_fixed
    buf.t[4] = 0
    buf.t[5] = 0
    buf.t[6] = 0
    buf.t[7] = (af - aMin)/jMax

    if check!(buf, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
        return true
    end

    return false
end

"""
Two-step ACC1_VEL profile (reaches aMin and vMax, skips aMax phase).
From reference: time_acc1_vel_two_step
"""
function time_acc1_vel_two_step!(buf::ProfileBuffer{T}, p0, v0, a0, pf, vf, af,
                                  jMax, vMax, vMin, aMax, aMin) where T
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    af_p3 = af^3
    af_p4 = af^4
    pd = pf - p0
    vf_vf = vf^2

    buf.t[1] = 0
    buf.t[2] = 0
    buf.t[3] = a0/jMax
    buf.t[4] = -(3*af_p4 - 8*aMin*(af_p3 - a0_p3) - 24*aMin*jMax*(a0*v0 - af*vf) +
                 6*af_af*(aMin^2 - 2*jMax*vf) -
                 12*jMax*(2*aMin*jMax*pd + aMin^2*(vf + vMax) + jMax*(vMax^2 - vf_vf) +
                          aMin*a0*(a0_a0 - 2*jMax*(v0 + vMax))/jMax)) / (24*aMin*jMax_jMax*vMax)
    buf.t[5] = -aMin/jMax
    buf.t[6] = -(af_af/2 - aMin^2 + jMax*(vMax - vf))/(aMin*jMax)
    buf.t[7] = buf.t[5] + af/jMax

    if check!(buf, UDDU, LIMIT_ACC1_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
        return true
    end

    return false
end

"""
Two-step VEL profile (simplified velocity-limited profile).
"""
function time_vel_two_step!(buf::ProfileBuffer{T}, p0, v0, a0, pf, vf, af,
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
    buf.t[1] = -a0/jMax
    buf.t[2] = 0
    buf.t[3] = 0
    buf.t[4] = (af_p3 - a0_p3)/(3*jMax_jMax*vMax) +
                   (a0*v0 - af*vf + (af_af*h1)/2)/(jMax*vMax) -
                   (vf/vMax + 1.0)*h1 + pd/vMax
    buf.t[5] = h1
    buf.t[6] = 0
    buf.t[7] = h1 + af/jMax

    if check!(buf, UDDU, LIMIT_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
        return true
    end

    # Solution 2: t[3] = a0/jMax (accelerate through zero)
    buf.t[1] = 0
    buf.t[2] = 0
    buf.t[3] = a0/jMax
    buf.t[4] = (af_p3 - a0_p3)/(3*jMax_jMax*vMax) +
                   (a0*v0 - af*vf + (af_af*h1 + a0_p3/jMax)/2)/(jMax*vMax) -
                   (v0/vMax + 1.0)*a0/jMax -
                   (vf/vMax + 1.0)*h1 + pd/vMax
    buf.t[5] = h1
    buf.t[6] = 0
    buf.t[7] = h1 + af/jMax

    if check!(buf, UDDU, LIMIT_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
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
    T_total = duration(profile)

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
    T = duration(profile)
    ts = 0:Ts:T
    pos, vel, acc, jerk = evaluate_at(profile, ts)
    pos, vel, acc, jerk, ts
end

#=============================================================================
 High-Level API
=============================================================================#

"""
    calculate_trajectory(lim::JerkLimiter; pf, p0=0, v0=0, a0=0, vf=0, af=0)

Calculate time-optimal trajectory from (p0, v0, a0) to (pf, vf, af).

# Arguments
- `lim`: JerkLimiter with velocity, acceleration, and jerk constraints
- `p0`: Initial position (default: 0)
- `v0`: Initial velocity (default: 0)
- `a0`: Initial acceleration (default: 0)
- `pf`: Target position (required)
- `vf`: Target velocity (default: 0)
- `af`: Target acceleration (default: 0)
"""
function calculate_trajectory(lim::JerkLimiter{T}; pf, p0=zero(T), v0=zero(T), a0=zero(T), vf=zero(T), af=zero(T)) where T

    (; vmax, vmin, amax, amin, jmax, buffer) = lim
    buf = buffer
    clear!(buf)

    # Determine primary and secondary direction based on displacement
    # For positive pd: primary uses standard limits
    # For negative pd: primary uses swapped limits (to move in negative direction)
    pd = pf - p0
    if pd >= 0
        jMax1, vMax1, vMin1, aMax1, aMin1 = jmax, vmax, vmin, amax, amin
        jMax2, vMax2, vMin2, aMax2, aMin2 = -jmax, vmin, vmax, amin, amax
    else
        jMax1, vMax1, vMin1, aMax1, aMin1 = -jmax, vmin, vmax, amin, amax
        jMax2, vMax2, vMin2, aMax2, aMin2 = jmax, vmax, vmin, amax, amin
    end

    # Try primary direction first
    if time_all_vel!(buf, p0, v0, a0, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_acc0_acc1!(buf, p0, v0, a0, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_all_none_acc0_acc1!(lim.roots, buf, p0, v0, a0, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_none_two_step!(buf, p0, v0, a0, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_acc0_two_step!(buf, p0, v0, a0, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_acc1_vel_two_step!(buf, p0, v0, a0, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_vel_two_step!(buf, p0, v0, a0, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
        return RuckigProfile(buf, pf, vf, af)
    end

    # Try secondary direction (for complex cases like starting with velocity
    # in the wrong direction)
    clear!(buf)

    if time_all_vel!(buf, p0, v0, a0, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_acc0_acc1!(buf, p0, v0, a0, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_all_none_acc0_acc1!(lim.roots, buf, p0, v0, a0, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_none_two_step!(buf, p0, v0, a0, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_acc0_two_step!(buf, p0, v0, a0, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_acc1_vel_two_step!(buf, p0, v0, a0, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
        return RuckigProfile(buf, pf, vf, af)
    end

    if time_vel_two_step!(buf, p0, v0, a0, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
        return RuckigProfile(buf, pf, vf, af)
    end

    error("No valid trajectory found from ($p0, $v0, $a0) to ($pf, $vf, $af), limiter: ", lim)

end

#=============================================================================
 Waypoint Trajectories
=============================================================================#

"""
Extract position, velocity, acceleration from a waypoint named tuple.
Defaults to v=0.0, a=0.0 if not specified.
"""
function get_waypoint_state(wp)
    p = wp.p
    v = hasproperty(wp, :v) ? wp.v : 0.0
    a = hasproperty(wp, :a) ? wp.a : 0.0
    (p, v, a)
end

"""
    calculate_waypoint_trajectory(lim, waypoints, Ts)

Calculate time-optimal trajectory passing through specified waypoints.

# Arguments
- `lim`: JerkLimiter with constraints
- `waypoints`: Vector of named tuples with fields:
  - `p`: Position at waypoint (required)
  - `v`: Velocity at waypoint (default: 0.0)
  - `a`: Acceleration at waypoint (default: 0.0)
- `Ts`: Sample interval for output

# Returns
`(ts, ps, vs, as, js)` - Arrays of time, position, velocity, acceleration, jerk

# Example
```julia
lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)
waypoints = [(p=0.0,), (p=2.0, v=5.0), (p=5.0,)]
ts, ps, vs, as, js = calculate_waypoint_trajectory(lim, waypoints)
```
"""
function calculate_waypoint_trajectory(lim::JerkLimiter, waypoints, Ts)
    n = length(waypoints)
    n < 2 && error("Need at least 2 waypoints")

    # Collect all segments
    all_ts = Float64[]
    all_ps = Float64[]
    all_vs = Float64[]
    all_as = Float64[]
    all_js = Float64[]

    t_offset = 0.0

    for i in 1:(n-1)
        # Extract states at waypoints
        p0, v0, a0 = get_waypoint_state(waypoints[i])
        pf, vf, af = get_waypoint_state(waypoints[i+1])

        # Calculate time-optimal trajectory for this segment
        profile = calculate_trajectory(lim; p0, v0, a0, pf, vf, af)

        # Sample at Ts intervals
        ps, vs, as, js, ts = evaluate_dt(profile, Ts)

        # Shift times by offset and append
        if i == 1
            append!(all_ts, ts .+ t_offset)
            append!(all_ps, ps)
            append!(all_vs, vs)
            append!(all_as, as)
            append!(all_js, js)
        else
            # Skip first point to avoid duplicates at waypoint boundaries
            append!(all_ts, (ts .+ t_offset)[2:end])
            append!(all_ps, ps[2:end])
            append!(all_vs, vs[2:end])
            append!(all_as, as[2:end])
            append!(all_js, js[2:end])
        end

        t_offset += duration(profile)
    end

    return all_ts, all_ps, all_vs, all_as, all_js
end

"""
Extract waypoint state arrays for multi-DOF trajectories.
"""
function get_waypoint_state_multidof(wp, ndof)
    p = wp.p
    v = hasproperty(wp, :v) ? wp.v : zeros(eltype(p), ndof)
    a = hasproperty(wp, :a) ? wp.a : zeros(eltype(p), ndof)
    (p, v, a)
end

"""
    calculate_waypoint_trajectory(lims::AbstractVector{<:JerkLimiter}, waypoints, Ts)

Calculate time-synchronized trajectory passing through specified waypoints for multiple DOFs.

Each waypoint is a named tuple with arrays for each state:
- `p`: position array (required)
- `v`: velocity array (optional, defaults to zeros)
- `a`: acceleration array (optional, defaults to zeros)

# Example
```julia
lims = [
    JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
    JerkLimiter(; vmax=5.0, amax=30.0, jmax=500.0),
]
waypoints = [
    (p = [0.0, 0.0],),
    (p = [1.0, 2.0], v = [2.0, 1.0]),
    (p = [3.0, 4.0],),
]
ts, ps, vs, as, js = calculate_waypoint_trajectory(lims, waypoints, 0.001)
```

Returns `(ts, ps, vs, as, js)` where `ps`, `vs`, `as`, `js` are matrices
with each column corresponding to a DOF.
"""
function calculate_waypoint_trajectory(lims::AbstractVector{<:JerkLimiter{T}}, waypoints, Ts) where T
    n = length(waypoints)
    n < 2 && error("Need at least 2 waypoints")
    ndof = length(lims)

    # Collect all segments
    all_ts = T[]
    all_ps = Vector{T}[]
    all_vs = Vector{T}[]
    all_as = Vector{T}[]
    all_js = Vector{T}[]

    t_offset = zero(T)

    for i in 1:(n-1)
        # Extract states at waypoints
        p0, v0, a0 = get_waypoint_state_multidof(waypoints[i], ndof)
        pf, vf, af = get_waypoint_state_multidof(waypoints[i+1], ndof)

        # Calculate synchronized trajectory for this segment
        profiles = calculate_trajectory(lims; p0, v0, a0, pf, vf, af)

        # Sample at Ts intervals
        ps, vs, as, js, ts = evaluate_dt(profiles, Ts)

        # Shift times by offset and append
        if i == 1
            append!(all_ts, ts .+ t_offset)
            for k in axes(ps, 1)
                push!(all_ps, ps[k, :])
                push!(all_vs, vs[k, :])
                push!(all_as, as[k, :])
                push!(all_js, js[k, :])
            end
        else
            # Skip first point to avoid duplicates at waypoint boundaries
            append!(all_ts, (ts .+ t_offset)[2:end])
            for k in 2:size(ps, 1)
                push!(all_ps, ps[k, :])
                push!(all_vs, vs[k, :])
                push!(all_as, as[k, :])
                push!(all_js, js[k, :])
            end
        end

        t_offset += duration(profiles[1])
    end

    # Convert vectors of vectors to matrices
    ps_mat = reduce(hcat, all_ps)'
    vs_mat = reduce(hcat, all_vs)'
    as_mat = reduce(hcat, all_as)'
    js_mat = reduce(hcat, all_js)'

    return all_ts, Matrix(ps_mat), Matrix(vs_mat), Matrix(as_mat), Matrix(js_mat)
end


#=============================================================================
 Step 2: Time-Synchronized Profile Calculation

 These functions calculate profiles for a GIVEN duration tf, rather than
 finding the minimum-time profile. Used for multi-DOF synchronization.
=============================================================================#

"""
Pre-computed expressions for Step2 calculations to avoid repeated computation.
"""
struct Step2PreComputed{T}
    pd::T       # pf - p0
    tf::T       # target duration
    tf_tf::T    # tf^2
    tf_p3::T    # tf^3
    tf_p4::T    # tf^4
    vd::T       # vf - v0
    vd_vd::T    # vd^2
    v0_v0::T    # v0^2
    vf_vf::T    # vf^2
    ad::T       # af - a0
    ad_ad::T    # ad^2
    a0_a0::T    # a0^2
    af_af::T    # af^2
    a0_p3::T    # a0^3
    a0_p4::T    # a0^4
    a0_p5::T    # a0^5
    a0_p6::T    # a0^6
    af_p3::T    # af^3
    af_p4::T    # af^4
    af_p5::T    # af^5
    af_p6::T    # af^6
    jMax_jMax::T  # jMax^2
    g1::T       # -pd + tf*v0
    g2::T       # -2pd + tf*(v0 + vf)
end

function Step2PreComputed(tf, p0, v0, a0, pf, vf, af, jMax)
    pd = pf - p0
    tf_tf = tf * tf
    tf_p3 = tf_tf * tf
    tf_p4 = tf_tf * tf_tf

    vd = vf - v0
    vd_vd = vd * vd
    v0_v0 = v0 * v0
    vf_vf = vf * vf

    ad = af - a0
    ad_ad = ad * ad
    a0_a0 = a0 * a0
    af_af = af * af

    a0_p3 = a0 * a0_a0
    a0_p4 = a0_a0 * a0_a0
    a0_p5 = a0_p3 * a0_a0
    a0_p6 = a0_p4 * a0_a0
    af_p3 = af * af_af
    af_p4 = af_af * af_af
    af_p5 = af_p3 * af_af
    af_p6 = af_p4 * af_af

    jMax_jMax = jMax * jMax
    g1 = -pd + tf * v0
    g2 = -2pd + tf * (v0 + vf)

    Step2PreComputed(pd, tf, tf_tf, tf_p3, tf_p4, vd, vd_vd, v0_v0, vf_vf,
                     ad, ad_ad, a0_a0, af_af, a0_p3, a0_p4, a0_p5, a0_p6,
                     af_p3, af_p4, af_p5, af_p6, jMax_jMax, g1, g2)
end

"""
Check profile for Step2 with target duration tf.
Returns true if profile is valid and matches duration.
"""
function check_step2!(buf::ProfileBuffer{T}, control_signs::ControlSigns, limits::ReachedLimits,
                      tf, jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af, jMax_limit=Inf) where T
    # Check jerk limit if provided
    abs(jf) > jMax_limit + EPS && return false

    # Use existing check function
    result = check!(buf, control_signs, limits, jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
    if !result
        return false
    end

    # Verify total duration matches tf
    abs(buf.t_sum[7] - tf) > T_PRECISION && return false

    return true
end

"""
    time_acc0_acc1_vel_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax)

Step2 profile reaching both acceleration limits and velocity limit.
"""
function time_acc0_acc1_vel_step2!(buf::ProfileBuffer{T}, pc::Step2PreComputed,
                                   p0, v0, a0, pf, vf, af,
                                   vMax, vMin, aMax, aMin, jMax) where T
    (; pd, tf, tf_tf, tf_p3, tf_p4, vd, vd_vd, ad, ad_ad,
       a0_a0, af_af, a0_p3, af_p3, a0_p4, af_p4, jMax_jMax, g1, g2) = pc

    # Profile UDDU, Solution 1
    if (2*(aMax - aMin) + ad)/jMax < tf
        h1_sq = (a0_p4 + af_p4 - 4*a0_p3*(2*aMax + aMin)/3 - 4*af_p3*(aMax + 2*aMin)/3 +
                 2*(a0_a0 - af_af)*aMax^2 +
                 (4*a0*aMax - 2*a0_a0)*(af_af - 2*af*aMin + (aMin - aMax)*aMin + 2*jMax*(aMin*tf - vd)) +
                 2*af_af*(aMin^2 + 2*jMax*(aMax*tf - vd)) +
                 4*jMax*(2*aMin*(af*vd + jMax*g1) + (aMax^2 - aMin^2)*vd + jMax*vd_vd) +
                 8*aMax*jMax_jMax*(pd - tf*vf))/(aMax*aMin) +
                4*af_af + 2*a0_a0 + (4*af + aMax - aMin)*(aMax - aMin) +
                4*jMax*(aMin - aMax + jMax*tf - 2*af)*tf

        h1_sq >= 0 || return false
        h1 = sqrt(h1_sq) * sign(jMax)

        buf.t[1] = (-a0 + aMax)/jMax
        buf.t[2] = (-(af_af - a0_a0 + 2*aMax^2 + aMin*(aMin - 2*ad - 3*aMax) + 2*jMax*(aMin*tf - vd)) + aMin*h1)/(2*(aMax - aMin)*jMax)
        buf.t[3] = aMax/jMax
        buf.t[4] = (aMin - aMax + h1)/(2*jMax)
        buf.t[5] = -aMin/jMax
        buf.t[6] = tf - (buf.t[1] + buf.t[2] + buf.t[3] + buf.t[4] + 2*buf.t[5] + af/jMax)
        buf.t[7] = buf.t[5] + af/jMax

        if check_step2!(buf, UDDU, LIMIT_ACC0_ACC1_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Profile UDUD
    if (-a0 + 4*aMax - af)/jMax < tf
        denom = a0_a0 + af_af - 2*(a0 + af)*aMax + 2*(aMax^2 - aMax*jMax*tf + jMax*vd)
        abs(denom) < EPS && return false

        buf.t[1] = (-a0 + aMax)/jMax
        buf.t[2] = (3*(a0_p4 + af_p4) - 4*(a0_p3 + af_p3)*aMax - 4*af_p3*aMax +
                    24*(a0 + af)*aMax^3 - 6*(af_af + a0_a0)*(aMax^2 - 2*jMax*vd) +
                    6*a0_a0*(af_af - 2*af*aMax - 2*aMax*jMax*tf) -
                    12*aMax^2*(2*aMax^2 - 2*aMax*jMax*tf + jMax*vd) -
                    24*af*aMax*jMax*vd + 12*jMax_jMax*(2*aMax*g1 + vd_vd))/(12*aMax*jMax*denom)
        buf.t[3] = aMax/jMax
        buf.t[4] = (-a0_a0 - af_af + 2*aMax*(a0 + af - 2*aMax) - 2*jMax*vd)/(2*aMax*jMax) + tf
        buf.t[5] = buf.t[3]
        buf.t[6] = tf - (buf.t[1] + buf.t[2] + buf.t[3] + buf.t[4] + 2*buf.t[5] - af/jMax)
        buf.t[7] = buf.t[5] - af/jMax

        if check_step2!(buf, UDUD, LIMIT_ACC0_ACC1_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
    time_acc1_vel_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax)

Step2 profile with ACC1 and velocity limit.
"""
function time_acc1_vel_step2!(roots::Roots, buf::ProfileBuffer{T}, pc::Step2PreComputed,
                              p0, v0, a0, pf, vf, af,
                              vMax, vMin, aMax, aMin, jMax) where T
    (; pd, tf, tf_tf, vd, vd_vd, ad, ad_ad,
       a0_a0, af_af, a0_p3, af_p3, a0_p4, af_p4, jMax_jMax, g1, g2) = pc

    # Profile UDDU
    ph1 = a0_a0 + af_af - aMin*(a0 + 2*af - aMin) - 2*jMax*(vd - aMin*tf)
    ph2 = 2*aMin*(jMax*g1 + af*vd) - aMin^2*vd + jMax*vd_vd
    ph3 = af_af + aMin*(aMin - 2*af) - 2*jMax*(vd - aMin*tf)

    polynom_0 = (2*(2*a0 - aMin))/jMax
    polynom_1 = (4*a0_a0 + ph1 - 3*a0*aMin)/jMax_jMax
    polynom_2 = (2*a0*ph1)/(jMax_jMax*jMax)
    polynom_3 = (3*(a0_p4 + af_p4) - 4*(a0_p3 + 2*af_p3)*aMin + 6*af_af*(aMin^2 - 2*jMax*vd) +
                 12*jMax*ph2 + 6*a0_a0*ph3)/(12*jMax_jMax*jMax_jMax)

    t_min = -a0/jMax
    t_max = min(tf + 2*aMin/jMax - (a0 + af)/jMax, 2*(aMax - a0)/jMax) / 2

    for t in solve_quartic_real!(roots, 1.0, polynom_0, polynom_1, polynom_2, polynom_3)
        (t < t_min || t > t_max) && continue

        # Newton refinement
        if abs(a0 + jMax*t) > 16*EPS
            h0 = jMax*t*t
            h1 = -((a0_a0 + af_af)/2 + jMax*(-vd + 2*a0*t + h0))/aMin
            orig = -pd + (3*(a0_p4 + af_p4) - 8*af_p3*aMin - 4*a0_p3*aMin +
                   6*af_af*(aMin^2 + 2*jMax*(h0 - vd)) +
                   6*a0_a0*(af_af - 2*af*aMin + aMin^2 + 2*aMin*jMax*(-2*t + tf) + 2*jMax*(5*h0 - vd)) +
                   24*a0*jMax*t*(a0_a0 + af_af - 2*af*aMin + aMin^2 + 2*jMax*(aMin*(-t + tf) + h0 - vd)) -
                   24*af*aMin*jMax*(h0 - vd) +
                   12*jMax*(aMin^2*(h0 - vd) + jMax*(h0 - vd)^2))/(24*aMin*jMax_jMax) +
                   h0*(tf - t) + tf*v0
            deriv = (a0 + jMax*t)*((a0_a0 + af_af)/(aMin*jMax) + (aMin - a0 - 2*af)/jMax +
                    (4*a0*t + 2*h0 - 2*vd)/aMin + 2*tf - 3*t)
            abs(deriv) > EPS && (t -= orig / deriv)
        end

        h1 = -((a0_a0 + af_af)/2 + jMax*(-vd + 2*a0*t + jMax*t*t))/aMin

        buf.t[1] = t
        buf.t[2] = 0
        buf.t[3] = a0/jMax + t
        buf.t[4] = tf - (h1 - aMin + a0 + af)/jMax - 2*t
        buf.t[5] = -aMin/jMax
        buf.t[6] = (h1 + aMin)/jMax
        buf.t[7] = buf.t[5] + af/jMax

        if check_step2!(buf, UDDU, LIMIT_ACC1_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Profile UDUD
    ph1 = a0_a0 - af_af + (2*af - a0)*aMax - aMax^2 - 2*jMax*(vd - aMax*tf)
    ph2 = aMax^2 + 2*jMax*vd
    ph3 = af_af + ph2 - 2*aMax*(af + jMax*tf)
    ph4 = 2*aMax*jMax*g1 + aMax^2*vd + jMax*vd_vd

    polynom_0 = (4*a0 - 2*aMax)/jMax
    polynom_1 = (4*a0_a0 - 3*a0*aMax + ph1)/jMax_jMax
    polynom_2 = (2*a0*ph1)/(jMax_jMax*jMax)
    polynom_3 = (3*(a0_p4 + af_p4) - 4*(a0_p3 + 2*af_p3)*aMax - 24*af*aMax*jMax*vd +
                 12*jMax*ph4 - 6*a0_a0*ph3 + 6*af_af*ph2)/(12*jMax_jMax*jMax_jMax)

    t_min = -a0/jMax
    t_max = min(tf + ad/jMax - 2*aMax/jMax, 2*(aMax - a0)/jMax) / 2

    for t in solve_quartic_real!(roots, 1.0, polynom_0, polynom_1, polynom_2, polynom_3)
        (t > t_max || t < t_min) && continue

        h1 = ((a0_a0 - af_af)/2 + jMax_jMax*t*t - jMax*(vd - 2*a0*t))/aMax

        buf.t[1] = t
        buf.t[2] = 0
        buf.t[3] = t + a0/jMax
        buf.t[4] = tf + (h1 + ad - aMax)/jMax - 2*t
        buf.t[5] = aMax/jMax
        buf.t[6] = -(h1 + aMax)/jMax
        buf.t[7] = buf.t[5] - af/jMax

        if check_step2!(buf, UDUD, LIMIT_ACC1_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
    time_acc0_vel_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax)

Step2 profile with ACC0 and velocity limit.
"""
function time_acc0_vel_step2!(roots::Roots, buf::ProfileBuffer{T}, pc::Step2PreComputed,
                              p0, v0, a0, pf, vf, af,
                              vMax, vMin, aMax, aMin, jMax) where T
    (; pd, tf, tf_tf, vd, vd_vd, ad, ad_ad,
       a0_a0, af_af, a0_p3, af_p3, a0_p4, af_p4, jMax_jMax, g1, g2) = pc

    # Early exit check
    tf < max((-a0 + aMax)/jMax, 0.0) + max(aMax/jMax, 0.0) && return false
    ph1 = 12*jMax*(-aMax^2*vd - jMax*vd_vd + 2*aMax*jMax*(-pd + tf*vf))

    # Profile UDDU
    polynom_0 = (2*aMax)/jMax
    polynom_1 = (a0_a0 - af_af + 2*ad*aMax + aMax^2 + 2*jMax*(vd - aMax*tf))/jMax_jMax
    polynom_2 = 0.0
    polynom_3 = -(-3*(a0_p4 + af_p4) + 4*(af_p3 + 2*a0_p3)*aMax -
                  12*a0*aMax*(af_af - 2*jMax*vd) + 6*a0_a0*(af_af - aMax^2 - 2*jMax*vd) +
                  6*af_af*(aMax^2 - 2*aMax*jMax*tf + 2*jMax*vd) + ph1)/(12*jMax_jMax*jMax_jMax)

    t_min = -af/jMax
    t_max = min(tf - (2*aMax - a0)/jMax, -aMin/jMax)

    for t in solve_quartic_real!(roots, 1.0, polynom_0, polynom_1, polynom_2, polynom_3)
        (t < t_min || t > t_max) && continue

        # Newton refinement
        if t > EPS
            h1 = jMax*t*t + vd
            orig = (-3*(a0_p4 + af_p4) + 4*(af_p3 + 2*a0_p3)*aMax - 24*af*aMax*jMax_jMax*t*t -
                    12*a0*aMax*(af_af - 2*jMax*h1) + 6*a0_a0*(af_af - aMax^2 - 2*jMax*h1) +
                    6*af_af*(aMax^2 - 2*aMax*jMax*tf + 2*jMax*h1) -
                    12*jMax*(aMax^2*h1 + jMax*h1^2 + 2*aMax*jMax*(pd + jMax*t*t*(t - tf) - tf*vf)))/(24*aMax*jMax_jMax)
            deriv = -t*(a0_a0 - af_af + 2*aMax*(ad - jMax*tf) + aMax^2 + 3*aMax*jMax*t + 2*jMax*h1)/aMax
            abs(deriv) > EPS && (t -= orig / deriv)
        end

        h1 = ((a0_a0 - af_af)/2 + jMax*(jMax*t*t + vd))/aMax

        buf.t[1] = (-a0 + aMax)/jMax
        buf.t[2] = (h1 - aMax)/jMax
        buf.t[3] = aMax/jMax
        buf.t[4] = tf - (h1 + ad + aMax)/jMax - 2*t
        buf.t[5] = t
        buf.t[6] = 0
        buf.t[7] = af/jMax + t

        if check_step2!(buf, UDDU, LIMIT_ACC0_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # Profile UDUD
    polynom_0 = (-2*aMax)/jMax
    polynom_1 = -(a0_a0 + af_af - 2*(a0 + af)*aMax + aMax^2 + 2*jMax*(vd - aMax*tf))/jMax_jMax
    polynom_2 = 0.0
    polynom_3 = (3*(a0_p4 + af_p4) - 4*(af_p3 + 2*a0_p3)*aMax +
                 6*a0_a0*(af_af + aMax^2 + 2*jMax*vd) - 12*a0*aMax*(af_af + 2*jMax*vd) +
                 6*af_af*(aMax^2 - 2*aMax*jMax*tf + 2*jMax*vd) - ph1)/(12*jMax_jMax*jMax_jMax)

    t_min = af/jMax
    t_max = min(tf - aMax/jMax, aMax/jMax)

    for t in solve_quartic_real!(roots, 1.0, polynom_0, polynom_1, polynom_2, polynom_3)
        (t < t_min || t > t_max) && continue

        # Newton refinement
        h1 = jMax*t*t - vd
        orig = -(3*(a0_p4 + af_p4) - 4*(2*a0_p3 + af_p3)*aMax + 24*af*aMax*jMax_jMax*t*t -
                 12*a0*aMax*(af_af - 2*jMax*h1) + 6*a0_a0*(af_af + aMax^2 - 2*jMax*h1) +
                 6*af_af*(aMax^2 - 2*jMax*(tf*aMax + h1)) +
                 12*jMax*(-aMax^2*h1 + jMax*h1^2 - 2*aMax*jMax*(-pd + jMax*t*t*(t - tf) + tf*vf)))/(24*aMax*jMax_jMax)
        deriv = t*(a0_a0 + af_af - 2*jMax*h1 - 2*(a0 + af + jMax*tf)*aMax + aMax^2 + 3*aMax*jMax*t)/aMax
        abs(deriv) > EPS && (t -= orig / deriv)

        h1 = ((a0_a0 + af_af)/2 + jMax*(vd - jMax*t*t))/aMax

        buf.t[1] = (-a0 + aMax)/jMax
        buf.t[2] = (h1 - aMax)/jMax
        buf.t[3] = aMax/jMax
        buf.t[4] = tf - (h1 - a0 - af + aMax)/jMax - 2*t
        buf.t[5] = t
        buf.t[6] = 0
        buf.t[7] = -(af/jMax) + t

        if check_step2!(buf, UDUD, LIMIT_ACC0_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
    time_acc0_acc1_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax)

Step2 profile reaching both acceleration limits (no velocity limit).
"""
function time_acc0_acc1_step2!(buf::ProfileBuffer{T}, pc::Step2PreComputed,
                               p0, v0, a0, pf, vf, af,
                               vMax, vMin, aMax, aMin, jMax) where T
    (; pd, tf, tf_tf, tf_p3, vd, vd_vd, ad, ad_ad,
       a0_a0, af_af, a0_p3, af_p3, a0_p4, af_p4, jMax_jMax, g1, g2) = pc

    # Simple case: a0 ≈ 0 and af ≈ 0
    if abs(a0) < EPS && abs(af) < EPS
        h1 = 2*aMin*g1 + vd_vd + aMax*(2*pd + aMin*tf_tf - 2*tf*vf)
        h2 = (aMax - aMin)*(-aMin*vd + aMax*(aMin*tf - vd))

        abs(h1) < EPS && return false
        jf = h2/h1

        abs(jf) < EPS && return false

        buf.t[1] = aMax/jf
        buf.t[2] = (-2*aMax*h1 + aMin^2*g2)/h2
        buf.t[3] = buf.t[1]
        buf.t[4] = 0
        buf.t[5] = -aMin/jf
        buf.t[6] = tf - (2*buf.t[1] + buf.t[2] + 2*buf.t[5])
        buf.t[7] = buf.t[5]

        if check_step2!(buf, UDDU, LIMIT_ACC0_ACC1, tf, jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af, abs(jMax))
            return true
        end
    end

    # UDDU general case
    h_denom = 2*aMin*g1 + vd_vd + aMax*(2*pd + aMin*tf_tf - 2*tf*vf)
    abs(h_denom) < EPS && return false

    h1_sq = 144*(((aMax - aMin)*(-aMin*vd + aMax*(aMin*tf - vd)) -
                   af_af*(aMax*tf - vd) + 2*af*aMin*(aMax*tf - vd) +
                   a0_a0*(aMin*tf + v0 - vf) - 2*a0*aMax*(aMin*tf - vd))^2) +
            48*ad*(3*a0_p3 - 3*af_p3 + 12*aMax*aMin*(-aMax + aMin) +
                   4*af_af*(aMax + 2*aMin) +
                   a0*(-3*af_af + 8*af*(aMin - aMax) + 6*(aMax^2 + 2*aMax*aMin - aMin^2)) +
                   6*af*(aMax^2 - 2*aMax*aMin - aMin^2) +
                   a0_a0*(3*af - 4*(2*aMax + aMin)))*h_denom

    h1_sq >= 0 || return false
    h1 = sqrt(h1_sq)

    jf = -(3*af_af*aMax*tf - 3*a0_a0*aMin*tf - 6*ad*aMax*aMin*tf +
           3*aMax*aMin*(aMin - aMax)*tf + 3*(a0_a0 - af_af)*vd +
           6*vd*(af*aMin - a0*aMax) + 3*(aMax^2 - aMin^2)*vd + h1/4)/(6*h_denom)

    abs(jf) < EPS && return false

    buf.t[1] = (aMax - a0)/jf
    buf.t[2] = (a0_a0 - af_af + 2*ad*aMin - 2*(aMax^2 - 2*aMax*aMin + aMin^2 + aMin*jf*tf - jf*vd))/(2*(aMax - aMin)*jf)
    buf.t[3] = aMax/jf
    buf.t[4] = 0
    buf.t[5] = -aMin/jf
    buf.t[6] = tf - (buf.t[1] + buf.t[2] + buf.t[3] + 2*buf.t[5] + af/jf)
    buf.t[7] = buf.t[5] + af/jf

    if check_step2!(buf, UDDU, LIMIT_ACC0_ACC1, tf, jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af, abs(jMax))
        return true
    end

    return false
end

"""
    time_acc0_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax)

Step2 profile with only ACC0 limit reached.
"""
function time_acc0_step2!(buf::ProfileBuffer{T}, pc::Step2PreComputed,
                          p0, v0, a0, pf, vf, af,
                          vMax, vMin, aMax, aMin, jMax) where T
    (; pd, tf, tf_tf, vd, vd_vd, ad, ad_ad,
       a0_a0, af_af, a0_p3, af_p3, a0_p4, jMax_jMax, g1, g2) = pc

    # UDUD case
    h1_sq = ad_ad/(2*jMax_jMax) - ad*(aMax - a0)/jMax_jMax + (aMax*tf - vd)/jMax
    if h1_sq >= 0
        h1 = sqrt(h1_sq)

        buf.t[1] = (aMax - a0)/jMax
        buf.t[2] = tf - ad/jMax - 2*h1
        buf.t[3] = h1
        buf.t[4] = 0
        buf.t[5] = (af - aMax)/jMax + h1
        buf.t[6] = 0
        buf.t[7] = 0

        if check_step2!(buf, UDUD, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # UDDU case with t[4] != 0
    h0a = -a0_a0 + af_af - 2*ad*aMax + 2*jMax*(aMax*tf - vd)
    h0b = a0_p3 + 2*af_p3 - 6*af_af*aMax - 3*a0_a0*(af - jMax*tf) -
          3*a0*aMax*(aMax - 2*af + 2*jMax*tf) -
          3*jMax*(jMax*(-2*pd + aMax*tf_tf + 2*tf*v0) + aMax*(aMax*tf - 2*vd)) +
          3*af*(aMax^2 + 2*aMax*jMax*tf - 2*jMax*vd)
    h0_sq = 4*h0b^2 - 18*h0a^3
    if h0_sq >= 0
        h0 = abs(jMax) * sqrt(h0_sq)
        h1 = 3*jMax*h0a

        abs(h1) < EPS && return false

        buf.t[1] = (-a0 + aMax)/jMax
        buf.t[2] = (-a0_p3 + af_p3 + af_af*(-6*aMax + 3*jMax*tf) +
                    a0_a0*(-3*af + 6*aMax + 3*jMax*tf) + 6*af*(aMax^2 - jMax*vd) +
                    3*a0*(af_af - 2*(aMax^2 + jMax*vd)) -
                    6*jMax*(aMax*(aMax*tf - 2*vd) + jMax*g2))/h1
        buf.t[3] = -(ad + h0/h1)/(2*jMax) + tf/2 - buf.t[2]/2
        buf.t[4] = h0/(jMax*h1)
        buf.t[5] = 0
        buf.t[6] = 0
        buf.t[7] = tf - (buf.t[1] + buf.t[2] + buf.t[3] + buf.t[4])

        if check_step2!(buf, UDDU, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # UDDU Solution 1
    h0b = a0_a0 + af_af + 2*(aMax^2 - (a0 + af)*aMax + jMax*(vd - aMax*tf))
    h0a = a0_p3 + 2*af_p3 - 6*(af_af + aMax^2)*aMax - 6*(a0 + af)*aMax*jMax*tf +
          9*aMax^2*(af + jMax*tf) + 3*a0*aMax*(-2*af + 3*aMax) +
          3*a0_a0*(af - 2*aMax + jMax*tf) - 6*jMax_jMax*g1 +
          6*(af - aMax)*jMax*vd - 3*aMax*jMax_jMax*tf_tf
    h0_sq = 4*h0a^2 - 18*h0b^3
    if h0_sq >= 0
        h1 = (jMax > 0 ? 1 : -1) * sqrt(h0_sq)
        h2 = 6*jMax*h0b

        abs(h2) < EPS && return false

        buf.t[1] = (-a0 + aMax)/jMax
        buf.t[2] = ad/jMax - 2*buf.t[1] - (2*h0a - h1)/h2 + tf
        buf.t[3] = -(2*h0a + h1)/h2
        buf.t[4] = (2*h0a - h1)/h2
        buf.t[5] = tf - (buf.t[1] + buf.t[2] + buf.t[3] + buf.t[4])
        buf.t[6] = 0
        buf.t[7] = 0

        if check_step2!(buf, UDDU, LIMIT_ACC0, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
    time_acc1_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax)

Step2 profile with only ACC1 limit reached.
"""
function time_acc1_step2!(buf::ProfileBuffer{T}, pc::Step2PreComputed,
                          p0, v0, a0, pf, vf, af,
                          vMax, vMin, aMax, aMin, jMax) where T
    (; pd, tf, tf_tf, vd, vd_vd, ad, ad_ad,
       a0_a0, af_af, a0_p3, af_p3, a0_p4, af_p4, jMax_jMax, g1, g2) = pc

    # UDDU case
    h0_sq = jMax_jMax*(a0_p4 + af_p4 - 4*af_p3*jMax*tf + 6*af_af*jMax_jMax*tf_tf -
            4*a0_p3*(af - jMax*tf) + 6*a0_a0*(af - jMax*tf)^2 + 24*af*jMax_jMax*g1 -
            4*a0*(af_p3 - 3*af_af*jMax*tf + 6*jMax_jMax*(-pd + tf*vf)) -
            12*jMax_jMax*(-vd_vd + jMax*tf*g2))/3
    if h0_sq >= 0
        h0 = sqrt(h0_sq)/jMax
        h1_sq = (a0_a0 + af_af - 2*a0*af - 2*ad*jMax*tf + 2*h0)/jMax_jMax + tf_tf
        if h1_sq >= 0
            h1 = sqrt(h1_sq)

            denom = 2*jMax*(-ad + jMax*tf)
            abs(denom) < EPS && return false

            buf.t[1] = -(a0_a0 + af_af + 2*a0*(jMax*tf - af) - 2*jMax*vd + h0)/denom
            buf.t[2] = 0
            buf.t[3] = (tf - h1)/2 - ad/(2*jMax)
            buf.t[4] = 0
            buf.t[5] = 0
            buf.t[6] = h1
            buf.t[7] = tf - (buf.t[1] + buf.t[3] + buf.t[6])

            if check_step2!(buf, UDDU, LIMIT_ACC1, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end

    # UDUD case
    h0_sq = jMax_jMax*(a0_p4 + af_p4 + 4*(af_p3 - a0_p3)*jMax*tf + 6*af_af*jMax_jMax*tf_tf +
            6*a0_a0*(af + jMax*tf)^2 + 24*af*jMax_jMax*g1 -
            4*a0*(a0_a0*af + af_p3 + 3*af_af*jMax*tf + 6*jMax_jMax*(-pd + tf*vf)) +
            12*jMax_jMax*(vd_vd + jMax*tf*g2))/3
    if h0_sq >= 0
        h0 = sqrt(h0_sq)/jMax
        h1_sq = (a0_a0 + af_af - 2*a0*af + 2*ad*jMax*tf + 2*h0)/jMax_jMax + tf_tf
        if h1_sq >= 0
            h1 = sqrt(h1_sq)

            denom = 2*jMax*(ad + jMax*tf)
            abs(denom) < EPS && return false

            buf.t[1] = 0
            buf.t[2] = 0
            buf.t[3] = -(a0_a0 + af_af - 2*a0*af + 2*jMax*(vd - a0*tf) + h0)/denom
            buf.t[4] = 0
            buf.t[5] = ad/(2*jMax) + (tf - h1)/2
            buf.t[6] = h1
            buf.t[7] = tf - (buf.t[6] + buf.t[5] + buf.t[3])

            if check_step2!(buf, UDUD, LIMIT_ACC1, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end

    # UDDU Solution 2
    h0b = a0_a0 + af_af - 2*(a0 + af)*aMin + 2*(aMin^2 - jMax*(-aMin*tf + vd))
    h0a = a0_p3 - af_p3 - 3*a0_a0*aMin + 3*aMin^2*(a0 + jMax*tf) +
          3*af*aMin*(-aMin - 2*jMax*tf) - 3*af_af*(-aMin - jMax*tf) -
          3*jMax_jMax*(-2*pd - aMin*tf_tf + 2*tf*vf)
    h0c = a0_p4 + 3*af_p4 - 4*(a0_p3 + 2*af_p3)*aMin + 6*a0_a0*aMin^2 +
          6*af_af*(aMin^2 - 2*jMax*vd) + 12*jMax*(2*aMin*jMax*g1 - aMin^2*vd + jMax*vd_vd) +
          24*af*aMin*jMax*vd -
          4*a0*(af_p3 - 3*af*aMin*(-aMin - 2*jMax*tf) + 3*af_af*(-aMin - jMax*tf) +
                3*jMax*(-aMin^2*tf + jMax*(-2*pd - aMin*tf_tf + 2*tf*vf)))
    h0_sq = 4*h0a^2 - 6*h0b*h0c
    if h0_sq >= 0
        h1 = (jMax > 0 ? 1 : -1) * sqrt(h0_sq)
        h2 = 6*jMax*h0b

        abs(h2) < EPS && return false

        buf.t[1] = 0
        buf.t[2] = 0
        buf.t[3] = (2*h0a + h1)/h2

        denom = 2*jMax*(a0 - aMin - jMax*buf.t[3])
        abs(denom) < EPS && return false

        buf.t[4] = -(a0_a0 + af_af - 2*(a0 + af)*aMin + 2*(aMin^2 + aMin*jMax*tf - jMax*vd))/denom
        buf.t[5] = (a0 - aMin)/jMax - buf.t[3]
        buf.t[6] = tf - (buf.t[3] + buf.t[4] + buf.t[5] + (af - aMin)/jMax)
        buf.t[7] = (af - aMin)/jMax

        if check_step2!(buf, UDDU, LIMIT_ACC1, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # UDUD Solution 1
    h0a = -a0_p3 + af_p3 + 3*(a0_a0 - af_af)*aMax - 3*ad*aMax^2 - 6*af*aMax*jMax*tf +
          3*af_af*jMax*tf + 3*jMax*(aMax^2*tf + jMax*(-2*pd - aMax*tf_tf + 2*tf*vf))
    h0b = a0_a0 - af_af + 2*ad*aMax + 2*jMax*(aMax*tf - vd)
    h0c = a0_p4 + 3*af_p4 - 4*(a0_p3 + 2*af_p3)*aMax + 6*a0_a0*aMax^2 - 24*af*aMax*jMax*vd +
          12*jMax*(2*aMax*jMax*g1 + jMax*vd_vd + aMax^2*vd) + 6*af_af*(aMax^2 + 2*jMax*vd) -
          4*a0*(af_p3 + 3*af*aMax*(aMax - 2*jMax*tf) - 3*af_af*(aMax - jMax*tf) +
                3*jMax*(aMax^2*tf + jMax*(-2*pd - aMax*tf_tf + 2*tf*vf)))
    h0_sq = 4*h0a^2 - 6*h0b*h0c
    if h0_sq >= 0
        h1 = (jMax > 0 ? 1 : -1) * sqrt(h0_sq)
        h2 = 6*jMax*h0b

        abs(h2) < EPS && return false

        buf.t[1] = 0
        buf.t[2] = 0
        buf.t[3] = -(2*h0a + h1)/h2
        buf.t[4] = 2*h1/h2
        buf.t[5] = (aMax - a0)/jMax + buf.t[3]
        buf.t[6] = tf - (buf.t[3] + buf.t[4] + buf.t[5] + (-af + aMax)/jMax)
        buf.t[7] = (-af + aMax)/jMax

        if check_step2!(buf, UDUD, LIMIT_ACC1, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
    time_none_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax)

Step2 profile with no limits reached.
"""
function time_none_step2!(roots::Roots, buf::ProfileBuffer{T}, pc::Step2PreComputed,
                          p0, v0, a0, pf, vf, af,
                          vMax, vMin, aMax, aMin, jMax) where T
    (; pd, tf, tf_tf, tf_p3, tf_p4, vd, vd_vd, v0_v0, vf_vf, ad, ad_ad,
       a0_a0, af_af, a0_p3, af_p3, a0_p4, af_p4, a0_p5, af_p5, a0_p6, af_p6, jMax_jMax, g1, g2) = pc

    # Special case: start from rest with zero acceleration (v0=a0=af=0)
    if abs(v0) < EPS && abs(a0) < EPS && abs(af) < EPS
        h1_sq = tf_tf*vf_vf + (4*pd - tf*vf)^2
        if h1_sq >= 0
            h1 = sqrt(h1_sq)
            jf = 4*(4*pd - 2*tf*vf + h1)/tf_p3

            if abs(jf) > EPS
                buf.t[1] = tf/4
                buf.t[2] = 0
                buf.t[3] = 2*buf.t[1]
                buf.t[4] = 0
                buf.t[5] = 0
                buf.t[6] = 0
                buf.t[7] = buf.t[1]

                if check_step2!(buf, UDDU, LIMIT_NONE, tf, jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af, abs(jMax))
                    return true
                end
            end
        end
    end

    # Case a0=af=0: Profile with constant velocity phase (t[3] != 0)
    if abs(a0) < EPS && abs(af) < EPS
        # UDDU with constant phase - quartic polynomial
        polynom_0 = -2*tf
        polynom_1 = 2*vd/jMax + tf_tf
        polynom_2 = 4*(pd - tf*vf)/jMax
        polynom_3 = (vd_vd + jMax*tf*g2)/jMax_jMax

        t_max = min(tf/2, (aMax - a0)/jMax)

        for t_root in solve_quartic_real!(roots, 1.0, polynom_0, polynom_1, polynom_2, polynom_3)
            (t_root > t_max || t_root < 0) && continue
            t = t_root

            # Newton refinement
            denom = jMax*(2*t - tf)
            if abs(denom) > EPS
                h1 = (jMax*t*(t - tf) + vd)/denom
                h2 = (2*jMax*t*(t - tf) + jMax*tf_tf - 2*vd)/(denom*(2*t - tf))
                orig = (-2*pd + 2*tf*v0 + h1^2*jMax*(tf - 2*t) + jMax*tf*(2*h1*t - t*t - (h1 - t)*tf))/2
                deriv = (jMax*tf*(2*t - tf)*(h2 - 1))/2 + h1*jMax*(tf - (2*t - tf)*h2 - h1)
                abs(deriv) > EPS && (t -= orig / deriv)
            end

            denom = jMax*(2*t - tf)
            abs(denom) < EPS && continue
            h1 = (jMax*t*(t - tf) + vd)/denom

            buf.t[1] = t
            buf.t[2] = 0
            buf.t[3] = h1
            buf.t[4] = tf - 2*t
            buf.t[5] = t - h1
            buf.t[6] = 0
            buf.t[7] = 0

            if check_step2!(buf, UDDU, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end

    # UDUD T 0246 - general case
    h0_inner = 2*(a0_p3 - af_p3 - 3*af_af*jMax*tf + 9*af*jMax_jMax*tf_tf - 3*a0_a0*(af + jMax*tf) +
                  3*a0*(af + jMax*tf)^2 + 3*jMax_jMax*(8*pd + jMax*tf_p3 - 8*tf*vf))^2 -
               3*(a0_a0 + af_af - 2*af*jMax*tf - 2*a0*(af + jMax*tf) - jMax*(jMax*tf_tf + 4*v0 - 4*vf))*
               (a0_p4 + af_p4 + 4*af_p3*jMax*tf + 6*af_af*jMax_jMax*tf_tf - 3*jMax_jMax*jMax_jMax*tf_p4 -
                4*a0_p3*(af + jMax*tf) + 6*a0_a0*(af + jMax*tf)^2 -
                12*af*jMax_jMax*(8*pd + jMax*tf_p3 - 8*tf*v0) + 48*jMax_jMax*vd_vd + 48*jMax_jMax*jMax*tf*g2 -
                4*a0*(af_p3 + 3*af_af*jMax*tf - 9*af*jMax_jMax*tf_tf - 3*jMax_jMax*(8*pd + jMax*tf_p3 - 8*tf*vf)))
    if h0_inner >= 0
        h0 = sqrt(2*jMax_jMax*h0_inner)/jMax
        h1 = 12*jMax*(-a0_a0 - af_af + 2*af*jMax*tf + 2*a0*(af + jMax*tf) + jMax*(jMax*tf_tf + 4*v0 - 4*vf))
        h2 = -4*a0_p3 + 4*af_p3 + 12*a0_a0*af - 12*a0*af_af + 48*jMax_jMax*pd +
             12*(a0_a0 - af_af)*jMax*tf - 24*jMax_jMax*tf*(v0 + vf) + 24*ad*jMax*vd
        h3 = 2*a0_p3 - 2*af_p3 - 6*a0_a0*af + 6*a0*af_af

        if abs(h1) > EPS
            buf.t[1] = (h3 - 48*jMax_jMax*(tf*vf - pd) - 6*(a0_a0 + af_af)*jMax*tf +
                        12*a0*af*jMax*tf + 6*(a0 + 3*af + jMax*tf)*tf_tf*jMax_jMax - h0)/h1
            buf.t[2] = 0
            buf.t[3] = (h2 + h0)/h1
            buf.t[4] = 0
            buf.t[5] = (-h2 + h0)/h1
            buf.t[6] = 0
            buf.t[7] = (-h3 + 48*jMax_jMax*(tf*v0 - pd) - 6*(a0_a0 + af_af)*jMax*tf +
                        12*a0*af*jMax*tf + 6*(af + 3*a0 + jMax*tf)*tf_tf*jMax_jMax - h0)/h1

            if check_step2!(buf, UDUD, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end

    # UDDU T 0234 - profiles with a3 != 0
    ph1 = af + jMax*tf

    polynom_0 = -2*(ad + jMax*tf)/jMax
    polynom_1 = 2*(a0_a0 + af_af + jMax*(af*tf + vd) - 2*a0*ph1)/jMax_jMax + tf_tf
    polynom_2 = 2*(a0_p3 - af_p3 - 3*af_af*jMax*tf + 3*a0*ph1*(ph1 - a0) - 6*jMax_jMax*(-pd + tf*vf))/(3*jMax_jMax*jMax)
    polynom_3 = (a0_p4 + af_p4 + 4*af_p3*jMax*tf - 4*a0_p3*ph1 + 6*a0_a0*ph1^2 + 24*jMax_jMax*af*g1 -
                 4*a0*(af_p3 + 3*af_af*jMax*tf + 6*jMax_jMax*(-pd + tf*vf)) + 6*jMax_jMax*af_af*tf_tf +
                 12*jMax_jMax*(vd_vd + jMax*tf*g2))/(12*jMax_jMax*jMax_jMax)

    t_min = ad/jMax
    t_max = min((aMax - a0)/jMax, (ad/jMax + tf) / 2)

    for t_root in solve_quartic_real!(roots, 1.0, polynom_0, polynom_1, polynom_2, polynom_3)
        (t_root < t_min || t_root > t_max) && continue
        t = t_root

        # Newton refinement
        h0 = jMax*(2*t - tf) - ad
        if abs(h0) > EPS
            h1 = (ad_ad - 2*af*jMax*t + 2*a0*jMax*(t - tf) + 2*jMax*(jMax*t*(t - tf) + vd))/(2*jMax*h0)
            h2 = (-ad_ad + 2*jMax_jMax*(tf_tf + t*(t - tf)) + (a0 + af)*jMax*tf - ad*h0 - 2*jMax*vd)/(h0*h0)
            orig = (-a0_p3 + af_p3 + 3*ad_ad*jMax*(h1 - t) + 3*ad*jMax_jMax*(h1 - t)^2 - 3*a0*af*ad +
                    3*jMax_jMax*(a0*tf_tf - 2*pd + 2*tf*v0 + h1^2*jMax*(tf - 2*t) + jMax*tf*(2*h1*t - t*t - (h1 - t)*tf)))/(6*jMax_jMax)
            deriv = (h0*(-ad + jMax*tf)*(h2 - 1))/(2*jMax) + h1*(-ad + jMax*(tf - h1) - h0*h2)
            abs(deriv) > EPS && (t -= orig / deriv)
        end

        h0 = jMax*(2*t - tf) - ad
        abs(h0) < EPS && continue
        h1 = (ad_ad + 2*jMax*(-a0*tf - ad*t + jMax*t*(t - tf) + vd))/(2*jMax*h0)

        buf.t[1] = t
        buf.t[2] = 0
        buf.t[3] = h1
        buf.t[4] = ad/jMax + tf - 2*t
        buf.t[5] = tf - (t + h1 + buf.t[4])
        buf.t[6] = 0
        buf.t[7] = 0

        if check_step2!(buf, UDDU, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # UDDU T 3456
    h1 = 3*jMax*(ad_ad + 2*jMax*(a0*tf - vd))
    h2 = ad_ad + 2*jMax*(a0*tf - vd)
    if abs(h1) > EPS && h2^3 >= 0
        h0_inner = 4*(2*(a0_p3 - af_p3) - 6*a0_a0*(af - jMax*tf) + 6*jMax_jMax*g1 +
                      3*a0*(2*af_af - 2*jMax*af*tf + jMax_jMax*tf_tf) + 6*ad*jMax*vd)^2 - 18*h2^3
        if h0_inner >= 0
            h0 = (jMax > 0 ? 1 : -1) * sqrt(h0_inner)/h1

            buf.t[1] = 0
            buf.t[2] = 0
            buf.t[3] = 0
            buf.t[4] = (af_p3 - a0_p3 + 3*(af_af - a0_a0)*jMax*tf - 3*ad*(a0*af + 2*jMax*vd) - 6*jMax_jMax*g2)/h1
            buf.t[5] = (tf - buf.t[4] - h0)/2 - ad/(2*jMax)
            buf.t[6] = h0
            buf.t[7] = (tf - buf.t[4] + ad/jMax - h0)/2

            if check_step2!(buf, UDDU, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end

    # T 2346 - UDDU with constant phase
    ph1 = ad_ad + 2*(af + a0)*jMax*tf - jMax*(jMax*tf_tf + 4*vd)
    if abs(ph1) > EPS
        ph2 = jMax*tf_tf*g1 - vd*(-2*pd - tf*v0 + 3*tf*vf)
        ph3 = 5*af_af - 8*af*jMax*tf + 2*jMax*(2*jMax*tf_tf - vd)
        ph4 = jMax_jMax*tf_p4 - 2*vd_vd + 8*jMax*tf*(-pd + tf*vf)
        ph5 = 5*af_p4 - 8*af_p3*jMax*tf - 12*af_af*jMax*(jMax*tf_tf + vd) +
              24*af*jMax_jMax*(-2*pd + jMax*tf_p3 + 2*tf*vf) - 6*jMax_jMax*ph4
        ph6 = -vd_vd + jMax*tf*(-2*pd + 3*tf*v0 - tf*vf) - af*g2

        polynom_0 = -(4*(a0_p3 - af_p3) - 12*a0_a0*(af - jMax*tf) +
                      6*a0*(2*af_af - 2*af*jMax*tf + jMax*(jMax*tf_tf - 2*vd)) +
                      6*af*jMax*(3*jMax*tf_tf + 2*vd) -
                      6*jMax_jMax*(-4*pd + jMax*tf_p3 - 2*tf*v0 + 6*tf*vf))/(3*jMax*ph1)
        polynom_1 = -(-a0_p4 - af_p4 + 4*a0_p3*(af - jMax*tf) +
                      a0_a0*(-6*af_af + 8*af*jMax*tf - 4*jMax*(jMax*tf_tf - vd)) +
                      2*af_af*jMax*(jMax*tf_tf + 2*vd) -
                      4*af*jMax_jMax*(-3*pd + jMax*tf_p3 + 2*tf*v0 + tf*vf) +
                      jMax_jMax*(jMax_jMax*tf_p4 - 8*vd_vd + 4*jMax*tf*(-3*pd + tf*v0 + 2*tf*vf)) +
                      2*a0*(2*af_p3 - 2*af_af*jMax*tf + af*jMax*(-3*jMax*tf_tf - 4*vd) +
                            jMax_jMax*(-6*pd + jMax*tf_p3 - 4*tf*v0 + 10*tf*vf)))/(jMax_jMax*ph1)
        polynom_2 = -(a0_p5 - af_p5 + af_p4*jMax*tf - 5*a0_p4*(af - jMax*tf) + 2*a0_p3*ph3 +
                      4*af_p3*jMax*(jMax*tf_tf + vd) + 12*jMax_jMax*af*ph6 -
                      2*a0_a0*(5*af_p3 - 9*af_af*jMax*tf - 6*af*jMax*vd +
                               6*jMax_jMax*(-2*pd - tf*v0 + 3*tf*vf)) -
                      12*jMax_jMax*jMax*ph2 + a0*ph5)/(3*jMax_jMax*jMax*ph1)
        polynom_3 = -(-a0_p6 - af_p6 + 6*a0_p5*(af - jMax*tf) - 48*af_p3*jMax_jMax*g1 +
                      72*jMax_jMax*jMax*(jMax*g1*g1 + vd_vd*vd + 2*af*g1*vd) - 3*a0_p4*ph3 -
                      36*af_af*jMax_jMax*vd_vd + 6*af_p4*jMax*vd +
                      4*a0_p3*(5*af_p3 - 9*af_af*jMax*tf - 6*af*jMax*vd +
                               6*jMax_jMax*(-2*pd - tf*v0 + 3*tf*vf)) - 3*a0_a0*ph5 +
                      6*a0*(af_p5 - af_p4*jMax*tf - 4*af_p3*jMax*(jMax*tf_tf + vd) +
                            12*jMax_jMax*(-af*ph6 + jMax*ph2)))/(18*jMax_jMax*jMax_jMax*ph1)

        t_max = (a0 - aMin)/jMax

        for t_root in solve_quartic_real!(roots, 1.0, polynom_0, polynom_1, polynom_2, polynom_3)
            t_root > t_max && continue
            t = t_root

            # Newton refinement
            h1_inner = ad_ad/2 + jMax*(af*t + (jMax*t - a0)*(t - tf) - vd)
            if h1_inner >= 0
                h2_tmp = -ad + jMax*(tf - 2*t)
                h3 = sqrt(h1_inner)
                orig = (af_p3 - a0_p3 + 3*af*jMax*t*(af + jMax*t) + 3*a0_a0*(af + jMax*t) -
                        3*a0*(af_af + 2*af*jMax*t + jMax_jMax*(t*t - tf_tf)) +
                        3*jMax_jMax*(-2*pd + jMax*t*(t - tf)*tf + 2*tf*v0))/(6*jMax_jMax) -
                       h3^3/(jMax*abs(jMax)) + ((-ad - jMax*t)*h1_inner)/jMax_jMax
                deriv = (6*jMax*h2_tmp*h3/abs(jMax) + 2*(-ad - jMax*tf)*h2_tmp -
                         2*(3*ad_ad + af*jMax*(8*t - 2*tf) + 4*a0*jMax*(-2*t + tf) +
                            2*jMax*(jMax*t*(3*t - 2*tf) - vd)))/(4*jMax)
                abs(deriv) > EPS && (t -= orig / deriv)
            end

            h1_inner = 2*ad_ad + 4*jMax*(ad*t + a0*tf + jMax*t*(t - tf) - vd)
            h1_inner < 0 && continue
            h1_val = sqrt(h1_inner)/abs(jMax)

            buf.t[1] = 0
            buf.t[2] = 0
            buf.t[3] = t
            buf.t[4] = tf - 2*t - ad/jMax - h1_val
            buf.t[5] = h1_val/2
            buf.t[6] = 0
            buf.t[7] = tf - (t + buf.t[4] + buf.t[5])

            if check_step2!(buf, UDDU, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end

    # T 0124 UDUD
    ph1 = -ad + jMax*tf
    if abs(ph1) > EPS
        ph0 = -2*pd - tf*v0 + 3*tf*vf
        ph2 = jMax*tf_tf*g1 - vd*ph0
        ph3 = 5*af_af + 2*jMax*(2*jMax*tf_tf - vd - 4*af*tf)
        ph4 = jMax_jMax*tf_p4 - 2*vd_vd + 8*jMax*tf*(-pd + tf*vf)
        ph5 = 5*af_p4 - 8*af_p3*jMax*tf - 12*af_af*jMax*(jMax*tf_tf + vd) +
              24*af*jMax_jMax*(-2*pd + jMax*tf_p3 + 2*tf*vf) - 6*jMax_jMax*ph4
        ph6 = -vd_vd + jMax*tf*(-2*pd + 3*tf*v0 - tf*vf)
        ph7 = 3*jMax_jMax*ph1^2

        abs(ph7) < EPS && @goto skip_t0124

        polynom_0 = (4*af*tf - 2*jMax*tf_tf - 4*vd)/ph1
        polynom_1 = (-2*(a0_p4 + af_p4) + 8*af_p3*jMax*tf + 6*af_af*jMax_jMax*tf_tf +
                     8*a0_p3*(af - jMax*tf) - 12*a0_a0*(af - jMax*tf)^2 -
                     12*af*jMax_jMax*(-pd + jMax*tf_p3 - 2*tf*v0 + 3*tf*vf) +
                     2*a0*(4*af_p3 - 12*af_af*jMax*tf + 9*af*jMax_jMax*tf_tf -
                           3*jMax_jMax*(2*pd + jMax*tf_p3 - 2*tf*vf)) +
                     3*jMax_jMax*(jMax_jMax*tf_p4 + 4*vd_vd - 4*jMax*tf*(pd + tf*v0 - 2*tf*vf)))/ph7
        polynom_2 = (-a0_p5 + af_p5 - af_p4*jMax*tf + 5*a0_p4*(af - jMax*tf) - 2*a0_p3*ph3 -
                     4*af_p3*jMax*(jMax*tf_tf + vd) + 12*af_af*jMax_jMax*g2 - 12*af*jMax_jMax*ph6 +
                     2*a0_a0*(5*af_p3 - 9*af_af*jMax*tf - 6*af*jMax*vd + 6*jMax_jMax*ph0) +
                     12*jMax_jMax*jMax*ph2 +
                     a0*(-5*af_p4 + 8*af_p3*jMax*tf + 12*af_af*jMax*(jMax*tf_tf + vd) -
                         24*af*jMax_jMax*(-2*pd + jMax*tf_p3 + 2*tf*vf) + 6*jMax_jMax*ph4))/(jMax*ph7)
        polynom_3 = -(a0_p6 + af_p6 - 6*a0_p5*(af - jMax*tf) + 48*af_p3*jMax_jMax*g1 -
                      72*jMax_jMax*jMax*(jMax*g1*g1 + vd_vd*vd + 2*af*g1*vd) + 3*a0_p4*ph3 -
                      6*af_p4*jMax*vd + 36*af_af*jMax_jMax*vd_vd -
                      4*a0_p3*(5*af_p3 - 9*af_af*jMax*tf - 6*af*jMax*vd + 6*jMax_jMax*ph0) + 3*a0_a0*ph5 -
                      6*a0*(af_p5 - af_p4*jMax*tf - 4*af_p3*jMax*(jMax*tf_tf + vd) +
                            12*jMax_jMax*(af_af*g2 - af*ph6 + jMax*ph2)))/(6*jMax_jMax*ph7)

        for t_root in solve_quartic_real!(roots, 1.0, polynom_0, polynom_1, polynom_2, polynom_3)
            (t_root > tf || t_root > (aMax - a0)/jMax) && continue
            t = t_root

            h1_inner = ad_ad/(2*jMax_jMax) + (a0*(t + tf) - af*t + jMax*t*tf - vd)/jMax
            h1_inner < 0 && continue
            h1_val = sqrt(h1_inner)

            buf.t[1] = t
            buf.t[2] = tf - ad/jMax - 2*h1_val
            buf.t[3] = h1_val
            buf.t[4] = 0
            buf.t[5] = ad/jMax + h1_val - t
            buf.t[6] = 0
            buf.t[7] = 0

            if check_step2!(buf, UDUD, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    end
    @label skip_t0124

    # 3-step profile (UZD)
    h1_sq = -ad_ad + jMax*(2*(a0 + af)*tf - 4*vd + jMax*tf_tf)
    if h1_sq >= 0
        h1 = sqrt(h1_sq) / abs(jMax)

        buf.t[1] = (tf - h1 + ad/jMax)/2
        buf.t[2] = h1
        buf.t[3] = (tf - h1 - ad/jMax)/2
        buf.t[4] = 0
        buf.t[5] = 0
        buf.t[6] = 0
        buf.t[7] = 0

        if check_step2!(buf, UDDU, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    # 3-step profile (UZU) - cubic polynomial
    polynom_0 = ad_ad
    polynom_1 = ad_ad*tf
    polynom_2 = (a0_a0 + af_af + 10*a0*af)*tf_tf + 24*(tf*(af*v0 - a0*vf) - pd*ad) + 12*vd_vd
    polynom_3 = -3*tf*((a0_a0 + af_af + 2*a0*af)*tf_tf - 4*vd*(a0 + af)*tf + 4*vd_vd)

    for t_root in solve_cubic_real!(roots, polynom_0, polynom_1, polynom_2, polynom_3)
        t_root > tf && continue
        t = t_root

        denom = tf - t
        abs(denom) < EPS && continue
        jf = ad/denom

        abs(jf) < EPS && continue
        denom2 = 2*jf*t
        abs(denom2) < EPS && continue

        buf.t[1] = (2*(vd - a0*tf) + ad*(t - tf))/denom2
        buf.t[2] = t
        buf.t[3] = 0
        buf.t[4] = 0
        buf.t[5] = 0
        buf.t[6] = 0
        buf.t[7] = tf - (buf.t[1] + buf.t[2])

        if check_step2!(buf, UDDU, LIMIT_NONE, tf, jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af, abs(jMax))
            return true
        end
    end

    # 3-step profile (UDU)
    denom = 4*(ad - jMax*tf)
    if abs(denom) > EPS
        buf.t[1] = (ad_ad/jMax + 2*(a0 + af)*tf - jMax*tf_tf - 4*vd)/denom
        buf.t[2] = 0
        buf.t[3] = -ad/(2*jMax) + tf/2
        buf.t[4] = 0
        buf.t[5] = 0
        buf.t[6] = 0
        buf.t[7] = tf - (buf.t[1] + buf.t[3])

        if check_step2!(buf, UDDU, LIMIT_NONE, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            return true
        end
    end

    return false
end

"""
    time_vel_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax)

Step2 profile with velocity limit only (no ACC0 or ACC1).
"""
function time_vel_step2!(roots::Roots, buf::ProfileBuffer{T}, pc::Step2PreComputed,
                         p0, v0, a0, pf, vf, af,
                         vMax, vMin, aMax, aMin, jMax) where T
    (; pd, tf, tf_tf, vd, vd_vd, ad, ad_ad,
       a0_a0, af_af, a0_p3, af_p3, a0_p4, af_p4, a0_p6, af_p6, jMax_jMax, g1, g2) = pc

    tz_min = max(0.0, -a0/jMax)
    tz_max = min((tf - a0/jMax)/2, (aMax - a0)/jMax)

    # Profile UDDU - simple case when v0=a0=vf=af=0
    if abs(v0) < EPS && abs(a0) < EPS && abs(vf) < EPS && abs(af) < EPS
        # Solve cubic: t³ - (tf/2)t² + pd/(2*jMax) = 0
        for t_root in solve_cubic_real!(roots, 1.0, -tf/2, 0.0, pd/(2*jMax))
            t_root > tf/4 && continue
            t = t_root

            # Newton refinement
            if t > EPS
                orig = -pd + jMax*t*t*(tf - 2*t)
                deriv = 2*jMax*t*(tf - 3*t)
                abs(deriv) > EPS && (t -= orig / deriv)
            end

            buf.t[1] = t
            buf.t[2] = 0
            buf.t[3] = t
            buf.t[4] = tf - 4*t
            buf.t[5] = t
            buf.t[6] = 0
            buf.t[7] = t

            if check_step2!(buf, UDDU, LIMIT_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end
        end
    else
        # UDDU - general case with 5th order polynomial
        p1 = af_af - 2*jMax*(-2*af*tf + jMax*tf_tf + 3*vd)
        ph1 = af_p3 - 3*jMax_jMax*g1 - 3*af*jMax*vd
        ph2 = af_p4 + 8*af_p3*jMax*tf + 12*jMax*(3*jMax*vd_vd - af_af*vd + 2*af*jMax*(g1 - tf*vd) - 2*jMax_jMax*tf*g1)
        ph3 = a0*(af - jMax*tf)
        ph4 = jMax*(-ad + jMax*tf)

        abs(ph4) < EPS && @goto udud_general

        # 5th order polynomial coefficients
        polynom_0 = (15*a0_a0 + af_af + 4*af*jMax*tf - 16*ph3 - 2*jMax*(jMax*tf_tf + 3*vd))/(4*ph4)
        polynom_1 = (29*a0_p3 - 2*af_p3 - 33*a0*ph3 + 6*jMax_jMax*g1 + 6*af*jMax*vd + 6*a0*p1)/(6*jMax*ph4)
        polynom_2 = (61*a0_p4 - 76*a0_a0*ph3 - 16*a0*ph1 + 30*a0_a0*p1 + ph2)/(24*jMax_jMax*ph4)
        polynom_3 = (a0*(7*a0_p4 - 10*a0_a0*ph3 - 4*a0*ph1 + 6*a0_a0*p1 + ph2))/(12*jMax_jMax*jMax*ph4)
        polynom_4 = (7*a0_p6 + af_p6 - 12*a0_p4*ph3 + 48*af_p3*jMax_jMax*g1 - 8*a0_p3*ph1 -
                     72*jMax_jMax*jMax*(jMax*g1*g1 + vd_vd*vd + 2*af*g1*vd) - 6*af_p4*jMax*vd +
                     36*af_af*jMax_jMax*vd_vd + 9*a0_p4*p1 + 3*a0_a0*ph2)/(144*jMax_jMax*jMax_jMax*ph4)

        # Solve quintic using quartic derivative to find intervals
        deriv_0 = 5.0
        deriv_1 = 4*polynom_0
        deriv_2 = 3*polynom_1
        deriv_3 = 2*polynom_2
        deriv_4 = polynom_3

        tz_current = tz_min

        for tz in solve_quartic_real!(roots, deriv_0, deriv_1, deriv_2, deriv_3, deriv_4)
            tz >= tz_max && continue
            tz < tz_min && continue

            # Evaluate polynomial at tz and tz_current
            val_current = polynom_4 + tz_current*(polynom_3 + tz_current*(polynom_2 + tz_current*(polynom_1 + tz_current*(polynom_0 + tz_current))))
            val_new = polynom_4 + tz*(polynom_3 + tz*(polynom_2 + tz*(polynom_1 + tz*(polynom_0 + tz))))

            if val_current * val_new < 0
                # Root in interval - use bisection
                t = shrink_interval_poly5(polynom_0, polynom_1, polynom_2, polynom_3, polynom_4, tz_current, tz)

                # Newton refinement
                h1_sq = (a0_a0 + af_af)/(2*jMax_jMax) + (2*a0*t + jMax*t*t - vd)/jMax
                if h1_sq >= 0
                    h1 = sqrt(h1_sq)
                    orig = -pd - (2*a0_p3 + 4*af_p3 + 24*a0*jMax*t*(af + jMax*(h1 + t - tf)) +
                           6*a0_a0*(af + jMax*(2*t - tf)) + 6*(a0_a0 + af_af)*jMax*h1 +
                           12*af*jMax*(jMax*t*t - vd) +
                           12*jMax_jMax*(jMax*t*t*(h1 + t - tf) - tf*v0 - h1*vd))/(12*jMax_jMax)
                    deriv_newton = -(a0 + jMax*t)*(3*(h1 + t) - 2*tf + (a0 + 2*af)/jMax)
                    if !isnan(orig) && !isnan(deriv_newton) && abs(deriv_newton) > EPS
                        t -= orig / deriv_newton
                    end
                end

                h1_sq = (a0_a0 + af_af)/(2*jMax_jMax) + (t*(2*a0 + jMax*t) - vd)/jMax
                h1_sq < 0 && (tz_current = tz; continue)
                h1 = sqrt(h1_sq)

                buf.t[1] = t
                buf.t[2] = 0
                buf.t[3] = t + a0/jMax
                buf.t[4] = tf - 2*(t + h1) - (a0 + af)/jMax
                buf.t[5] = h1
                buf.t[6] = 0
                buf.t[7] = h1 + af/jMax

                if check_step2!(buf, UDDU, LIMIT_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                    return true
                end
            end
            tz_current = tz
        end

        # Check interval from last extrema to tz_max
        val_current = polynom_4 + tz_current*(polynom_3 + tz_current*(polynom_2 + tz_current*(polynom_1 + tz_current*(polynom_0 + tz_current))))
        val_max = polynom_4 + tz_max*(polynom_3 + tz_max*(polynom_2 + tz_max*(polynom_1 + tz_max*(polynom_0 + tz_max))))
        if val_current * val_max < 0
            t = shrink_interval_poly5(polynom_0, polynom_1, polynom_2, polynom_3, polynom_4, tz_current, tz_max)

            h1_sq = (a0_a0 + af_af)/(2*jMax_jMax) + (t*(2*a0 + jMax*t) - vd)/jMax
            if h1_sq >= 0
                h1 = sqrt(h1_sq)

                buf.t[1] = t
                buf.t[2] = 0
                buf.t[3] = t + a0/jMax
                buf.t[4] = tf - 2*(t + h1) - (a0 + af)/jMax
                buf.t[5] = h1
                buf.t[6] = 0
                buf.t[7] = h1 + af/jMax

                if check_step2!(buf, UDDU, LIMIT_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                    return true
                end
            end
        end
    end

    @label udud_general
    # Profile UDUD - general case with 6th order polynomial
    if !(abs(v0) < EPS && abs(a0) < EPS && abs(vf) < EPS && abs(af) < EPS)
        ph1 = af_af - 2*jMax*(2*af*tf + jMax*tf_tf - 3*vd)
        ph2 = af_p3 - 3*jMax_jMax*g1 + 3*af*jMax*vd
        ph3 = 2*jMax*tf*g1 + 3*vd_vd
        ph4 = af_p4 - 8*af_p3*jMax*tf + 12*jMax*(jMax*ph3 + af_af*vd + 2*af*jMax*(g1 - tf*vd))
        ph5 = af + jMax*tf

        # 6th order polynomial coefficients
        polynom_0 = (5*a0 - ph5)/jMax
        polynom_1 = (39*a0_a0 - ph1 - 16*a0*ph5)/(4*jMax_jMax)
        polynom_2 = (55*a0_p3 - 33*a0_a0*ph5 - 6*a0*ph1 + 2*ph2)/(6*jMax_jMax*jMax)
        polynom_3 = (101*a0_p4 + ph4 - 76*a0_p3*ph5 - 30*a0_a0*ph1 + 16*a0*ph2)/(24*jMax_jMax*jMax_jMax)
        polynom_4 = (a0*(11*a0_p4 + ph4 - 10*a0_p3*ph5 - 6*a0_a0*ph1 + 4*a0*ph2))/(12*jMax_jMax*jMax_jMax*jMax)
        polynom_5 = (11*a0_p6 - af_p6 - 12*a0_p4*(a0*ph5) - 48*af_p3*jMax_jMax*g1 - 9*a0_p4*ph1 +
                     72*jMax_jMax*jMax*(jMax*g1*g1 - vd_vd*vd - 2*af*g1*vd) - 6*af_p4*jMax*vd -
                     36*af_af*jMax_jMax*vd_vd + 8*a0_p3*ph2 + 3*a0_a0*ph4)/(144*jMax_jMax*jMax_jMax*jMax_jMax)

        # Solve using quintic derivative to find intervals
        deriv_0 = 6.0
        deriv_1 = 5*polynom_0
        deriv_2 = 4*polynom_1
        deriv_3 = 3*polynom_2
        deriv_4 = 2*polynom_3
        deriv_5 = polynom_4

        # Use quartic solution on 4th derivative to find extrema
        dderiv_0 = 30.0
        dderiv_1 = 20*polynom_0
        dderiv_2 = 12*polynom_1
        dderiv_3 = 6*polynom_2
        dderiv_4 = 2*polynom_3

        dd_tz_current = tz_min
        intervals = NTuple{2,T}[]

        for tz in solve_quartic_real!(roots, dderiv_0, dderiv_1, dderiv_2, dderiv_3, dderiv_4)
            tz >= tz_max && continue
            tz < tz_min && continue

            # Check sign change in 1st derivative
            val_current = deriv_5 + dd_tz_current*(deriv_4 + dd_tz_current*(deriv_3 + dd_tz_current*(deriv_2 + dd_tz_current*(deriv_1 + dd_tz_current*deriv_0))))
            val_new = deriv_5 + tz*(deriv_4 + tz*(deriv_3 + tz*(deriv_2 + tz*(deriv_1 + tz*deriv_0))))

            if val_current * val_new < 0
                push!(intervals, (dd_tz_current, tz))
            end
            dd_tz_current = tz
        end

        # Check last interval
        val_current = deriv_5 + dd_tz_current*(deriv_4 + dd_tz_current*(deriv_3 + dd_tz_current*(deriv_2 + dd_tz_current*(deriv_1 + dd_tz_current*deriv_0))))
        val_max = deriv_5 + tz_max*(deriv_4 + tz_max*(deriv_3 + tz_max*(deriv_2 + tz_max*(deriv_1 + tz_max*deriv_0))))
        if val_current * val_max < 0
            push!(intervals, (dd_tz_current, tz_max))
        end

        tz_current = tz_min

        for (int_lo, int_hi) in intervals
            # Find root of derivative in interval (extremum of polynomial)
            tz = shrink_interval_poly5(deriv_1/deriv_0, deriv_2/deriv_0, deriv_3/deriv_0, deriv_4/deriv_0, deriv_5/deriv_0, int_lo, int_hi)

            tz >= tz_max && continue

            # Evaluate polynomial at tz
            p_val = polynom_5 + tz*(polynom_4 + tz*(polynom_3 + tz*(polynom_2 + tz*(polynom_1 + tz*(polynom_0 + tz)))))

            # Check if extremum is close to zero (root)
            ddval = dderiv_4 + tz*(dderiv_3 + tz*(dderiv_2 + tz*(dderiv_1 + tz*dderiv_0)))
            if abs(p_val) < 64 * abs(ddval) * 1e-12
                t = tz
            else
                # Check for sign change
                p_val_current = polynom_5 + tz_current*(polynom_4 + tz_current*(polynom_3 + tz_current*(polynom_2 + tz_current*(polynom_1 + tz_current*(polynom_0 + tz_current)))))
                if p_val_current * p_val < 0
                    t = shrink_interval_poly6(polynom_0, polynom_1, polynom_2, polynom_3, polynom_4, polynom_5, tz_current, tz)
                else
                    tz_current = tz
                    continue
                end
            end

            # Double Newton step
            h1_sq = (af_af - a0_a0)/(2*jMax_jMax) - ((2*a0 + jMax*t)*t - vd)/jMax
            if h1_sq >= 0
                h1 = sqrt(h1_sq)
                orig = -pd + (af_p3 - a0_p3 + 3*a0_a0*jMax*(tf - 2*t))/(6*jMax_jMax) + (2*a0 + jMax*t)*t*(tf - t) + (jMax*h1 - af)*h1*h1 + tf*v0
                deriv_newton = (a0 + jMax*t)*(2*(af + jMax*tf) - 3*jMax*(h1 + t) - a0)/jMax

                if abs(deriv_newton) > EPS
                    t -= orig / deriv_newton

                    h1_sq = (af_af - a0_a0)/(2*jMax_jMax) - ((2*a0 + jMax*t)*t - vd)/jMax
                    if h1_sq >= 0
                        h1 = sqrt(h1_sq)
                        orig = -pd + (af_p3 - a0_p3 + 3*a0_a0*jMax*(tf - 2*t))/(6*jMax_jMax) + (2*a0 + jMax*t)*t*(tf - t) + (jMax*h1 - af)*h1*h1 + tf*v0
                        if abs(orig) > 1e-9
                            deriv_newton = (a0 + jMax*t)*(2*(af + jMax*tf) - 3*jMax*(h1 + t) - a0)/jMax
                            abs(deriv_newton) > EPS && (t -= orig / deriv_newton)
                        end
                    end
                end
            end

            h1_sq = (af_af - a0_a0)/(2*jMax_jMax) - ((2*a0 + jMax*t)*t - vd)/jMax
            h1_sq < 0 && (tz_current = tz; continue)
            h1 = sqrt(h1_sq)

            buf.t[1] = t
            buf.t[2] = 0
            buf.t[3] = t + a0/jMax
            buf.t[4] = tf - 2*(t + h1) + ad/jMax
            buf.t[5] = h1
            buf.t[6] = 0
            buf.t[7] = h1 - af/jMax

            if check_step2!(buf, UDUD, LIMIT_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                return true
            end

            tz_current = tz
        end

        # Check last interval to tz_max
        p_val_current = polynom_5 + tz_current*(polynom_4 + tz_current*(polynom_3 + tz_current*(polynom_2 + tz_current*(polynom_1 + tz_current*(polynom_0 + tz_current)))))
        p_val_max = polynom_5 + tz_max*(polynom_4 + tz_max*(polynom_3 + tz_max*(polynom_2 + tz_max*(polynom_1 + tz_max*(polynom_0 + tz_max)))))
        if p_val_current * p_val_max < 0
            t = shrink_interval_poly6(polynom_0, polynom_1, polynom_2, polynom_3, polynom_4, polynom_5, tz_current, tz_max)

            h1_sq = (af_af - a0_a0)/(2*jMax_jMax) - ((2*a0 + jMax*t)*t - vd)/jMax
            if h1_sq >= 0
                h1 = sqrt(h1_sq)

                buf.t[1] = t
                buf.t[2] = 0
                buf.t[3] = t + a0/jMax
                buf.t[4] = tf - 2*(t + h1) + ad/jMax
                buf.t[5] = h1
                buf.t[6] = 0
                buf.t[7] = h1 - af/jMax

                if check_step2!(buf, UDUD, LIMIT_VEL, tf, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
                    return true
                end
            end
        end
    end

    return false
end

# Helper for bisection on 5th degree polynomial
function shrink_interval_poly5(c0, c1, c2, c3, c4, lo, hi)
    for _ in 1:64
        mid = (lo + hi) / 2
        val_mid = c4 + mid*(c3 + mid*(c2 + mid*(c1 + mid*(c0 + mid))))
        val_lo = c4 + lo*(c3 + lo*(c2 + lo*(c1 + lo*(c0 + lo))))
        if val_lo * val_mid < 0
            hi = mid
        else
            lo = mid
        end
        abs(hi - lo) < 1e-14 && break
    end
    return (lo + hi) / 2
end

# Helper for bisection on 6th degree polynomial
function shrink_interval_poly6(c0, c1, c2, c3, c4, c5, lo, hi)
    for _ in 1:64
        mid = (lo + hi) / 2
        val_mid = c5 + mid*(c4 + mid*(c3 + mid*(c2 + mid*(c1 + mid*(c0 + mid)))))
        val_lo = c5 + lo*(c4 + lo*(c3 + lo*(c2 + lo*(c1 + lo*(c0 + lo)))))
        if val_lo * val_mid < 0
            hi = mid
        else
            lo = mid
        end
        abs(hi - lo) < 1e-14 && break
    end
    return (lo + hi) / 2
end

"""
    calculate_profile_step2!(roots, buf, tf, p0, v0, a0, pf, vf, af, jMax, vmax, vmin, amax, amin)

Calculate a profile that achieves the target duration tf.
This is Step 2 of the Ruckig algorithm - time synchronization.

Returns true if a valid profile was found.
"""
function calculate_profile_step2!(roots::Roots, buf::ProfileBuffer{T}, tf, p0, v0, a0, pf, vf, af,
                                  jMax, vmax, vmin, amax, amin) where T
    pc = Step2PreComputed(tf, p0, v0, a0, pf, vf, af, jMax)
    pd = pf - p0

    # Determine primary direction
    up_first = pd > tf * v0

    if up_first
        vMax, vMin = vmax, vmin
        aMax, aMin = amax, amin
        jMax_dir = jMax
    else
        vMax, vMin = vmin, vmax
        aMax, aMin = amin, amax
        jMax_dir = -jMax
    end

    # Try velocity-limited profiles (UP direction)
    time_acc0_acc1_vel_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax_dir) && return true
    time_vel_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax_dir) && return true
    time_acc0_vel_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax_dir) && return true
    time_acc1_vel_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax_dir) && return true

    # Try velocity-limited profiles (DOWN direction)
    time_acc0_acc1_vel_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMin, vMax, aMin, aMax, -jMax_dir) && return true
    time_vel_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMin, vMax, aMin, aMax, -jMax_dir) && return true
    time_acc0_vel_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMin, vMax, aMin, aMax, -jMax_dir) && return true
    time_acc1_vel_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMin, vMax, aMin, aMax, -jMax_dir) && return true

    # Try acceleration-limited profiles (UP direction)
    time_acc0_acc1_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax_dir) && return true
    time_acc0_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax_dir) && return true
    time_acc1_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax_dir) && return true
    time_none_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMax, vMin, aMax, aMin, jMax_dir) && return true

    # Try acceleration-limited profiles (DOWN direction)
    time_acc0_acc1_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMin, vMax, aMin, aMax, -jMax_dir) && return true
    time_acc0_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMin, vMax, aMin, aMax, -jMax_dir) && return true
    time_acc1_step2!(buf, pc, p0, v0, a0, pf, vf, af, vMin, vMax, aMin, aMax, -jMax_dir) && return true
    time_none_step2!(roots, buf, pc, p0, v0, a0, pf, vf, af, vMin, vMax, aMin, aMax, -jMax_dir) && return true

    return false
end


#=============================================================================
 Multi-DOF Trajectory Calculation
=============================================================================#

"""
    calculate_trajectory(lims::AbstractVector{<:JerkLimiter}; pf, p0, v0, a0, vf, af)

Calculate time-synchronized trajectories for multiple degrees of freedom.
All DOFs will have the same total duration.

# Arguments
- `lims`: Vector of JerkLimiter, one per DOF
- `pf`: Vector of target positions (required)
- `p0`: Vector of initial positions (default: zeros)
- `v0`: Vector of initial velocities (default: zeros)
- `a0`: Vector of initial accelerations (default: zeros)
- `vf`: Vector of final velocities (default: zeros)
- `af`: Vector of final accelerations (default: zeros)

# Returns
Vector of RuckigProfile, one per DOF, all with the same duration.
"""
function calculate_trajectory(lims::AbstractVector{<:JerkLimiter{T}};
    pf::AbstractVector,
    p0::AbstractVector = zeros(T, length(lims)),
    v0::AbstractVector = zeros(T, length(lims)),
    a0::AbstractVector = zeros(T, length(lims)),
    vf::AbstractVector = zeros(T, length(lims)),
    af::AbstractVector = zeros(T, length(lims)),
) where T
    ndof = length(lims)
    length(pf) == ndof || throw(ArgumentError("pf must have length $ndof"))
    length(p0) == ndof || throw(ArgumentError("p0 must have length $ndof"))
    length(v0) == ndof || throw(ArgumentError("v0 must have length $ndof"))
    length(a0) == ndof || throw(ArgumentError("a0 must have length $ndof"))
    length(vf) == ndof || throw(ArgumentError("vf must have length $ndof"))
    length(af) == ndof || throw(ArgumentError("af must have length $ndof"))

    # Step 1: Calculate minimum-time profile for each DOF independently
    blocks = Vector{Block{Float64}}(undef, ndof)
    for i in 1:ndof
        profile = calculate_trajectory(lims[i];
            p0=p0[i], v0=v0[i], a0=a0[i],
            pf=pf[i], vf=vf[i], af=af[i])
        blocks[i] = Block(profile)
    end

    # Find synchronization time (maximum of all minimum times)
    t_sync = maximum(block.t_min for block in blocks)
    limiting_dof = argmax([block.t_min for block in blocks])

    # Check if t_sync is blocked for any DOF
    for i in 1:ndof
        if is_blocked(blocks[i], t_sync)
            # TODO: Try next candidate time from block intervals
            error("Synchronization time $t_sync is blocked for DOF $i")
        end
    end

    # Step 2: Recalculate non-limiting DOFs for synchronized duration
    profiles = Vector{RuckigProfile{Float64}}(undef, ndof)

    for i in 1:ndof
        if i == limiting_dof
            # Limiting DOF uses its minimum-time profile
            profiles[i] = blocks[i].p_min
        elseif abs(t_sync - blocks[i].t_min) < T_PRECISION
            # Duration matches, use existing profile
            profiles[i] = blocks[i].p_min
        else
            # Need to recalculate for synchronized duration (Step 2)
            buf = lims[i].buffer
            clear!(buf)

            success = calculate_profile_step2!(lims[i].roots, buf, t_sync,
                p0[i], v0[i], a0[i], pf[i], vf[i], af[i],
                lims[i].jmax, lims[i].vmax, lims[i].vmin,
                lims[i].amax, lims[i].amin)

            if !success
                error("Failed to find synchronized profile for DOF $i at duration $t_sync")
            end

            profiles[i] = RuckigProfile(buf, pf[i], vf[i], af[i])
        end
    end

    return profiles
end

"""
    evaluate_at(profiles::AbstractVector{<:RuckigProfile}, t)

Evaluate all DOF profiles at time t.
Returns (positions, velocities, accelerations, jerks) as vectors.
"""
function evaluate_at(profiles::AbstractVector{<:RuckigProfile{T}}, t::Real) where T
    ndof = length(profiles)
    ps = Vector{T}(undef, ndof)
    vs = Vector{T}(undef, ndof)
    as = Vector{T}(undef, ndof)
    js = Vector{T}(undef, ndof)

    for i in 1:ndof
        ps[i], vs[i], as[i], js[i] = evaluate_at(profiles[i], t)
    end

    return ps, vs, as, js
end

"""
    evaluate_dt(profiles::AbstractVector{<:RuckigProfile}, Ts)

Evaluate all DOF profiles at regular time intervals.
Returns matrices (pos, vel, acc, jerk) where each column is a DOF,
plus the time vector ts.
"""
function evaluate_dt(profiles::AbstractVector{<:RuckigProfile{T}}, Ts) where T
    ndof = length(profiles)
    Tf = duration(profiles[1])  # All profiles have same duration (synchronized)
    ts = 0:Ts:Tf
    n = length(ts)

    pos = Matrix{T}(undef, n, ndof)
    vel = Matrix{T}(undef, n, ndof)
    acc = Matrix{T}(undef, n, ndof)
    jerk = Matrix{T}(undef, n, ndof)

    for j in 1:ndof
        for (i, t) in enumerate(ts)
            pos[i, j], vel[i, j], acc[i, j], jerk[i, j] = evaluate_at(profiles[j], t)
        end
    end

    return pos, vel, acc, jerk, ts
end
