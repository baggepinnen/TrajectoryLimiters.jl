# Ruckig: Time-optimal jerk-limited trajectory generation
# Based on: Berscheid & Kröger, "Jerk-limited Real-time Trajectory Generation
# with Arbitrary Target States", 2021
# Reference implementation: https://github.com/pantor/ruckig
# License of reference: MIT License https://github.com/pantor/ruckig/blob/main/LICENSE

export JerkLimiter, RuckigProfile, BrakeProfile, Block, BlockInterval
export calculate_trajectory, calculate_waypoint_trajectory, calculate_velocity_trajectory
export calculate_block_with_collection
export evaluate_at, evaluate_dt, duration, main_duration

#=============================================================================
 Constants (matching reference implementation)
=============================================================================#

const EPS = 1e-12  # Matching reference implementation's v_eps/a_eps/j_eps
const P_PRECISION = 1e-8
const V_PRECISION = 1e-8
const A_PRECISION = 1e-10
const T_PRECISION = 1e-12
const NEWTON_TOL = 1e-9  # Newton refinement tolerance (C++ uses 1e-9 for profile refinement)

# Include brake profile implementation
include("ruckig_brake.jl")

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

@enum Direction begin
    DIR_UP    # Positive jMax direction
    DIR_DOWN  # Negative jMax direction
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
    direction::Direction
end

function ProfileBuffer{T}() where T
    ProfileBuffer{T}(
        Memory{T}(undef, 7), Memory{T}(undef, 7), Memory{T}(undef, 7),
        Memory{T}(undef, 8), Memory{T}(undef, 8), Memory{T}(undef, 8),
        LIMIT_NONE, UDDU, DIR_UP
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
    buf.direction = DIR_UP
    buf
end

"""Copy all contents from src buffer to dst buffer."""
function copy_buffer!(dst::ProfileBuffer{T}, src::ProfileBuffer{T}) where T
    for i in 1:7
        dst.t[i] = src.t[i]
        dst.t_sum[i] = src.t_sum[i]
        dst.j[i] = src.j[i]
    end
    for i in 1:8
        dst.a[i] = src.a[i]
        dst.v[i] = src.v[i]
        dst.p[i] = src.p[i]
    end
    dst.limits = src.limits
    dst.control_signs = src.control_signs
    dst.direction = src.direction
    dst
end

"""
    RuckigProfile{T}

A 7-phase jerk-limited trajectory profile (immutable result).
Optionally includes a pre-pended brake profile if initial state was outside limits.
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
    direction::Direction
    brake_duration::T     # Duration of pre-pended brake profile (0 if no braking)
    brake::Union{Nothing, BrakeProfile{T}}  # Brake profile for initial states outside limits
end

"""
Create RuckigProfile from ProfileBuffer.
"""
function RuckigProfile(buf::ProfileBuffer{T}, pf, vf, af; brake_duration=zero(T), brake::Union{Nothing, BrakeProfile{T}}=nothing) where T
    RuckigProfile{T}(
        NTuple{7,T}(buf.t),
        NTuple{7,T}(buf.t_sum),
        NTuple{7,T}(buf.j),
        NTuple{8,T}(buf.a),
        NTuple{8,T}(buf.v),
        NTuple{8,T}(buf.p),
        pf, vf, af,
        buf.limits, buf.control_signs, buf.direction,
        T(brake_duration),
        brake
    )
end

# Allow RuckigProfile to broadcast as a scalar
Base.Broadcast.broadcastable(p::RuckigProfile) = Ref(p)

"""Total duration including any brake profile."""
duration(p::RuckigProfile) = p.brake_duration + p.t_sum[end]

"""Duration of just the main profile (excluding brake)."""
main_duration(p::RuckigProfile) = p.t_sum[end]

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
Create a BlockInterval from two profiles.
Automatically determines left/right based on durations (matching C++ Interval constructor).
The profile stored is the one with the longer duration (corresponding to `right`).
"""
function BlockInterval(profile_left::RuckigProfile{T}, profile_right::RuckigProfile{T}) where T
    left_duration = profile_left.t_sum[end]
    right_duration = profile_right.t_sum[end]
    if left_duration < right_duration
        BlockInterval{T}(left_duration, right_duration, profile_right)
    else
        BlockInterval{T}(right_duration, left_duration, profile_left)
    end
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

"""String representation of Block (matches C++ Block::to_string)."""
function Base.show(io::IO, block::Block)
    print(io, "[", block.t_min, " ")
    if !isnothing(block.a)
        print(io, block.a.left, "] [", block.a.right, " ")
    end
    if !isnothing(block.b)
        print(io, block.b.left, "] [", block.b.right, " ")
    end
    print(io, "-")
end

#=============================================================================
 ValidProfileCollection: Collects valid profiles during Step 1 for Block calculation
=============================================================================#

"""
    ValidProfileCollection{T}

Mutable storage for collecting valid profiles during Step 1.
Used to compute blocked intervals. C++ uses at most 5 valid profiles.
"""
mutable struct ValidProfileCollection{T}
    profiles::Vector{RuckigProfile{T}}
    count::Int
end

ValidProfileCollection{T}() where T = ValidProfileCollection{T}(Vector{RuckigProfile{T}}(undef, 5), 0)

function clear!(vpc::ValidProfileCollection)
    vpc.count = 0
    vpc
end

function add_profile!(vpc::ValidProfileCollection{T}, profile::RuckigProfile{T}) where T
    vpc.count += 1
    if vpc.count <= length(vpc.profiles)
        vpc.profiles[vpc.count] = profile
    else
        push!(vpc.profiles, profile)
    end
    vpc
end

"""
    calculate_block!(vpc::ValidProfileCollection{T}) -> Block{T}

Compute a Block from a collection of valid profiles, determining blocked intervals.
This mirrors the C++ Block::calculate_block function (block.hpp lines 59-132).

Logic:
- 1 profile: Just the minimum, no blocked intervals
- 2 profiles: One blocked interval between them (if durations differ significantly)
- 3 profiles: Find minimum, create one blocked interval from the other two
- 4 profiles: Handle as numerical degenerate case (remove near-duplicates)
- 5 profiles: Find minimum, create two blocked intervals
"""
function calculate_block!(vpc::ValidProfileCollection{T}) where T
    count = vpc.count

    if count == 0
        error("No valid profiles found")
    end

    # Work with a copy of the profiles array to allow modification
    profiles = vpc.profiles
    valid_count = count

    if valid_count == 1
        return Block(profiles[1])
    end

    if valid_count == 2
        # Check if durations are essentially equal (numerical tolerance)
        # C++ block.hpp line 71
        if abs(profiles[1].t_sum[end] - profiles[2].t_sum[end]) < 8 * eps(Float64)
            return Block(profiles[1])
        end

        # C++ block.hpp lines 76-83 (numerical_robust = true)
        idx_min = profiles[1].t_sum[end] < profiles[2].t_sum[end] ? 1 : 2
        idx_other = 3 - idx_min

        p_min = profiles[idx_min]
        t_min = p_min.t_sum[end]
        interval_a = BlockInterval(profiles[idx_min], profiles[idx_other])
        return Block{T}(p_min, t_min, interval_a, nothing)
    end

    # For 4 profiles, try to remove near-duplicates (numerical issue handling)
    # C++ block.hpp lines 86-100
    if valid_count == 4
        # Check specific pairs for near-identical durations with opposite directions
        # C++ checks (0,1), (2,3), (0,3) in 0-based indexing
        if abs(profiles[1].t_sum[end] - profiles[2].t_sum[end]) < 32 * eps(Float64) &&
           profiles[1].direction != profiles[2].direction
            # Remove profile 2 by shifting
            profiles[2] = profiles[valid_count]
            valid_count -= 1
        elseif abs(profiles[3].t_sum[end] - profiles[4].t_sum[end]) < 256 * eps(Float64) &&
               profiles[3].direction != profiles[4].direction
            # Remove profile 4
            valid_count -= 1
        elseif abs(profiles[1].t_sum[end] - profiles[4].t_sum[end]) < 256 * eps(Float64) &&
               profiles[1].direction != profiles[4].direction
            # Remove profile 4
            valid_count -= 1
        else
            error("Invalid state: 4 valid profiles with no near-duplicates to remove")
        end
    end

    # Check for valid odd count (C++ block.hpp lines 98-100)
    if valid_count % 2 == 0
        error("Invalid state: even number of valid profiles ($valid_count)")
    end

    # Find index of minimum duration profile (C++ block.hpp lines 103-106)
    idx_min = 1
    for i in 2:valid_count
        if profiles[i].t_sum[end] < profiles[idx_min].t_sum[end]
            idx_min = i
        end
    end

    p_min = profiles[idx_min]
    t_min = p_min.t_sum[end]

    if valid_count == 3
        # C++ block.hpp lines 108-113
        # C++ uses (idx_min + 1) % 3 and (idx_min + 2) % 3 with 0-based indexing
        # For Julia 1-based: if idx_min=1, others are 2,3; if idx_min=2, others are 3,1; if idx_min=3, others are 1,2
        idx_else_1 = mod1(idx_min, 3) == idx_min ? mod1(idx_min + 1, 3) : mod1(idx_min + 1, 3)
        idx_else_2 = mod1(idx_min + 2, 3)
        # Simpler: get the two indices that aren't idx_min
        others = filter(i -> i != idx_min, 1:3)
        idx_else_1, idx_else_2 = others[1], others[2]

        interval_a = BlockInterval(profiles[idx_else_1], profiles[idx_else_2])
        return Block{T}(p_min, t_min, interval_a, nothing)

    elseif valid_count == 5
        # C++ block.hpp lines 115-128
        # Get the 4 indices that aren't idx_min, in order
        others = filter(i -> i != idx_min, 1:5)
        idx_else_1, idx_else_2, idx_else_3, idx_else_4 = others[1], others[2], others[3], others[4]

        # Check direction pairing (C++ line 121)
        if profiles[idx_else_1].direction == profiles[idx_else_2].direction
            interval_a = BlockInterval(profiles[idx_else_1], profiles[idx_else_2])
            interval_b = BlockInterval(profiles[idx_else_3], profiles[idx_else_4])
        else
            interval_a = BlockInterval(profiles[idx_else_1], profiles[idx_else_4])
            interval_b = BlockInterval(profiles[idx_else_2], profiles[idx_else_3])
        end
        return Block{T}(p_min, t_min, interval_a, interval_b)
    end

    # Fallback: just use minimum profile
    return Block(p_min)
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

# Helper to get root by index (1-based)
@inline function _get_root(r::Roots, i)
    i == 1 && return r.r1
    i == 2 && return r.r2
    i == 3 && return r.r3
    return r.r4
end

# Iterator interface for for-loop consumption
# Matches C++ PositiveSet behavior: only iterate over non-negative roots
function Base.iterate(r::Roots)
    for i in 1:r.count
        val = _get_root(r, i)
        val >= 0 && return (val, i + 1)
    end
    return nothing
end

function Base.iterate(r::Roots, state)
    for i in state:r.count
        val = _get_root(r, i)
        val >= 0 && return (val, i + 1)
    end
    return nothing
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
    buffer::ProfileBuffer{Float64}  # Always Float64 for computation (stores best profile)
    candidate::ProfileBuffer{Float64}  # Candidate buffer for profile search
    valid_profiles::ValidProfileCollection{Float64}  # Collects valid profiles for blocked intervals
    brake::BrakeProfile{Float64}    # Brake profile for initial states outside limits
end

function JerkLimiter(; vmax, amax, jmax, vmin=-vmax, amin=-amax)
    T = promote_type(typeof(vmax), typeof(vmin), typeof(amax), typeof(amin), typeof(jmax))
    JerkLimiter(T(vmax), T(vmin), T(amax), T(amin), T(jmax), Roots{Float64}(), ProfileBuffer{Float64}(), ProfileBuffer{Float64}(), ValidProfileCollection{Float64}(), BrakeProfile{Float64}())
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
 VelocityLimiter: First-Order Profiles (velocity-limited only)
=============================================================================#

"""
    VelocityLimiter{T}

Limiter for first-order profiles (constant velocity only).
Use when acceleration and jerk limits don't apply.

# Fields
- `vmax`: Maximum velocity
- `vmin`: Minimum velocity (default: `-vmax`)
"""
struct VelocityLimiter{T}
    vmax::T
    vmin::T
    buffer::ProfileBuffer{Float64}
end

function VelocityLimiter(; vmax, vmin=-vmax)
    T = promote_type(typeof(vmax), typeof(vmin))
    VelocityLimiter(T(vmax), T(vmin), ProfileBuffer{Float64}())
end

function Base.show(io::IO, lim::VelocityLimiter)
    if lim.vmin == -lim.vmax
        print(io, "VelocityLimiter(; vmax=$(lim.vmax))")
    else
        print(io, "VelocityLimiter(; vmax=$(lim.vmax), vmin=$(lim.vmin))")
    end
end

"""
    calculate_trajectory(lim::VelocityLimiter; pf, p0=0)

Calculate first-order trajectory (constant velocity) from p0 to pf.
The trajectory moves at maximum velocity in the appropriate direction.

# Returns
`RuckigProfile` with only phase 4 (coast) active.
"""
function calculate_trajectory(lim::VelocityLimiter{T}; pf, p0=zero(T)) where T
    (; vmax, vmin, buffer) = lim
    buf = buffer
    clear!(buf)

    pd = pf - p0

    # Choose velocity direction
    if pd >= 0
        vf = vmax
    else
        vf = vmin
    end

    if abs(vf) < EPS
        if abs(pd) < EPS
            # Already at target
            return RuckigProfile(buf, pf, zero(T), zero(T))
        else
            error("Cannot reach target position with zero velocity limit")
        end
    end

    # Time to reach target at constant velocity
    t_coast = pd / vf

    if t_coast < 0
        error("Invalid first-order trajectory: negative time")
    end

    # Set profile: only coast phase (index 4)
    for i in 1:7
        buf.t[i] = zero(Float64)
        buf.j[i] = zero(Float64)
    end
    buf.t[4] = t_coast

    # Set boundary states
    buf.p[1] = p0
    buf.v[1] = vf  # Constant velocity throughout
    buf.a[1] = zero(Float64)

    for i in 1:7
        buf.p[i+1] = buf.p[i] + buf.t[i] * buf.v[i]
        buf.v[i+1] = buf.v[i]
        buf.a[i+1] = zero(Float64)
    end

    # Compute cumulative times
    buf.t_sum[1] = buf.t[1]
    for i in 2:7
        buf.t_sum[i] = buf.t_sum[i-1] + buf.t[i]
    end

    buf.limits = LIMIT_VEL
    buf.control_signs = UDDU
    buf.direction = vf > 0 ? DIR_UP : DIR_DOWN

    return RuckigProfile(buf, pf, zero(T), zero(T))
end

"""
    calculate_trajectory(lim::VelocityLimiter; pf, p0=0, tf)

Calculate first-order trajectory with specified duration tf.
The velocity is computed as pd/tf and must be within limits.

# Returns
`RuckigProfile` with only phase 4 (coast) active.
"""
function calculate_trajectory(lim::VelocityLimiter{T}; pf, p0=zero(T), tf) where T
    (; vmax, vmin, buffer) = lim
    buf = buffer
    clear!(buf)

    pd = pf - p0
    vf = pd / tf

    # Check velocity limits
    if vf > vmax + V_PRECISION
        error("Required velocity $vf exceeds vmax=$(lim.vmax)")
    end
    if vf < vmin - V_PRECISION
        error("Required velocity $vf is below vmin=$(lim.vmin)")
    end

    # Set profile: only coast phase (index 4)
    for i in 1:7
        buf.t[i] = zero(Float64)
        buf.j[i] = zero(Float64)
    end
    buf.t[4] = tf

    # Set boundary states
    buf.p[1] = p0
    buf.v[1] = vf
    buf.a[1] = zero(Float64)

    for i in 1:7
        buf.p[i+1] = buf.p[i] + buf.t[i] * buf.v[i]
        buf.v[i+1] = buf.v[i]
        buf.a[i+1] = zero(Float64)
    end

    # Compute cumulative times
    buf.t_sum[1] = buf.t[1]
    for i in 2:7
        buf.t_sum[i] = buf.t_sum[i-1] + buf.t[i]
    end

    buf.limits = LIMIT_NONE
    buf.control_signs = UDDU
    buf.direction = vf > 0 ? DIR_UP : DIR_DOWN

    return RuckigProfile(buf, pf, zero(T), zero(T))
end

export VelocityLimiter

#=============================================================================
 AccelerationLimiter: Second-Order Profiles (acceleration-limited, no jerk)
=============================================================================#

"""
    AccelerationLimiter{T}

Limiter for second-order profiles (acceleration-limited, no jerk limit).
Use when jerk limits don't apply but acceleration limits do.

# Fields
- `vmax`: Maximum velocity
- `vmin`: Minimum velocity (default: `-vmax`)
- `amax`: Maximum acceleration
- `amin`: Minimum acceleration (default: `-amax`)
"""
struct AccelerationLimiter{T}
    vmax::T
    vmin::T
    amax::T
    amin::T
    buffer::ProfileBuffer{Float64}
    candidate::ProfileBuffer{Float64}
    brake::BrakeProfile{Float64}
end

function AccelerationLimiter(; vmax, amax, vmin=-vmax, amin=-amax)
    T = promote_type(typeof(vmax), typeof(vmin), typeof(amax), typeof(amin))
    AccelerationLimiter(T(vmax), T(vmin), T(amax), T(amin), ProfileBuffer{Float64}(), ProfileBuffer{Float64}(), BrakeProfile{Float64}())
end

function Base.show(io::IO, lim::AccelerationLimiter)
    if lim.vmin == -lim.vmax && lim.amin == -lim.amax
        print(io, "AccelerationLimiter(; vmax=$(lim.vmax), amax=$(lim.amax))")
    else
        print(io, "AccelerationLimiter(; vmax=$(lim.vmax), vmin=$(lim.vmin), amax=$(lim.amax), amin=$(lim.amin))")
    end
end

"""
Second-order ACC0 profile: accelerate to vMax, coast, decelerate to vf.
"""
function time_acc0_second_order!(buf::ProfileBuffer{T}, p0, v0, pf, vf, vMax, vMin, aMax, aMin) where T
    pd = pf - p0

    # t[1] = time to accelerate from v0 to vMax
    # t[2] = coast time at vMax
    # t[3] = time to decelerate from vMax to vf
    t1 = (vMax - v0) / aMax
    t3 = (vf - vMax) / aMin

    if t1 < -EPS || t3 < -EPS
        return false
    end

    # Coast distance: total distance - acceleration distance - deceleration distance
    # d_accel = v0*t1 + 0.5*aMax*t1²
    # d_decel = vMax*t3 + 0.5*aMin*t3²
    d_accel = v0 * t1 + 0.5 * aMax * t1^2
    d_decel = vMax * t3 + 0.5 * aMin * t3^2
    d_coast = pd - d_accel - d_decel

    if abs(vMax) < EPS
        return false
    end

    t2 = d_coast / vMax

    if t2 < -EPS
        return false
    end

    # Set profile times (using phases 1, 2, 3 for accel, coast, decel)
    buf.t[1] = max(t1, zero(T))
    buf.t[2] = max(t2, zero(T))
    buf.t[3] = max(t3, zero(T))
    for i in 4:7
        buf.t[i] = zero(T)
    end

    # Check profile validity
    check_for_second_order!(buf, p0, v0, pf, vf, aMax, aMin, vMax, vMin, LIMIT_ACC0, UDDU)
end

"""
Second-order NONE profile: direct acceleration/deceleration (no coast).
"""
function time_none_second_order!(buf::ProfileBuffer{T}, p0, v0, pf, vf, vMax, vMin, aMax, aMin) where T
    pd = pf - p0

    # Solve for peak velocity: v_peak² = (aMax*vf² - aMin*v0² - 2*aMax*aMin*pd) / (aMax - aMin)
    denom = aMax - aMin
    if abs(denom) < EPS
        return false
    end

    h1_sq = (aMax * vf^2 - aMin * v0^2 - 2 * aMax * aMin * pd) / denom
    if h1_sq < 0
        return false
    end

    h1 = sqrt(h1_sq)

    # Solution 1: peak velocity = -h1
    t1_s1 = -(v0 + h1) / aMax
    t3_s1 = (vf + h1) / aMin

    if t1_s1 >= -EPS && t3_s1 >= -EPS
        buf.t[1] = max(t1_s1, zero(T))
        buf.t[2] = zero(T)
        buf.t[3] = max(t3_s1, zero(T))
        for i in 4:7
            buf.t[i] = zero(T)
        end

        if check_for_second_order!(buf, p0, v0, pf, vf, aMax, aMin, vMax, vMin, LIMIT_NONE, UDDU)
            return true
        end
    end

    # Solution 2: peak velocity = +h1
    t1_s2 = (-v0 + h1) / aMax
    t3_s2 = (vf - h1) / aMin

    if t1_s2 >= -EPS && t3_s2 >= -EPS
        buf.t[1] = max(t1_s2, zero(T))
        buf.t[2] = zero(T)
        buf.t[3] = max(t3_s2, zero(T))
        for i in 4:7
            buf.t[i] = zero(T)
        end

        if check_for_second_order!(buf, p0, v0, pf, vf, aMax, aMin, vMax, vMin, LIMIT_NONE, UDDU)
            return true
        end
    end

    return false
end

"""
Check and finalize a second-order profile.
"""
function check_for_second_order!(buf::ProfileBuffer{T}, p0, v0, pf, vf, aMax, aMin, vMax, vMin, limits, control_signs) where T
    # Set all jerks to zero (second-order profile)
    for i in 1:7
        buf.j[i] = zero(T)
    end

    # Compute cumulative times
    buf.t_sum[1] = buf.t[1]
    for i in 2:7
        buf.t_sum[i] = buf.t_sum[i-1] + buf.t[i]
    end

    # Check for negative times
    for i in 1:7
        if buf.t[i] < -EPS
            return false
        end
    end

    # Set accelerations for each phase
    # Phase 1: accelerate at aMax
    # Phase 2: coast (a=0)
    # Phase 3: decelerate at aMin
    buf.a[1] = aMax
    buf.a[2] = aMax  # End of phase 1
    buf.a[3] = zero(T)  # Coast
    buf.a[4] = aMin  # Deceleration
    for i in 5:8
        buf.a[i] = zero(T)
    end

    # Set initial state
    buf.p[1] = p0
    buf.v[1] = v0

    # Integrate through phases (second-order: v' = a, no jerk)
    for i in 1:7
        ai = i == 1 ? aMax : (i == 3 ? aMin : zero(T))
        if i == 2  # Coast
            ai = zero(T)
        elseif i == 1  # Accelerate
            ai = aMax
        elseif i == 3  # Decelerate
            ai = aMin
        end

        buf.v[i+1] = buf.v[i] + buf.t[i] * ai
        buf.p[i+1] = buf.p[i] + buf.t[i] * (buf.v[i] + buf.t[i] * ai / 2)
    end

    # Check final position and velocity
    if abs(buf.p[8] - pf) > P_PRECISION
        return false
    end
    if abs(buf.v[8] - vf) > V_PRECISION
        return false
    end

    # Check velocity limits
    for i in 1:8
        if buf.v[i] > vMax + V_PRECISION || buf.v[i] < vMin - V_PRECISION
            return false
        end
    end

    buf.limits = limits
    buf.control_signs = control_signs
    buf.direction = vMax > 0 ? DIR_UP : DIR_DOWN

    return true
end

"""
    calculate_trajectory(lim::AccelerationLimiter; pf, p0=0, v0=0, vf=0)

Calculate second-order trajectory (acceleration-limited, no jerk) from (p0, v0) to (pf, vf).

# Returns
`RuckigProfile` with phases 1-3 active (accelerate, coast, decelerate).
"""
function calculate_trajectory(lim::AccelerationLimiter{T}; pf, p0=zero(T), v0=zero(T), vf=zero(T)) where T
    (; vmax, vmin, amax, amin, buffer, brake) = lim
    buf = buffer
    clear!(buf)

    # Compute brake profile if initial velocity is outside limits
    get_second_order_position_brake_trajectory!(brake, v0, vmax, vmin, amax, amin)
    ps, vs, as = finalize_second_order_brake!(brake, p0, v0, zero(T))
    brake_duration = brake.duration
    brake_copy = brake_duration > 0 ? deepcopy(brake) : nothing

    p0_eff, v0_eff = ps, vs
    pd = pf - p0_eff

    # Determine direction
    if pd >= 0
        vMax, vMin, aMax, aMin = vmax, vmin, amax, amin
    else
        vMax, vMin, aMax, aMin = vmin, vmax, amin, amax
    end

    # Try ACC0 profile (accelerate to vMax, coast, decelerate)
    if time_acc0_second_order!(buf, p0_eff, v0_eff, pf, vf, vMax, vMin, aMax, aMin)
        return RuckigProfile(buf, pf, vf, zero(T); brake_duration, brake=brake_copy)
    end

    # Try NONE profile (direct acceleration/deceleration)
    clear!(buf)
    if time_none_second_order!(buf, p0_eff, v0_eff, pf, vf, vMax, vMin, aMax, aMin)
        return RuckigProfile(buf, pf, vf, zero(T); brake_duration, brake=brake_copy)
    end

    # Try opposite direction
    if pd >= 0
        vMax, vMin, aMax, aMin = vmin, vmax, amin, amax
    else
        vMax, vMin, aMax, aMin = vmax, vmin, amax, amin
    end

    clear!(buf)
    if time_acc0_second_order!(buf, p0_eff, v0_eff, pf, vf, vMax, vMin, aMax, aMin)
        return RuckigProfile(buf, pf, vf, zero(T); brake_duration, brake=brake_copy)
    end

    clear!(buf)
    if time_none_second_order!(buf, p0_eff, v0_eff, pf, vf, vMax, vMin, aMax, aMin)
        return RuckigProfile(buf, pf, vf, zero(T); brake_duration, brake=brake_copy)
    end

    error("No valid second-order trajectory found from ($p0, $v0) to ($pf, $vf), limiter: ", lim)
end

export AccelerationLimiter

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
Matches C++ reference implementation in roots.hpp.
"""
function solve_cubic_real!(roots::Roots, a, b, c, d)
    clear!(roots)

    # Special case: d ≈ 0 means x = 0 is a root (matching C++ roots.hpp lines 63-72)
    # Factor out x: x(ax² + bx + c) = 0, so solve the quadratic for remaining roots
    if abs(d) < EPS
        push!(roots, 0.0)
        # Solve ax² + bx + c = 0 for remaining roots
        if abs(a) < EPS
            # Linear: bx + c = 0
            if abs(b) > EPS
                push!(roots, -c/b)
            end
        else
            disc = b^2 - 4a*c
            if disc >= 0
                if disc < EPS
                    push!(roots, -b / (2a))
                else
                    sqrt_disc = sqrt(disc)
                    push!(roots, (-b - sqrt_disc) / (2a))
                    push!(roots, (-b + sqrt_disc) / (2a))
                end
            end
        end
        return roots
    end

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
    # Note: jerk is set after times are clamped below (C++ sets jerk based on t > 0)

    # Check all times non-negative (NaN < x is false, so check explicitly)
    @inbounds for i in 1:7
        (isnan(buf.t[i]) || buf.t[i] < -T_PRECISION) && return false
        buf.t[i] = max(buf.t[i], zero(T))
    end

    # Reference implementation (profile.hpp lines 188-204):
    # Check that certain phase times are strictly positive for specific profile types
    # For velocity profiles: t[4] (Julia 1-based) must be > epsilon
    if limits in (LIMIT_ACC0_ACC1_VEL, LIMIT_ACC0_VEL, LIMIT_ACC1_VEL, LIMIT_VEL)
        buf.t[4] < EPS && return false
    end
    # For ACC0/ACC0_ACC1: t[2] must be > epsilon (cruise time at aMax)
    if limits in (LIMIT_ACC0, LIMIT_ACC0_ACC1)
        buf.t[2] < EPS && return false
    end
    # For ACC1/ACC0_ACC1: t[6] must be > epsilon (cruise time at aMin)
    if limits in (LIMIT_ACC1, LIMIT_ACC0_ACC1)
        buf.t[6] < EPS && return false
    end

    # Set jerk pattern based on control signs (matching C++ profile.hpp lines 210-214)
    # Jerk is 0 if corresponding phase time is 0
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

        # Reference implementation (profile.hpp lines 225-246):
        # For velocity-limited profiles, explicitly set a[4] = 0 after phase 3
        # This is done ALWAYS for velocity profiles (not inside set_limits block)
        if limits in (LIMIT_ACC0_ACC1_VEL, LIMIT_ACC0_VEL, LIMIT_ACC1_VEL, LIMIT_VEL) && i == 3
            buf.a[4] = zero(T)
        end

        # For ACC1 profiles, set a[4] = aMin (C++ uses set_limits=true for ACC1)
        if limits == LIMIT_ACC1 && i == 3
            buf.a[4] = aMin
        end

        # For ACC0_ACC1 profiles, set a[2] = aMax and a[6] = aMin
        if limits == LIMIT_ACC0_ACC1
            if i == 1
                buf.a[2] = aMax
            elseif i == 5
                buf.a[6] = aMin
            end
        end

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

    # Check velocity at acceleration zero-crossings (matching C++ profile.hpp lines 249-254)
    # C++ checks for i > 1 in 0-based indexing (i=2,3,4,5,6), which is Julia i=3,4,5,6,7
    # When acceleration changes sign within a phase, check that velocity at the zero-crossing
    # doesn't violate limits. Formula: v_at_zero = v[i] - a[i]²/(2*j[i])
    @inbounds for i in 3:7
        ai = buf.a[i]
        # Check if acceleration changes sign between start and end of phase
        if ai * buf.a[i+1] < -EPS
            ji = buf.j[i]
            v_at_zero = buf.v[i] - ai^2 / (2ji)
            (v_at_zero > vUppLim || v_at_zero < vLowLim) && return false
        end
    end

    buf.limits = limits
    buf.control_signs = control_signs
    # Direction is UP when vMax > 0, DOWN when vMax <= 0 (matching C++ profile.hpp line 216)
    buf.direction = vMax > 0 ? DIR_UP : DIR_DOWN

    return true
end

#=============================================================================
 Profile Time Calculations - Zero-Limits Special Case
=============================================================================#

"""
    time_all_single_step!(buf, p0, v0, a0, pf, vf, af, jMax, vMax, vMin, aMax, aMin) -> Bool

Handle zero-limits special case when jMax=0, aMax=0, or aMin=0.
This computes a single-phase trajectory with constant acceleration.

C++ Reference: position_third_step1.cpp lines 467-508
"""
function time_all_single_step!(buf::ProfileBuffer{T}, p0, v0, a0, pf, vf, af,
                               jMax, vMax, vMin, aMax, aMin) where T
    pd = pf - p0

    # Acceleration must be constant (a0 = af)
    if abs(af - a0) > EPS
        return false
    end

    # All phase times are zero except t[4] (coast phase, 1-indexed)
    for i in 1:7
        buf.t[i] = zero(T)
        buf.j[i] = zero(T)
    end

    if abs(a0) > EPS
        # Constant acceleration case: solve p = p0 + v0*t + 0.5*a0*t² for t
        # Quadratic: 0.5*a0*t² + v0*t - pd = 0
        # t = (-v0 ± sqrt(v0² + 2*a0*pd)) / a0
        disc = v0^2 + 2 * a0 * pd
        if disc < 0
            return false
        end
        q = sqrt(disc)

        # Solution 1: (-v0 + q) / a0
        t_coast = (-v0 + q) / a0
        if t_coast >= 0
            buf.t[4] = t_coast
            if check_for_profile!(buf, p0, v0, a0, pf, vf, af, zero(T), vMax, vMin, aMax, aMin, LIMIT_NONE, UDDU)
                return true
            end
        end

        # Solution 2: -(v0 + q) / a0
        t_coast = -(v0 + q) / a0
        if t_coast >= 0
            buf.t[4] = t_coast
            if check_for_profile!(buf, p0, v0, a0, pf, vf, af, zero(T), vMax, vMin, aMax, aMin, LIMIT_NONE, UDDU)
                return true
            end
        end

    elseif abs(v0) > EPS
        # Constant velocity case: t = pd / v0
        t_coast = pd / v0
        buf.t[4] = t_coast
        if check_for_profile!(buf, p0, v0, a0, pf, vf, af, zero(T), vMax, vMin, aMax, aMin, LIMIT_NONE, UDDU)
            return true
        end

    elseif abs(pd) < EPS
        # Already at target
        if check_for_profile!(buf, p0, v0, a0, pf, vf, af, zero(T), vMax, vMin, aMax, aMin, LIMIT_NONE, UDDU)
            return true
        end
    end

    return false
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
function time_acc0_acc1!(buf::ProfileBuffer{T}, candidate::ProfileBuffer{T}, p0, v0, a0, pf, vf, af,
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

    # Track the best (shortest duration) profile found
    best_duration = T(Inf)
    found_valid = false

    # Try two solutions (from reference implementation)
    # Solution 2: h2 > h1/aMax, h3 > -h1/aMin => t[2] = h2 - h1/aMax, t[6] = h3 + h1/aMin
    # Solution 1: h2 > -h1/aMax, h3 > h1/aMin => t[2] = h2 + h1/aMax, t[6] = h3 - h1/aMin
    for h1_sign in (1, -1)
        t1_cond = h2 > h1_sign * h1 / aMax
        t5_cond = h3 > -h1_sign * h1 / aMin

        t1_cond && t5_cond || continue

        candidate.t[1] = (-a0 + aMax) / jMax
        candidate.t[2] = h2 - h1_sign * h1 / aMax
        candidate.t[3] = aMax / jMax
        candidate.t[4] = 0
        candidate.t[5] = -aMin / jMax
        candidate.t[6] = h3 + h1_sign * h1 / aMin
        candidate.t[7] = candidate.t[5] + af / jMax

        if check!(candidate, UDDU, LIMIT_ACC0_ACC1, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            dur = sum(candidate.t)
            if dur < best_duration
                best_duration = dur
                found_valid = true
                copy_buffer!(buf, candidate)
            end
        end
    end

    return found_valid
end

"""
Try ACC0, ACC1, and NONE profiles (no velocity limit reached).
Returns the SHORTEST valid profile found (not the first).
Uses candidate buffer for checking, copies best result to buf.
"""
function time_all_none_acc0_acc1!(roots::Roots, buf::ProfileBuffer{T}, candidate::ProfileBuffer{T},
                                  p0, v0, a0, pf, vf, af,
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

    # Track the best (shortest duration) profile found
    best_duration = T(Inf)
    found_valid = false

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
        candidate.t[1] = h0 + t/2 - a0/jMax
        candidate.t[2] = 0
        candidate.t[3] = t
        candidate.t[4] = 0
        candidate.t[5] = 0
        candidate.t[6] = 0
        candidate.t[7] = -h0 + t/2 + af/jMax

        if check!(candidate, UDDU, LIMIT_NONE, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            dur = sum(candidate.t)
            if dur < best_duration
                best_duration = dur
                found_valid = true
                copy_buffer!(buf, candidate)
            end
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

        candidate.t[1] = (-a0 + aMax)/jMax
        candidate.t[2] = h3_acc0 - 2*t + jMax/aMax*t^2
        candidate.t[3] = t
        candidate.t[4] = 0
        candidate.t[5] = 0
        candidate.t[6] = 0
        candidate.t[7] = (af - aMax)/jMax + t

        if check!(candidate, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            dur = sum(candidate.t)
            if dur < best_duration
                best_duration = dur
                found_valid = true
                copy_buffer!(buf, candidate)
            end
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

            if abs(orig) > NEWTON_TOL
                deriv = (aMin - a0 - h1)*(h2_acc1 + h1*(4*a0 - aMin + 2*h1))
                t -= orig / deriv  # Second step: no min

                h1 = jMax*t
                orig = -(h0_acc1/2 + h1*(h5 + a0*(aMin - 2*h1)*(aMin - h1) + a0_a0*(5*h1/2 - 2*aMin) + aMin^2*h1/2 + jMax*(h1/2 - aMin)*(h1*t + 2*v0)))/jMax

                if abs(orig) > NEWTON_TOL
                    deriv = (aMin - a0 - h1)*(h2_acc1 + h1*(4*a0 - aMin + 2*h1))
                    t -= orig / deriv  # Third step: no min
                end
            end
        end

        candidate.t[1] = t
        candidate.t[2] = 0
        candidate.t[3] = (a0 - aMin)/jMax + t
        candidate.t[4] = 0
        candidate.t[5] = 0
        candidate.t[6] = h3_acc1 - (2*a0 + jMax*t)*t/aMin
        candidate.t[7] = (af - aMin)/jMax

        if check!(candidate, UDDU, LIMIT_ACC1, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            dur = sum(candidate.t)
            if dur < best_duration
                best_duration = dur
                found_valid = true
                copy_buffer!(buf, candidate)
            end
        end
    end

    return found_valid
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
    brake_dur = profile.brake_duration

    if t <= 0
        # Return initial state (before brake if any)
        if profile.brake !== nothing && brake_dur > 0
            bp = profile.brake
            return bp.p[1], bp.v[1], bp.a[1], bp.j[1]
        else
            return profile.p[1], profile.v[1], profile.a[1], profile.j[1]
        end
    end

    if t >= T_total
        return profile.p[8], profile.v[8], profile.a[8], zero(T)
    end

    # Handle brake phase if present
    if profile.brake !== nothing && brake_dur > 0 && t < brake_dur
        bp = profile.brake
        # Find which brake phase we're in
        if t <= bp.t[1]
            # First brake phase
            dt = t
            pk, vk, ak, jk = bp.p[1], bp.v[1], bp.a[1], bp.j[1]
        else
            # Second brake phase
            dt = t - bp.t[1]
            pk, vk, ak, jk = bp.p[2], bp.v[2], bp.a[2], bp.j[2]
        end

        p = pk + dt * (vk + dt * (ak / 2 + dt * jk / 6))
        v = vk + dt * (ak + dt * jk / 2)
        a = ak + dt * jk

        return p, v, a, jk
    end

    # Main profile evaluation (subtract brake duration)
    t_main = t - brake_dur

    # Find phase in main profile
    phase = 1
    @inbounds for k in 1:7
        if t_main <= profile.t_sum[k]
            phase = k
            break
        end
    end

    t_start = phase == 1 ? zero(T) : profile.t_sum[phase-1]
    dt = t_main - t_start

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

    (; vmax, vmin, amax, amin, jmax, buffer, brake) = lim
    buf = buffer
    clear!(buf)

    # Validate target constraints (initial state can be outside limits - handled by brake)
    if vf < vmin || vf > vmax
        error("Target velocity vf=$vf is outside allowed range [$vmin, $vmax]")
    end
    if af < amin || af > amax
        error("Target acceleration af=$af is outside allowed range [$amin, $amax]")
    end

    # Compute brake profile if initial state is outside limits
    # This handles a0 > amax, a0 < amin, v0 > vmax, or v0 < vmin
    get_position_brake_trajectory!(brake, v0, a0, vmax, vmin, amax, amin, jmax)
    ps, vs, as = finalize_brake!(brake, p0, v0, a0)
    brake_duration = brake.duration

    # Create a copy of the brake profile to store in the result (if braking occurred)
    brake_copy = brake_duration > 0 ? deepcopy(brake) : nothing

    # Use post-brake state as effective initial state
    p0_eff, v0_eff, a0_eff = ps, vs, as

    # Determine primary and secondary direction based on displacement
    # For positive pd: primary uses standard limits
    # For negative pd: primary uses swapped limits (to move in negative direction)
    pd = pf - p0_eff
    if pd >= 0
        jMax1, vMax1, vMin1, aMax1, aMin1 = jmax, vmax, vmin, amax, amin
        jMax2, vMax2, vMin2, aMax2, aMin2 = -jmax, vmin, vmax, amin, amax
    else
        jMax1, vMax1, vMin1, aMax1, aMin1 = -jmax, vmin, vmax, amin, amax
        jMax2, vMax2, vMin2, aMax2, aMin2 = jmax, vmax, vmin, amax, amin
    end

    # Zero-limits special case: jMax=0, aMax=0, or aMin=0
    # C++ Reference: position_third_step1.cpp lines 511-525
    if jmax == 0 || amax == 0 || amin == 0
        if time_all_single_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
        end
        error("No valid trajectory found for zero-limits case from ($p0, $v0, $a0) to ($pf, $vf, $af), limiter: ", lim)
    end

    # Reference implementation behavior:
    # - When vf == 0 && af == 0: return first valid profile (fast path)
    # - When vf != 0 || af != 0: collect all profiles and return minimum
    # See position_third_step1.cpp lines 531-585

    if abs(vf) < EPS && abs(af) < EPS
        # Fast path: return first valid profile found
        # Try primary direction first
        if time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
        end

        if time_all_none_acc0_acc1!(lim.roots, buf, lim.candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
        end

        if time_acc0_acc1!(buf, lim.candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
        end

        # Try secondary direction
        clear!(buf)
        if time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
            return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
        end

        if time_all_none_acc0_acc1!(lim.roots, buf, lim.candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
            return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
        end

        if time_acc0_acc1!(buf, lim.candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
            return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
        end

        # Fall through to two-step profiles
    else
        # Full collection mode: try all profiles and return minimum
        # This matches C++ behavior when vf != 0 || af != 0
        # IMPORTANT: C++ uses original limits here, NOT pd-swapped limits
        # See position_third_step1.cpp lines 558-563
        best_duration = T(Inf)
        best_profile = nothing

        # Helper to check if current buffer has a shorter profile
        function try_save_best!()
            dur = sum(buf.t)
            if dur < best_duration
                best_duration = dur
                best_profile = RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
            end
        end

        # Collect from time_all_none_acc0_acc1 (both directions) - using ORIGINAL limits
        if time_all_none_acc0_acc1!(lim.roots, buf, lim.candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
            try_save_best!()
        end
        clear!(buf)
        if time_all_none_acc0_acc1!(lim.roots, buf, lim.candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
            try_save_best!()
        end

        # Collect from time_acc0_acc1 (both directions) - using ORIGINAL limits
        clear!(buf)
        if time_acc0_acc1!(buf, lim.candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
            try_save_best!()
        end
        clear!(buf)
        if time_acc0_acc1!(buf, lim.candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
            try_save_best!()
        end

        # Collect from time_all_vel (both directions) - using ORIGINAL limits
        clear!(buf)
        if time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
            try_save_best!()
        end
        clear!(buf)
        if time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
            try_save_best!()
        end

        if best_profile !== nothing
            return best_profile
        end

        # Fall through to two-step profiles if no profile found
    end

    # Two-step profiles (fallback, only if no regular profile found)
    # C++ uses original limits for two-step profiles (lines 567-581)
    clear!(buf)
    if time_none_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
        return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
    end
    clear!(buf)
    if time_none_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
        return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
    end

    clear!(buf)
    if time_acc0_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
        return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
    end
    clear!(buf)
    if time_acc0_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
        return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
    end

    clear!(buf)
    if time_vel_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
        return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
    end
    clear!(buf)
    if time_vel_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
        return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
    end

    clear!(buf)
    if time_acc1_vel_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
        return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
    end
    clear!(buf)
    if time_acc1_vel_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
        return RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy)
    end

    error("No valid trajectory found from ($p0, $v0, $a0) to ($pf, $vf, $af), limiter: ", lim)

end

"""
    calculate_trajectory_with_block(lim; pf, p0=0, v0=0, a0=0, vf=0, af=0) -> Block{Float64}

Calculate a time-optimal trajectory and return a Block struct containing:
- The minimum-time profile
- Any blocked time intervals (for multi-DOF synchronization)

This function collects ALL valid profiles during Step 1 calculation,
matching the C++ reference implementation behavior for computing blocked intervals.
"""
function calculate_trajectory_with_block(lim::JerkLimiter{T}; pf, p0=zero(T), v0=zero(T), a0=zero(T), vf=zero(T), af=zero(T)) where T

    (; vmax, vmin, amax, amin, jmax, buffer, candidate, valid_profiles, brake) = lim
    buf = buffer
    clear!(buf)
    clear!(valid_profiles)

    # Validate target constraints (initial state can be outside limits - handled by brake)
    if vf < vmin || vf > vmax
        error("Target velocity vf=$vf is outside allowed range [$vmin, $vmax]")
    end
    if af < amin || af > amax
        error("Target acceleration af=$af is outside allowed range [$amin, $amax]")
    end

    # Compute brake profile if initial state is outside limits
    get_position_brake_trajectory!(brake, v0, a0, vmax, vmin, amax, amin, jmax)
    ps, vs, as = finalize_brake!(brake, p0, v0, a0)
    brake_duration = brake.duration
    brake_copy = brake_duration > 0 ? deepcopy(brake) : nothing

    # Use post-brake state as effective initial state
    p0_eff, v0_eff, a0_eff = ps, vs, as
    pd = pf - p0_eff

    # Set direction-dependent limits
    if pd >= 0
        jMax1, vMax1, vMin1, aMax1, aMin1 = jmax, vmax, vmin, amax, amin
        jMax2, vMax2, vMin2, aMax2, aMin2 = -jmax, vmin, vmax, amin, amax
    else
        jMax1, vMax1, vMin1, aMax1, aMin1 = -jmax, vmin, vmax, amin, amax
        jMax2, vMax2, vMin2, aMax2, aMin2 = jmax, vmax, vmin, amax, amin
    end

    # Zero-limits special case: jMax=0, aMax=0, or aMin=0
    if jmax == 0 || amax == 0 || amin == 0
        if time_all_single_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        error("No valid trajectory found for zero-limits case from ($p0, $v0, $a0) to ($pf, $vf, $af), limiter: ", lim)
    end

    # Reference implementation behavior (position_third_step1.cpp lines 531-585):
    # - When vf == 0 && af == 0: return first valid profile (no blocked intervals possible)
    # - When vf != 0 || af != 0: collect ALL profiles for blocked interval computation
    # See C++ comment at line 542: "There is no blocked interval when vf==0 && af==0"

    if abs(vf) < EPS && abs(af) < EPS
        # Fast path: no blocked intervals when vf==0 && af==0
        # Return first valid profile found as the block

        # Try primary direction
        if time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        if time_all_none_acc0_acc1!(lim.roots, buf, candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        if time_acc0_acc1!(buf, candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end

        # Try secondary direction
        clear!(buf)
        if time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        if time_all_none_acc0_acc1!(lim.roots, buf, candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        if time_acc0_acc1!(buf, candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
    else
        # Full collection mode: collect ALL valid profiles for blocked interval computation
        # C++ uses original limits (NOT pd-swapped), see position_third_step1.cpp lines 558-563
        # Use the _collect functions which add ALL valid profiles (not just the shortest)

        # Collect from time_all_none_acc0_acc1 (both directions)
        time_all_none_acc0_acc1_collect!(valid_profiles, lim.roots, buf, candidate,
                                         p0_eff, v0_eff, a0_eff, pf, vf, af,
                                         jmax, vmax, vmin, amax, amin;
                                         brake_duration, brake=brake_copy)

        time_all_none_acc0_acc1_collect!(valid_profiles, lim.roots, buf, candidate,
                                         p0_eff, v0_eff, a0_eff, pf, vf, af,
                                         -jmax, vmin, vmax, amin, amax;
                                         brake_duration, brake=brake_copy)

        # Collect from time_acc0_acc1 (both directions)
        time_acc0_acc1_collect!(valid_profiles, buf, candidate,
                               p0_eff, v0_eff, a0_eff, pf, vf, af,
                               jmax, vmax, vmin, amax, amin;
                               brake_duration, brake=brake_copy)

        time_acc0_acc1_collect!(valid_profiles, buf, candidate,
                               p0_eff, v0_eff, a0_eff, pf, vf, af,
                               -jmax, vmin, vmax, amin, amax;
                               brake_duration, brake=brake_copy)

        # Collect from time_all_vel (both directions)
        time_all_vel_collect!(valid_profiles, buf,
                             p0_eff, v0_eff, a0_eff, pf, vf, af,
                             jmax, vmax, vmin, amax, amin;
                             brake_duration, brake=brake_copy)

        time_all_vel_collect!(valid_profiles, buf,
                             p0_eff, v0_eff, a0_eff, pf, vf, af,
                             -jmax, vmin, vmax, amin, amax;
                             brake_duration, brake=brake_copy)

        # If valid profiles found, compute block with blocked intervals
        if valid_profiles.count > 0
            return calculate_block!(valid_profiles)
        end
    end

    # Two-step profiles (fallback) - these don't contribute to blocked intervals
    # (they're only used when no regular profile is found)
    clear!(buf)
    if time_none_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
        return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
    end
    clear!(buf)
    if time_none_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
        return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
    end

    clear!(buf)
    if time_acc0_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
        return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
    end
    clear!(buf)
    if time_acc0_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
        return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
    end

    clear!(buf)
    if time_vel_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
        return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
    end
    clear!(buf)
    if time_vel_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
        return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
    end

    clear!(buf)
    if time_acc1_vel_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jmax, vmax, vmin, amax, amin)
        return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
    end
    clear!(buf)
    if time_acc1_vel_two_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, -jmax, vmin, vmax, amin, amax)
        return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
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
                        if abs(orig) > NEWTON_TOL
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
 Position Extrema Functions
=============================================================================#

"""
    PositionBound{T}

Stores minimum and maximum position along a trajectory and the times at which they occur.
"""
struct PositionBound{T}
    min::T
    max::T
    t_min::T
    t_max::T
end

"""
    check_position_extremum!(t_ext, t_sum, t, p, v, a, j, ext_min, ext_max, t_min, t_max)

Check if position at time t_ext (within phase) is an extremum.
Updates extremum values if a new min/max is found.
Returns (new_min, new_max, new_t_min, new_t_max).
"""
function check_position_extremum(t_ext, t_sum, t, p, v, a, j, ext_min, ext_max, t_min, t_max)
    if 0 < t_ext < t
        # Integrate to t_ext
        p_ext = p + t_ext * (v + t_ext * (a / 2 + t_ext * j / 6))
        a_ext = a + t_ext * j

        if a_ext > 0 && p_ext < ext_min
            ext_min = p_ext
            t_min = t_sum + t_ext
        elseif a_ext < 0 && p_ext > ext_max
            ext_max = p_ext
            t_max = t_sum + t_ext
        end
    end
    return ext_min, ext_max, t_min, t_max
end

"""
    check_step_for_position_extremum(t_sum, t, p, v, a, j, ext_min, ext_max, t_min, t_max)

Check phase boundary and find velocity zeros within phase for position extrema.
"""
function check_step_for_position_extremum(t_sum, t, p, v, a, j, ext_min, ext_max, t_min, t_max)
    # Check boundary position
    if p < ext_min
        ext_min = p
        t_min = t_sum
    end
    if p > ext_max
        ext_max = p
        t_max = t_sum
    end

    # Find velocity zeros within phase (where position extrema can occur)
    if j != 0
        # Velocity: v(t) = v + a*t + j*t²/2 = 0
        # Quadratic: j/2 * t² + a*t + v = 0
        # t = (-a ± sqrt(a² - 2*j*v)) / j
        D = a^2 - 2 * j * v

        if abs(D) < eps(Float64)
            # Single root: t = -a/j
            t_ext = -a / j
            ext_min, ext_max, t_min, t_max = check_position_extremum(t_ext, t_sum, t, p, v, a, j, ext_min, ext_max, t_min, t_max)

        elseif D > 0
            D_sqrt = sqrt(D)
            t_ext1 = (-a - D_sqrt) / j
            t_ext2 = (-a + D_sqrt) / j
            ext_min, ext_max, t_min, t_max = check_position_extremum(t_ext1, t_sum, t, p, v, a, j, ext_min, ext_max, t_min, t_max)
            ext_min, ext_max, t_min, t_max = check_position_extremum(t_ext2, t_sum, t, p, v, a, j, ext_min, ext_max, t_min, t_max)
        end
    end

    return ext_min, ext_max, t_min, t_max
end

"""
    get_position_extrema(profile::RuckigProfile) -> PositionBound

Find the minimum and maximum positions along the trajectory.
Useful for collision checking.

# Returns
`PositionBound{T}` with fields: `min`, `max`, `t_min`, `t_max`
"""
function get_position_extrema(profile::RuckigProfile{T}) where T
    ext_min = T(Inf)
    ext_max = T(-Inf)
    t_min = zero(T)
    t_max = zero(T)

    # Check brake profile phases if present
    if profile.brake !== nothing && profile.brake_duration > 0
        bp = profile.brake
        if bp.t[1] > 0
            ext_min, ext_max, t_min, t_max = check_step_for_position_extremum(
                zero(T), bp.t[1], bp.p[1], bp.v[1], bp.a[1], bp.j[1],
                ext_min, ext_max, t_min, t_max)

            if bp.t[2] > 0
                ext_min, ext_max, t_min, t_max = check_step_for_position_extremum(
                    bp.t[1], bp.t[2], bp.p[2], bp.v[2], bp.a[2], bp.j[2],
                    ext_min, ext_max, t_min, t_max)
            end
        end
    end

    # Check main profile phases
    t_current_sum = profile.brake_duration
    for i in 1:7
        ext_min, ext_max, t_min, t_max = check_step_for_position_extremum(
            t_current_sum, profile.t[i], profile.p[i], profile.v[i], profile.a[i], profile.j[i],
            ext_min, ext_max, t_min, t_max)
        t_current_sum += profile.t[i]
    end

    # Check final position
    pf = profile.pf
    total_duration = duration(profile)
    if pf < ext_min
        ext_min = pf
        t_min = total_duration
    end
    if pf > ext_max
        ext_max = pf
        t_max = total_duration
    end

    return PositionBound{T}(ext_min, ext_max, t_min, t_max)
end

"""
    get_first_state_at_position(profile::RuckigProfile, pt; time_after=0.0) -> (found::Bool, time::Float64)

Find the first time at which the trajectory reaches position `pt`.
Optionally search only after `time_after`.

# Returns
- `(true, time)` if position is reached
- `(false, NaN)` if position is never reached
"""
function get_first_state_at_position(profile::RuckigProfile{T}, pt::Real; time_after::Real=0.0) where T
    t_cum = profile.brake_duration  # Start after brake

    for i in 1:7
        if profile.t[i] == 0
            continue
        end

        # Check if we're at the position at phase start
        if abs(profile.p[i] - pt) < EPS && t_cum >= time_after
            return (true, t_cum)
        end

        # Solve cubic: p + v*t + a*t²/2 + j*t³/6 = pt
        # j/6 * t³ + a/2 * t² + v*t + (p - pt) = 0
        j, a, v, p = profile.j[i], profile.a[i], profile.v[i], profile.p[i]

        if abs(j) > EPS
            # Full cubic
            roots = solve_cubic_real(j/6, a/2, v, p - pt)
            for t_root in roots
                if 0 < t_root <= profile.t[i] && (time_after - t_cum) <= t_root
                    return (true, t_root + t_cum)
                end
            end
        elseif abs(a) > EPS
            # Quadratic: a/2 * t² + v*t + (p - pt) = 0
            disc = v^2 - 2*a*(p - pt)
            if disc >= 0
                disc_sqrt = sqrt(disc)
                for t_root in ((-v - disc_sqrt) / a, (-v + disc_sqrt) / a)
                    if 0 < t_root <= profile.t[i] && (time_after - t_cum) <= t_root
                        return (true, t_root + t_cum)
                    end
                end
            end
        elseif abs(v) > EPS
            # Linear: v*t + (p - pt) = 0
            t_root = (pt - p) / v
            if 0 < t_root <= profile.t[i] && (time_after - t_cum) <= t_root
                return (true, t_root + t_cum)
            end
        end

        t_cum += profile.t[i]
    end

    # Check final position
    total_dur = duration(profile)
    if (profile.t[7] > 0 || main_duration(profile) == 0) && abs(profile.pf - pt) < NEWTON_TOL && total_dur >= time_after
        return (true, total_dur)
    end

    return (false, T(NaN))
end

"""
Helper to solve cubic equation: a*x³ + b*x² + c*x + d = 0
Returns real roots only.
"""
function solve_cubic_real(a, b, c, d)
    roots = Float64[]

    if abs(a) < EPS
        # Quadratic or lower
        if abs(b) < EPS
            # Linear
            if abs(c) > EPS
                push!(roots, -d / c)
            end
        else
            # Quadratic
            disc = c^2 - 4*b*d
            if disc >= 0
                disc_sqrt = sqrt(disc)
                push!(roots, (-c - disc_sqrt) / (2*b))
                push!(roots, (-c + disc_sqrt) / (2*b))
            end
        end
        return roots
    end

    # Normalize to monic: x³ + p*x² + q*x + r = 0
    p, q, r = b/a, c/a, d/a

    # Cardano's formula with Vieta's substitution: x = t - p/3
    # t³ + pt + q = 0 where p = q - p²/3, q = r - pq/3 + 2p³/27
    p2 = p^2
    Q = (3*q - p2) / 9
    R = (9*p*q - 27*r - 2*p2*p) / 54

    D = Q^3 + R^2  # Discriminant

    if D > 0
        # One real root
        D_sqrt = sqrt(D)
        S = cbrt(R + D_sqrt)
        T_val = cbrt(R - D_sqrt)
        push!(roots, S + T_val - p/3)
    elseif abs(D) < EPS
        # Three real roots, at least two equal
        if abs(R) < EPS
            push!(roots, -p/3)
        else
            S = cbrt(R)
            push!(roots, 2*S - p/3)
            push!(roots, -S - p/3)
        end
    else
        # Three distinct real roots
        theta = acos(R / sqrt(-Q^3))
        sqrt_neg_Q = sqrt(-Q)
        push!(roots, 2*sqrt_neg_Q*cos(theta/3) - p/3)
        push!(roots, 2*sqrt_neg_Q*cos((theta + 2π)/3) - p/3)
        push!(roots, 2*sqrt_neg_Q*cos((theta + 4π)/3) - p/3)
    end

    return roots
end

export PositionBound, get_position_extrema, get_first_state_at_position

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

    # Step 1: Calculate minimum-time profile for each DOF with blocked intervals
    blocks = Vector{Block{Float64}}(undef, ndof)
    for i in 1:ndof
        blocks[i] = calculate_trajectory_with_block(lims[i];
            p0=p0[i], v0=v0[i], a0=a0[i],
            pf=pf[i], vf=vf[i], af=af[i])
    end

    # Synchronization: find valid t_sync that isn't blocked for any DOF
    # Candidate times are: t_min for each DOF, a.right, b.right for each DOF
    # See C++ calculator_target.hpp lines 123-200

    # Build list of candidate synchronization times
    # Format: (time, dof_index, source) where source: 0=t_min, 1=a.right, 2=b.right
    candidates = Tuple{Float64, Int, Int}[]

    for i in 1:ndof
        push!(candidates, (blocks[i].t_min, i, 0))
        if !isnothing(blocks[i].a)
            push!(candidates, (blocks[i].a.right, i, 1))
        end
        if !isnothing(blocks[i].b)
            push!(candidates, (blocks[i].b.right, i, 2))
        end
    end

    # Sort candidates by time
    sort!(candidates, by=first)

    # Find minimum of all t_min values - can't synchronize faster than the slowest minimum
    min_t_min = maximum(block.t_min for block in blocks)

    # Find valid t_sync: first candidate >= min_t_min that isn't blocked for any DOF
    t_sync = NaN
    limiting_dof = 0
    limiting_source = 0

    for (candidate_time, dof, source) in candidates
        # Skip candidates faster than the slowest minimum-time profile
        candidate_time < min_t_min - T_PRECISION && continue
        isinf(candidate_time) && continue

        # Check if this time is blocked for any DOF
        is_any_blocked = false
        for j in 1:ndof
            if is_blocked(blocks[j], candidate_time)
                is_any_blocked = true
                break
            end
        end

        if !is_any_blocked
            t_sync = candidate_time
            limiting_dof = dof
            limiting_source = source
            break
        end
    end

    if isnan(t_sync)
        error("Failed to find valid synchronization time. Blocks: ", blocks)
    end

    # Step 2: Recalculate non-limiting DOFs for synchronized duration
    profiles = Vector{RuckigProfile{Float64}}(undef, ndof)

    for i in 1:ndof
        # Check if this DOF can use an existing profile (from Step 1 or blocked interval)
        # Use 2*eps tolerance for numerical robustness (matching C++ line 480)
        if abs(t_sync - blocks[i].t_min) < 2 * T_PRECISION
            profiles[i] = blocks[i].p_min
        elseif !isnothing(blocks[i].a) && abs(t_sync - blocks[i].a.right) < 2 * T_PRECISION
            profiles[i] = blocks[i].a.profile
        elseif !isnothing(blocks[i].b) && abs(t_sync - blocks[i].b.right) < 2 * T_PRECISION
            profiles[i] = blocks[i].b.profile
        else
            # Need to recalculate for synchronized duration (Step 2)
            buf = lims[i].buffer
            clear!(buf)

            success = calculate_profile_step2!(lims[i].roots, buf, t_sync,
                p0[i], v0[i], a0[i], pf[i], vf[i], af[i],
                lims[i].jmax, lims[i].vmax, lims[i].vmin,
                lims[i].amax, lims[i].amin)

            if !success
                error("Failed to find synchronized profile for DOF $i at duration $t_sync. Blocks: $blocks, lims = $lims, p0 = $p0, v0 = $v0, a0 = $a0, pf = $pf, vf = $vf, af = $af")
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

# Include velocity control interface
include("ruckig_velocity.jl")

# Include enhanced block interval collection
include("ruckig_block.jl")
