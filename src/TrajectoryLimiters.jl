"""
Quick example follows below, see the [readme](https://github.com/baggepinnen/TrajectoryLimiters.jl) for more details.

```julia
using TrajectoryLimiters

ẍM   = 50                       # Maximum acceleration
ẋM   = 10                       # Maximum velocity
Ts   = 0.005                    # Sample time
r(t) = 2.5 + 3 * (t - floor(t)) # Reference to be smoothed
t    = 0:Ts:3                   # Time vector
R    = r.(t)                    # An array of sampled position references

limiter = TrajectoryLimiter(Ts, ẋM, ẍM)

X, Ẋ, Ẍ = limiter(R)

plot(
    t,
    [X Ẋ Ẍ],
    plotu = true,
    c = :black,
    title = ["Position \$x(t)\$" "Velocity \$ẋ(t)\$" "Acceleration \$u(t)\$"],
    ylabel = "",
    layout = (3,1),
)
plot!(r, extrema(t)..., sp = 1, lab = "", l = (:black, :dashdot))
```
"""
module TrajectoryLimiters

export TrajectoryLimiter, JerkTrajectoryLimiter

sat(x) = clamp(x, -one(x), one(x))

"""
    State{T}

# Fields:
- `x`: Filtered reference position
- `ẋ`: Filtered reference velocity
- `r`: Reference position
- `ṙ`: Reference velocity
"""
struct State{T}
    x::T
    ẋ::T
    r::T
    ṙ::T
end

State(args...) = State(promote(args...)...)
State(R) = State(0*R[1], 0, R[1], 0)

"""
    JerkState{T}

State for the jerk-limited trajectory filter.

# Fields:
- `x`: Filtered reference position
- `ẋ`: Filtered reference velocity
- `ẍ`: Filtered reference acceleration
- `r`: Reference position
- `ṙ`: Reference velocity
"""
struct JerkState{T}
    x::T
    ẋ::T
    ẍ::T
    r::T
    ṙ::T
end

JerkState(args...) = JerkState(promote(args...)...)
JerkState(R) = JerkState(0*R[1], 0, 0, R[1], 0)

struct TrajectoryLimiter{T}
    Ts::T
    ẋM::T
    ẍM::T
end

"""
    limiter = TrajectoryLimiter(Ts, ẋM, ẍM)

Create a trajectory limiter that can be called like so
```julia
rlim = limiter(state, r::Number)
# or
X, Ẋ, Ẍ = limiter(state, R::Vector)
# or
X, Ẋ, Ẍ = limiter(R::Vector) # Uses a zero initial state
```

- `ẋM`: Upper bound on the magnitude of the velocity
- `ẍM`: Upper bound on the magnitude of the acceleration (in the reference paper, this bound is denoted by U)
"""
function TrajectoryLimiter(args...)
    TrajectoryLimiter(promote(args...)...)
end


"""
    state, ẍ = trajlim(state, r::Number, Ts, ẋM, ẍM)

Return an updated state and the acceleration

# Arguments:
- `state`: An instance of `TrajectoryLimiters.State`
- `r`: The reference input
- `Ts`: The sample time
- `ẋM`: The maximum velocity
- `ẍM`: The maximum acceleration
"""
function trajlim(state, rt, Ts, ẋM, ẍM)
    (; x, ẋ, r, ṙ) = state

    TU = Ts*ẍM
    ṙ = (rt-r)/Ts
    r = rt
    # ṙ = 2/Ts*(r-rold) - ṙold # The expression in the paper doesn't work

    e = x-r
    ė = ẋ-ṙ

    z = 1/TU * (e/Ts + ė/2)
    ż = ė/TU
    m = floor((1 + √(1 + 8abs(z))) / 2)
    σ = ż + z/m + (m-1)/2*sign(z)
    u = -ẍM*sat(σ)*(1 + sign(ẋ*sign(σ) + ẋM-TU))/2

    # x⁺ - x = Ts u
    # x⁺ = Ts*u + x
    ẋ1 = Ts*u + ẋ

    # x⁺ - x = Ts/2 (u⁺ + u)
    # x⁺ = Ts/2 (u⁺ + u) + x
    x1 = Ts/2*(ẋ1 + ẋ) + x

    State(x1, ẋ1, r, ṙ), u
end

function (limiter::TrajectoryLimiter)(state, r::Number)
    trajlim(state, r, limiter.Ts, limiter.ẋM, limiter.ẍM)
end

(limiter::TrajectoryLimiter)(R::AbstractArray) = limiter(State(R), R)
(limiter::TrajectoryLimiter)(X, Ẋ, Ẍ, R::AbstractArray) = limiter(State(R), X, Ẋ, Ẍ, R)

function (limiter::TrajectoryLimiter)(state, R::AbstractArray)
    X = similar(R)
    Ẋ = similar(R)
    Ẍ = similar(R)
    limiter(state, X, Ẋ, Ẍ, R)
end

function (limiter::TrajectoryLimiter)(state, X, Ẋ, Ẍ, R::AbstractArray)
    T = length(R)
    length(X) == length(Ẋ) == length(Ẍ) == T || throw(ArgumentError("Inconsistent array lengths"))
    @inbounds for i = 1:T
        X[i] = state.x
        Ẋ[i] = state.ẋ
        state, u = limiter(state, R[i])
        Ẍ[i] = u
    end
    X, Ẋ, Ẍ
end

# =============================================================================
# Jerk-limited trajectory filter (Third-order)
# Based on: "Third Order Trajectory Generator Satisfying Velocity, Acceleration
# and Jerk Constraints" - R. Zanasi, R. Morselli (2002)
# =============================================================================

struct JerkTrajectoryLimiter{T}
    Ts::T
    ẋM::T
    ẍM::T
    x⃛M::T  # Jerk bound (called U in the paper)
end

"""
    limiter = JerkTrajectoryLimiter(Ts, ẋM, ẍM, x⃛M)

Create a jerk-limited trajectory limiter that can be called like so
```julia
state, u = limiter(state, r::Number)
# or
X, Ẋ, Ẍ, X⃛ = limiter(state, R::Vector)
# or
X, Ẋ, Ẍ, X⃛ = limiter(R::Vector) # Uses a zero initial state
```

# Arguments
- `Ts`: Sample time
- `ẋM`: Upper bound on the magnitude of the velocity
- `ẍM`: Upper bound on the magnitude of the acceleration
- `x⃛M`: Upper bound on the magnitude of the jerk (called U in the reference paper)
"""
function JerkTrajectoryLimiter(args...)
    JerkTrajectoryLimiter(promote(args...)...)
end

# Helper functions for the jerk-limited control law
# These implement equations from the paper

# δ* (eq. 8): Curve in error space
_δstar(ẏ, ÿ) = ẏ + ÿ * abs(ÿ) / 2

# σ* (eq. 7): Sliding surface for unconstrained minimum-time control
function _σstar(y, ẏ, ÿ)
    δs = _δstar(ẏ, ÿ)
    sδ = sign(δs)
    abs_sδ = abs(sδ)
    term1 = y + ẏ * ÿ
    term2 = -ÿ^3 / 6 * (1 - 3 * abs_sδ)
    inner = ÿ^2 + 2 * ẏ * sδ
    term3 = inner > 0 ? sδ / 4 * sqrt(2 * inner^3) : zero(y)
    term1 + term2 + term3
end

# ν⁺* (eq. 17): Modified surface for max velocity bound
function _νplus_star(y, ẏ, ÿ, ÿM)
    term1 = ÿM * (ÿ^2 - 2ẏ) / 4
    term2 = (ÿ^2 - 2ẏ)^2 / (8 * ÿM)
    term3 = ÿ * (3ẏ - ÿ^2) / 3
    y - term1 - term2 - term3
end

# ν⁻* (eq. 18): Modified surface for min velocity bound
function _νminus_star(y, ẏ, ÿ, ÿM)
    term1 = ÿM * (ÿ^2 + 2ẏ) / 4
    term2 = (ÿ^2 + 2ẏ)^2 / (8 * ÿM)
    term3 = ÿ * (3ẏ + ÿ^2) / 3
    y - term1 - term2 + term3
end

# uₐ(a) (eq. 12): Control for reaching acceleration a
_ua(ÿ, a, U) = -U * sign(ÿ - a)

# δ(v) for velocity control (eq. 13)
_δv(ẏ, ÿ, v) = ÿ * abs(ÿ) + 2 * (ẏ - v)

# uᵥ(v) (eq. 13): Control for reaching velocity v with acceleration bounds
function _uv(ẏ, ÿ, v, ÿm, ÿM, U)
    δv = _δv(ẏ, ÿ, v)
    sδv = sign(δv)
    ucv = -U * sign(δv + (1 - abs(sδv)) * ÿ)
    clamp(ucv, _ua(ÿ, ÿm, U), _ua(ÿ, ÿM, U))
end

# Σ* (eq. 19): Selects appropriate surface based on region
function _Σstar(y, ẏ, ÿ, ÿm, ÿM)
    # Check which region we're in
    # R_ν+ : ÿ ≤ ÿM and ẏ ≤ ÿ²/2 - ÿM²
    # R_ν- : ÿ ≥ ÿm and ẏ ≥ ÿm² - ÿ²/2
    if ÿ <= ÿM && ẏ <= ÿ^2 / 2 - ÿM^2
        return _νplus_star(y, ẏ, ÿ, ÿM)
    elseif ÿ >= ÿm && ẏ >= ÿm^2 - ÿ^2 / 2
        return _νminus_star(y, ẏ, ÿ, abs(ÿm))  # Use |ÿm| since ÿM in paper is positive
    else
        return _σstar(y, ẏ, ÿ)
    end
end

# Full control law (eq. 19)
function _jerk_control(y, ẏ, ÿ, ẏm, ẏM, ÿm, ÿM, U)
    Σs = _Σstar(y, ẏ, ÿ, ÿm, ÿM)
    δs = _δstar(ẏ, ÿ)
    sΣ = sign(Σs)
    sδ = sign(δs)

    # Inner control: u_c = -U sgn{Σ* + (1-|sgn(Σ*)|)·[δ* + (1-|sgn(δ*)|)ÿ]}
    inner = δs + (1 - abs(sδ)) * ÿ
    arg = Σs + (1 - abs(sΣ)) * inner
    uc = -U * sign(arg)

    # Apply velocity limits via u_v
    u_lower = _uv(ẏ, ÿ, ẏm, ÿm, ÿM, U)
    u_upper = _uv(ẏ, ÿ, ẏM, ÿm, ÿM, U)

    clamp(uc, u_lower, u_upper)
end

"""
    state, u = trajlim(state::JerkState, r::Number, Ts, ẋM, ẍM, x⃛M)

Return an updated state and the jerk for the jerk-limited trajectory filter.

# Arguments:
- `state`: An instance of `TrajectoryLimiters.JerkState`
- `r`: The reference input
- `Ts`: The sample time
- `ẋM`: The maximum velocity
- `ẍM`: The maximum acceleration
- `x⃛M`: The maximum jerk (called U in the paper)
"""
function trajlim(state::JerkState, rt, Ts, ẋM, ẍM, x⃛M)
    (; x, ẋ, ẍ, r, ṙ) = state
    U = x⃛M

    # Update reference velocity estimate
    ṙ_new = (rt - r) / Ts
    r_new = rt

    # Normalized error coordinates (eq. 4 area)
    # The paper assumes r̈ = 0 for admissible references
    y = (x - r_new) / U
    ẏ = (ẋ - ṙ_new) / U
    ÿ = ẍ / U

    # Normalized bounds
    ẏM = (ẋM - ṙ_new) / U  # Corresponds to eq. 11 with ṙ instead of ṙmax
    ẏm = (-ẋM - ṙ_new) / U
    ÿM = ẍM / U
    ÿm = -ẍM / U

    # Compute control (jerk)
    u = _jerk_control(y, ẏ, ÿ, ẏm, ẏM, ÿm, ÿM, U)

    # Discrete-time integration (trapezoidal rule, matching existing code pattern)
    ẍ1 = Ts * u + ẍ
    ẋ1 = Ts / 2 * (ẍ1 + ẍ) + ẋ
    x1 = Ts / 2 * (ẋ1 + ẋ) + x

    JerkState(x1, ẋ1, ẍ1, r_new, ṙ_new), u
end

function (limiter::JerkTrajectoryLimiter)(state::JerkState, r::Number)
    trajlim(state, r, limiter.Ts, limiter.ẋM, limiter.ẍM, limiter.x⃛M)
end

(limiter::JerkTrajectoryLimiter)(R::AbstractArray) = limiter(JerkState(R), R)
(limiter::JerkTrajectoryLimiter)(X, Ẋ, Ẍ, X⃛, R::AbstractArray) = limiter(JerkState(R), X, Ẋ, Ẍ, X⃛, R)

function (limiter::JerkTrajectoryLimiter)(state::JerkState, R::AbstractArray)
    X = similar(R)
    Ẋ = similar(R)
    Ẍ = similar(R)
    X⃛ = similar(R)
    limiter(state, X, Ẋ, Ẍ, X⃛, R)
end

function (limiter::JerkTrajectoryLimiter)(state::JerkState, X, Ẋ, Ẍ, X⃛, R::AbstractArray)
    T = length(R)
    length(X) == length(Ẋ) == length(Ẍ) == length(X⃛) == T || throw(ArgumentError("Inconsistent array lengths"))
    @inbounds for i = 1:T
        X[i] = state.x
        Ẋ[i] = state.ẋ
        Ẍ[i] = state.ẍ
        state, u = limiter(state, R[i])
        X⃛[i] = u
    end
    X, Ẋ, Ẍ, X⃛
end

end
