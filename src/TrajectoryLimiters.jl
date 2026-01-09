"""
Quick example follows below, see the [readme](https://github.com/baggepinnen/TrajectoryLimiters.jl) for more details.
    
```julia
using TrajectoryLimiters

ẍM   = 50                       # Maximum acceleration
ẋM   = 10                       # Maximum velocity
Ts   = 0.005                    # Sample time
r(t) = 2.5 + 3 * (t - floor(t)) # Reference to be smoothed
t    = 0:Ts:3                   # Time vector
R    = r.(t)                    # An array of sampled position references 

limiter = TrajectoryLimiter(Ts, ẋM, ẍM)

X, Ẋ, Ẍ = limiter(R)

plot(
    t,
    [X Ẋ Ẍ],
    plotu = true,
    c = :black,
    title = ["Position \$x(t)\$" "Velocity \$ẋ(t)\$" "Acceleration \$u(t)\$"],
    ylabel = "",
    layout = (3,1),
)
plot!(r, extrema(t)..., sp = 1, lab = "", l = (:black, :dashdot))
```
"""
module TrajectoryLimiters

export TrajectoryLimiter

include("ruckig.jl")

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
    ẋ::T
    r::T
    ṙ::T
end

State(args...) = State(promote(args...)...)
State(R) = State(0*R[1], 0, R[1], 0)

struct TrajectoryLimiter{T}
    Ts::T
    ẋM::T
    ẍM::T
end

"""
    limiter = TrajectoryLimiter(Ts, ẋM, ẍM)

Create a trajectory limiter that can be called like so
```julia
rlim = limiter(state, r::Number)
# or
X, Ẋ, Ẍ = limiter(state, R::Vector)
# or 
X, Ẋ, Ẍ = limiter(R::Vector) # Uses a zero initial state
```

- `ẋM`: Upper bound on the magnitude of the velocity
- `ẍM`: Upper bound on the magnitude of the acceleration (in the reference paper, this bound is denoted by U)
"""
function TrajectoryLimiter(args...)
    TrajectoryLimiter(promote(args...)...)
end


"""
    state, ẍ = trajlim(state, r::Number, Ts, ẋM, ẍM)

Return an updated state and the acceleration

# Arguments:
- `state`: An instance of `TrajectoryLimiters.State`
- `r`: The reference input
- `Ts`: The sample time
- `ẋM`: The maximum velocity
- `ẍM`: The maximum acceleration
"""
function trajlim(state, rt, Ts, ẋM, ẍM)
    (; x, ẋ, r, ṙ) = state

    TU = Ts*ẍM
    ṙ = (rt-r)/Ts
    r = rt
    # ṙ = 2/Ts*(r-rold) - ṙold # The expression in the paper doesn't work

    e = x-r
    ė = ẋ-ṙ

    z = 1/TU * (e/Ts + ė/2)
    ż = ė/TU
    m = floor((1 + √(1 + 8abs(z))) / 2)
    σ = ż + z/m + (m-1)/2*sign(z)
    u = -ẍM*sat(σ)*(1 + sign(ẋ*sign(σ) + ẋM-TU))/2

    # x⁺ - x = Ts u
    # x⁺ = Ts*u + x
    ẋ1 = Ts*u + ẋ

    # x⁺ - x = Ts/2 (u⁺ + u)
    # x⁺ = Ts/2 (u⁺ + u) + x
    x1 = Ts/2*(ẋ1 + ẋ) + x

    State(x1, ẋ1, r, ṙ), u
end

function (limiter::TrajectoryLimiter)(state, r::Number)
    trajlim(state, r, limiter.Ts, limiter.ẋM, limiter.ẍM)
end

(limiter::TrajectoryLimiter)(R::AbstractArray) = limiter(State(R), R)
(limiter::TrajectoryLimiter)(X, Ẋ, Ẍ, R::AbstractArray) = limiter(State(R), X, Ẋ, Ẍ, R)

function (limiter::TrajectoryLimiter)(state, R::AbstractArray)
    X = similar(R)
    Ẋ = similar(R)
    Ẍ = similar(R)
    limiter(state, X, Ẋ, Ẍ, R)
end

function (limiter::TrajectoryLimiter)(state, X, Ẋ, Ẍ, R::AbstractArray)
    T = length(R)
    length(X) == length(Ẋ) == length(Ẍ) == T || throw(ArgumentError("Inconsistent array lengths"))
    @inbounds for i = 1:T
        X[i] = state.x
        Ẋ[i] = state.ẋ
        state, u = limiter(state, R[i])
        Ẍ[i] = u
    end
    X, Ẋ, Ẍ
end

end
