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

using LinearMPC: compute_control, set_objective!, set_input_bounds!, add_constraint!, setup!
import LinearMPC
using ControlSystemsBase

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
# =============================================================================

struct JerkTrajectoryLimiter{T, M}
    Ts::T
    ẋM::T
    ẍM::T
    x⃛M::T  # Jerk bound
    mpc::M  # MPC controller
end

"""
    limiter = JerkTrajectoryLimiter(Ts, ẋM, ẍM, x⃛M; Np=20)

Create a jerk-limited trajectory limiter using Model Predictive Control.

The limiter uses a discrete-time triple integrator model with state `[position, velocity, acceleration]`
and input `jerk`. It minimizes position tracking error while respecting velocity, acceleration, and jerk bounds.

Can be called like so:
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
- `x⃛M`: Upper bound on the magnitude of the jerk

# Keyword Arguments
- `Np`: Prediction horizon
"""
function JerkTrajectoryLimiter(Ts, ẋM, ẍM, x⃛M; Np=100)
    Ts, ẋM, ẍM, x⃛M = promote(Ts, ẋM, ẍM, x⃛M)

    # Continuous-time triple integrator: ẋ = Ac*x + Bc*u
    # State: [position, velocity, acceleration], Input: jerk
    Ac = [0 1.0 0; 0 0 1; 0 0 0]
    Bc = [0.0; 0; 1]

    # Output matrix: track position and velocity
    Cc = [1.0 0 0; 0 1 0; 0 0 1]
    # C = [1.0 0 0]
    # M = exp([Ac Bc zeros(3,1); zeros(1,3) zeros(1,1) I(1); zeros(1,5)] * Ts)
    # F = M[1:3, 1:3]
    # G = M[1:3, 4]
    # G1 = M[1:3, 4]

    # Create MPC (LinearMPC will discretize automatically)
    # mpc = LinearMPC.MPC(A, B; C, Ts=Float64(Ts), Np, Nc)
    mpc = LinearMPC.MPC(Ac, Bc, Float64(Ts); C=Cc, Np)

    # Set objective: minimize position and velocity tracking error
    set_objective!(mpc; Q=[1e4, 0, 0], R=[1e-6], Rr=[1e-4], Qf=[1e8, 1e-2, 1e-3])
    # Set jerk (input) bounds
    set_input_bounds!(mpc; umin=[-x⃛M], umax=[x⃛M])

    # add_constraint!(mpc;
    #     Ax=[1 0.0 0.0],
    #     Ar=[-1 0.0 0.0],
    #     lb=[0], ub=[0], ks=Np:Np,
    #     soft=true)

    add_constraint!(mpc;
        Ax=[0.0 1.0 0.0],
        lb=[-ẋM], ub=[ẋM],
        soft=false)

    # Add acceleration constraint: -ẍM ≤ x₃ ≤ ẍM (soft to avoid infeasibility)
    add_constraint!(mpc;
        Ax=[0.0 0.0 1.0],
        lb=[-ẍM], ub=[ẍM],
        soft=false)

    sysd = ss(mpc.model.F, mpc.model.G, Cc, 0)
    K = place(sysd, [0.3, 0.3, 0.3])
    # LinearMPC.set_prestabilizing_feedback!(mpc, K)
    mpc.settings.reference_preview = true

    LinearMPC.move_block!(mpc,[ones(40); 1ones(100)])

    # parameter_range = LinearMPC.ParameterRange(mpc)
    # parameter_range.xmax .= [10, ẋM, ẍM]
    # parameter_range.xmin .= -[10, ẋM, ẍM]
    # parameter_range.rmax .= [10, ẋM, ẍM]
    # parameter_range.rmin .= -[10, ẋM, ẍM]
    # mpc = LinearMPC.ExplicitMPC(mpc; build_tree=true, range=parameter_range)

    # Setup the MPC (converts to QP form)
    setup!(mpc)

    LinearMPC.DAQP.settings(mpc.opt_model, Dict(
        # :iter_limit => 2000,
        # :primal_tol => 1e-8,
        :dual_tol => 1e-10,
        :pivot_tol => 1e-5, # A higher value improves stability. 1e-6
    ))

    JerkTrajectoryLimiter(Ts, ẋM, ẍM, x⃛M, mpc)
end


# =============================================================================
# MPC-based jerk-limited trajectory filter
# =============================================================================

"""
    state, u = trajlim(state::JerkState, r, limiter::JerkTrajectoryLimiter)

Compute one step of the jerk-limited trajectory filter using MPC.

Returns updated state and the jerk control input.
"""
function trajlim(state::JerkState, rt::Number, limiter::JerkTrajectoryLimiter)
    T = limiter.Ts

    # Current state vector [position, velocity, acceleration]
    x = [state.x, state.ẋ, state.ẍ]
    A = limiter.mpc.model.F
    B = limiter.mpc.model.G

    ṙ_new = (rt - state.r) / T
    r̈_new = (ṙ_new - state.ṙ) / T

    r = [rt, ṙ_new, r̈_new]
    R = zeros(3, limiter.mpc.Np)
    R[:, 1] .= r
    for i = 1:limiter.mpc.Np-1
        R[:, i+1] .= A*R[:, i] # predict reference assuming zero jerk
    end

    u_vec = compute_control(limiter.mpc, x; r=R, check=true)
    u = u_vec[1]
    x1 = A*x + B*u
    x1, ẋ1, ẍ1 = x1

    JerkState(x1, ẋ1, ẍ1, rt, ṙ_new), u
end

function trajlim(state::JerkState, R::AbstractMatrix, limiter::JerkTrajectoryLimiter)
    T = limiter.Ts

    # Current state vector [position, velocity, acceleration]
    x = [state.x, state.ẋ, state.ẍ]
    A = limiter.mpc.model.F
    B = limiter.mpc.model.G
    rt = R[1]
    ṙ_new = (rt - state.r) / T

    u_vec = compute_control(limiter.mpc, x; r=R, check=true)
    u = u_vec[1]
    x1 = A*x + B*u
    x1, ẋ1, ẍ1 = x1

    JerkState(x1, ẋ1, ẍ1, rt, ṙ_new), u
end

function (limiter::JerkTrajectoryLimiter)(state::JerkState, r)
    trajlim(state, r, limiter)
end

(limiter::JerkTrajectoryLimiter)(R::AbstractArray; kwargs...) = limiter(JerkState(R), R; kwargs...)
(limiter::JerkTrajectoryLimiter)(X, Ẋ, Ẍ, X⃛, R::AbstractArray; kwargs...) = limiter(JerkState(R), X, Ẋ, Ẍ, X⃛, R; kwargs...)

function (limiter::JerkTrajectoryLimiter)(state::JerkState, R::AbstractVector; kwargs...)
    X = similar(R)
    Ẋ = similar(R)
    Ẍ = similar(R)
    X⃛ = similar(R)
    limiter(state, X, Ẋ, Ẍ, X⃛, R; kwargs...)
end

function (limiter::JerkTrajectoryLimiter)(state::JerkState, X, Ẋ, Ẍ, X⃛, R::AbstractVector; causal=true)
    T = length(R)
    length(X) == length(Ẋ) == length(Ẍ) == length(X⃛) == T || throw(ArgumentError("Inconsistent array lengths"))
    if !causal
        Rd = centraldiff(R) ./ limiter.Ts
        Rdd = centraldiff(Rd) ./ limiter.Ts
        Rmat = [R Rd Rdd]'
    end
    @inbounds for i = 1:T
        X[i] = state.x
        Ẋ[i] = state.ẋ
        Ẍ[i] = state.ẍ
        state, u = limiter(state, causal ? R[i] : Rmat[:, i:end])
        X⃛[i] = u
    end
    X, Ẋ, Ẍ, X⃛
end

function centraldiff(v::AbstractVector)
    dv = Base.diff(v)/2
    a1 = [dv[1];dv]
    a2 = [dv;dv[end]]
    a = a1+a2
end

end


# Based on: "Discrete-Time Third Order Trajectory Generator Satisfying Velocity,
# Acceleration and Jerk Constraints" - O. Gerelli, C. Guarino Lo Bianco (2010)
# IEEE Int. Conf. on Robotics and Automation (ICRA)
# Helper functions for the discrete-time jerk-limited control law
# These implement equations from jerk2.pdf (Gerelli & Guarino Lo Bianco, 2010)

# Velocity limit curves in z-coordinates (eq. 17-18)
# z̄₂⁺(z₃) = -⌈z₃⌉(z₃ - (⌈z₃⌉-1)/2) for upper velocity bound
# z̄₂⁻(z₃) = ⌈-z₃⌉(-z₃ - (⌈-z₃⌉-1)/2) for lower velocity bound
# function _z̄2_plus(z3)
#     c = ceil(z3)
#     -c * (z3 - (c - 1) / 2)
# end

# function _z̄2_minus(z3)
#     c = ceil(-z3)
#     c * (-z3 - (c - 1) / 2)
# end

# # Compute sliding surface σₙ (eq. 22-23)
# function _compute_σ(γ)
#     m = floor((1 + sqrt(1 + 8 * abs(γ))) / 2)
#     -(m - 1) / 2 * sign(γ) - γ / m
# end

# # Minimum-time sliding surface σ₃ from reference [16] (Zanasi & Morselli, Automatica 2003)
# # The control law from [16] eq. 16 is: α = z₃ + c_z2*z₂ + c_z1*z₁ + c_η*η, u = -sat(α)
# # So: σ₃ = -(c_z2*z₂ + c_z1*z₁ + c_η*η)

# # Compute B⁺ₕ,ₖ vertex coordinates (eq. 11 from [16])
# function _B_plus(h, k)
#     z1 = k*(k-1)*(k-2)//6 + k*(k-1)*h//2 + h*(h-1)*k//2 - h*(h-1)*(h-2)//6
#     z2 = h*(h-1)//2 - h*k - k*(k-1)//2
#     Float64(z1), Float64(z2)
# end

# # Check if point (z1, z2) is inside parallelogram with vertices v1, v2, v3, v4
# function _point_in_parallelogram(z1, z2, v1, v2, v3, v4)
#     # Cross product sign check
#     cross_sign(ax, ay, bx, by, px, py) = sign((bx - ax) * (py - ay) - (by - ay) * (px - ax))

#     s1 = cross_sign(v1[1], v1[2], v2[1], v2[2], z1, z2)
#     s2 = cross_sign(v2[1], v2[2], v3[1], v3[2], z1, z2)
#     s3 = cross_sign(v3[1], v3[2], v4[1], v4[2], z1, z2)
#     s4 = cross_sign(v4[1], v4[2], v1[1], v1[2], z1, z2)

#     (s1 >= 0 && s2 >= 0 && s3 >= 0 && s4 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0 && s4 <= 0)
# end

# # Find (h, k, η) for a given (z1, z2) point - Figure 7 in [16]
# function _find_hkη(z1, z2)
#     # Handle symmetry: P⁻ₕ,ₖ = -P⁺ₕ,ₖ
#     if z1 < 0 || (z1 == 0 && z2 > 0)
#         η = -one(z1)
#         z1, z2 = -z1, -z2
#     else
#         η = one(z1)
#     end

#     # Special case near origin
#     if abs(z1) < 0.5 && abs(z2) < 0.5
#         return one(z1), one(z1), η
#     end

#     # Estimate search range based on scaling
#     scale = sqrt(abs(z1) + abs(z2) + 1)
#     max_search = max(5, Int(ceil(3 * scale)))

#     # Search for parallelogram containing (z1, z2)
#     for total in 2:max_search+100
#         for h in 1:total-1
#             k = total - h
#             k < 1 && continue

#             # Vertices of P⁺ₕ,ₖ in counter-clockwise order (non-self-intersecting)
#             # From Figure 4: B⁺ₕ,ₖ → B⁺ₕ,ₖ₊₁ → B⁺ₕ₊₁,ₖ → B⁺ₕ₊₁,ₖ₋₁
#             v1 = _B_plus(h, k)
#             v2 = _B_plus(h, k+1)
#             v3 = _B_plus(h+1, k)
#             v4 = _B_plus(h+1, k-1)

#             if _point_in_parallelogram(z1, z2, v1, v2, v3, v4)
#                 return typeof(z1)(h), typeof(z1)(k), η
#             end
#         end
#     end

#     # Fallback
#     return one(z1), one(z1), η
# end

# # Compute σ₃ using the formula from [16] eq. 16
# function _compute_σ3(z1, z2)
#     h, k, η = _find_hkη(z1, z2)

#     # From [16] eq. 16: α = z₃ + (2h+k-1)/(h(h+k))*z₂ + 2/(h(h+k))*z₁ + c_η*η
#     # So σ₃ = -[those terms] since we want α = z₃ - σ₃
#     denom = h * (h + k)
#     c_z2 = (2h + k - 1) / denom
#     c_z1 = 2 / denom
#     c_η = (2h^3 + k^3 + 3h^2*k - 3h*k - 3h^2 + h - k) / (6 * denom)

#     σ3 = -(c_z2 * z2 + c_z1 * z1 + c_η * η)
#     σ3
# end

# """
#     state, u = trajlim(state::JerkState, r::Number, Ts, ẋM, ẍM, x⃛M)

# Return an updated state and the jerk for the jerk-limited trajectory filter.

# # Arguments:
# - `state`: An instance of `TrajectoryLimiters.JerkState`
# - `r`: The reference input
# - `Ts`: The sample time
# - `ẋM`: The maximum velocity
# - `ẍM`: The maximum acceleration
# - `x⃛M`: The maximum jerk (called U in the paper)
# """
# function trajlim(state::JerkState, rt, Ts, ẋM, ẍM, x⃛M)
#     (; x, ẋ, ẍ, r, ṙ) = state
#     T = Ts
#     U = x⃛M
#     TU = T * U

#     # Update reference velocity estimate
#     ṙ_new = (rt - r) / T
#     r_new = rt

#     # Error coordinates (y = x - r)
#     y = x - r_new
#     ẏ = ẋ - ṙ_new
#     ÿ = ẍ  # r̈ = 0 for admissible references

#     # Normalized coordinates z = W⁻¹ y (eq. 12)
#     z1 = (1 / TU) * (y / T^2 + ẏ / T + ÿ / 3)
#     z2 = (1 / TU) * (ẏ / T + ÿ / 2)
#     z3 = ÿ / TU

#     # Bounds in z-coordinates (eq. 13-16)
#     z2_plus = (ẋM - ṙ_new) / (T^2 * U)   # max velocity bound
#     z2_minus = (-ẋM - ṙ_new) / (T^2 * U)  # min velocity bound
#     z3_plus = ẍM / (T * U)                 # max acceleration bound
#     z3_minus = -ẍM / (T * U)               # min acceleration bound

#     # Velocity limit curves evaluated at current z3 (eq. 17-18)
#     z̄2_plus = _z̄2_plus(z3_plus)
#     z̄2_minus = _z̄2_minus(z3_minus)

#     # d₁ and d₂: distances to velocity curves (eq. 19-20)
#     d1 = z2 - z̄2_plus
#     d2 = z2 - z̄2_minus

#     # γ values clamped to acceleration bounds (eq. 21)
#     γ1 = clamp(d1, z2_minus, z2_plus)
#     γ2 = clamp(d2, z2_minus, z2_plus)

#     # Compute sliding surfaces (eq. 22-23)
#     σ1 = _compute_σ(γ1)  # Upper velocity constraint surface
#     σ2 = _compute_σ(γ2)  # Lower velocity constraint surface
#     σ3 = _compute_σ3(z1, z2)  # Minimum-time surface (eq. 24)

#     # Select sliding surface (eq. 25): σ = max(σ₁, min(σ₃, σ₂))
#     # σ₁ is the lower bound (from upper velocity limit)
#     # σ₂ is the upper bound (from lower velocity limit)
#     # σ₃ is clamped to [σ₁, σ₂]
#     σ = clamp(σ3, σ1, σ2)

#     # Explicit acceleration constraint: only intervene when about to violate bounds
#     # z₃_next = z₃ - sat(α), so to prevent z₃_next > z₃⁺ we need sat(α) ≥ z₃ - z₃⁺
#     # This is only binding when z₃ ≥ z₃⁺ - 1 (one step from bound)
#     if z3 >= z3_plus - 1
#         # Need α ≥ z3 - z3_plus to ensure z3_next ≤ z3_plus
#         # α = z3 - σ, so σ ≤ z3_plus
#         σ = min(σ, z3_plus)
#     end
#     if z3 <= z3_minus + 1
#         # Need α ≤ z3 - z3_minus to ensure z3_next ≥ z3_minus
#         # α = z3 - σ, so σ ≥ z3_minus
#         σ = max(σ, z3_minus)
#     end

#     # Control law (eq. 26-27)
#     α = z3 - σ
#     u = -U * sat(α)

#     # Debug output - check if z3 respects bounds and region classification
#     z3_in_bounds = z3_minus ≤ z3 ≤ z3_plus
#     z3_next = z3 - sat(α)  # predicted next z3
#     z3_next_in_bounds = z3_minus ≤ z3_next ≤ z3_plus
#     @info "JerkLimiter" z3 z3_minus z3_plus z3_in_bounds z3_next z3_next_in_bounds z2 z̄2_plus z̄2_minus σ1 σ2 σ3 σ α u

#     # Discrete-time integration (eq. 2-3 from paper)
#     # x_{i+1} = x_i + T*ẋ_i + T²/2*ẍ_i + T³/6*u_i
#     # ẋ_{i+1} = ẋ_i + T*ẍ_i + T²/2*u_i
#     # ẍ_{i+1} = ẍ_i + T*u_i
#     x1 = x + T * ẋ + T^2 / 2 * ẍ + T^3 / 6 * u
#     ẋ1 = ẋ + T * ẍ + T^2 / 2 * u
#     ẍ1 = ẍ + T * u

#     JerkState(x1, ẋ1, ẍ1, r_new, ṙ_new), u
# end