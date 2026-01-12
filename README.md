# TrajectoryLimiters

[![Build Status](https://github.com/baggepinnen/TrajectoryLimiters.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/TrajectoryLimiters.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/baggepinnen/TrajectoryLimiters.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/baggepinnen/TrajectoryLimiters.jl)

Contains implementations of 
> Nonlinear filters for the generation of smooth trajectories
> R. Zanasi, C. Guarino Lo Bianco, A. Tonielli

and the _ruckig_ algorithm from

> Jerk-limited Real-time Trajectory Generation with Arbitrary Target States
> M. Berscheid, T. Kröger

The ruckig algorithm is a time-optimal jerk-limited trajectory generator for point-to-point motions with arbitrary initial and target states.

The nonlinear trajectory filter takes a pre-defined reference trajectory $r(t)$ (uniformly sampled in $t$) and filters it (causally) such that the velocity and acceleration are bounded by $ẋ_M$ and $ẍ_M$.

What is this good for? Some applications call for a dynamically feasible reference trajectory, i.e., a trajectory with bounded velocity and acceleration, but all you have access to is an instantaneous reference $r(t)$ that might change abruptly, e.g., from an operator changing a set point. In such situations, this filter performs the required pre-processing of the reference to provide a smoother, dynamically feasible reference trajectory. If you already have a trajectory planner that outputs dynamically feasible trajectories (like the _ruckig_ algorithm), you do not need this filter.

## Trajectory filtering

To filter an entire preexisting trajectory, create a `TrajectoryLimiter` and call it like a function:
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
![limited trajectory](https://user-images.githubusercontent.com/3797491/204131020-c0dbcfa5-33f2-44df-b12d-528f3f4e7132.png)

The figure above reproduces figure 10 from the reference, except that we did not increase the acceleration bound (which we call $ẍ_M$ but they call $U$) at time $t=2$ like they did. To do this, use the lower-level interface explained below.

The figure indicates that the limited (solid lines) trajectory follows the original reference trajectory (dashed line) whenever possible, but deviates whenever the original trajectory violates the velocity or acceleration constraints. When it has deviated, the limited trajectory converges to the original reference trajectory again with a time-optimal behavior whenever the velocity and acceleration profiles allow.

Since the trajectory limiter outputs position, velocity and acceleration, it is easy to use inverse-based feedforward models to improve the trajectory tracking compared to purely feedback-based controllers (*always* use some form of feedforward if trajectory-tracking performance is important).


To limit a trajectory online, i.e., one step at a time, call the limiter like so
```julia
state, ẍ = limiter(state, r(t))
```
this outputs a new state, containing $x, ẋ, r, ṙ$ as well as the acceleration $ẍ$.

One can also call the lower-level function
```julia
state, ẍ = TrajectoryLimiter.trajlim(state, rt, Ts, ẋM, ẍM)
```
directly in case one would like to change any of the parameters online.

To set the initial state of the trajectory limiter, create a
```julia
TrajectoryLimiters.State(x, ẋ, r, ṙ)
```
manually. The default choice if no initial state is given when batch filtering an array `R` is `TrajectoryLimiters.State(0, 0, r, 0)` where `r` is the first value in the array `R`.


### Performance
On a laptop from 2021, filtering a trajectory `R` of length 601 samples takes
```julia
julia> length(R)
601

julia> @btime $limiter($R);
  23.745 μs (3 allocations: 14.62 KiB)
```

With preallocated output arrays, you can avoid the allocations completely:
```julia
julia> X, Ẋ, Ẍ = similar.((R,R,R));

julia> @btime $limiter($X, $Ẋ, $Ẍ, $R);
  20.813 μs (0 allocations: 0 bytes)
```

Taking a single step takes
```julia
julia> @btime $limiter(TrajectoryLimiters.State(0.0), 0.0);
  17.372 ns (0 allocations: 0 bytes)
  ```

## Ruckig: Jerk-Limited Trajectory Generation

This package also includes a `JerkLimiter` for generating time-optimal jerk-limited trajectories. This is based on the Ruckig algorithm:

> Berscheid & Kröger, "Jerk-limited Real-time Trajectory Generation with Arbitrary Target States", pedestrians 2021

Unlike the `TrajectoryLimiter` which filters an existing reference signal, `JerkLimiter` plans point-to-point trajectories that are time-optimal while respecting constraints on:
- **Velocity**: $v_{\min} \leq v(t) \leq v_{\max}$
- **Acceleration**: $a_{\min} \leq a(t) \leq a_{\max}$
- **Jerk** (rate of change of acceleration): $|j(t)| \leq j_{\max}$

### Features

- **Time-optimal**: Generates the fastest possible trajectory within the given constraints
- **Jerk-limited**: Ensures smooth acceleration profiles (no discontinuities in acceleration)
- **Arbitrary initial state**: Supports starting from any position, velocity, and acceleration
- **Non-zero target velocity and acceleration**: Can plan trajectories that end at a specified velocity and acceleration (not just rest-to-rest)
- **Asymmetric limits**: Supports different limits for positive and negative directions

### Basic Usage

```julia
using TrajectoryLimiters

# Create a jerk limiter with symmetric limits
lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

# Plan a trajectory from rest at position 0 to rest at position 1
profile = calculate_trajectory(lim; pf=1.0)

Tf = duration(profile)  # Total trajectory duration

# Evaluate the trajectory at any time
t = 0.05 # May also be a vector of time points
p, v, a, j = evaluate_at(profile, t)

# Evaluate entire trajectory at fixed time interval 0.001
P, V, A, J, ts = evaluate_dt(profile, 0.001)
```

### Example: Comparing Different Constraint Levels

The following example demonstrates how different jerk limits affect the trajectory smoothness:

```julia
using TrajectoryLimiters
using Plots

# Common velocity and acceleration limits
vmax, amax = 10.0, 50.0

# Three different jerk limits: high, medium, low
jmax_high = 5000.0   # Sharp acceleration changes
jmax_med  = 1000.0   # Moderate smoothness
jmax_low  = 200.0    # Very smooth acceleration

lim_high = JerkLimiter(; vmax, amax, jmax=jmax_high)
lim_med  = JerkLimiter(; vmax, amax, jmax=jmax_med)
lim_low  = JerkLimiter(; vmax, amax, jmax=jmax_low)

# Plan trajectories to position 2 (defaults: p0=0, v0=0, a0=0, vf=0)
prof_high = calculate_trajectory(lim_high; pf=2.0)
prof_med  = calculate_trajectory(lim_med; pf=2.0)
prof_low  = calculate_trajectory(lim_low; pf=2.0)

# Sample trajectories for plotting
Ts = 0.001
p1, v1, a1, j1, t1 = evaluate_dt(prof_high, Ts)
p2, v2, a2, j2, t2 = evaluate_dt(prof_med, Ts)
p3, v3, a3, j3, t3 = evaluate_dt(prof_low, Ts)

plot(
    plot(t1, p1, label="jmax=5000", ylabel="Position"),
    plot(t1, v1, label="jmax=5000", ylabel="Velocity"),
    plot(t1, a1, label="jmax=5000", ylabel="Acceleration"),
    plot(t1, j1, label="jmax=5000", ylabel="Jerk"),
    layout=(4,1), size=(600,600), legend=:right
)
plot!(t2, p2, label="jmax=1000", sp=1)
plot!(t2, v2, label="jmax=1000", sp=2)
plot!(t2, a2, label="jmax=1000", sp=3)
plot!(t2, j2, label="jmax=1000", sp=4)
plot!(t3, p3, label="jmax=200", sp=1)
plot!(t3, v3, label="jmax=200", sp=2)
plot!(t3, a3, label="jmax=200", sp=3)
plot!(t3, j3, label="jmax=200", sp=4)
```

![jerk comparison](https://github.com/user-attachments/assets/392a2c6e-9f39-46a9-a361-b775ce70709b)

Lower jerk limits produce smoother acceleration profiles at the cost of longer trajectory duration. The jerk (bottom plot) shows how the rate of acceleration change is bounded.

### Example: Trajectory with Initial Velocity

The algorithm handles arbitrary initial states, including non-zero velocity and acceleration:

```julia
using TrajectoryLimiters
using Plots

lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

# Start with initial velocity v0=5, plan to rest at position 3
profile = calculate_trajectory(lim; v0=5.0, pf=3.0)

# Sample and plot
pos, vel, acc, jerk, ts = evaluate_dt(profile, 0.001)

plot(
    plot(ts, pos, ylabel="Position", label=""),
    plot(ts, vel, ylabel="Velocity", label=""),
    plot(ts, acc, ylabel="Acceleration", label=""),
    layout=(3,1), xlabel="Time [s]", size=(600,450)
)
```

![initial velocity](https://github.com/user-attachments/assets/67ab9661-6ca0-4d5f-82dc-6d3b9cb9bbea)

### Velocity Control

For applications where you need to reach a target velocity (rather than a target position), use `calculate_velocity_trajectory`. This is useful for, e.g.,:
- Conveyor belt speed control
- Spindle speed changes

```julia
using TrajectoryLimiters

# Create a jerk limiter
lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

# Accelerate from rest to velocity 5.0
profile = calculate_velocity_trajectory(lim; vf=5.0)

# Evaluate the trajectory
p, v, a, j = profile(duration(profile))
# v ≈ 5.0 (target velocity reached)

# With initial velocity and acceleration
profile2 = calculate_velocity_trajectory(lim; v0=2.0, a0=10.0, vf=8.0, af=0.0)

# Time-synchronized: reach target velocity in exactly 0.5 seconds
profile3 = calculate_velocity_trajectory(lim; vf=5.0, tf=0.5)
```

The `AccelerationLimiter` (second-order, no jerk limit) and `VelocityLimiter` (first-order, velocity limit only) also support velocity control:

```julia
# AccelerationLimiter: acceleration-limited trajectories (no jerk constraint)
lim2 = AccelerationLimiter(; vmax=100.0, amax=50.0)
profile = calculate_velocity_trajectory(lim2; vf=10.0)

# VelocityLimiter: velocity-limited trajectories (constant velocity, instantaneous changes)
lim3 = VelocityLimiter(; vmax=10.0)
profile = calculate_trajectory(lim3; pf=100.0)  # Travels at vmax=10 for 10 seconds
```

The `VelocityLimiter` is useful for simple motion planning where acceleration constraints are not important, or as a baseline for comparison.

### Asymmetric Limits

For applications where positive and negative motion have different constraints (e.g., gravity-affected systems), specify the directional limits:

```julia
# Different limits for positive/negative directions
lim = JerkLimiter(;
    vmax=10.0, vmin=-5.0,    # Can move faster in positive direction
    amax=50.0, amin=-30.0,   # Can accelerate faster than decelerate
    jmax=1000.0
)
```

### Multi-DOF Synchronized Trajectories

For systems with multiple degrees of freedom (e.g., robotic arms, CNC machines), trajectories can be synchronized so that all DOFs reach their targets simultaneously. Each DOF can have different constraints, and the algorithm finds the time-optimal synchronized duration.

The multi-DOF algorithm is invoked by passing an array of `JerkLimiter` instances to `calculate_trajectory`, one for each DOF:

```julia
using TrajectoryLimiters

# Create limiters with different constraints for each DOF
lims = [
    JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),  # DOF 1
    JerkLimiter(; vmax=5.0,  amax=40.0, jmax=500.0),   # DOF 2
    JerkLimiter(; vmax=8.0,  amax=30.0, jmax=800.0),   # DOF 3
]

# Plan synchronized trajectories to target positions
profiles = calculate_trajectory(lims;
    pf = [1.0, 2.0, 0.5],  # Target positions for each DOF
)

# Evaluate all DOFs at a specific time
t = 0.1
ps, vs, as, js = evaluate_at(profiles, t)  # Returns vectors of length 3

# Evaluate with initial states
profiles = calculate_trajectory(lims;
    p0 = [0.0, 1.0, 0.0],   # Initial positions
    v0 = [1.0, 0.0, 0.5],   # Initial velocities
    pf = [5.0, 4.0, 2.0],   # Target positions
)

# Sample and plot
pos, vel, acc, jerk, ts = evaluate_dt(profiles, 0.001)

plot(
    plot(ts, pos, ylabel="Position", label=""),
    plot(ts, vel, ylabel="Velocity", label=""),
    plot(ts, acc, ylabel="Acceleration", label=""),
    plot(ts, jerk, ylabel="Jerk", label=""),
    layout=(4,1), xlabel="Time [s]", size=(600,700)
)
hline!([10 5 8], sp=2, ls=:dash, label="", c=(1:3)')
hline!([50 40 30], sp=3, ls=:dash, label="", c=(1:3)')
hline!([1000 500 800], sp=4, ls=:dash, label="", c=(1:3)')
```

![multi DOF synch](https://github.com/user-attachments/assets/4c2bf06e-d985-4aac-8286-bc517f72ef51)

The image shows how different parts of the trajectory is limited by different degrees of freedom, all three DOF start and reach the end at the same time.

### Multi-DOF Waypoint Trajectories

For multi-DOF systems that need to pass through multiple waypoints, use `calculate_waypoint_trajectory` with an array of limiters. Each waypoint specifies arrays for position (and optionally velocity/acceleration) for all DOFs:

```julia
using TrajectoryLimiters
using Plots

# Two DOFs with different constraints
lims = [
    JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),  # DOF 1 (e.g., X axis)
    JerkLimiter(; vmax=8.0, amax=40.0, jmax=800.0),    # DOF 2 (e.g., Y axis)
]

# Define 4 waypoints - each with position arrays for both DOFs
waypoints = [
    (p = [0.0, 0.0],),                      # Start at origin
    (p = [2.0, 1.0],),                      # First waypoint
    (p = [3.0, 3.0],),                      # Second waypoint
    (p = [-0.5, 2.0],), 
    (p = [0.0, 0.0],),                      # Final position
]

# Calculate synchronized trajectory through all waypoints
ts, pos, vel, acc, jerk = calculate_waypoint_trajectory(lims, waypoints, 0.001)

# Plot the trajectory in XY space
scatter([wp.p[1] for wp in waypoints], [wp.p[2] for wp in waypoints],
         label="Waypoints", ms=8, c=:red)
plot!(pos[:, 1], pos[:, 2], label="Path", xlabel="X", ylabel="Y", aspect_ratio=:equal, c=1)
```

Each segment between waypoints is time-synchronized across all DOFs, ensuring coordinated motion through the entire path. Note, the trajectory is in general not globally time optimal since the velocity and acceleration targets at the waypoints are not optimized.

#### Optimizing for global time-optimality
We could try to optimize the waypoint velocities in order to minimize the total trajectory time (not a supported feature yet, this is only an example):
```julia
using Optim

# Fixed waypoint positions (first and last have fixed v=0, a=0)
positions = [
    [0.0, 0.0],   # Start
    [2.0, 1.0],   # Intermediate 1
    [3.0, 3.0],   # Intermediate 2
    [-0.5, 2.0],  # Intermediate 3
    [0.0, 0.0],   # End
]

ndof = length(positions[1])
n_intermediate = length(positions) - 2  # Exclude first and last

# Build waypoints from optimization parameters
# params layout: [v1..., v2..., ..., a1..., a2..., ...] for each intermediate waypoint
function get_waypoints(params)
    wps = [(p = positions[1], v=zeros(ndof), a=zeros(ndof))]  # First waypoint (fixed)
    np = length(positions)
    for i in 1:n_intermediate
        push!(wps, (
            p = positions[i + 1],
            v = params[(i - 1) * ndof + 1 : i * ndof],
            a = params[(n_intermediate * ndof) + (i - 1) * ndof + 1 : (n_intermediate * ndof) + i * ndof]
        ))
    end
    push!(wps, (p = positions[end], v=zeros(ndof), a=zeros(ndof)))  # Last waypoint (fixed)
    return wps
end

function cost(params)
    try
        ts, _ = calculate_waypoint_trajectory(lims, get_waypoints(params), 0.001)
        return last(ts)  # Total trajectory time
    catch
        return Inf  # Infeasible trajectory
    end
end

n_params = n_intermediate * ndof * 2  # velocities + accelerations
initial_params = zeros(n_params)

res = Optim.optimize(
    cost,
    initial_params,
    ParticleSwarm(),
    Optim.Options(
        store_trace       = true,
        show_trace        = true,
        show_every        = 1,
        iterations        = 1000,
        allow_f_increases = false,
        time_limit        = 100,
        x_abstol          = 0,
        f_abstol          = 0,
        g_tol             = 1e-8,
        f_calls_limit     = 0,
        g_calls_limit     = 0,
    ),
)

opt = Optim.minimizer(res)
ts_opt, pos_opt, vel_opt, acc_opt, jerk_opt = calculate_waypoint_trajectory(lims, get_waypoints(opt), 0.001)
plot!(pos_opt[:, 1], pos_opt[:, 2], label="Optimized Path")
```
![optimized trajectory](https://github.com/user-attachments/assets/02161a30-0a75-4471-8212-e8d9488192d0)

### Profile Structure

The algorithm generates a 7-phase trajectory where each phase has constant jerk. The phases are:
1. Jerk to reach maximum acceleration
2. Coast at maximum acceleration
3. Jerk to reduce acceleration
4. Coast at maximum velocity (if reached)
5. Jerk to reach minimum acceleration
6. Coast at minimum acceleration
7. Jerk to reach target acceleration (zero)

Not all phases are present in every trajectory—shorter moves may skip the coasting phases.