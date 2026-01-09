# TrajectoryLimiters

[![Build Status](https://github.com/baggepinnen/TrajectoryLimiters.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/TrajectoryLimiters.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/baggepinnen/TrajectoryLimiters.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/baggepinnen/TrajectoryLimiters.jl)

Contains implementations of 
> Nonlinear filters for the generation of smooth trajectories
> R. Zanasi, C. Guarino Lo Bianco, A. Tonielli

and the _ruckig_ algorithm from

> Jerk-limited Real-time Trajectory Generation with Arbitrary Target States
> M. Berscheid, T. Kröger

This nonlinear trajectory filter takes a pre-defined reference trajectory $r(t)$ (uniformly sampled in $t$) and filters it (causally) such that the velocity and acceleration are bounded by $ẋ_M$ and $ẍ_M$.

What is this good for? Some applications call for a dynamically feasible reference trajectory, i.e., a trajectory with bounded velocity and acceleration, but all you have access to is an instantaneous reference $r(t)$ that might change abruptly, e.g., from an operator changing a set point. In such situations, this filter performs the required pre-processing of the reference to provide a smoother, dynamically feasible reference trajectory. If you already have a trajectory planner that outputs dynamically feasible trajectories, you do not need this package. 

## Usage

To filter an entire trajectory, create a `TrajectoryLimiter` and call it like a function:
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
- **Non-zero target velocity**: Can plan trajectories that end at a specified velocity (not just rest-to-rest)
- **Asymmetric limits**: Supports different limits for positive and negative directions

### Basic Usage

```julia
using TrajectoryLimiters

# Create a jerk limiter with symmetric limits
lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

# Plan a trajectory from rest at position 0 to rest at position 1
profile = calculate_trajectory(lim; pf=1.0)

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

![jerk comparison](https://private-user-images.githubusercontent.com/3797491/533803613-392a2c6e-9f39-46a9-a361-b775ce70709b.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc5NDkyNjMsIm5iZiI6MTc2Nzk0ODk2MywicGF0aCI6Ii8zNzk3NDkxLzUzMzgwMzYxMy0zOTJhMmM2ZS05ZjM5LTQ2YTktYTM2MS1iNzc1Y2U3MDcwOWIucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDEwOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjAxMDlUMDg1NjAzWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YTg1ODg2MmM3M2VmNDhjMDU5MWFlNTI4YjQ0NjkzZWQyMzMxMDAyOTc0NjY5NGY0MDIxNDk1YmQ5NjllYTliMCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.w0P7_MfhlPrd6tog8eowHvZ5H31mzSbL0tFbY6Wf9_A)

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
hline!([0], sp=2, ls=:dash, c=:gray, label="")
hline!([0], sp=3, ls=:dash, c=:gray, label="")
```

![initial velocity](https://private-user-images.githubusercontent.com/3797491/533803584-67ab9661-6ca0-4d5f-82dc-6d3b9cb9bbea.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc5NDkyNjMsIm5iZiI6MTc2Nzk0ODk2MywicGF0aCI6Ii8zNzk3NDkxLzUzMzgwMzU4NC02N2FiOTY2MS02Y2EwLTRkNWYtODJkYy02ZDNiOWNiOWJiZWEucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDEwOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjAxMDlUMDg1NjAzWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NWI2NzkxODNkMWQ1OTNjYmZlNzk0MDM1ZTY2OTEyZDQzOTdmMmQ5NDcxNTBmZGMxMTQ1YWNlN2JlMzk0ODM4OSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.W3uS4WPk5_24CBGcerWHkRdTM6LKKbyGlTTxwrja9dg)

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