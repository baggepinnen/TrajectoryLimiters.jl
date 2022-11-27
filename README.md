# TrajectoryLimiters

[![Build Status](https://github.com/baggepinnen/TrajectoryLimiters.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/TrajectoryLimiters.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/baggepinnen/TrajectoryLimiters.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/baggepinnen/TrajectoryLimiters.jl)

Contains an implementation of 
> Nonlinear filters for the generation of smooth trajectories
> R. Zanasi, C. Guarino Lo Bianco, A. Tonielli

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