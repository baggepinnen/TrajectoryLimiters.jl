# TrajectoryLimiters JuliaC Example

This example demonstrates how to compile TrajectoryLimiters.jl into a C-callable shared library using [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl).

## Prerequisites

- **Julia 1.12+** (required for `--trim` support)
- **JuliaC.jl** installed as an app:
  ```bash
  julia -e 'using Pkg; Pkg.add("JuliaC")'
  ```
- **C compiler** (gcc or clang)

## Project Structure

```
TrajectoryLimitersC/
├── Project.toml                 # Package definition
├── src/
│   └── TrajectoryLimitersC.jl   # Module with @ccallable functions
├── example/
│   └── caller.c                 # C program demonstrating usage
└── README.md                    # This file
```

## Exposed C API

### TrajectoryLimiter (Online Filtering)

```c
// Create a trajectory limiter
// Ts: sample time, xdotM: max velocity, xddotM: max acceleration
void* trajectory_limiter_create(double Ts, double xdotM, double xddotM);

// Destroy a trajectory limiter
void trajectory_limiter_destroy(void* handle);

// Take one filtering step
// state: array of 4 doubles [x, xdot, r, rdot] - modified in place
// r: new reference position
// Returns: acceleration command
double trajectory_limiter_step(void* handle, double* state, double r);
```

### JerkLimiter (Time-Optimal Trajectory Generation)

```c
// Create a jerk limiter with symmetric limits
void* jerk_limiter_create(double vmax, double amax, double jmax);

// Destroy a jerk limiter
void jerk_limiter_destroy(void* handle);

// Create a trajectory profile
// p0, v0, a0: initial position, velocity, acceleration
// pf, vf, af: target position, velocity, acceleration
void* profile_create(void* limiter, double p0, double v0, double a0,
                     double pf, double vf, double af);

// Destroy a profile
void profile_destroy(void* handle);

// Get trajectory duration in seconds
double profile_duration(void* handle);

// Evaluate profile at time t
// result: array of 4 doubles to store [position, velocity, acceleration, jerk]
void profile_evaluate(void* handle, double t, double* result);
```

## Building the Shared Library

From the `TrajectoryLimitersC/` directory:

```bash
# Install dependencies first
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.develop(path="..")'

# Build with JuliaC
juliac --output-lib trajlim --trim=safe --compile-ccallable .
```

This creates:
- `build/lib/libtrajlim.so` (Linux) or `build/lib/libtrajlim.dylib` (macOS)
- Bundled Julia runtime in `build/`

## Building the C Example

After building the shared library:

### Linux

```bash
gcc -o example/caller example/caller.c \
    -L./build/lib -ltrajlim \
    -Wl,-rpath,'$ORIGIN/../build/lib'
```

### macOS

```bash
clang -o example/caller example/caller.c \
    -L./build/lib -ltrajlim \
    -Wl,-rpath,@executable_path/../build/lib
```

## Running the Example

```bash
./example/caller
```

Expected output:
```
========================================
TrajectoryLimiters C API Example
========================================

=== TrajectoryLimiter Example ===
Filtering a step reference from 0 to 5

Step  Reference    Position    Velocity    Acceleration
----  ---------    --------    --------    ------------
   0       5.00      0.0025      0.5000         50.0000
   1       5.00      0.0100      1.0000         50.0000
   ...

=== JerkLimiter Example ===
Computing trajectory from (0,0,0) to (2,0,0)

Profile duration: 0.750000 seconds

Time        Position    Velocity    Acceleration    Jerk
----        --------    --------    ------------    ----
0.000000      0.0000      0.0000          0.0000   1000.00
...

Done!
```

## Design Notes

### Memory Management

Julia objects (TrajectoryLimiter, JerkLimiter, profiles) are stored in a global dictionary to prevent garbage collection. The C API uses opaque pointers (`void*`) to reference these objects.

**Important:** Always call the corresponding `*_destroy()` function when done with an object to free memory.

### State Passing

The `trajectory_limiter_step` function uses an in-out parameter for state:
- Input: current state `[x, xdot, r, rdot]`
- Output: updated state (modified in place)
- Return value: acceleration command

### Type Safety

All numeric parameters use `double` (64-bit float) to match Julia's default `Float64`.

## Troubleshooting

### Library not found at runtime

Set the library path environment variable:

```bash
# Linux
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_FALLBACK_LIBRARY_PATH=$PWD/build/lib:$DYLD_FALLBACK_LIBRARY_PATH
```

### JuliaC not found

Make sure JuliaC is installed as an app:
```bash
julia -e 'using Pkg; Pkg.add("JuliaC")'
```

The `juliac` command should be available in your PATH after installation.
