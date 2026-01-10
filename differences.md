# Algorithmic Differences: C++ Ruckig vs Julia Implementation

This document systematically compares the C++ reference implementation at `/home/fredrikb/repos/ruckig/` with the Julia implementation in `src/ruckig.jl`.

## Function Inventory Table

### Control Interfaces

| Feature | C++ | Julia | Notes |
|---------|-----|-------|-------|
| Position Control (3rd order) | Yes | Yes | Main implementation |
| Position Control (2nd order) | Yes | No | Missing in Julia |
| Position Control (1st order) | Yes | No | Missing in Julia |
| Velocity Control (3rd order) | Yes | No | Missing in Julia |
| Velocity Control (2nd order) | Yes | No | Missing in Julia |

### Step 1 Functions (Minimum-Time Profile Generation)

| C++ Function | Julia Function | Status |
|--------------|----------------|--------|
| `PositionThirdOrderStep1::time_all_vel()` | `time_all_vel!()` | Present |
| `PositionThirdOrderStep1::time_acc0_acc1()` | `time_acc0_acc1!()` | Present |
| `PositionThirdOrderStep1::time_all_none_acc0_acc1()` | `time_all_none_acc0_acc1!()` | Present |
| `PositionThirdOrderStep1::time_none_two_step()` | `time_none_two_step!()` | Present |
| `PositionThirdOrderStep1::time_acc0_two_step()` | `time_acc0_two_step!()` | Present |
| `PositionThirdOrderStep1::time_vel_two_step()` | `time_vel_two_step!()` | Present |
| `PositionThirdOrderStep1::time_acc1_vel_two_step()` | `time_acc1_vel_two_step!()` | Present |
| `PositionThirdOrderStep1::time_all_single_step()` | N/A | **Missing** |
| `PositionThirdOrderStep1::get_profile()` | `calculate_trajectory()` | Different structure |

### Step 2 Functions (Time Synchronization)

| C++ Function | Julia Function | Status |
|--------------|----------------|--------|
| `PositionThirdOrderStep2::time_acc0_acc1_vel()` | `time_acc0_acc1_vel_step2!()` | Present |
| `PositionThirdOrderStep2::time_acc1_vel()` | `time_acc1_vel_step2!()` | Present |
| `PositionThirdOrderStep2::time_acc0_vel()` | `time_acc0_vel_step2!()` | Present |
| `PositionThirdOrderStep2::time_vel()` | `time_vel_step2!()` | Present |
| `PositionThirdOrderStep2::time_acc0_acc1()` | `time_acc0_acc1_step2!()` | Present |
| `PositionThirdOrderStep2::time_acc0()` | `time_acc0_step2!()` | Present |
| `PositionThirdOrderStep2::time_acc1()` | `time_acc1_step2!()` | Present |
| `PositionThirdOrderStep2::time_none()` | `time_none_step2!()` | Present |
| `PositionThirdOrderStep2::get_profile()` | `calculate_profile_step2!()` | Present |

### Root Finding Functions

| C++ Function | Julia Function | Status |
|--------------|----------------|--------|
| `roots::solve_cubic()` | `solve_cubic_real!()` | Present |
| `roots::solve_quart_monic()` | `solve_quartic_real!()` | Present |
| `roots::solve_resolvent()` | `solve_cubic_all_real()` | Present (different name) |
| `roots::shrink_interval<N>()` | `shrink_interval_poly5()`, `shrink_interval_poly6()` | Present (specialized) |
| `roots::poly_eval<N>()` | Inline evaluation | Present (inlined) |
| `roots::poly_derivative<N>()` | Inline computation | Present (inlined) |

### Profile Validation

| C++ Function | Julia Function | Status |
|--------------|----------------|--------|
| `Profile::check<ControlSigns, ReachedLimits>()` | `check!()` | Present |
| `Profile::check_with_timing<>()` | `check_step2!()` | Present |
| `Profile::get_position_extrema()` | N/A | **Missing** |
| `Profile::get_first_state_at_position()` | N/A | **Missing** |
| `Profile::set_boundary()` | N/A | **Missing** (implicit in constructor) |

### Brake Profiles

| C++ Function | Julia Function | Status |
|--------------|----------------|--------|
| `BrakeProfile::acceleration_brake()` | N/A | **Missing** |
| `BrakeProfile::velocity_brake()` | N/A | **Missing** |
| `BrakeProfile::get_position_brake_trajectory()` | N/A | **Missing** |
| `BrakeProfile::get_velocity_brake_trajectory()` | N/A | **Missing** |
| `BrakeProfile::finalize()` | N/A | **Missing** |

### Block Interval Calculation

| C++ Function | Julia Function | Status |
|--------------|----------------|--------|
| `Block::calculate_block<N>()` | N/A | **Missing** - Julia uses simplified Block |
| `Block::is_blocked()` | `is_blocked()` | Present |
| `Block::get_profile()` | `get_profile()` | Present |

---

## Detailed Algorithmic Differences

### 1. Missing Features in Julia

#### 1.1 Brake Profiles (brake.cpp)
The C++ implementation includes brake trajectories to handle initial states outside kinematic limits. These are pre-pended to the main trajectory to bring the system into a valid state before computing the optimal profile.

**C++ Implementation** (`brake.cpp`, `brake.hpp`):
```cpp
// Called when a0 > aMax (acceleration too high)
void BrakeProfile::acceleration_brake(double v0, double a0, double vMax, double vMin,
                                       double aMax, double aMin, double jMax) {
    j[0] = -jMax;  // Apply negative jerk to reduce acceleration
    const double t_to_a_max = (a0 - aMax) / jMax;
    const double t_to_a_zero = a0 / jMax;
    // Compute braking time, check velocity constraints during braking
    t[0] = t_to_a_max + eps;
    // May need second phase if velocity limit would be violated
}

// Called when v0 > vMax or v0 < vMin (velocity out of bounds)
void BrakeProfile::velocity_brake(double v0, double a0, double vMax, double vMin,
                                   double aMax, double aMin, double jMax) {
    // Apply jerk to reduce velocity while respecting acceleration limits
}

// Main entry point - decides which brake type is needed
void BrakeProfile::get_position_brake_trajectory(...) {
    if (a0 > aMax) {
        acceleration_brake(v0, a0, vMax, vMin, aMax, aMin, jMax);
    } else if (a0 < aMin) {
        acceleration_brake(v0, a0, vMin, vMax, aMin, aMax, -jMax);
    } else if ((v0 > vMax && ...) || (a0 > 0 && ...)) {
        velocity_brake(v0, a0, vMax, vMin, aMax, aMin, jMax);
    } else if ((v0 < vMin && ...) || (a0 < 0 && ...)) {
        velocity_brake(v0, a0, vMin, vMax, aMin, aMax, -jMax);
    }
}
```

The brake profile duration is added to the main trajectory duration, and the state after braking becomes the effective initial state for the main profile calculation.

**Julia Implementation**: No equivalent. The `JerkLimiter` assumes initial states are within bounds.

**Impact**:
- If `a0 > amax` or `a0 < amin`: Julia will produce an invalid trajectory or fail
- If `v0 > vmax` or `v0 < vmin`: Julia will produce an invalid trajectory or fail
- This is particularly problematic for real-time applications where the current state may temporarily exceed limits due to disturbances

#### 1.2 Zero-Limits Special Case (time_all_single_step)
The C++ implementation handles degenerate cases where one or more kinematic limits are zero, reducing to simpler motion types.

**C++ Implementation** (`position_third_step1.cpp:467-508`):
```cpp
bool PositionThirdOrderStep1::time_all_single_step(Profile* profile,
    double vMax, double vMin, double aMax, double aMin, double jMax) const {

    if (std::abs(af - a0) > DBL_EPSILON) {
        return false;  // Cannot handle if af != a0
    }

    // All phase times are zero - pure constant-acceleration or constant-velocity
    profile->t[0] = profile->t[1] = ... = profile->t[6] = 0;

    if (std::abs(a0) > DBL_EPSILON) {
        // Constant acceleration: solve p = p0 + v0*t + 0.5*a0*t²
        const double q = std::sqrt(2*a0*pd + v0_v0);
        profile->t[3] = (-v0 + q) / a0;  // or -(v0 + q) / a0
    } else if (std::abs(v0) > DBL_EPSILON) {
        // Constant velocity: t = pd / v0
        profile->t[3] = pd / v0;
    } else if (std::abs(pd) < DBL_EPSILON) {
        // Already at target
        return true;
    }
}
```

This is called at the start of `get_profile()` when `_jMax == 0.0 || _aMax == 0.0 || _aMin == 0.0`.

**Julia Implementation**: No equivalent. The quartic/cubic solvers will fail or produce NaN when limits are zero.

**Impact**:
- `jmax = 0`: Division by zero in polynomial coefficients
- `amax = 0` or `amin = 0`: Division by zero in `time_all_vel!` and other functions
- Cannot handle "infinite jerk" (bang-bang acceleration) profiles
- Cannot handle pure constant-velocity motion

#### 1.3 Block Interval Calculation
When computing multi-DOF synchronized trajectories, there may be multiple valid profiles for a single DOF with different durations. The C++ implementation tracks "blocked intervals" where certain durations are invalid due to competing profile transitions.

**C++ Implementation** (`block.hpp:60-132`):
```cpp
template<size_t N, bool numerical_robust = true>
static bool calculate_block(Block& block, std::array<Profile, N>& valid_profiles,
                            size_t valid_profile_counter) {
    if (valid_profile_counter == 1) {
        block.set_min_profile(valid_profiles[0]);
        return true;
    }
    else if (valid_profile_counter == 2) {
        // Two profiles: one blocked interval between them
        const size_t idx_min = (valid_profiles[0].t_sum.back() < valid_profiles[1].t_sum.back()) ? 0 : 1;
        block.set_min_profile(valid_profiles[idx_min]);
        block.a = Interval(valid_profiles[idx_min], valid_profiles[idx_else]);
    }
    else if (valid_profile_counter == 3) {
        // Three profiles: one blocked interval from the two non-minimum profiles
        block.a = Interval(valid_profiles[idx_else_1], valid_profiles[idx_else_2]);
    }
    else if (valid_profile_counter == 5) {
        // Five profiles: two blocked intervals
        block.a = Interval(...);
        block.b = Interval(...);
    }
    // Also handles 4 profiles (numerical edge case - removes near-duplicate)
}
```

The synchronization algorithm then finds a time that is NOT blocked for any DOF:
```cpp
// In calculator_target.hpp
for (double t : candidate_times) {
    bool is_valid = true;
    for (size_t dof = 0; dof < DOFs; ++dof) {
        if (blocks[dof].is_blocked(t)) {
            is_valid = false;
            break;
        }
    }
    if (is_valid) {
        sync_time = t;
        break;
    }
}
```

**Julia Implementation** (`ruckig.jl:173-194`):
```julia
struct Block{T}
    p_min::RuckigProfile{T}       # Only stores ONE profile
    t_min::T
    a::Union{Nothing, BlockInterval{T}}  # Always Nothing in practice
    b::Union{Nothing, BlockInterval{T}}  # Always Nothing in practice
end

function Block(p_min::RuckigProfile{T}) where T
    Block{T}(p_min, duration(p_min), nothing, nothing)  # No intervals computed
end
```

Julia only stores the minimum-time profile and throws an error if the sync time happens to be blocked:
```julia
if is_blocked(blocks[i], t_sync)
    error("Synchronization time $t_sync is blocked for DOF $i")
end
```

**Impact**:
- Multi-DOF trajectories may fail for valid inputs when profiles have multiple solutions
- The Julia implementation doesn't collect all valid profiles from Step 1, so it can't build intervals
- Edge cases where the minimum-time profile of one DOF falls within a blocked interval of another DOF will error instead of finding an alternative valid time

#### 1.4 Velocity Control Interface
The C++ implementation supports targeting a final velocity (ignoring position) as an alternative control mode.

**C++ Implementation** (`velocity_third_step1.cpp`, `velocity_third_step2.cpp`):
```cpp
// Step 1: Find minimum-time profile to reach target velocity
class VelocityThirdOrderStep1 {
    void time_acc0(ProfileIter& profile, double aMax, double aMin, double jMax);
    void time_none(ProfileIter& profile, double aMax, double aMin, double jMax);
    bool get_profile(const Profile& input, Block& block);
};

// Step 2: Time synchronization for velocity profiles
class VelocityThirdOrderStep2 {
    bool time_acc0(Profile& profile, double aMax, double aMin, double jMax);
    bool time_none(Profile& profile, double aMax, double aMin, double jMax);
    bool get_profile(Profile& profile);
};
```

Use case: Controlling a conveyor belt or spindle where you want to reach a target velocity as fast as possible, and the final position doesn't matter.

**Julia Implementation**: Not present.

**Impact**:
- Cannot use the library for velocity-only control applications
- Would need to implement ~260 lines of additional Step1/Step2 functions
- Multi-DOF velocity synchronization also not available

#### 1.5 Second-Order (Acceleration-Limited) Profiles
The C++ implementation supports profiles where jerk is unlimited (instantaneous acceleration changes), using only velocity and acceleration limits.

**C++ Implementation** (`position_second_step1.cpp`, `position_second_step2.cpp`):
```cpp
// 5-phase trapezoidal velocity profile (no jerk limits)
// Phases: accelerate, coast at vMax, decelerate (with acc/dec ramps)
class PositionSecondOrderStep1 {
    void time_acc0(ProfileIter& profile, double vMax, double vMin, double aMax, double aMin);
    void time_none(ProfileIter& profile, double vMax, double vMin, double aMax, double aMin);
    bool time_all_single_step(Profile* profile, double vMax, double vMin, double aMax, double aMin);
    bool get_profile(const Profile& input, Block& block);
};

class PositionSecondOrderStep2 {
    // Time-synchronized versions
    bool time_acc0(Profile& profile, double vMax, double vMin, double aMax, double aMin);
    bool time_none(Profile& profile, double vMax, double vMin, double aMax, double aMin);
    bool get_profile(Profile& profile);
};
```

The profile structure is simpler (effectively 3-5 non-zero phases instead of 7) with discontinuous acceleration at phase boundaries.

**Julia Implementation**: Not present.

**Impact**:
- Cannot use simplified "bang-bang" acceleration profiles
- For systems where jerk doesn't matter (e.g., some pneumatic actuators), this is unnecessarily complex
- Cannot match behavior of systems configured for 2nd-order control

#### 1.6 First-Order (Velocity-Limited) Profiles
The C++ implementation supports profiles where both jerk and acceleration are unlimited, using only velocity limits.

**C++ Implementation** (`position_first_step1.cpp`, `position_first_step2.cpp`):
```cpp
// Simple constant-velocity profile (instantaneous velocity changes)
// Only one phase at vMax (or vMin for negative displacement)
class PositionFirstOrderStep1 {
    bool get_profile(const Profile& input, Block& block) {
        // t[3] = pd / vMax (coast phase only)
        // All other phases are zero
    }
};

class PositionFirstOrderStep2 {
    bool get_profile(Profile& profile) {
        // Adjust coast time to match target duration tf
    }
};
```

This is the simplest profile type with only a single constant-velocity phase.

**Julia Implementation**: Not present.

**Impact**:
- Cannot generate pure constant-velocity trajectories
- For some simple motion systems, this would be the appropriate mode
- Cannot match behavior of systems configured for 1st-order control (e.g., some simple servo modes)

---

### 2. Control Sign Patterns (UDDU vs UDUD)

#### C++ Implementation
Both control sign patterns are tried systematically:
- **UDDU** (↑↓↓↑): Standard pattern for most motions
- **UDUD** (↑↓↑↓): Alternative pattern, especially for certain Step2 profiles

In Step2 functions, C++ explicitly tries both:
```cpp
// position_third_step2.cpp:41-75
bool time_acc0_acc1_vel(...) {
    // Profile UDDU, Solution 1
    if (...) {
        if (profile.check_with_timing<ControlSigns::UDDU, ...>(...)) return true;
    }
    // Profile UDUD
    if (...) {
        if (profile.check_with_timing<ControlSigns::UDUD, ...>(...)) return true;
    }
}
```

#### Julia Implementation
Julia implements both UDDU and UDUD patterns in Step2 functions, matching C++ behavior:
```julia
# ruckig.jl:1810-1833
if check_step2!(buf, UDDU, LIMIT_ACC0_ACC1_VEL, ...)
    return true
end
# Profile UDUD
if check_step2!(buf, UDUD, LIMIT_ACC0_ACC1_VEL, ...)
    return true
end
```

**Status**: Equivalent

---

### 3. Profile Collection vs First-Found

#### C++ Implementation (`position_third_step1.cpp:531-585`)
```cpp
if (std::abs(vf) < DBL_EPSILON && std::abs(af) < DBL_EPSILON) {
    // Fast path: return first valid profile
    time_all_vel(profile, ..., true);  // return_after_found=true
    if (profile > start) { goto return_block; }
    ...
} else {
    // Collect all valid profiles
    time_all_none_acc0_acc1(profile, ..., false);  // return_after_found=false
    ...
}
```

#### Julia Implementation (`ruckig.jl:1381-1462`)
```julia
if abs(vf) < EPS && abs(af) < EPS
    # Fast path: return first valid profile found
    if time_all_vel!(buf, ...)
        return RuckigProfile(buf, pf, vf, af)
    end
    ...
else
    # Full collection mode: try all profiles and return minimum
    best_duration = T(Inf)
    ...
end
```

**Status**: Equivalent logic - both use fast path when vf=af=0, otherwise collect all valid profiles.

---

### 4. Newton Refinement Steps

#### C++ Implementation
Uses single or double Newton steps with explicit tolerance checking:
```cpp
// position_third_step1.cpp:206-211
if (t > DBL_EPSILON) {
    const double orig = ...;
    const double deriv = ...;
    t -= orig / deriv;
}
```

Some functions use triple Newton refinement:
```cpp
// position_third_step1.cpp:267-288 (time_all_none_acc0_acc1 for ACC1)
// Double Newton step (regarding pd)
t -= std::min(orig / deriv, t);
if (std::abs(orig) > 1e-9) {
    t -= orig / deriv;
    if (std::abs(orig) > 1e-9) {
        t -= orig / deriv;
    }
}
```

#### Julia Implementation
Similar Newton refinement but sometimes simplified:
```julia
# ruckig.jl, line ~1869-1881 (time_acc1_vel_step2!)
if abs(a0 + jMax*t) > 16*EPS
    # Newton refinement
    orig = ...
    deriv = ...
    abs(deriv) > EPS && (t -= orig / deriv)
end
```

**Difference**:
- C++ uses `std::min(orig/deriv, t)` to prevent negative time in some cases
- C++ uses explicit `1e-9` tolerance for second/third Newton steps
- Julia uses uniform `EPS` tolerance

---

### 5. Polynomial Root Finding

#### Cubic Solver

##### C++ (`roots.hpp:60-149`)
```cpp
inline PositiveSet<double, 3> solve_cubic(double a, double b, double c, double d) {
    // Handle d≈0: insert x=0, reduce to quadratic
    // Handle a≈0: linear or quadratic
    // Use Cardano's formula with polar form for 3 real roots
    constexpr double cos120 = -0.50;
    constexpr double sin120 = 0.866025403784438646764;
}
```

##### Julia (`ruckig.jl:337-372`)
```julia
function solve_cubic_real!(roots::Roots, a, b, c, d)
    # Normalize to monic: t³ + pt² + qr + r = 0
    # Use depressed cubic substitution
    # Cardano's formula with discriminant branching
end
```

**Difference**:
- C++ reduces to lower-degree if `d≈0` first, then uses explicit cos120/sin120 constants
- Julia uses standard depressed cubic transformation without d≈0 special case
- Different but mathematically equivalent approaches

#### Quartic Solver

##### C++ (`roots.hpp:197-283`)
Uses resolvent cubic method with special cases for `d≈0` and `c≈0`:
```cpp
inline PositiveSet<double, 4> solve_quart_monic(double a, double b, double c, double d) {
    if (std::abs(d) < DBL_EPSILON) {
        if (std::abs(c) < DBL_EPSILON) {
            // x² factor out
        }
        if (std::abs(a) < DBL_EPSILON && std::abs(b) < DBL_EPSILON) {
            // x^4 + cx = 0
        }
    }
    // Resolvent cubic: x³ - b*x² + (a*c - 4*d)*x - (a²*d - 4*b*d + c²) = 0
}
```

##### Julia (`ruckig.jl:377-475`)
Ferrari's method with similar special cases:
```julia
function solve_quartic_real!(roots::Roots, a, b, c, d, e)
    # Special cases from reference implementation
    if abs(s) < EPS
        if abs(r) < EPS
            # x² factor
        end
    end
    # Resolvent cubic approach
end
```

**Difference**: Both use essentially the same resolvent cubic approach but with slightly different formulations. Julia also handles the `a ≈ 0` case (non-quartic) explicitly.

---

### 6. Profile Check/Validation

#### C++ (`profile.hpp`)
Template-based with control signs and limits as compile-time parameters:
```cpp
template<ControlSigns control_signs, ReachedLimits limits, bool skip_acc_check = false>
bool check(double jf, double vMax, double vMin, double aMax, double aMin) {
    // Set jerk pattern based on control_signs
    // Integrate 7-phase profile
    // Check final state precision
    // Check velocity limits
    // Check acceleration at zero-crossings
}
```

#### Julia (`ruckig.jl:534-669`)
Runtime dispatch based on control signs and limits:
```julia
function check!(buf::ProfileBuffer{T}, control_signs::ControlSigns, limits::ReachedLimits,
                jf, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af=0) where T
    # Set jerk pattern based on control_signs
    # Integrate 7-phase profile
    # Check final state precision
    # Check velocity limits
    # Check acceleration at zero-crossings
end
```

**Difference**:
- C++ uses template specialization for compile-time optimization
- Julia uses runtime enum values (likely JIT-optimized)
- Both implement identical validation logic

---

### 7. Profile Direction Handling

#### C++ (`position_third_step1.cpp:532-537`)
```cpp
const double vMax = (pd >= 0) ? _vMax : _vMin;
const double vMin = (pd >= 0) ? _vMin : _vMax;
const double aMax = (pd >= 0) ? _aMax : _aMin;
const double aMin = (pd >= 0) ? _aMin : _aMax;
const double jMax = (pd >= 0) ? _jMax : -_jMax;
```

#### Julia (`ruckig.jl:1367-1374`)
```julia
pd = pf - p0
if pd >= 0
    jMax1, vMax1, vMin1, aMax1, aMin1 = jmax, vmax, vmin, amax, amin
    jMax2, vMax2, vMin2, aMax2, aMin2 = -jmax, vmin, vmax, amin, amax
else
    jMax1, vMax1, vMin1, aMax1, aMin1 = -jmax, vmin, vmax, amin, amax
    jMax2, vMax2, vMin2, aMax2, aMin2 = jmax, vmax, vmin, amax, amin
end
```

**Status**: Equivalent - both swap limits based on displacement sign.

---

### 8. Multi-DOF Synchronization

#### C++ (`calculator_target.hpp`)
- Collects multiple valid profiles per DOF
- Builds Block intervals using `Block::calculate_block()`
- Handles 1-5 competing profiles
- Selects synchronization time considering blocked intervals
- Falls back to next candidate if blocked

#### Julia (`ruckig.jl:3094-3160`)
```julia
# Step 1: Calculate minimum-time profile for each DOF independently
blocks = Vector{Block{Float64}}(undef, ndof)
for i in 1:ndof
    profile = calculate_trajectory(lims[i]; ...)
    blocks[i] = Block(profile)  # Single profile only
end

# Find synchronization time (maximum of all minimum times)
t_sync = maximum(block.t_min for block in blocks)

# Check if t_sync is blocked for any DOF
for i in 1:ndof
    if is_blocked(blocks[i], t_sync)
        error("Synchronization time $t_sync is blocked for DOF $i")
    end
end
```

**Difference**:
- C++ collects multiple competing profiles and builds blocked intervals
- Julia only stores single minimum-time profile per DOF
- Julia throws error if sync time is blocked; C++ tries alternatives
- **This is a significant algorithmic difference for edge cases**

---

### 9. Precision Constants

| Constant | C++ Value | Julia Value | Status |
|----------|-----------|-------------|--------|
| EPS / v_eps / a_eps / j_eps | `DBL_EPSILON` (~2.2e-16) | `1e-12` | Different |
| p_precision | `1e-8` | `1e-8` | Same |
| v_precision | `1e-8` | `1e-8` | Same |
| a_precision | `1e-10` | `1e-10` | Same |
| t_precision | `1e-12` | `1e-12` | Same |
| Newton tolerance | `1e-9` (explicit) | `EPS` (1e-12) | Different |
| shrink_interval tolerance | `1e-14` | `1e-14` | Same |

---

### 10. Data Structures

#### Profile Storage

| Aspect | C++ | Julia |
|--------|-----|-------|
| Phase times | `double t[7]` | `Memory{T}` (mutable) / `NTuple{7,T}` (immutable) |
| Cumulative times | `double t_sum[7]` | `Memory{T}` / `NTuple{7,T}` |
| Jerk values | `double j[7]` | `Memory{T}` / `NTuple{7,T}` |
| Boundary states | `double a[8], v[8], p[8]` | `Memory{T}` / `NTuple{8,T}` |
| Brake profile | `BrakeProfile brake, accel` | N/A |
| Direction | `Direction direction` | N/A |

#### Root Storage

| Aspect | C++ | Julia |
|--------|-----|-------|
| Container | `PositiveSet<double, N>` (stack array) | `Roots{T}` (mutable struct) |
| Filtering | Only inserts non-negative | Filters during iteration |
| Sorting | Sorts on begin() access | Not sorted |

---

## Summary of Critical Differences

### High Priority (May cause incorrect results)

1. **No Brake Profiles**: Initial states outside limits will fail
2. **No Block Interval Calculation**: Multi-DOF synchronization may fail for competing profiles
3. **No Zero-Limits Handling**: jMax=0, aMax=0, or aMin=0 will fail

### Medium Priority (Missing functionality)

4. **No Velocity Control Interface**: Cannot target velocity
5. **No 2nd-Order Profiles**: Cannot ignore jerk limits
6. **No 1st-Order Profiles**: Cannot do pure constant-velocity motion
7. **No Position Extrema Functions**: Cannot query min/max position on trajectory

### Low Priority (Numerical differences)

8. **Different EPS values**: C++ uses DBL_EPSILON, Julia uses 1e-12 (more conservative)
9. **Newton refinement tolerance**: C++ uses explicit 1e-9, Julia uses 1e-12
10. **Triple Newton steps**: C++ uses in some functions, Julia typically uses single

---

## Recommendations

1. **Add brake profile support** for handling initial states outside limits
2. **Implement `time_all_single_step`** for zero-limits edge case
3. **Improve Block calculation** to handle multiple competing profiles
4. **Consider adding velocity control** if needed for applications
5. **Add position extrema functions** if needed for collision checking
