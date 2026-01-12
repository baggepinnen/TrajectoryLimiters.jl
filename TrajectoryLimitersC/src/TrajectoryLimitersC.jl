module TrajectoryLimitersC

using TrajectoryLimiters
using TrajectoryLimiters: State, trajlim, RuckigProfile

# Global storage to prevent GC of Julia objects accessed via opaque pointers
const OBJECT_STORE = Dict{UInt, Any}()
const NEXT_HANDLE = Ref{UInt}(1)

function store_object(obj)::Ptr{Cvoid}
    handle = NEXT_HANDLE[]
    NEXT_HANDLE[] += 1
    OBJECT_STORE[handle] = obj
    return Ptr{Cvoid}(handle)
end

function get_object(handle::Ptr{Cvoid})
    return OBJECT_STORE[UInt(handle)]
end

function remove_object(handle::Ptr{Cvoid})
    delete!(OBJECT_STORE, UInt(handle))
    return nothing
end

#=============================================================================
 TrajectoryLimiter API
=============================================================================#

"""
Create a TrajectoryLimiter with given sample time and limits.
Returns an opaque handle.
"""
Base.@ccallable function trajectory_limiter_create(Ts::Cdouble, xdotM::Cdouble, xddotM::Cdouble)::Ptr{Cvoid}
    limiter = TrajectoryLimiter(Ts, xdotM, xddotM)
    return store_object(limiter)
end

"""
Destroy a TrajectoryLimiter and free associated resources.
"""
Base.@ccallable function trajectory_limiter_destroy(handle::Ptr{Cvoid})::Cvoid
    remove_object(handle)
    return
end

"""
Take one filtering step with the TrajectoryLimiter.

Arguments:
- handle: TrajectoryLimiter handle from trajectory_limiter_create
- state_ptr: Pointer to array of 4 doubles [x, xdot, r, rdot] - modified in place
- r: New reference position

Returns: acceleration command
"""
Base.@ccallable function trajectory_limiter_step(handle::Ptr{Cvoid}, state_ptr::Ptr{Cdouble}, r::Cdouble)::Cdouble
    limiter = get_object(handle)::TrajectoryLimiter{Float64}

    # Read current state from C array
    state_arr = unsafe_wrap(Array, state_ptr, 4)
    state = State(state_arr[1], state_arr[2], state_arr[3], state_arr[4])

    # Take one step
    new_state, accel = trajlim(state, r, limiter.Ts, limiter.ẋM, limiter.ẍM)

    # Write new state back to C array
    state_arr[1] = new_state.x
    state_arr[2] = new_state.ẋ
    state_arr[3] = new_state.r
    state_arr[4] = new_state.ṙ

    return accel
end

#=============================================================================
 JerkLimiter API
=============================================================================#

"""
Create a JerkLimiter with symmetric limits.
Returns an opaque handle.
"""
Base.@ccallable function jerk_limiter_create(vmax::Cdouble, amax::Cdouble, jmax::Cdouble)::Ptr{Cvoid}
    limiter = JerkLimiter(; vmax, amax, jmax)
    return store_object(limiter)
end

"""
Destroy a JerkLimiter and free associated resources.
"""
Base.@ccallable function jerk_limiter_destroy(handle::Ptr{Cvoid})::Cvoid
    remove_object(handle)
    return
end

#=============================================================================
 Profile API
=============================================================================#

"""
Create a trajectory profile using a JerkLimiter.

Arguments:
- limiter_handle: JerkLimiter handle from jerk_limiter_create
- p0, v0, a0: Initial position, velocity, acceleration
- pf, vf, af: Target position, velocity, acceleration

Returns: opaque profile handle
"""
Base.@ccallable function profile_create(limiter_handle::Ptr{Cvoid},
                                        p0::Cdouble, v0::Cdouble, a0::Cdouble,
                                        pf::Cdouble, vf::Cdouble, af::Cdouble)::Ptr{Cvoid}
    limiter = get_object(limiter_handle)::JerkLimiter{Float64}
    profile = calculate_trajectory(limiter; p0, v0, a0, pf, vf, af)
    return store_object(profile)
end

"""
Destroy a profile and free associated resources.
"""
Base.@ccallable function profile_destroy(handle::Ptr{Cvoid})::Cvoid
    remove_object(handle)
    return
end

"""
Get the total duration of a trajectory profile.
"""
Base.@ccallable function profile_duration(handle::Ptr{Cvoid})::Cdouble
    profile = get_object(handle)::RuckigProfile{Float64}
    return duration(profile)
end

"""
Evaluate a trajectory profile at time t.

Arguments:
- handle: Profile handle from profile_create
- t: Time to evaluate at
- result_ptr: Pointer to array of 4 doubles to store [p, v, a, j]
"""
Base.@ccallable function profile_evaluate(handle::Ptr{Cvoid}, t::Cdouble, result_ptr::Ptr{Cdouble})::Cvoid
    profile = get_object(handle)::RuckigProfile{Float64}
    p, v, a, j = profile(t)

    result_arr = unsafe_wrap(Array, result_ptr, 4)
    result_arr[1] = p
    result_arr[2] = v
    result_arr[3] = a
    result_arr[4] = j

    return
end

end # module
