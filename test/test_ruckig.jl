using TrajectoryLimiters
using Test

@testset "JerkLimiter constructors" begin
    # Symmetric constructor
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)
    @test lim.vmax == 10.0
    @test lim.vmin == -10.0
    @test lim.amax == 50.0
    @test lim.amin == -50.0
    @test lim.jmax == 1000.0

    # Directional constructor
    lim2 = JerkLimiter(; vmax=10.0, vmin=-5.0, amax=50.0, amin=-30.0, jmax=1000.0)
    @test lim2.vmax == 10.0
    @test lim2.vmin == -5.0
    @test lim2.amax == 50.0
    @test lim2.amin == -30.0

    # Type promotion
    lim3 = JerkLimiter(; vmax=10, amax=50, jmax=1000)
    @test lim3.vmax isa Int
end

@testset "Simple trajectory: rest to rest" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Move from 0 to 1
    profile = calculate_trajectory(lim; pf=1.0)

    @test duration(profile) > 0
    @test profile.p[1] ≈ 0.0
    @test profile.v[1] ≈ 0.0
    @test profile.a[1] ≈ 0.0
    @test profile.p[8] ≈ 1.0 atol=1e-6
    @test profile.v[8] ≈ 0.0 atol=1e-6
    @test profile.a[8] ≈ 0.0 atol=1e-6

    # Check trajectory stays within limits
    for t in range(0, duration(profile), length=100)
        p, v, a, j = evaluate_at(profile, t)
        @test lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6
        @test lim.amin - 1e-6 <= a <= lim.amax + 1e-6
        @test -lim.jmax - 1e-6 <= j <= lim.jmax + 1e-6
    end
end

@testset "Simple trajectory: negative direction" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Move from 1 to 0
    profile = calculate_trajectory(lim; p0=1.0, pf=0.0)

    @test duration(profile) > 0
    @test profile.p[1] ≈ 1.0
    @test profile.p[8] ≈ 0.0 atol=1e-6
    @test profile.v[8] ≈ 0.0 atol=1e-6
    @test profile.a[8] ≈ 0.0 atol=1e-6
end

@testset "evaluate_at boundary conditions" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)
    profile = calculate_trajectory(lim; pf=1.0)

    # At t=0
    p, v, a, j = evaluate_at(profile, 0.0)
    @test p ≈ 0.0
    @test v ≈ 0.0
    @test a ≈ 0.0

    # At t=T_total
    T_total = duration(profile)
    p, v, a, j = evaluate_at(profile, T_total)
    @test p ≈ 1.0 atol=1e-6
    @test v ≈ 0.0 atol=1e-6
    @test a ≈ 0.0 atol=1e-6

    # Beyond end
    p, v, a, j = evaluate_at(profile, T_total + 1.0)
    @test p ≈ 1.0 atol=1e-6
    @test j ≈ 0.0
end

@testset "Trajectory with initial velocity" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Start with v0 = 5, move forward
    profile = calculate_trajectory(lim; v0=5.0, pf=2.0)

    @test duration(profile) > 0
    @test profile.p[1] ≈ 0.0
    @test profile.v[1] ≈ 5.0
    @test profile.p[8] ≈ 2.0 atol=1e-6
    @test profile.v[8] ≈ 0.0 atol=1e-6
end

@testset "Velocity-limited trajectory" begin
    # Use low velocity limit to force velocity plateau
    lim = JerkLimiter(; vmax=2.0, amax=50.0, jmax=1000.0)

    # Long distance should hit velocity limit
    profile = calculate_trajectory(lim; pf=10.0)

    @test duration(profile) > 0

    # Check max velocity reached is close to limit
    max_v = maximum(profile.v)
    @test max_v ≈ lim.vmax atol=0.1

    # Final state
    @test profile.p[8] ≈ 10.0 atol=1e-6
    @test profile.v[8] ≈ 0.0 atol=1e-6
end

@testset "Profile duration monotonicity" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Longer distances should take longer
    p1 = calculate_trajectory(lim; pf=1.0)
    p2 = calculate_trajectory(lim; pf=2.0)
    p3 = calculate_trajectory(lim; pf=4.0)

    @test p1.t_sum[7] < p2.t_sum[7] < p3.t_sum[7]
end

@testset "Continuous trajectory sampling" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)
    profile = calculate_trajectory(lim; pf=1.0)

    # Sample finely and check continuity
    Ts = 0.001
    prev_p, prev_v, prev_a, _ = evaluate_at(profile, 0.0)

    for t in Ts:Ts:duration(profile)
        p, v, a, j = evaluate_at(profile, t)

        # Position should be monotonic for this trajectory
        @test p >= prev_p - 1e-10

        # Changes should be smooth (bounded by limits * Ts with margin)
        @test abs(v - prev_v) < (lim.amax + lim.jmax * Ts) * Ts * 2
        @test abs(a - prev_a) < lim.jmax * Ts * 2

        prev_p, prev_v, prev_a = p, v, a
    end
end

@testset "Low jerk trajectories (two-step profiles)" begin
    # Low jerk requires two-step profile fallbacks
    vmax, amax = 10.0, 50.0

    for jmax in (5000.0, 1000.0, 500.0, 200.0)
        lim = JerkLimiter(; vmax, amax, jmax)
        profile = calculate_trajectory(lim; pf=2.0)

        @test duration(profile) > 0
        @test profile.p[8] ≈ 2.0 atol=1e-6
        @test profile.v[8] ≈ 0.0 atol=1e-6
        @test profile.a[8] ≈ 0.0 atol=1e-6

        # Lower jerk should result in longer duration
        # (smoother acceleration requires more time)
    end

    # Verify duration ordering: lower jerk → longer time
    p_high = calculate_trajectory(JerkLimiter(; vmax, amax, jmax=5000.0); pf=2.0)
    p_med  = calculate_trajectory(JerkLimiter(; vmax, amax, jmax=1000.0); pf=2.0)
    p_low  = calculate_trajectory(JerkLimiter(; vmax, amax, jmax=200.0); pf=2.0)

    @test p_high.t_sum[7] < p_med.t_sum[7] < p_low.t_sum[7]
end

@testset "Various jerk levels with limits check" begin
    vmax, amax = 10.0, 50.0

    for jmax in (5000.0, 1000.0, 200.0)
        lim = JerkLimiter(; vmax, amax, jmax)
        profile = calculate_trajectory(lim; pf=2.0)

        # Check trajectory stays within limits
        for t in range(0, duration(profile), length=100)
            p, v, a, j = evaluate_at(profile, t)
            @test lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6
            @test lim.amin - 1e-6 <= a <= lim.amax + 1e-6
            @test -lim.jmax - 1e-6 <= j <= lim.jmax + 1e-6
        end
    end
end

@testset "evaluate_at with vector of times" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)
    profile = calculate_trajectory(lim; pf=2.0)

    ts = range(0, duration(profile), length=50)
    pos, vel, acc, jerk = evaluate_at(profile, ts)

    @test length(pos) == 50
    @test length(vel) == 50
    @test length(acc) == 50
    @test length(jerk) == 50

    # Check that vector version matches scalar version
    for (i, t) in enumerate(ts)
        p, v, a, j = evaluate_at(profile, t)
        @test pos[i] ≈ p
        @test vel[i] ≈ v
        @test acc[i] ≈ a
        @test jerk[i] ≈ j
    end

    # Check boundary values
    @test pos[1] ≈ 0.0
    @test pos[end] ≈ 2.0 atol=1e-6
    @test vel[end] ≈ 0.0 atol=1e-6
end

@testset "evaluate_dt" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)
    profile = calculate_trajectory(lim; pf=2.0)

    # Default Ts=0.001
    pos, vel, acc, jerk, ts = evaluate_dt(profile, 0.001)

    @test ts[1] == 0.0
    @test ts[end] <= duration(profile)
    @test step(ts) == 0.001

    # Check boundary values
    @test pos[1] ≈ 0.0
    @test pos[end] ≈ 2.0 atol=1e-3  # May not hit exactly due to Ts discretization
    @test vel[1] ≈ 0.0
    @test acc[1] ≈ 0.0

    # Custom Ts
    pos2, vel2, acc2, jerk2, ts2 = evaluate_dt(profile, 0.01)
    @test step(ts2) == 0.01
    @test length(ts2) < length(ts)  # Coarser sampling = fewer points
end

@testset "Trajectory with non-zero final velocity" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # End at vf=3.0 (moving when we arrive)
    profile = calculate_trajectory(lim; pf=5.0, vf=3.0)

    @test duration(profile) > 0
    @test profile.p[1] ≈ 0.0
    @test profile.v[1] ≈ 0.0
    @test profile.p[8] ≈ 5.0 atol=1e-6
    @test profile.v[8] ≈ 3.0 atol=1e-6
    @test profile.a[8] ≈ 0.0 atol=1e-6

    # Verify via evaluate_at at final time
    T_total = duration(profile)
    p, v, a, j = evaluate_at(profile, T_total)
    @test p ≈ 5.0 atol=1e-6
    @test v ≈ 3.0 atol=1e-6
    @test a ≈ 0.0 atol=1e-6
end

@testset "Trajectory with non-zero final acceleration" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # End at af=10.0 (accelerating when we arrive)
    profile = calculate_trajectory(lim; pf=2.0, af=10.0)

    @test duration(profile) > 0
    @test profile.p[1] ≈ 0.0
    @test profile.v[1] ≈ 0.0
    @test profile.a[1] ≈ 0.0
    @test profile.p[8] ≈ 2.0 atol=1e-6
    @test profile.v[8] ≈ 0.0 atol=1e-6
    @test profile.a[8] ≈ 10.0 atol=1e-6

    # Verify via evaluate_at at final time
    T_total = duration(profile)
    p, v, a, j = evaluate_at(profile, T_total)
    @test p ≈ 2.0 atol=1e-6
    @test v ≈ 0.0 atol=1e-6
    @test a ≈ 10.0 atol=1e-6
end

@testset "Trajectory with non-zero final velocity and acceleration" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # End at vf=2.0 and af=5.0
    profile = calculate_trajectory(lim; pf=3.0, vf=2.0, af=5.0)

    @test duration(profile) > 0
    @test profile.p[8] ≈ 3.0 atol=1e-6
    @test profile.v[8] ≈ 2.0 atol=1e-6
    @test profile.a[8] ≈ 5.0 atol=1e-6

    # Check trajectory stays within limits
    for t in range(0, duration(profile), length=100)
        p, v, a, j = evaluate_at(profile, t)
        @test lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6
        @test lim.amin - 1e-6 <= a <= lim.amax + 1e-6
        @test -lim.jmax - 1e-6 <= j <= lim.jmax + 1e-6
    end
end

@testset "Trajectory with all non-zero boundary conditions" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Start and end with non-zero states
    profile = calculate_trajectory(lim; p0=1.0, v0=2.0, a0=5.0, pf=4.0, vf=1.0, af=3.0)

    @test duration(profile) > 0
    @test profile.p[1] ≈ 1.0
    @test profile.v[1] ≈ 2.0
    @test profile.a[1] ≈ 5.0
    @test profile.p[8] ≈ 4.0 atol=1e-6
    @test profile.v[8] ≈ 1.0 atol=1e-6
    @test profile.a[8] ≈ 3.0 atol=1e-6
end

@testset "Waypoint trajectories" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Basic 2-waypoint (equivalent to single segment)
    waypoints = [(p=0.0,), (p=1.0,)]
    ts, ps, vs, as, js = calculate_waypoint_trajectory(lim, waypoints, 0.001)
    @test ps[1] ≈ 0.0
    @test ps[end] ≈ 1.0 atol=1e-3
    @test vs[end] ≈ 0.0 atol=1e-3

    # 3 waypoints
    waypoints = [(p=0.0,), (p=1.0,), (p=3.0,)]
    ts, ps, vs, as, js = calculate_waypoint_trajectory(lim, waypoints, 0.001)
    @test ps[1] ≈ 0.0
    @test ps[end] ≈ 3.0 atol=1e-3

    # With non-zero velocities at waypoints
    waypoints = [(p=0.0,), (p=2.0, v=5.0), (p=5.0,)]
    ts, ps, vs, as, js = calculate_waypoint_trajectory(lim, waypoints, 0.001)
    @test ps[1] ≈ 0.0
    @test ps[end] ≈ 5.0 atol=1e-3

    # Check limits are respected
    for i in eachindex(ts)
        @test lim.vmin - 1e-6 <= vs[i] <= lim.vmax + 1e-6
        @test lim.amin - 1e-6 <= as[i] <= lim.amax + 1e-6
        @test -lim.jmax - 1e-6 <= js[i] <= lim.jmax + 1e-6
    end

    # Check time is monotonically increasing
    for i in 2:length(ts)
        @test ts[i] > ts[i-1]
    end
end

@testset "Waypoint trajectory with all states specified" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # All waypoints with full state specification
    waypoints = [
        (p=0.0, v=0.0, a=0.0),
        (p=2.0, v=3.0, a=5.0),
        (p=5.0, v=0.0, a=0.0),
    ]
    ts, ps, vs, as, js = calculate_waypoint_trajectory(lim, waypoints, 0.001)

    @test ps[1] ≈ 0.0
    @test vs[1] ≈ 0.0
    @test as[1] ≈ 0.0
    @test ps[end] ≈ 5.0 atol=1e-3
    @test vs[end] ≈ 0.0 atol=1e-3
end

@testset "Multi-DOF synchronized trajectories" begin
    # Two DOFs with different constraints
    lims = [
        JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
        JerkLimiter(; vmax=5.0, amax=30.0, jmax=500.0),
    ]

    # Basic test: both DOFs start at rest, end at rest
    profiles = calculate_trajectory(lims; pf=[1.0, 2.0])

    # All profiles should have the same duration
    @test duration(profiles[1]) ≈ duration(profiles[2])

    # Check boundary conditions for each DOF
    @test profiles[1].p[1] ≈ 0.0
    @test profiles[1].v[1] ≈ 0.0
    @test profiles[1].a[1] ≈ 0.0
    @test profiles[1].p[8] ≈ 1.0 atol=1e-6
    @test profiles[1].v[8] ≈ 0.0 atol=1e-6
    @test profiles[1].a[8] ≈ 0.0 atol=1e-6

    @test profiles[2].p[1] ≈ 0.0
    @test profiles[2].v[1] ≈ 0.0
    @test profiles[2].a[1] ≈ 0.0
    @test profiles[2].p[8] ≈ 2.0 atol=1e-6
    @test profiles[2].v[8] ≈ 0.0 atol=1e-6
    @test profiles[2].a[8] ≈ 0.0 atol=1e-6

    # Check limits are respected for each DOF
    T_total = duration(profiles[1])
    for t in range(0, T_total, length=100)
        for (i, lim) in enumerate(lims)
            p, v, a, j = evaluate_at(profiles[i], t)
            @test lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6
            @test lim.amin - 1e-6 <= a <= lim.amax + 1e-6
            @test -lim.jmax - 1e-6 <= j <= lim.jmax + 1e-6
        end
    end
end

@testset "Multi-DOF evaluate_at" begin
    lims = [
        JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
        JerkLimiter(; vmax=5.0, amax=30.0, jmax=500.0),
    ]

    profiles = calculate_trajectory(lims; pf=[1.0, 2.0])

    # Test vector evaluate_at
    t = 0.05
    ps, vs, as, js = evaluate_at(profiles, t)

    @test length(ps) == 2
    @test length(vs) == 2
    @test length(as) == 2
    @test length(js) == 2

    # Check that vector version matches scalar version
    for i in 1:2
        p, v, a, j = evaluate_at(profiles[i], t)
        @test ps[i] ≈ p
        @test vs[i] ≈ v
        @test as[i] ≈ a
        @test js[i] ≈ j
    end

    # Test at boundaries
    ps0, vs0, as0, js0 = evaluate_at(profiles, 0.0)
    @test ps0[1] ≈ 0.0
    @test ps0[2] ≈ 0.0
    @test vs0[1] ≈ 0.0
    @test vs0[2] ≈ 0.0

    T_total = duration(profiles[1])
    psf, vsf, asf, jsf = evaluate_at(profiles, T_total)
    @test psf[1] ≈ 1.0 atol=1e-6
    @test psf[2] ≈ 2.0 atol=1e-6
    @test vsf[1] ≈ 0.0 atol=1e-6
    @test vsf[2] ≈ 0.0 atol=1e-6
end

@testset "Multi-DOF evaluate_dt" begin
    lims = [
        JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
        JerkLimiter(; vmax=5.0, amax=30.0, jmax=500.0),
    ]

    profiles = calculate_trajectory(lims; pf=[1.0, 2.0])

    # Test matrix evaluate_dt
    pos, vel, acc, jerk, ts = evaluate_dt(profiles, 0.001)

    @test size(pos, 2) == 2  # 2 DOFs
    @test size(vel, 2) == 2
    @test size(acc, 2) == 2
    @test size(jerk, 2) == 2
    @test size(pos, 1) == length(ts)

    # Check boundary values
    @test pos[1, 1] ≈ 0.0
    @test pos[1, 2] ≈ 0.0
    @test pos[end, 1] ≈ 1.0 atol=1e-3
    @test pos[end, 2] ≈ 2.0 atol=1e-3

    # Check that matrix version matches scalar version
    for (i, t) in enumerate(ts[1:10:end])
        for j in 1:2
            p, v, a, jk = evaluate_at(profiles[j], t)
            idx = 1 + (i-1)*10
            @test pos[idx, j] ≈ p
            @test vel[idx, j] ≈ v
            @test acc[idx, j] ≈ a
            @test jerk[idx, j] ≈ jk
        end
    end
end

@testset "Multi-DOF with initial states" begin
    lims = [
        JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
        JerkLimiter(; vmax=8.0, amax=40.0, jmax=800.0),
    ]

    # Start with different initial positions and velocities
    profiles = calculate_trajectory(lims;
        p0=[1.0, 2.0],
        v0=[2.0, 1.0],
        pf=[5.0, 6.0],
    )

    # All profiles should have the same duration
    @test duration(profiles[1]) ≈ duration(profiles[2])

    # Check initial conditions
    @test profiles[1].p[1] ≈ 1.0
    @test profiles[1].v[1] ≈ 2.0
    @test profiles[2].p[1] ≈ 2.0
    @test profiles[2].v[1] ≈ 1.0

    # Check final conditions
    @test profiles[1].p[8] ≈ 5.0 atol=1e-6
    @test profiles[2].p[8] ≈ 6.0 atol=1e-6
    @test profiles[1].v[8] ≈ 0.0 atol=1e-6
    @test profiles[2].v[8] ≈ 0.0 atol=1e-6
end

@testset "Multi-DOF with 3 DOFs" begin
    lims = [
        JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
        JerkLimiter(; vmax=5.0, amax=30.0, jmax=500.0),
        JerkLimiter(; vmax=8.0, amax=40.0, jmax=800.0),
    ]

    profiles = calculate_trajectory(lims; pf=[1.0, 2.0, 0.5])

    # All profiles should have the same duration
    T = duration(profiles[1])
    @test duration(profiles[2]) ≈ T
    @test duration(profiles[3]) ≈ T

    # Check final positions
    @test profiles[1].p[8] ≈ 1.0 atol=1e-6
    @test profiles[2].p[8] ≈ 2.0 atol=1e-6
    @test profiles[3].p[8] ≈ 0.5 atol=1e-6

    # Check limits are respected for each DOF
    for t in range(0, T, length=100)
        for (i, lim) in enumerate(lims)
            p, v, a, j = evaluate_at(profiles[i], t)
            @test lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6
            @test lim.amin - 1e-6 <= a <= lim.amax + 1e-6
            @test -lim.jmax - 1e-6 <= j <= lim.jmax + 1e-6
        end
    end
end

@testset "Multi-DOF limiting DOF identification" begin
    # DOF 2 has tighter velocity constraint, so for equal distances
    # it should be the limiting DOF
    lims = [
        JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
        JerkLimiter(; vmax=2.0, amax=50.0, jmax=1000.0),  # Much slower
    ]

    # Same distance for both DOFs
    profiles = calculate_trajectory(lims; pf=[2.0, 2.0])

    # Synchronized duration should be longer than DOF 1 alone would need
    profile_single = calculate_trajectory(lims[1]; pf=2.0)
    @test duration(profiles[1]) > duration(profile_single)

    # Both DOFs still reach their targets
    @test profiles[1].p[8] ≈ 2.0 atol=1e-6
    @test profiles[2].p[8] ≈ 2.0 atol=1e-6
end

@testset "Multi-DOF waypoint trajectories" begin
    lims = [
        JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
        JerkLimiter(; vmax=5.0, amax=30.0, jmax=500.0),
    ]

    # Basic 2-waypoint (equivalent to single segment)
    waypoints = [(p=[0.0, 0.0],), (p=[1.0, 2.0],)]
    ts, ps, vs, as, js = calculate_waypoint_trajectory(lims, waypoints, 0.001)

    @test size(ps, 2) == 2  # 2 DOFs
    @test ps[1, 1] ≈ 0.0
    @test ps[1, 2] ≈ 0.0
    @test ps[end, 1] ≈ 1.0 atol=1e-3
    @test ps[end, 2] ≈ 2.0 atol=1e-3
    @test vs[end, 1] ≈ 0.0 atol=1e-3
    @test vs[end, 2] ≈ 0.0 atol=1e-3

    # 3 waypoints
    waypoints = [
        (p = [0.0, 0.0],),
        (p = [1.0, 2.0],),
        (p = [3.0, 4.0],),
    ]
    ts, ps, vs, as, js = calculate_waypoint_trajectory(lims, waypoints, 0.001)

    @test ps[1, 1] ≈ 0.0
    @test ps[1, 2] ≈ 0.0
    @test ps[end, 1] ≈ 3.0 atol=1e-3
    @test ps[end, 2] ≈ 4.0 atol=1e-3

    # With non-zero velocities at waypoints
    waypoints = [
        (p = [0.0, 0.0],),
        (p = [2.0, 3.0], v = [0.1, 0.5]),
        (p = [5.0, 6.0],),
    ]
    ts, ps, vs, as, js = calculate_waypoint_trajectory(lims, waypoints, 0.001)

    @test ps[1, 1] ≈ 0.0
    @test ps[1, 2] ≈ 0.0
    @test ps[end, 1] ≈ 5.0 atol=1e-3
    @test ps[end, 2] ≈ 6.0 atol=1e-3

    # Check limits are respected for each DOF
    for i in eachindex(ts)
        for (j, lim) in enumerate(lims)
            @test lim.vmin - 1e-6 <= vs[i, j] <= lim.vmax + 1e-6
            @test lim.amin - 1e-6 <= as[i, j] <= lim.amax + 1e-6
            @test -lim.jmax - 1e-6 <= js[i, j] <= lim.jmax + 1e-6
        end
    end

    # Check time is monotonically increasing
    for i in 2:length(ts)
        @test ts[i] > ts[i-1]
    end
end

@testset "Multi-DOF waypoint trajectory with 3 DOFs" begin
    lims = [
        JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
        JerkLimiter(; vmax=5.0, amax=30.0, jmax=500.0),
        JerkLimiter(; vmax=8.0, amax=40.0, jmax=800.0),
    ]

    waypoints = [
        (p = [0.0, 0.0, 0.0],),
        (p = [1.0, 2.0, 1.5], v = [0.1, 0.3, 0.4]),
        (p = [3.0, 4.0, 3.0],),
    ]
    ts, ps, vs, as, js = calculate_waypoint_trajectory(lims, waypoints, 0.001)

    @test size(ps, 2) == 3  # 3 DOFs
    @test ps[end, 1] ≈ 3.0 atol=1e-3
    @test ps[end, 2] ≈ 4.0 atol=1e-3
    @test ps[end, 3] ≈ 3.0 atol=1e-3

    # All DOFs end at rest
    @test vs[end, 1] ≈ 0.0 atol=1e-3
    @test vs[end, 2] ≈ 0.0 atol=1e-3
    @test vs[end, 3] ≈ 0.0 atol=1e-3
end

@testset "random waypoint traj" begin
    for _ in 1:5
        lims = [
            JerkLimiter(; vmax=10.0*rand(), amax=50.0*rand(), jmax=1000.0*rand()) for i = 1:7
        ]
        waypoints = [(p = randn(7),) for i = 1:100]
        @test_nowarn calculate_waypoint_trajectory(lims, waypoints, 0.001)


        lims = [
            JerkLimiter(; vmax=10.0*rand(), amax=50.0*rand(), jmax=1000.0*rand(), vmin=-10.0*rand(), amin=-50.0*rand()) for i = 1:2
        ]
        waypoints = [(p = randn(2),) for i = 1:100]
        @test_nowarn calculate_waypoint_trajectory(lims, waypoints, 0.001)


        lim = JerkLimiter(; vmax=10.0*rand(), amax=50.0*rand(), jmax=1000.0*rand())
        waypoints = [(p = randn(),) for i = 1:100]
        @test_nowarn calculate_waypoint_trajectory(lim, waypoints, 0.001)


        lims = [
            JerkLimiter(; vmax=10.0*rand()+5, amax=50.0*rand()+5, jmax=1000.0*rand(), vmin=-10.0*rand()-5, amin=-50.0*rand()-5) for i = 1:2
        ]
        waypoints = [(p = randn(2), v = randn(2)) for i = 1:100]
        @test_nowarn calculate_waypoint_trajectory(lims, waypoints, 0.001)

        waypoints = [(p = randn(2), v = randn(2), a = randn(2)) for i = 1:100]
        @test_nowarn calculate_waypoint_trajectory(lims, waypoints, 0.001)

        GC.gc(true); sleep(0.1)
    end
end


@testset "Initial velocity outside limits" begin
    # Test case where initial velocity exceeds vmax - requires brake profile
    lim = JerkLimiter(; vmax=5.0, amax=50.0, jmax=1000.0)

    # v0 = 8.0 > vmax = 5.0, need to brake to get within limits
    profile = calculate_trajectory(lim; v0=8.0, pf=10.0)

    @test duration(profile) > 0
    p0, v0, a0, _ = evaluate_at(profile, 0.0)
    @test v0 ≈ 8.0  # Starts above vmax
    T_total = duration(profile)
    pf, vf, af, _ = evaluate_at(profile, T_total)
    @test pf ≈ 10.0 atol=1e-6
    @test vf ≈ 0.0 atol=1e-6
    @test af ≈ 0.0 atol=1e-6

    # Check trajectory eventually comes within limits
    for t in range(T_total/2, T_total, length=50)
        p, v, a, j = evaluate_at(profile, t)
        @test lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6
        @test lim.amin - 1e-6 <= a <= lim.amax + 1e-6
    end

    # Test negative direction: v0 < vmin
    lim2 = JerkLimiter(; vmax=5.0, vmin=-3.0, amax=50.0, jmax=1000.0)
    profile2 = calculate_trajectory(lim2; v0=-6.0, pf=-5.0)

    _, v0_2, _, _ = evaluate_at(profile2, 0.0)
    @test v0_2 ≈ -6.0  # Starts below vmin
    pf_2, vf_2, _, _ = evaluate_at(profile2, duration(profile2))
    @test pf_2 ≈ -5.0 atol=1e-6
    @test vf_2 ≈ 0.0 atol=1e-6
end

@testset "Initial acceleration outside limits" begin
    # Test case where initial acceleration exceeds amax - requires brake profile
    lim = JerkLimiter(; vmax=10.0, amax=20.0, jmax=1000.0)

    # a0 = 35.0 > amax = 20.0, need to brake to get within limits
    profile = calculate_trajectory(lim; a0=35.0, pf=5.0)

    @test duration(profile) > 0
    _, _, a0, _ = profile(0.0)
    @test a0 ≈ 35.0  # Starts above amax
    T_total = duration(profile)
    pf, vf, af, _ = profile(T_total)
    @test pf ≈ 5.0 atol=1e-6
    @test vf ≈ 0.0 atol=1e-6
    @test af ≈ 0.0 atol=1e-6

    # Check trajectory eventually comes within limits
    for t in range(T_total/2, T_total, length=50)
        p, v, a, j = profile(t)
        @test lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6
        @test lim.amin - 1e-6 <= a <= lim.amax + 1e-6
    end

    # Test negative direction: a0 < amin
    lim2 = JerkLimiter(; vmax=10.0, amax=20.0, amin=-15.0, jmax=1000.0)
    profile2 = calculate_trajectory(lim2; a0=-25.0, pf=5.0)

    _, _, a0_2, _ = profile2(0.0)
    @test a0_2 ≈ -25.0  # Starts below amin
    pf_2, _, af_2, _ = profile2(duration(profile2))
    @test pf_2 ≈ 5.0 atol=1e-6
    @test af_2 ≈ 0.0 atol=1e-6
end

@testset "Initial velocity and acceleration both outside limits" begin
    # Test case where both initial velocity and acceleration exceed limits
    lim = JerkLimiter(; vmax=5.0, amax=20.0, jmax=1000.0)

    # v0 = 8.0 > vmax = 5.0 and a0 = 30.0 > amax = 20.0
    profile = calculate_trajectory(lim; v0=8.0, a0=30.0, pf=15.0)

    @test duration(profile) > 0
    _, v0, a0, _ = profile(0.0)
    @test v0 ≈ 8.0   # Starts above vmax
    @test a0 ≈ 30.0  # Starts above amax
    T_total = duration(profile)
    pf, vf, af, _ = profile(T_total)
    @test pf ≈ 15.0 atol=1e-6
    @test vf ≈ 0.0 atol=1e-6
    @test af ≈ 0.0 atol=1e-6

    # Check trajectory eventually comes within limits
    for t in range(T_total/2, T_total, length=50)
        p, v, a, j = profile(t)
        @test lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6
        @test lim.amin - 1e-6 <= a <= lim.amax + 1e-6
    end

    # Test with asymmetric limits and both negative out-of-range
    lim2 = JerkLimiter(; vmax=5.0, vmin=-3.0, amax=20.0, amin=-15.0, jmax=1000.0)
    profile2 = calculate_trajectory(lim2; v0=-7.0, a0=-25.0, pf=-10.0)

    _, v0_2, a0_2, _ = profile2(0.0)
    @test v0_2 ≈ -7.0   # Starts below vmin
    @test a0_2 ≈ -25.0  # Starts below amin
    pf_2, vf_2, af_2, _ = profile2(duration(profile2))
    @test pf_2 ≈ -10.0 atol=1e-6
    @test vf_2 ≈ 0.0 atol=1e-6
    @test af_2 ≈ 0.0 atol=1e-6

    # Mixed: v0 > vmax but a0 < amin
    lim3 = JerkLimiter(; vmax=5.0, vmin=-5.0, amax=20.0, amin=-15.0, jmax=1000.0)
    profile3 = calculate_trajectory(lim3; v0=8.0, a0=-25.0, pf=5.0)

    _, v0_3, a0_3, _ = profile3(0.0)
    @test v0_3 ≈ 8.0    # Starts above vmax
    @test a0_3 ≈ -25.0  # Starts below amin
    pf_3, vf_3, _, _ = evaluate_at(profile3, duration(profile3))
    @test pf_3 ≈ 5.0 atol=1e-6
    @test vf_3 ≈ 0.0 atol=1e-6
end

@testset "Target velocity at limit" begin
    lim = JerkLimiter(; vmax=5.0, vmin=-3.0, amax=50.0, jmax=1000.0)

    # Target velocity equals vmax
    profile = calculate_trajectory(lim; pf=10.0, vf=5.0)
    @test duration(profile) > 0
    pf, vf, af, _ = profile(duration(profile))
    @test pf ≈ 10.0 atol=1e-6
    @test vf ≈ 5.0 atol=1e-6  # Exactly at vmax
    @test af ≈ 0.0 atol=1e-6

    # Target velocity equals vmin (asymmetric)
    profile2 = calculate_trajectory(lim; pf=-5.0, vf=-3.0)
    pf_2, vf_2, af_2, _ = profile2(duration(profile2))
    @test pf_2 ≈ -5.0 atol=1e-6
    @test vf_2 ≈ -3.0 atol=1e-6  # Exactly at vmin
    @test af_2 ≈ 0.0 atol=1e-6
end

@testset "Target acceleration at limit" begin
    lim = JerkLimiter(; vmax=10.0, amax=20.0, amin=-15.0, jmax=1000.0)

    # Target acceleration equals amax
    profile = calculate_trajectory(lim; pf=5.0, af=20.0)
    @test duration(profile) > 0
    pf, vf, af, _ = profile(duration(profile))
    @test pf ≈ 5.0 atol=1e-6
    @test vf ≈ 0.0 atol=1e-6
    @test af ≈ 20.0 atol=1e-6  # Exactly at amax

    # Target acceleration equals amin (asymmetric)
    profile2 = calculate_trajectory(lim; pf=5.0, af=-15.0)
    pf_2, vf_2, af_2, _ = profile2(duration(profile2))
    @test pf_2 ≈ 5.0 atol=1e-6
    @test vf_2 ≈ 0.0 atol=1e-6
    @test af_2 ≈ -15.0 atol=1e-6  # Exactly at amin
end

@testset "Target velocity and acceleration both at limits" begin
    lim = JerkLimiter(; vmax=5.0, vmin=-3.0, amax=20.0, amin=-15.0, jmax=1000.0)

    # vf = vmax, af = amax
    profile = calculate_trajectory(lim; pf=10.0, vf=5.0, af=20.0)
    @test duration(profile) > 0
    pf, vf, af, _ = profile(duration(profile))
    @test pf ≈ 10.0 atol=1e-6
    @test vf ≈ 5.0 atol=1e-6   # Exactly at vmax
    @test af ≈ 20.0 atol=1e-6  # Exactly at amax

    # vf = vmin, af = amin
    profile2 = calculate_trajectory(lim; pf=-5.0, vf=-3.0, af=-15.0)
    pf_2, vf_2, af_2, _ = profile2(duration(profile2))
    @test pf_2 ≈ -5.0 atol=1e-6
    @test vf_2 ≈ -3.0 atol=1e-6   # Exactly at vmin
    @test af_2 ≈ -15.0 atol=1e-6  # Exactly at amin

    # Mixed: vf = vmax, af = amin
    @test_skip begin
        profile3 = calculate_trajectory(lim; pf=8.0, vf=5.0, af=-15.0)
        pf_3, vf_3, af_3, _ = profile3(duration(profile3))
        @test pf_3 ≈ 8.0 atol=1e-6
        @test vf_3 ≈ 5.0 atol=1e-6    # Exactly at vmax
        @test af_3 ≈ -15.0 atol=1e-6  # Exactly at amin
    end
end

@testset "known failure case" begin
    lim = JerkLimiter(; vmax=5.378090911418406, amax=21.580739221501887, jmax=250.48205176578452)
    p0,v0,a0 = (0.48825150691793306, 0.0, 0.0)
    pf,vf,af = (-1.3966905677540724, 0.0, 0.0)
    calculate_trajectory(lim; p0, v0, a0, pf, vf, af)

    # Asymmetric limits case
    lim2 = JerkLimiter(; vmax=3.63206191841991, vmin=-8.818152052375963, amax=49.05606954525809, amin=-12.662503905178562, jmax=640.562496516735)
    p0,v0,a0 = (-0.43474443878557517, 0.0, 0.0)
    pf,vf,af = (-0.5629190782888567, 0.0, 0.0)
    calculate_trajectory(lim2; p0, v0, a0, pf, vf, af)

    # Asymmetric limits case 2
    lim3 = JerkLimiter(; vmax=6.568803887579402, vmin=-5.873698934812709, amax=16.355405347028825, amin=-28.737465951611775, jmax=663.5605816234907)
    p0,v0,a0 = (0.1675166567390261, 0.0, 0.0)
    pf,vf,af = (0.05958563199475451, 0.0, 0.0)
    calculate_trajectory(lim3; p0, v0, a0, pf, vf, af)

    # Asymmetric limits case 3 - positive displacement with asymmetric limits
    lim4 = JerkLimiter(; vmax=3.63206191841991, vmin=-8.818152052375963, amax=49.05606954525809, amin=-12.662503905178562, jmax=640.562496516735)
    p0,v0,a0 = (-0.5315788502269256, 0.0, 0.0)
    pf,vf,af = (0.6323634408240384, 0.0, 0.0)
    calculate_trajectory(lim4; p0, v0, a0, pf, vf, af)

    # Asymmetric limits case 4 - invalid target velocity (outside allowed range)
    # vf=-0.938 < vmin=-0.887, so this should error
    lim5 = JerkLimiter(; vmax=7.179733007056405, vmin=-0.8871410693710302, amax=33.08407145600285, amin=-6.791789891210365, jmax=563.0719879954868)
    p0,v0,a0 = (-0.22161140141696192, -0.29449744543659623, 0.0)
    pf,vf,af = (-0.2962092284890949, -0.9377700029012783, 0.0)
    @test_throws ErrorException calculate_trajectory(lim5; p0, v0, a0, pf, vf, af)

    # Asymmetric limits case 5 - single DOF with non-zero velocities (C++ gets 0.086s)
    lim6 = JerkLimiter(; vmax=9.893427138005233, vmin=-4.195612678348535, amax=12.078328237289949, amin=-39.36255340641591, jmax=507.34452830286716)
    p0,v0,a0 = (1.1027816486815265, -1.5479008715462181, 0.5968646706547006)
    pf,vf,af = (0.9930833291182205, -0.8985940436554194, 1.4276116384379833)
    prof = calculate_trajectory(lim6; p0, v0, a0, pf, vf, af)
    @test duration(prof) ≈ 0.0859329038189459 rtol=1e-6

    # Asymmetric limits case 6 - Multi-DOF synchronization
    lims_multi = [
        JerkLimiter(; vmax=9.893427138005233, vmin=-4.195612678348535, amax=12.078328237289949, amin=-39.36255340641591, jmax=507.34452830286716),
        JerkLimiter(; vmax=1.1765171949042518, vmin=-0.9686298251680693, amax=47.31266645698063, amin=-8.68587820773526, jmax=927.2892655068578),
    ]
    p0_multi = [1.1027816486815265, -0.24704027944893318]
    v0_multi = [-1.5479008715462181, -0.5009063759259685]
    a0_multi = [0.5968646706547006, -1.2339254009498102]
    pf_multi = [0.9930833291182205, -0.4813668823155839]
    vf_multi = [-0.8985940436554194, -0.06858932645285804]
    af_multi = [1.4276116384379833, 3.7285908168024676]
    profiles = calculate_trajectory(lims_multi; p0=p0_multi, v0=v0_multi, a0=a0_multi, pf=pf_multi, vf=vf_multi, af=af_multi)
    @test duration(profiles[1]) ≈ duration(profiles[2])  # synchronized
    @test duration(profiles[1]) ≈ 0.315 rtol=0.01  # approximately C++ result

    # Asymmetric limits case 7 - Multi-DOF synchronization, DOF 0 is limiting
    # DOF 0 min duration: 0.782, DOF 1 min duration: 0.379
    # C++ synchronizes both to 0.782 successfully
    lims_multi7 = [
        JerkLimiter(; vmax=9.893427138005233, vmin=-4.195612678348535, amax=12.078328237289949, amin=-39.36255340641591, jmax=507.34452830286716),
        JerkLimiter(; vmax=13.291034736601903, vmin=-3.8842062448080252, amax=31.775176417549602, amin=-8.718562920546741, jmax=823.3195381668564),
    ]
    p0_multi7 = [-1.6478335226310068, -1.8288458926404993]
    v0_multi7 = [-0.559765177860057, 0.28380553260563673]
    a0_multi7 = [0.0, 0.0]
    pf_multi7 = [0.5109185999108988, -1.1461809393561588]
    vf_multi7 = [0.10670168637106899, 0.8160643008030284]
    af_multi7 = [0.0, 0.0]
    profiles7 = calculate_trajectory(lims_multi7; p0=p0_multi7, v0=v0_multi7, a0=a0_multi7, pf=pf_multi7, vf=vf_multi7, af=af_multi7)
    @test duration(profiles7[1]) ≈ duration(profiles7[2])  # synchronized
    @test duration(profiles7[1]) ≈ 0.78215181077265 rtol=1e-6  # C++ result
end

#=============================================================================
 Velocity Control Tests (ruckig_velocity.jl)
=============================================================================#

@testset "Velocity control: JerkLimiter basic" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Accelerate from rest to vf=5
    profile = calculate_velocity_trajectory(lim; vf=5.0)
    @test duration(profile) > 0
    _, v0, a0, _ = profile(0.0)
    @test v0 ≈ 0.0
    @test a0 ≈ 0.0
    _, vf, af, _ = profile(duration(profile))
    @test vf ≈ 5.0 atol=1e-6
    @test af ≈ 0.0 atol=1e-6

    # Decelerate from rest to negative velocity
    profile2 = calculate_velocity_trajectory(lim; vf=-5.0)
    @test duration(profile2) > 0
    _, vf2, af2, _ = profile2(duration(profile2))
    @test vf2 ≈ -5.0 atol=1e-6
    @test af2 ≈ 0.0 atol=1e-6

    # Start with initial velocity
    profile3 = calculate_velocity_trajectory(lim; v0=3.0, vf=8.0)
    @test duration(profile3) > 0
    _, v0_3, _, _ = profile3(0.0)
    @test v0_3 ≈ 3.0
    _, vf3, af3, _ = profile3(duration(profile3))
    @test vf3 ≈ 8.0 atol=1e-6
    @test af3 ≈ 0.0 atol=1e-6
end

@testset "Velocity control: JerkLimiter with initial/final acceleration" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Start with initial acceleration
    profile = calculate_velocity_trajectory(lim; v0=0.0, a0=10.0, vf=5.0)
    @test duration(profile) > 0
    _, v0, a0, _ = profile(0.0)
    @test v0 ≈ 0.0
    @test a0 ≈ 10.0
    _, vf, af, _ = profile(duration(profile))
    @test vf ≈ 5.0 atol=1e-6
    @test af ≈ 0.0 atol=1e-6

    # End with non-zero target acceleration
    profile2 = calculate_velocity_trajectory(lim; v0=0.0, vf=5.0, af=10.0)
    @test duration(profile2) > 0
    _, vf2, af2, _ = profile2(duration(profile2))
    @test vf2 ≈ 5.0 atol=1e-6
    @test af2 ≈ 10.0 atol=1e-6

    # Both initial and final acceleration
    profile3 = calculate_velocity_trajectory(lim; v0=2.0, a0=5.0, vf=7.0, af=15.0)
    @test duration(profile3) > 0
    _, v0_3, a0_3, _ = profile3(0.0)
    @test v0_3 ≈ 2.0
    @test a0_3 ≈ 5.0
    _, vf3, af3, _ = profile3(duration(profile3))
    @test vf3 ≈ 7.0 atol=1e-6
    @test af3 ≈ 15.0 atol=1e-6
end

@testset "Velocity control: JerkLimiter time-synchronized" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Specify target time
    tf_target = 0.5
    profile = calculate_velocity_trajectory(lim; vf=5.0, tf=tf_target)
    @test duration(profile) ≈ tf_target atol=1e-6
    _, vf, af, _ = profile(duration(profile))
    @test vf ≈ 5.0 atol=1e-6
    @test af ≈ 0.0 atol=1e-6

    # Negative velocity with specified time
    profile2 = calculate_velocity_trajectory(lim; vf=-3.0, tf=0.4)
    @test duration(profile2) ≈ 0.4 atol=1e-6
    _, vf2, _, _ = profile2(duration(profile2))
    @test vf2 ≈ -3.0 atol=1e-6
end

@testset "Velocity control: JerkLimiter asymmetric limits" begin
    lim = JerkLimiter(; vmax=10.0, vmin=-5.0, amax=50.0, amin=-30.0, jmax=1000.0)

    # Positive velocity change
    profile = calculate_velocity_trajectory(lim; vf=8.0)
    @test duration(profile) > 0
    _, vf, af, _ = profile(duration(profile))
    @test vf ≈ 8.0 atol=1e-6
    @test af ≈ 0.0 atol=1e-6

    # Negative velocity change (constrained by vmin)
    profile2 = calculate_velocity_trajectory(lim; vf=-4.0)
    @test duration(profile2) > 0
    _, vf2, af2, _ = profile2(duration(profile2))
    @test vf2 ≈ -4.0 atol=1e-6
    @test af2 ≈ 0.0 atol=1e-6
end

@testset "Velocity control: JerkLimiter limits respected" begin
    lim = JerkLimiter(; vmax=5.0, amax=20.0, jmax=500.0)

    profile = calculate_velocity_trajectory(lim; v0=0.0, vf=4.0)

    # Check trajectory stays within limits
    for t in range(0, duration(profile), length=50)
        _, v, a, j = profile(t)
        @test lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6
        @test lim.amin - 1e-6 <= a <= lim.amax + 1e-6
        @test -lim.jmax - 1e-6 <= j <= lim.jmax + 1e-6
    end
end

@testset "Velocity control: JerkLimiter initial state outside limits" begin
    lim = JerkLimiter(; vmax=10.0, amax=20.0, jmax=1000.0)

    # Initial acceleration exceeds amax - requires brake
    profile = calculate_velocity_trajectory(lim; a0=35.0, vf=5.0)
    @test duration(profile) > 0
    _, _, a0, _ = profile(0.0)
    @test a0 ≈ 35.0  # Starts above amax
    _, vf, af, _ = profile(duration(profile))
    @test vf ≈ 5.0 atol=1e-6
    @test af ≈ 0.0 atol=1e-6
end

@testset "Velocity control: AccelerationLimiter basic" begin
    lim = AccelerationLimiter(; vmax=100.0, amax=50.0)

    # Accelerate from rest
    profile = calculate_velocity_trajectory(lim; vf=10.0)
    @test duration(profile) > 0
    _, v0, _, _ = profile(0.0)
    @test v0 ≈ 0.0
    _, vf, _, _ = profile(duration(profile))
    @test vf ≈ 10.0 atol=1e-6

    # Decelerate
    profile2 = calculate_velocity_trajectory(lim; v0=10.0, vf=0.0)
    @test duration(profile2) > 0
    _, vf2, _, _ = profile2(duration(profile2))
    @test vf2 ≈ 0.0 atol=1e-6

    # Negative velocity
    profile3 = calculate_velocity_trajectory(lim; vf=-8.0)
    @test duration(profile3) > 0
    _, vf3, _, _ = profile3(duration(profile3))
    @test vf3 ≈ -8.0 atol=1e-6
end

@testset "Velocity control: AccelerationLimiter time-synchronized" begin
    lim = AccelerationLimiter(; vmax=100.0, amax=50.0)

    # Specify target time
    tf_target = 0.5
    profile = calculate_velocity_trajectory(lim; vf=10.0, tf=tf_target)
    @test duration(profile) ≈ tf_target atol=1e-6
    _, vf, _, _ = profile(duration(profile))
    @test vf ≈ 10.0 atol=1e-6

    # Longer time requires lower acceleration
    profile2 = calculate_velocity_trajectory(lim; vf=10.0, tf=1.0)
    @test duration(profile2) ≈ 1.0 atol=1e-6
    _, vf2, _, _ = profile2(duration(profile2))
    @test vf2 ≈ 10.0 atol=1e-6
end

@testset "Velocity control: AccelerationLimiter asymmetric limits" begin
    lim = AccelerationLimiter(; vmax=100.0, amax=50.0, amin=-30.0)

    # Positive acceleration
    profile = calculate_velocity_trajectory(lim; vf=10.0)
    @test duration(profile) > 0

    # Negative acceleration (uses amin)
    profile2 = calculate_velocity_trajectory(lim; vf=-10.0)
    @test duration(profile2) > 0
    _, vf2, _, _ = profile2(duration(profile2))
    @test vf2 ≈ -10.0 atol=1e-6
end