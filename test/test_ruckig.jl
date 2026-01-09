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
    lims = [
        JerkLimiter(; vmax=10.0*rand(), amax=50.0*rand(), jmax=1000.0*rand()) for i = 1:7
    ]
    waypoints = [(p = randn(7),) for i = 1:1000]
    @test_nowarn calculate_waypoint_trajectory(lims, waypoints, 0.001)
end


@testset "known failure case" begin
    lim = JerkLimiter(; vmax=5.378090911418406, amax=21.580739221501887, jmax=250.48205176578452)
    p0,v0,a0 = (0.48825150691793306, 0.0, 0.0)
    pf,vf,af = (-1.3966905677540724, 0.0, 0.0)
    calculate_trajectory(lim; p0, v0, a0, pf, vf, af)
end