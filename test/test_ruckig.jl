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

    @test profile.t_sum[7] > 0
    @test profile.p[1] ≈ 0.0
    @test profile.v[1] ≈ 0.0
    @test profile.a[1] ≈ 0.0
    @test profile.p[8] ≈ 1.0 atol=1e-6
    @test profile.v[8] ≈ 0.0 atol=1e-6
    @test profile.a[8] ≈ 0.0 atol=1e-6

    # Check trajectory stays within limits
    for t in range(0, profile.t_sum[7], length=100)
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

    @test profile.t_sum[7] > 0
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
    T_total = profile.t_sum[7]
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

    @test profile.t_sum[7] > 0
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

    @test profile.t_sum[7] > 0

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

    for t in Ts:Ts:profile.t_sum[7]
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

        @test profile.t_sum[7] > 0
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
        for t in range(0, profile.t_sum[7], length=100)
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

    ts = range(0, profile.t_sum[7], length=50)
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
    pos, vel, acc, jerk, ts = evaluate_dt(profile)

    @test ts[1] == 0.0
    @test ts[end] <= profile.t_sum[7]
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
