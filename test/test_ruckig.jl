using TrajectoryLimiters
using Test

@testset "JerkLimiter constructors" begin
    # Symmetric constructor
    lim = JerkLimiter(10.0, 50.0, 1000.0)
    @test lim.vmax == 10.0
    @test lim.vmin == -10.0
    @test lim.amax == 50.0
    @test lim.amin == -50.0
    @test lim.jmax == 1000.0

    # Directional constructor
    lim2 = JerkLimiter(10.0, -5.0, 50.0, -30.0, 1000.0)
    @test lim2.vmax == 10.0
    @test lim2.vmin == -5.0
    @test lim2.amax == 50.0
    @test lim2.amin == -30.0

    # Type promotion
    lim3 = JerkLimiter(10, 50, 1000)
    @test lim3.vmax isa Int
end

@testset "Simple trajectory: rest to rest" begin
    lim = JerkLimiter(10.0, 50.0, 1000.0)

    # Move from 0 to 1
    profile = calculate_trajectory(lim, 0.0, 0.0, 0.0, 1.0)

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
    lim = JerkLimiter(10.0, 50.0, 1000.0)

    # Move from 1 to 0
    profile = calculate_trajectory(lim, 1.0, 0.0, 0.0, 0.0)

    @test profile.t_sum[7] > 0
    @test profile.p[1] ≈ 1.0
    @test profile.p[8] ≈ 0.0 atol=1e-6
    @test profile.v[8] ≈ 0.0 atol=1e-6
    @test profile.a[8] ≈ 0.0 atol=1e-6
end

@testset "evaluate_at boundary conditions" begin
    lim = JerkLimiter(10.0, 50.0, 1000.0)
    profile = calculate_trajectory(lim, 0.0, 0.0, 0.0, 1.0)

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
    lim = JerkLimiter(10.0, 50.0, 1000.0)

    # Start with v0 = 5, move forward
    profile = calculate_trajectory(lim, 0.0, 5.0, 0.0, 2.0)

    @test profile.t_sum[7] > 0
    @test profile.p[1] ≈ 0.0
    @test profile.v[1] ≈ 5.0
    @test profile.p[8] ≈ 2.0 atol=1e-6
    @test profile.v[8] ≈ 0.0 atol=1e-6
end

@testset "Velocity-limited trajectory" begin
    # Use low velocity limit to force velocity plateau
    lim = JerkLimiter(2.0, 50.0, 1000.0)

    # Long distance should hit velocity limit
    profile = calculate_trajectory(lim, 0.0, 0.0, 0.0, 10.0)

    @test profile.t_sum[7] > 0

    # Check max velocity reached is close to limit
    max_v = maximum(profile.v)
    @test max_v ≈ lim.vmax atol=0.1

    # Final state
    @test profile.p[8] ≈ 10.0 atol=1e-6
    @test profile.v[8] ≈ 0.0 atol=1e-6
end

@testset "Profile duration monotonicity" begin
    lim = JerkLimiter(10.0, 50.0, 1000.0)

    # Longer distances should take longer
    p1 = calculate_trajectory(lim, 0.0, 0.0, 0.0, 1.0)
    p2 = calculate_trajectory(lim, 0.0, 0.0, 0.0, 2.0)
    p3 = calculate_trajectory(lim, 0.0, 0.0, 0.0, 4.0)

    @test p1.t_sum[7] < p2.t_sum[7] < p3.t_sum[7]
end

@testset "Continuous trajectory sampling" begin
    lim = JerkLimiter(10.0, 50.0, 1000.0)
    profile = calculate_trajectory(lim, 0.0, 0.0, 0.0, 1.0)

    # Sample finely and check continuity
    dt = 0.001
    prev_p, prev_v, prev_a, _ = evaluate_at(profile, 0.0)

    for t in dt:dt:profile.t_sum[7]
        p, v, a, j = evaluate_at(profile, t)

        # Position should be monotonic for this trajectory
        @test p >= prev_p - 1e-10

        # Changes should be smooth (bounded by limits * dt with margin)
        @test abs(v - prev_v) < (lim.amax + lim.jmax * dt) * dt * 2
        @test abs(a - prev_a) < lim.jmax * dt * 2

        prev_p, prev_v, prev_a = p, v, a
    end
end
