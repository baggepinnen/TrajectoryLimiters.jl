# Tests for the phase-synchronization mode of the multi-DOF position interface
# (port of the Synchronization::Phase mode of C++ ruckig)

using TrajectoryLimiters
using TrajectoryLimiters: is_phase_synchronized

# Maximum orthogonal deviation of the sampled multi-DOF path from the straight
# line (chord) between the start and end points
function max_line_deviation(profiles, p0v, pfv; n = 1001)
    d = collect(Float64, pfv .- p0v)
    d ./= sqrt(sum(abs2, d))
    T = duration(profiles[1])
    worst = 0.0
    for t in range(0, T, length = n)
        r = [evaluate_at(p, t).p for p in profiles] .- p0v
        proj = sum(r .* d)
        worst = max(worst, sqrt(sum(abs2, r .- proj .* d)))
    end
    worst
end

same_timing(a, b) = a.t == b.t && a.t_sum == b.t_sum && a.j == b.j &&
                    a.brake_duration == b.brake_duration

@testset "Phase synchronization" begin

    @testset "canonical copy-scale case" begin
        lims = [JerkLimiter(vmax = 1.0, amax = 2.0, jmax = 20.0) for _ in 1:3]
        pf = [1.0, 2.0, 0.5]

        pt = calculate_trajectory(lims; pf)
        pp = calculate_trajectory(lims; pf, synchronization = :phase)

        # Phase sync keeps the time-synchronized (optimal) duration
        @test duration(pp[1]) ≈ duration(pt[1]) atol = 1e-12
        @test all(p -> isapprox(duration(p), duration(pp[1]); atol = 1e-12), pp)

        @test is_phase_synchronized(pp)
        @test !is_phase_synchronized(pt)

        @test max_line_deviation(pp, zeros(3), pf) < 1e-9
        @test max_line_deviation(pt, zeros(3), pf) > 1e-3  # time sync is curved here

        # Targets reached and per-DOF limits respected
        for (i, p) in enumerate(pp)
            st = evaluate_at(p, duration(p))
            @test st.p ≈ pf[i] atol = 1e-6
            @test abs(st.v) < 1e-6
            @test abs(st.a) < 1e-6
            for t in range(0, duration(p), length = 500)
                s = evaluate_at(p, t)
                @test -1.0 - 1e-9 <= s.v <= 1.0 + 1e-9
                @test -2.0 - 1e-9 <= s.a <= 2.0 + 1e-9
                @test abs(s.j) <= 20.0 + 1e-9
            end
        end

        # phase_strict succeeds for a colinear input
        ps = calculate_trajectory(lims; pf, synchronization = :phase_strict)
        @test is_phase_synchronized(ps)
    end

    @testset "non-colinear input falls back to time sync" begin
        lims = [JerkLimiter(vmax = 1.0, amax = 2.0, jmax = 20.0) for _ in 1:3]
        pf = [1.0, 2.0, 0.5]
        v0 = [0.5, 0.0, 0.0]  # not proportional to pf - p0

        pt = calculate_trajectory(lims; pf, v0)
        pp = calculate_trajectory(lims; pf, v0, synchronization = :phase)

        @test !is_phase_synchronized(pp)
        @test all(same_timing(pp[i], pt[i]) for i in 1:3)  # identical to :time output
        @test_throws ErrorException calculate_trajectory(lims; pf, v0, synchronization = :phase_strict)
    end

    @testset "mixed binding constraints fall back (C++-faithful)" begin
        # DOF 1 is velocity-bound, DOF 2 jerk-bound: the limiting DOF's scaled
        # profile violates the other DOF's jerk limit, so phase sync is not
        # possible and the duration-optimal (curved) trajectory is returned
        lims = [JerkLimiter(vmax = 1.0, amax = 50.0, jmax = 1000.0),
                JerkLimiter(vmax = 10.0, amax = 50.0, jmax = 20.0)]
        pf = [1.0, 1.0]

        pt = calculate_trajectory(lims; pf)
        pp = calculate_trajectory(lims; pf, synchronization = :phase)

        @test !is_phase_synchronized(pp)
        @test duration(pp[1]) ≈ duration(pt[1]) atol = 1e-12
        @test all(same_timing(pp[i], pt[i]) for i in 1:2)
        @test_throws ErrorException calculate_trajectory(lims; pf, synchronization = :phase_strict)
    end

    @testset "colinear nonzero boundary states" begin
        lims = [JerkLimiter(vmax = 1.0, amax = 2.0, jmax = 20.0) for _ in 1:3]
        pf = [1.0, 2.0, 0.5]
        d = pf  # p0 = 0, so the direction vector is pf itself
        v0 = 0.3 .* d
        vf = 0.1 .* d

        pp = calculate_trajectory(lims; pf, v0, vf, synchronization = :phase)
        pp0 = calculate_trajectory(lims; pf, synchronization = :phase)

        @test is_phase_synchronized(pp)
        @test max_line_deviation(pp, zeros(3), pf) < 1e-9
        @test duration(pp[1]) != duration(pp0[1])  # boundary states took effect
        for (i, p) in enumerate(pp)
            st = evaluate_at(p, duration(p))
            @test st.p ≈ pf[i] atol = 1e-6
            @test st.v ≈ vf[i] atol = 1e-6
        end
    end

    @testset "negative direction and asymmetric limits" begin
        lims = [JerkLimiter(vmax = 1.0, vmin = -0.6, amax = 2.0, amin = -3.0, jmax = 20.0) for _ in 1:2]
        pf = [-1.0, -2.0]

        pp = calculate_trajectory(lims; pf, synchronization = :phase)
        @test is_phase_synchronized(pp)
        @test max_line_deviation(pp, zeros(2), pf) < 1e-9
        for (i, p) in enumerate(pp)
            @test evaluate_at(p, duration(p)).p ≈ pf[i] atol = 1e-6
            for t in range(0, duration(p), length = 500)
                s = evaluate_at(p, t)
                @test -0.6 - 1e-9 <= s.v <= 1.0 + 1e-9
                @test -3.0 - 1e-9 <= s.a <= 2.0 + 1e-9
            end
        end
    end

    @testset "zero-travel DOF" begin
        lims = [JerkLimiter(vmax = 1.0, amax = 2.0, jmax = 20.0) for _ in 1:3]
        pf = [1.0, 0.0, 0.5]

        pp = calculate_trajectory(lims; pf, synchronization = :phase)
        @test is_phase_synchronized(pp)
        for t in range(0, duration(pp[2]), length = 200)
            @test abs(evaluate_at(pp[2], t).p) < 1e-12
        end

        # A moving zero-travel DOF is not colinear: falls back
        pnc = calculate_trajectory(lims; pf, v0 = [0.0, 0.3, 0.0], synchronization = :phase)
        @test !is_phase_synchronized(pnc)
    end

    @testset "brake input falls back" begin
        lims = [JerkLimiter(vmax = 1.0, amax = 2.0, jmax = 20.0) for _ in 1:2]
        pf = [1.0, 2.0]
        v0 = [0.75, 1.5]  # colinear with pf, but DOF 2 starts outside vmax

        pp = calculate_trajectory(lims; pf, v0, synchronization = :phase)
        @test !is_phase_synchronized(pp)
        @test all(p -> isapprox(duration(p), duration(pp[1]); atol = 1e-9), pp)
        for (i, p) in enumerate(pp)
            @test evaluate_at(p, duration(p)).p ≈ pf[i] atol = 1e-6
        end
    end

    @testset "single DOF degenerates to the scalar solution" begin
        lim = JerkLimiter(vmax = 1.0, amax = 2.0, jmax = 20.0)
        pp = calculate_trajectory([lim]; pf = [1.0], synchronization = :phase)
        ps = calculate_trajectory(lim; pf = 1.0)
        @test duration(pp[1]) == duration(ps)
        @test pp[1].t == ps.t
        @test is_phase_synchronized(pp)
    end

    @testset "argument validation" begin
        lims = [JerkLimiter(vmax = 1.0, amax = 2.0, jmax = 20.0) for _ in 1:2]
        @test_throws ArgumentError calculate_trajectory(lims; pf = [1.0, 2.0], synchronization = :none)
    end
end
