using TrajectoryLimiters
using Test

# Deterministic pseudo-random generator (LCG) to avoid a Random test dependency.
# Returns a closure producing uniform values in [0, 1).
function make_lcg(seed::UInt64)
    state = Ref(seed)
    function ()
        state[] = state[] * 0x5851f42d4c957f2d + 0x14057b7ef767814f
        return (state[] >> 11) / 9.007199254740992e15
    end
end

# Collect the Roots iterator via explicit iteration (Base.length reports the raw
# stored count including negative roots, so collect cannot be used directly)
function roots_to_vector(r)
    vals = Float64[]
    for x in r
        push!(vals, x)
    end
    vals
end

@testset "Regression tests for C++-alignment fixes" begin

    @testset "Brake with nonzero target velocity" begin
        # v0 outside vmax triggers a brake; nonzero vf triggers collection mode.
        # This combination used to throw a TypeError from a mistyped brake kwarg.
        lim = JerkLimiter(; vmax=1.0, amax=1.0, jmax=1.0)
        prof = calculate_trajectory(lim; p0=0.0, pf=5.0, v0=3.0, vf=0.5)
        @test prof isa RuckigProfile
        @test prof.brake_duration > 0
        p, v, a, _ = evaluate_at(prof, duration(prof))
        @test p ≈ 5.0 atol=1e-6
        @test v ≈ 0.5 atol=1e-6
        @test a ≈ 0.0 atol=1e-6

        # Same input through the multi-DOF entry point
        profs = calculate_trajectory([lim]; pf=[5.0], v0=[3.0], vf=[0.5])
        @test length(profs) == 1
        p, v, a, _ = evaluate_at(profs[1], duration(profs[1]))
        @test p ≈ 5.0 atol=1e-6
        @test v ≈ 0.5 atol=1e-6
    end

    @testset "Zero-limits coast with negative displacement" begin
        # jmax of zero takes the zero-limits path; the negative-displacement case
        # used to be rejected because pd-swapped limits were passed to the
        # direction-unaware single-step check
        lim = JerkLimiter(; vmax=10.0, vmin=-10.0, amax=1.0, amin=-1.0, jmax=0.0)
        prof_neg = calculate_trajectory(lim; p0=0.0, pf=-1.0, v0=-1.0, vf=-1.0)
        @test duration(prof_neg) ≈ 1.0
        p, v, _, _ = evaluate_at(prof_neg, duration(prof_neg))
        @test p ≈ -1.0 atol=1e-6
        @test v ≈ -1.0 atol=1e-6

        # Mirrored positive case must give the same duration
        prof_pos = calculate_trajectory(lim; p0=0.0, pf=1.0, v0=1.0, vf=1.0)
        @test duration(prof_pos) ≈ duration(prof_neg)
    end

    @testset "Multi-DOF synchronization with a braking DOF" begin
        # DOF 1 starts above its velocity limit and needs a brake pre-trajectory;
        # the brake used to be dropped during step-2 re-timing
        lims = [
            JerkLimiter(; vmax=1.0, amax=2.0, jmax=20.0),
            JerkLimiter(; vmax=10.0, amax=20.0, jmax=200.0),
        ]
        pf = [3.0, 1.0]
        v0 = [2.5, 0.0]
        profs = calculate_trajectory(lims; pf, v0)
        @test profs[1].brake_duration > 0

        T1 = duration(profs[1])
        T2 = duration(profs[2])
        @test T1 ≈ T2 atol=1e-9

        for i in 1:2
            p, v, a, _ = evaluate_at(profs[i], T1)
            @test p ≈ pf[i] atol=1e-6
            @test v ≈ 0.0 atol=1e-6
            @test a ≈ 0.0 atol=1e-6
        end

        # After the brake, DOF 1 must respect its velocity limits
        bd = profs[1].brake_duration
        vs = [evaluate_at(profs[1], t).v for t in range(bd, T1, length=2000)]
        @test maximum(vs) <= lims[1].vmax + 1e-6
        @test minimum(vs) >= lims[1].vmin - 1e-6
    end

    @testset "AccelerationLimiter evaluate_at accelerations" begin
        # Per-phase accelerations from evaluate_at used to be wrong
        lim = AccelerationLimiter(; vmax=1.0, amax=2.0)
        prof = calculate_trajectory(lim; pf=3.0)
        Ttot = duration(prof)
        bd = prof.brake_duration

        # Three-phase profile: accelerate, coast, decelerate
        @test prof.t[1] > 0
        @test prof.t[2] > 0
        @test prof.t[3] > 0

        t_acc = bd + prof.t[1] / 2
        t_coast = bd + prof.t_sum[1] + prof.t[2] / 2
        t_dec = bd + prof.t_sum[2] + prof.t[3] / 2

        @test evaluate_at(prof, t_acc).a ≈ lim.amax atol=1e-9
        @test abs(evaluate_at(prof, t_coast).v) ≈ lim.vmax atol=1e-9
        @test evaluate_at(prof, t_coast).a ≈ 0.0 atol=1e-9
        @test evaluate_at(prof, t_dec).a ≈ lim.amin atol=1e-9

        # a(t) must match d/dt v(t) away from phase boundaries
        h = 1e-5
        breaks = [bd; bd .+ cumsum(collect(prof.t))]
        fd_err = 0.0
        for t in range(2h, Ttot - 2h, length=400)
            any(b -> abs(t - b) < 2h, breaks) && continue
            fd = (evaluate_at(prof, t + h).v - evaluate_at(prof, t - h).v) / (2h)
            fd_err = max(fd_err, abs(evaluate_at(prof, t).a - fd))
        end
        @test fd_err < 1e-6
    end

    @testset "Velocity interface brake displacement" begin
        # a0 outside amax forces a brake; the brake displacement used to be
        # dropped from the position samples
        lim = JerkLimiter(; vmax=1.0, amax=1.0, jmax=1.0)
        prof = calculate_velocity_trajectory(lim; v0=0.0, a0=3.0, vf=0.0)
        @test prof.brake_duration > 0

        Ttot = duration(prof)
        ts = range(0, Ttot, length=20_001)
        _, vs, _, _ = evaluate_at(prof, ts)
        dt = step(ts)
        pd_int = dt * (sum(vs) - (vs[1] + vs[end]) / 2)

        p_end = evaluate_at(prof, Ttot).p
        p_start = evaluate_at(prof, 0.0).p
        @test p_end - p_start ≈ pd_int atol=1e-4
        @test prof.pf ≈ p_end atol=1e-9
    end

    @testset "Velocity interface min-time optimality" begin
        # The solver used to return the first valid strategy instead of the
        # minimal-duration one. If the returned duration is minimal, requesting a
        # slightly shorter tf must be infeasible
        lim = JerkLimiter(; vmax=1.0, amax=1.0, jmax=1.0)
        rand01 = make_lcg(UInt64(1))
        n_checked = 0
        for _ in 1:200
            v0 = 1.8 * rand01() - 0.9
            a0 = 1.8 * rand01() - 0.9
            vf = 1.8 * rand01() - 0.9
            af = 1.8 * rand01() - 0.9
            prof = calculate_velocity_trajectory(lim; v0, a0, vf, af)
            T = duration(prof)
            _, v, a, _ = evaluate_at(prof, T)
            @test v ≈ vf atol=1e-6
            @test a ≈ af atol=1e-6
            T < 1e-2 && continue
            n_checked += 1
            @test_throws Exception calculate_velocity_trajectory(lim; v0, a0, vf, af, tf=0.9T)
        end
        @test n_checked > 150
    end

    @testset "Waypoint trajectory sample integrity" begin
        # The first sample of each segment (the exact waypoint state) used to be
        # dropped; segment durations are generally not multiples of Ts
        lim = JerkLimiter(; vmax=1.0, amax=5.0, jmax=50.0)
        Ts = 0.001
        waypoints = [(p=0.0,), (p=1.0,), (p=0.5,)]
        ts, ps, vs, as, js = calculate_waypoint_trajectory(lim, waypoints, Ts)

        @test all(>(0), diff(ts))
        @test maximum(diff(ts)) <= 1.5Ts

        @test ps[1] ≈ 0.0 atol=1e-6
        @test any(p -> isapprox(p, 1.0, atol=1e-6), ps)
        @test any(p -> isapprox(p, 0.5, atol=1e-6), ps)
        @test ps[end] ≈ 0.5 atol=1e-6
    end

    @testset "get_first_state_at_position returns first crossing" begin
        # High initial velocity forces an overshoot past pf; the earliest
        # crossing must be returned, not a later one
        lim = JerkLimiter(; vmax=5.0, amax=10.0, jmax=100.0)
        pf = 1.0
        prof = calculate_trajectory(lim; pf, v0=4.5)
        ext = get_position_extrema(prof)
        @test ext.max > pf + 1e-3

        found, t_first = get_first_state_at_position(prof, pf)
        @test found
        @test evaluate_at(prof, t_first).p ≈ pf atol=1e-6
        # The first crossing happens before the trajectory ends at pf
        @test t_first < duration(prof) - 1e-6

        # No crossing on a fine grid before the reported time
        ts = range(0, duration(prof), length=100_000)
        ps, _, _, _ = evaluate_at(prof, ts)
        i_cross = findfirst(>=(pf - 1e-9), ps)
        @test i_cross !== nothing
        @test abs(ts[i_cross] - t_first) <= 2 * step(ts)
    end

    @testset "Cubic solver domain robustness and Roots iteration" begin
        r = TrajectoryLimiters.Roots{Float64}()

        # Triple root at unity with the constant coefficient perturbed by ulps;
        # the acos argument used to be able to drift outside its domain
        for k in -5:5
            d = -1.0 + k * eps(1.0)
            TrajectoryLimiters.solve_cubic_real!(r, 1.0, -3.0, 3.0, d)
            vals = roots_to_vector(r)
            @test !isempty(vals)
            @test all(x -> abs(x - 1) < 1e-3, vals)
        end

        # Near-double root: (x-1)^2 (x-1-delta) with shrinking delta
        for k in 1:8, s in (-1.0, 1.0)
            delta = s * 10.0^-k
            TrajectoryLimiters.solve_cubic_real!(r, 1.0, -(3 + delta), 3 + 2delta, -(1 + delta))
            vals = roots_to_vector(r)
            @test !isempty(vals)
            @test all(x -> abs(x - 1) <= abs(delta) + 1e-3, vals)
        end

        # Symmetric near-triple roots (x-1)^3 - delta^2 (x-1) with negative
        # discriminant exercise the trigonometric branch
        for delta in (0.02, 0.05, 0.1)
            TrajectoryLimiters.solve_cubic_real!(r, 1.0, -3.0, 3 - delta^2, -1 + delta^2)
            vals = roots_to_vector(r)
            @test length(vals) == 3
            @test all(x -> abs(x - 1) <= delta + 1e-6, vals)
        end

        # Iteration must return non-negative roots in ascending order
        TrajectoryLimiters.clear!(r)
        for x in (3.0, -1.0, 0.5, 2.0)
            push!(r, x)
        end
        @test roots_to_vector(r) == [0.5, 2.0, 3.0]
    end

    @testset "Multi-DOF step-2 re-timing with nonzero target velocities" begin
        # Nonzero vf takes collection mode and forces step-2 re-timing of the
        # faster DOF; the duration gate and step-2 fallbacks must cooperate
        lims = [
            JerkLimiter(; vmax=2.0, amax=5.0, jmax=50.0),
            JerkLimiter(; vmax=3.0, amax=8.0, jmax=80.0),
        ]
        pf = [1.0, 0.4]
        vf = [0.5, 0.2]
        profs = calculate_trajectory(lims; pf, vf)

        T1 = duration(profs[1])
        T2 = duration(profs[2])
        @test T1 ≈ T2 atol=1e-9

        for i in 1:2
            p, v, a, _ = evaluate_at(profs[i], T1)
            @test p ≈ pf[i] atol=1e-6
            @test v ≈ vf[i] atol=1e-6

            lim = lims[i]
            _, vs, as, js = evaluate_at(profs[i], range(0, T1, length=2000))
            @test all(v -> lim.vmin - 1e-6 <= v <= lim.vmax + 1e-6, vs)
            @test all(a -> lim.amin - 1e-6 <= a <= lim.amax + 1e-6, as)
            @test all(j -> abs(j) <= lim.jmax + 1e-6, js)
        end
    end

end
