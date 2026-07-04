# Randomized property-based tests for the Ruckig-style jerk-limited trajectory
# generator. This file is include()d from runtests.jl inside a parent testset.
#
# Each sub-testset seeds the global RNG for reproducibility. Violations are
# accumulated per property and asserted in aggregate; on failure, the exact
# offending inputs are printed (repr round-trips Float64) as reproducers.

using TrajectoryLimiters
using Test
using Random

#=============================================================================
 Helpers (prefixed _pt_ to avoid clashes with other included test files)
=============================================================================#

_pt_rand_in(rng, lo, hi) = lo + (hi - lo) * rand(rng)
_pt_loguniform(rng, lo, hi) = exp(_pt_rand_in(rng, log(lo), log(hi)))

"Random limiter with log-uniform limits; asymmetric vmin/amin half of the time."
function _pt_rand_limiter(rng)
    vmax = _pt_loguniform(rng, 0.05, 50.0)
    amax = _pt_loguniform(rng, 0.05, 50.0)
    jmax = _pt_loguniform(rng, 0.05, 50.0)
    if rand(rng) < 0.5
        vmin = -vmax * _pt_rand_in(rng, 0.3, 1.5)
        amin = -amax * _pt_rand_in(rng, 0.3, 1.5)
    else
        vmin = -vmax
        amin = -amax
    end
    JerkLimiter(; vmax, amax, jmax, vmin, amin)
end

"Value strictly outside [lo, hi] (up to 2.5x beyond the limit), random side."
_pt_outside(rng, lo, hi) =
    rand(rng) < 0.5 ? hi * _pt_rand_in(rng, 1.05, 2.5) : lo * _pt_rand_in(rng, 1.05, 2.5)

"""
Random target acceleration that is dynamically feasible for the target velocity
`vf`. Ending at (vf, af) with af > 0 forces the velocity to dip to
vf - af^2/(2 jmax) just before the final jerk ramp (and symmetrically toward
vmax for af < 0), so |af| must satisfy af^2 <= 2 jmax (vf - vmin) resp.
af^2 <= 2 jmax (vmax - vf). The C++ Ruckig reference rejects targets outside
this set in input validation; this port throws for them. `frac` leaves a
safety margin to the boundary.
"""
function _pt_rand_af(rng, lim, vf; frac)
    (; vmax, vmin, amax, amin, jmax) = lim
    hi = frac * min(amax, sqrt(2 * jmax * max(vf - vmin, 0.0)))
    lo = -frac * min(-amin, sqrt(2 * jmax * max(vmax - vf, 0.0)))
    return _pt_rand_in(rng, lo, hi)
end

"Reproducer string: limiter parameters merged with problem inputs, exact via repr."
_pt_repro(lim; kw...) = string(merge(
    (; vmax = lim.vmax, vmin = lim.vmin, amax = lim.amax, amin = lim.amin, jmax = lim.jmax),
    values(kw)))

"Closeness with atol 1e-6 + rtol 1e-9 (positions can have larger scale)."
_pt_close(x, y) = abs(x - y) <= 1e-6 + 1e-9 * max(abs(x), abs(y))

"""
Unavoidable velocity excursion beyond the velocity limits from the initial
state (v0, a0), in multiples of the velocity band width. Acceleration can only
decay at rate jmax, so the velocity overshoots v0 by up to a0^2/(2 jmax) even
under immediate maximal braking. A brake pre-trajectory handles such states,
but when this ratio is large the post-brake velocity is still far outside the
limits and the main-profile search operates at an extreme feasibility boundary.
"""
function _pt_excess_ratio(lim, v0, a0)
    (; vmax, vmin, jmax) = lim
    exc = a0 > 0 ? (v0 + a0^2 / (2jmax)) - vmax : vmin - (v0 - a0^2 / (2jmax))
    exc = max(exc, v0 - vmax, vmin - v0, 0.0)
    return exc / (vmax - vmin)
end

# FIXME(known solver limitation): calculate_trajectory can throw
# "No valid trajectory found" for initial states whose unavoidable velocity
# excursion exceeds the velocity band by more than roughly 20x
# (_pt_excess_ratio > 20; e.g. a0 = 24.3 with jmax = 0.155 and vmax = 2.0).
# A 20k-case probe found NO failures below ratio 20 and none without a brake
# phase, while failures above that boundary are probabilistic (successes exist
# up to ratio 7665). Example reproducer:
#   JerkLimiter(; vmax=2.0259680650948524, vmin=-2.007782731708046,
#               amax=43.56327529769432, amin=-18.600971789243044,
#               jmax=0.15457607796292946)
#   calculate_trajectory(lim; p0=9.893146566489744, v0=-0.7362226673099832,
#       a0=24.34693412809056, pf=-3.1589588894504717, vf=0.8406854613874348,
#       af=-0.4033399687373154)
# Exceptions from cases above the threshold below are tracked separately and
# only bounded loosely instead of being asserted to zero.
const _PT_EXTREME_RATIO = 10.0

"Print collected reproducers (capped) and return the violation count."
function _pt_dump(name, fails; max_print = 20)
    if !isempty(fails)
        println("  [", name, "]: ", length(fails), " violation(s)")
        for r in Iterators.take(fails, max_print)
            println("    REPRODUCER: ", r)
        end
        length(fails) > max_print && println("    ... and ", length(fails) - max_print, " more")
    end
    return length(fails)
end

_pt_new_viol() = (; err = String[], final = String[], vlim = String[],
    alim = String[], jerk = String[], trapz = String[])

"Print known-issue reproducers (bounded, not asserted to zero); returns count."
function _pt_dump_known(name, fails, n_class; max_print = 3)
    if !isempty(fails)
        println("  [KNOWN ISSUE, ", name, "]: ", length(fails), " of ", n_class,
            " extreme cases (see FIXME at _PT_EXTREME_RATIO)")
        for r in Iterators.take(fails, max_print)
            println("    REPRODUCER: ", r)
        end
    end
    return length(fails)
end

"""
Check final state and sampled kinematics of `prof` against the limits of `lim`,
pushing violation reproducers into the vectors of `viol`.

- `pf === nothing` skips the final-position check (velocity interface).
- Limits are only checked after the brake pre-trajectory, which may legally
  exceed them. Note that a brake phase can be triggered even when (v0, a0) are
  individually within limits: if v0 + a0^2/(2 jmax) is outside the velocity
  limits, a velocity excursion beyond the limits is dynamically unavoidable.
- Velocity check: matching the C++ reference brake semantics, the post-brake
  velocity itself can still be outside the limits for extreme initial states
  (the brake reserves exactly the jerk-limited recovery margin). The invariant
  tested is therefore that the main profile never violates the velocity limits
  by MORE than the post-brake state already does. For in-limit initial states
  (post-brake velocity within limits) this reduces to a strict limit check.
- `check_v = false` skips velocity-limit checks (the velocity interface does
  not enforce vmax).
- Acceleration is checked strictly against [amin, amax] after the brake.
- Jerk is checked on all samples (brake profiles also use jmax).
- `check_trapz` verifies trapezoid consistency of consecutive (p, v) samples,
  which catches discontinuities in the evaluated trajectory.
"""
function _pt_check_profile!(viol, lim, prof, repro; vf, af, pf = nothing,
        nsamples::Int, check_v::Bool = true, check_trapz::Bool = false)
    (; vmax, vmin, amax, amin, jmax) = lim
    Ttot = duration(prof)
    if !isfinite(Ttot) || Ttot < 0
        push!(viol.final, string(repro, " | invalid duration ", repr(Ttot)))
        return
    end

    s = evaluate_at(prof, Ttot)
    final_ok = (pf === nothing || _pt_close(s.p, pf)) && _pt_close(s.v, vf) && _pt_close(s.a, af)
    final_ok || push!(viol.final, string(repro, " | final state ", s, " != target"))

    Ttot > 0 || return
    ts = range(0.0, Ttot, length = nsamples)
    P, V, A, J = evaluate_at(prof, ts)

    jworst = maximum(abs, J)
    jworst <= jmax + 1e-9 ||
        push!(viol.jerk, string(repro, " | max|j|=", repr(jworst), " > jmax"))

    bd = prof.brake_duration
    limit_start = bd + max(1e-9, 1e-9 * bd)
    # Post-brake state: evaluate_at at t == brake_duration hits the main
    # profile (initial state when there is no brake)
    spb = evaluate_at(prof, bd)
    vhi_allow = vmax + max(0.0, spb.v - vmax) + 1e-6 + 1e-9 * abs(spb.v)
    vlo_allow = vmin - max(0.0, vmin - spb.v) - 1e-6 - 1e-9 * abs(spb.v)

    vworst = -Inf
    aworst = -Inf
    for i in eachindex(ts)
        ts[i] >= limit_start || continue
        if check_v
            vworst = max(vworst, V[i] - vhi_allow, vlo_allow - V[i])
        end
        aworst = max(aworst, A[i] - amax, amin - A[i])
    end
    check_v && vworst > 0 &&
        push!(viol.vlim, string(repro, " | v exceeds allowed band by ", repr(vworst),
            " (post-brake v=", repr(spb.v), ")"))
    aworst > 1e-6 &&
        push!(viol.alim, string(repro, " | a exceeds limits by ", repr(aworst)))

    if check_trapz
        h = step(ts)
        maxa = max(abs(amax), abs(amin), maximum(abs, A))
        tol = max(1e-8, h^2 * maxa) * 10
        worst = 0.0
        for i in 1:length(ts)-1
            worst = max(worst, abs(P[i+1] - P[i] - h * (V[i] + V[i+1]) / 2))
        end
        worst <= tol ||
            push!(viol.trapz, string(repro, " | trapezoid error ", repr(worst), " > tol ", repr(tol)))
    end
    return
end

#=============================================================================
 Property tests
=============================================================================#

@testset "Property tests" begin

    @testset "Single-DOF position, states within limits" begin
        Random.seed!(0x20260704)
        rng = Random.default_rng()
        viol = _pt_new_viol()
        known_err = String[]
        n_extreme = 0
        for _ in 1:2000
            lim = _pt_rand_limiter(rng)
            (; vmax, vmin, amax, amin) = lim
            p0 = _pt_rand_in(rng, -10.0, 10.0)
            pf = _pt_rand_in(rng, -10.0, 10.0)
            v0 = _pt_rand_in(rng, 0.9vmin, 0.9vmax)
            vf = _pt_rand_in(rng, 0.9vmin, 0.9vmax)
            a0 = _pt_rand_in(rng, 0.9amin, 0.9amax)
            af = _pt_rand_af(rng, lim, vf; frac = 0.9)
            extreme = _pt_excess_ratio(lim, v0, a0) > _PT_EXTREME_RATIO
            n_extreme += extreme
            repro = _pt_repro(lim; p0, v0, a0, pf, vf, af)
            prof = try
                calculate_trajectory(lim; pf, p0, v0, a0, vf, af)
            catch e
                msg = string(repro, " | threw: ", sprint(showerror, e))
                push!(extreme ? known_err : viol.err, msg)
                continue
            end
            _pt_check_profile!(viol, lim, prof, repro; pf, vf, af,
                nsamples = 500, check_trapz = true)
        end
        @test _pt_dump("solver exceptions", viol.err) == 0
        @test _pt_dump("final state", viol.final) == 0
        @test _pt_dump("velocity limits", viol.vlim) == 0
        @test _pt_dump("acceleration limits", viol.alim) == 0
        @test _pt_dump("jerk limit", viol.jerk) == 0
        @test _pt_dump("trapezoid consistency", viol.trapz) == 0
        # Loose ceiling on the known extreme-state failure class (FIXME above)
        @test _pt_dump_known("solver exceptions", known_err, n_extreme) <= 0.5 * n_extreme
    end

    @testset "Single-DOF brake cases (initial state outside limits)" begin
        Random.seed!(0x20260704)
        rng = Random.default_rng()
        viol = _pt_new_viol()
        known_err = String[]
        n_extreme = 0
        for _ in 1:1000
            lim = _pt_rand_limiter(rng)
            (; vmax, vmin, amax, amin) = lim
            p0 = _pt_rand_in(rng, -10.0, 10.0)
            pf = _pt_rand_in(rng, -10.0, 10.0)
            mode = rand(rng, 1:3) # 1: v0 outside, 2: a0 outside, 3: both outside
            v0 = mode == 2 ? _pt_rand_in(rng, 0.9vmin, 0.9vmax) : _pt_outside(rng, vmin, vmax)
            a0 = mode == 1 ? _pt_rand_in(rng, 0.9amin, 0.9amax) : _pt_outside(rng, amin, amax)
            if rand(rng) < 0.5
                vf = 0.0
                af = 0.0
            else
                vf = _pt_rand_in(rng, 0.5vmin, 0.5vmax)
                af = _pt_rand_af(rng, lim, vf; frac = 0.5)
            end
            extreme = _pt_excess_ratio(lim, v0, a0) > _PT_EXTREME_RATIO
            n_extreme += extreme
            repro = _pt_repro(lim; p0, v0, a0, pf, vf, af)
            prof = try
                calculate_trajectory(lim; pf, p0, v0, a0, vf, af)
            catch e
                msg = string(repro, " | threw: ", sprint(showerror, e))
                push!(extreme ? known_err : viol.err, msg)
                continue
            end
            _pt_check_profile!(viol, lim, prof, repro; pf, vf, af,
                nsamples = 500, check_trapz = true)
        end
        @test _pt_dump("solver exceptions", viol.err) == 0
        @test _pt_dump("final state", viol.final) == 0
        @test _pt_dump("velocity limits after brake", viol.vlim) == 0
        @test _pt_dump("acceleration limits after brake", viol.alim) == 0
        @test _pt_dump("jerk limit", viol.jerk) == 0
        @test _pt_dump("trapezoid consistency", viol.trapz) == 0
        @test _pt_dump_known("solver exceptions", known_err, n_extreme) <= 0.5 * n_extreme
    end

    @testset "Multi-DOF synchronization" begin
        Random.seed!(0x20260704)
        rng = Random.default_rng()
        viol = _pt_new_viol()
        sync = String[]
        known_err = String[]
        n_extreme = 0
        ncases = 400
        nbrake = 100 # cases where exactly one DOF has v0 outside its limits
        for case in 1:ncases
            ndof = rand(rng, 2:5)
            lims = [_pt_rand_limiter(rng) for _ in 1:ndof]
            p0 = [_pt_rand_in(rng, -10.0, 10.0) for _ in 1:ndof]
            pf = [_pt_rand_in(rng, -10.0, 10.0) for _ in 1:ndof]
            v0 = [_pt_rand_in(rng, 0.9l.vmin, 0.9l.vmax) for l in lims]
            vf = [_pt_rand_in(rng, 0.9l.vmin, 0.9l.vmax) for l in lims]
            a0 = [_pt_rand_in(rng, 0.9l.amin, 0.9l.amax) for l in lims]
            af = [_pt_rand_af(rng, lims[i], vf[i]; frac = 0.9) for i in 1:ndof]
            if case > ncases - nbrake
                bdof = rand(rng, 1:ndof)
                v0[bdof] = _pt_outside(rng, lims[bdof].vmin, lims[bdof].vmax)
            end
            extreme = any(_pt_excess_ratio(lims[i], v0[i], a0[i]) > _PT_EXTREME_RATIO for i in 1:ndof)
            n_extreme += extreme
            repro = string("lims=[", join((_pt_repro(l) for l in lims), ", "),
                "], p0=", repr(p0), ", v0=", repr(v0), ", a0=", repr(a0),
                ", pf=", repr(pf), ", vf=", repr(vf), ", af=", repr(af))
            profs = try
                calculate_trajectory(lims; pf, p0, v0, a0, vf, af)
            catch e
                msg = string(repro, " | threw: ", sprint(showerror, e))
                push!(extreme ? known_err : viol.err, msg)
                continue
            end
            durs = duration.(profs)
            if !all(isfinite, durs) || maximum(durs) - minimum(durs) > 1e-9
                push!(sync, string(repro, " | durations ", repr(durs)))
            end
            for i in 1:ndof
                _pt_check_profile!(viol, lims[i], profs[i], string(repro, " | dof=", i);
                    pf = pf[i], vf = vf[i], af = af[i], nsamples = 300)
            end
        end
        @test _pt_dump("solver exceptions", viol.err) == 0
        @test _pt_dump("duration synchronization", sync) == 0
        @test _pt_dump("final state", viol.final) == 0
        @test _pt_dump("velocity limits", viol.vlim) == 0
        @test _pt_dump("acceleration limits", viol.alim) == 0
        @test _pt_dump("jerk limit", viol.jerk) == 0
        @test _pt_dump_known("solver exceptions", known_err, n_extreme) <= 0.5 * n_extreme
    end

    @testset "Velocity interface" begin
        Random.seed!(0x20260704)
        rng = Random.default_rng()
        viol = _pt_new_viol()
        tf_dur = String[]
        tf_final = String[]
        tf_throw_repros = String[]
        tf_attempts = 0
        ncases = 800
        for case in 1:ncases
            lim = _pt_rand_limiter(rng)
            (; vmax, vmin, amax, amin) = lim
            v0 = _pt_rand_in(rng, 0.9vmin, 0.9vmax)
            vf = _pt_rand_in(rng, 0.9vmin, 0.9vmax)
            # Last quarter of the cases: a0 outside limits (velocity brake)
            a0 = case > 600 ? _pt_outside(rng, amin, amax) : _pt_rand_in(rng, 0.9amin, 0.9amax)
            af = _pt_rand_in(rng, 0.9amin, 0.9amax)
            repro = _pt_repro(lim; v0, a0, vf, af)
            prof = try
                calculate_velocity_trajectory(lim; vf, v0, a0, af)
            catch e
                push!(viol.err, string(repro, " | threw: ", sprint(showerror, e)))
                continue
            end
            # The velocity interface does not enforce vmax; do not check v limits
            _pt_check_profile!(viol, lim, prof, repro; vf, af,
                nsamples = 300, check_v = false)

            # Half the cases: also request a comfortably longer duration.
            # Throwing is allowed here (blocked duration intervals exist for
            # af != 0); the throw-rate is asserted below.
            if case % 2 == 0
                tf = duration(prof) + 0.5
                tf_attempts += 1
                prof2 = try
                    calculate_velocity_trajectory(lim; vf, v0, a0, af, tf)
                catch e
                    push!(tf_throw_repros, string(repro, " | tf=", repr(tf)))
                    nothing
                end
                if prof2 !== nothing
                    abs(duration(prof2) - tf) <= 1e-6 ||
                        push!(tf_dur, string(repro, " | tf=", repr(tf),
                            " but duration=", repr(duration(prof2))))
                    s2 = evaluate_at(prof2, duration(prof2))
                    (_pt_close(s2.v, vf) && _pt_close(s2.a, af)) ||
                        push!(tf_final, string(repro, " | tf=", repr(tf),
                            " final state ", s2, " != target"))
                end
            end
        end
        @test _pt_dump("solver exceptions", viol.err) == 0
        @test _pt_dump("final state", viol.final) == 0
        @test _pt_dump("acceleration limits after brake", viol.alim) == 0
        @test _pt_dump("jerk limit", viol.jerk) == 0
        @test _pt_dump("tf-case duration", tf_dur) == 0
        @test _pt_dump("tf-case final state", tf_final) == 0
        tf_throws = length(tf_throw_repros)
        if tf_throws > 0
            println("  [tf-case throws]: ", tf_throws, " of ", tf_attempts,
                " (allowed below 20%)")
        end
        @test tf_throws < 0.2 * tf_attempts
    end

    @testset "Determinism (no stale work-buffer state)" begin
        Random.seed!(0x20260704)
        rng = Random.default_rng()
        mismatch = String[]
        for _ in 1:200
            lim = _pt_rand_limiter(rng)
            (; vmax, vmin, amax, amin) = lim
            p0 = _pt_rand_in(rng, -10.0, 10.0)
            pf = _pt_rand_in(rng, -10.0, 10.0)
            # Quarter of the cases start outside the limits to exercise the brake buffer
            v0 = rand(rng) < 0.25 ? _pt_outside(rng, vmin, vmax) : _pt_rand_in(rng, 0.9vmin, 0.9vmax)
            vf = _pt_rand_in(rng, 0.9vmin, 0.9vmax)
            a0 = _pt_rand_in(rng, 0.9amin, 0.9amax)
            af = _pt_rand_af(rng, lim, vf; frac = 0.9)
            # A different problem solved in between must not change the result
            dirty = (; pf = _pt_rand_in(rng, -10.0, 10.0),
                v0 = _pt_rand_in(rng, 0.5vmin, 0.5vmax),
                a0 = _pt_rand_in(rng, 0.5amin, 0.5amax))
            repro = _pt_repro(lim; p0, v0, a0, pf, vf, af)
            prof1 = try
                calculate_trajectory(lim; pf, p0, v0, a0, vf, af)
            catch
                continue # exceptions are covered by the other testsets
            end
            try
                calculate_trajectory(lim; pf = dirty.pf, v0 = dirty.v0, a0 = dirty.a0)
            catch
            end
            prof2 = try
                calculate_trajectory(lim; pf, p0, v0, a0, vf, af)
            catch e
                push!(mismatch, string(repro, " | second call threw: ", sprint(showerror, e)))
                continue
            end
            if duration(prof1) !== duration(prof2)
                push!(mismatch, string(repro, " | durations ", repr(duration(prof1)),
                    " vs ", repr(duration(prof2))))
            end
            for _ in 1:10
                t = _pt_rand_in(rng, 0.0, 1.05 * duration(prof1))
                s1 = evaluate_at(prof1, t)
                s2 = evaluate_at(prof2, t)
                if !(s1.p === s2.p && s1.v === s2.v && s1.a === s2.a && s1.j === s2.j)
                    push!(mismatch, string(repro, " | t=", repr(t), ": ", s1, " vs ", s2))
                end
            end
        end
        @test _pt_dump("determinism", mismatch) == 0
    end

end
