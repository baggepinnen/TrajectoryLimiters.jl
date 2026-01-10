# Block Interval Profile Collection for Ruckig
# Enhanced profile search functions that collect ALL valid profiles
# for blocked interval computation when vf != 0 || af != 0
# Ported from C++ reference: ruckig/src/ruckig/position_third_step1.cpp

#=============================================================================
 Collecting Profile Search Functions

 These variants collect ALL valid profiles into a ValidProfileCollection,
 rather than returning after finding the first/shortest profile.
 Used for blocked interval computation in multi-DOF synchronization.
=============================================================================#

"""
    time_all_vel_collect!(vpc, buf, p0, v0, a0, pf, vf, af, jMax, vMax, vMin, aMax, aMin;
                          brake_duration=0, brake=nothing)

Try all velocity-limited profiles and add ALL valid ones to the collection.
Unlike `time_all_vel!`, this doesn't return after finding the first valid profile.

Strategies tried:
1. ACC0_ACC1_VEL - reach aMax, vMax, aMin
2. ACC1_VEL - reach vMax and aMin, not aMax
3. ACC0_VEL - reach aMax and vMax, not aMin
4. VEL - reach vMax only
"""
function time_all_vel_collect!(vpc::ValidProfileCollection{T}, buf::ProfileBuffer{T},
                               p0, v0, a0, pf, vf, af,
                               jMax, vMax, vMin, aMax, aMin;
                               brake_duration=zero(T), brake=nothing) where T
    # Pre-compute common terms
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    af_p3 = af^3
    a0_p4 = a0^4
    af_p4 = af^4
    v0_v0 = v0^2
    vf_vf = vf^2
    pd = pf - p0

    found_any = false

    # Strategy 1: ACC0_ACC1_VEL (reach aMax, vMax, aMin)
    begin
        buf.t[1] = (-a0 + aMax) / jMax
        buf.t[2] = (a0_a0/2 - aMax^2 - jMax*(v0 - vMax)) / (aMax * jMax)
        buf.t[3] = aMax / jMax
        buf.t[5] = -aMin / jMax
        buf.t[6] = -(af_af/2 - aMin^2 - jMax*(vf - vMax)) / (aMin * jMax)
        buf.t[7] = buf.t[5] + af / jMax

        buf.t[4] = (3*(a0_p4*aMin - af_p4*aMax) +
                    8*aMax*aMin*(af_p3 - a0_p3 + 3*jMax*(a0*v0 - af*vf)) +
                    6*a0_a0*aMin*(aMax^2 - 2*jMax*v0) -
                    6*af_af*aMax*(aMin^2 - 2*jMax*vf) -
                    12*jMax*(aMax*aMin*(aMax*(v0 + vMax) - aMin*(vf + vMax) - 2*jMax*pd) +
                            (aMin - aMax)*jMax*vMax^2 +
                            jMax*(aMax*vf_vf - aMin*v0_v0))) / (24*aMax*aMin*jMax_jMax*vMax)

        if check!(buf, UDDU, LIMIT_ACC0_ACC1_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            add_profile!(vpc, RuckigProfile(buf, pf, vf, af; brake_duration, brake))
            found_any = true
        end
    end

    # Strategy 2: ACC1_VEL (reach vMax and aMin, not aMax)
    begin
        t_acc0 = sqrt(max(0.0, a0_a0/(2*jMax_jMax) + (vMax - v0)/jMax))
        buf.t[1] = t_acc0 - a0/jMax
        buf.t[2] = 0
        buf.t[3] = t_acc0
        buf.t[4] = -(3*af_p4 - 8*aMin*(af_p3 - a0_p3) - 24*aMin*jMax*(a0*v0 - af*vf) +
                     6*af_af*(aMin^2 - 2*jMax*vf) -
                     12*jMax*(2*aMin*jMax*pd + aMin^2*(vf + vMax) + jMax*(vMax^2 - vf_vf) +
                              aMin*t_acc0*(a0_a0 - 2*jMax*(v0 + vMax))))/(24*aMin*jMax_jMax*vMax)
        buf.t[5] = -aMin / jMax
        buf.t[6] = -(af_af/2 - aMin^2 - jMax*(vf - vMax)) / (aMin * jMax)
        buf.t[7] = buf.t[5] + af / jMax

        if check!(buf, UDDU, LIMIT_ACC1_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            add_profile!(vpc, RuckigProfile(buf, pf, vf, af; brake_duration, brake))
            found_any = true
        end
    end

    # Strategy 3: ACC0_VEL (reach aMax and vMax, not aMin)
    begin
        t_acc1 = sqrt(max(0.0, af_af/(2*jMax_jMax) + (vMax - vf)/jMax))

        buf.t[1] = (-a0 + aMax) / jMax
        buf.t[2] = (a0_a0/2 - aMax^2 - jMax*(v0 - vMax)) / (aMax * jMax)
        buf.t[3] = aMax / jMax
        buf.t[4] = (3*a0_p4 + 8*aMax*(af_p3 - a0_p3) + 24*aMax*jMax*(a0*v0 - af*vf) +
                    6*a0_a0*(aMax^2 - 2*jMax*v0) -
                    12*jMax*(-2*aMax*jMax*pd + aMax^2*(v0 + vMax) + jMax*(vMax^2 - v0_v0) +
                             aMax*t_acc1*(-af_af + 2*(vf + vMax)*jMax)))/(24*aMax*jMax_jMax*vMax)
        buf.t[5] = t_acc1
        buf.t[6] = 0
        buf.t[7] = t_acc1 + af/jMax

        if check!(buf, UDDU, LIMIT_ACC0_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            add_profile!(vpc, RuckigProfile(buf, pf, vf, af; brake_duration, brake))
            found_any = true
        end
    end

    # Strategy 4: VEL (reach vMax only, no acceleration limits)
    begin
        t_acc0 = sqrt(max(0.0, a0_a0/(2*jMax_jMax) + (vMax - v0)/jMax))
        t_acc1 = sqrt(max(0.0, af_af/(2*jMax_jMax) + (vMax - vf)/jMax))

        buf.t[1] = t_acc0 - a0/jMax
        buf.t[2] = 0
        buf.t[3] = t_acc0
        buf.t[5] = t_acc1
        buf.t[6] = 0
        buf.t[7] = t_acc1 + af/jMax

        buf.t[4] = (af_p3 - a0_p3)/(3*jMax_jMax*vMax) +
                   (a0*v0 - af*vf + (af_af*t_acc1 + a0_a0*t_acc0)/2)/(jMax*vMax) -
                   (v0/vMax + 1.0)*t_acc0 - (vf/vMax + 1.0)*t_acc1 + pd/vMax

        if check!(buf, UDDU, LIMIT_VEL, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            add_profile!(vpc, RuckigProfile(buf, pf, vf, af; brake_duration, brake))
            found_any = true
        end
    end

    return found_any
end

"""
    time_acc0_acc1_collect!(vpc, buf, candidate, p0, v0, a0, pf, vf, af, jMax, vMax, vMin, aMax, aMin;
                            brake_duration=0, brake=nothing)

Try ACC0_ACC1 profile (reach both aMax and aMin) and add ALL valid solutions to collection.
Unlike `time_acc0_acc1!`, this collects both solutions if valid.
"""
function time_acc0_acc1_collect!(vpc::ValidProfileCollection{T}, buf::ProfileBuffer{T}, candidate::ProfileBuffer{T},
                                 p0, v0, a0, pf, vf, af,
                                 jMax, vMax, vMin, aMax, aMin;
                                 brake_duration=zero(T), brake=nothing) where T
    # Pre-compute common terms
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    af_p3 = af^3
    a0_p4 = a0^4
    af_p4 = af^4
    v0_v0 = v0^2
    vf_vf = vf^2
    pd = pf - p0

    # Compute h1
    h1 = (3*(af_p4*aMax - a0_p4*aMin) +
          aMax*aMin*(8*(a0_p3 - af_p3) + 3*aMax*aMin*(aMax - aMin) + 6*aMin*af_af - 6*aMax*a0_a0) +
          12*jMax*(aMax*aMin*((aMax - 2*a0)*v0 - (aMin - 2*af)*vf) + aMin*a0_a0*v0 - aMax*af_af*vf)) /
         (3*(aMax - aMin)*jMax_jMax) +
         4*(aMax*vf_vf - aMin*v0_v0 - 2*aMin*aMax*pd) / (aMax - aMin)

    h1 < 0 && return false
    h1 = sqrt(h1) / 2

    h2 = a0_a0/(2*aMax*jMax) + (aMin - 2*aMax)/(2*jMax) - v0/aMax
    h3 = -af_af/(2*aMin*jMax) - (aMax - 2*aMin)/(2*jMax) + vf/aMin

    found_any = false

    # Try both solutions
    for h1_sign in (1, -1)
        t1_cond = h2 > h1_sign * h1 / aMax
        t5_cond = h3 > -h1_sign * h1 / aMin

        t1_cond && t5_cond || continue

        candidate.t[1] = (-a0 + aMax) / jMax
        candidate.t[2] = h2 - h1_sign * h1 / aMax
        candidate.t[3] = aMax / jMax
        candidate.t[4] = 0
        candidate.t[5] = -aMin / jMax
        candidate.t[6] = h3 + h1_sign * h1 / aMin
        candidate.t[7] = candidate.t[5] + af / jMax

        if check!(candidate, UDDU, LIMIT_ACC0_ACC1, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            add_profile!(vpc, RuckigProfile(candidate, pf, vf, af; brake_duration, brake))
            found_any = true
            # Don't return - continue to check other solution
        end
    end

    return found_any
end

"""
    time_all_none_acc0_acc1_collect!(vpc, roots, buf, candidate, p0, v0, a0, pf, vf, af,
                                     jMax, vMax, vMin, aMax, aMin;
                                     brake_duration=0, brake=nothing)

Try NONE, ACC0, ACC1 profiles and add ALL valid ones to collection.
Unlike `time_all_none_acc0_acc1!`, this collects all valid profiles from all roots.
"""
function time_all_none_acc0_acc1_collect!(vpc::ValidProfileCollection{T}, roots::Roots,
                                          buf::ProfileBuffer{T}, candidate::ProfileBuffer{T},
                                          p0, v0, a0, pf, vf, af,
                                          jMax, vMax, vMin, aMax, aMin;
                                          brake_duration=zero(T), brake=nothing) where T
    # Pre-compute common terms
    jMax_jMax = jMax^2
    a0_a0 = a0^2
    af_af = af^2
    a0_p3 = a0^3
    af_p3 = af^3
    a0_p4 = a0_a0^2
    af_p4 = af_af^2
    v0_v0 = v0^2
    vf_vf = vf^2
    pd = pf - p0

    found_any = false

    # NONE profile: t7 == 0 strategy
    h2_none = (a0_a0 - af_af)/(2*jMax) + (vf - v0)
    h2_h2 = h2_none^2

    t_min_none = (a0 - af)/jMax
    t_max_none = (aMax - aMin)/jMax

    polynom_none_1 = -2*(a0_a0 + af_af - 2*jMax*(v0 + vf)) / jMax_jMax
    polynom_none_2 = 4*(a0_p3 - af_p3 + 3*jMax*(af*vf - a0*v0)) / (3*jMax*jMax_jMax) - 4*pd/jMax
    polynom_none_3 = -h2_h2 / jMax_jMax

    for t in solve_quartic_real!(roots, 1.0, 0.0, polynom_none_1, polynom_none_2, polynom_none_3)
        (t < t_min_none || t > t_max_none) && continue

        # Single Newton step for refinement
        if t > EPS
            h1 = jMax*t*t
            orig = -h2_h2/(4*jMax*t) + h2_none*(af/jMax + t) + (4*a0_p3 + 2*af_p3 - 6*a0_a0*(af + 2*jMax*t) + 12*(af - a0)*jMax*v0 + 3*jMax_jMax*(-4*pd + (h1 + 8*v0)*t))/(12*jMax_jMax)
            deriv = h2_none + 2*v0 - a0_a0/jMax + h2_h2/(4*h1) + (3*h1)/4
            t -= orig / deriv
        end

        h0 = h2_none/(2*jMax*t)
        candidate.t[1] = h0 + t/2 - a0/jMax
        candidate.t[2] = 0
        candidate.t[3] = t
        candidate.t[4] = 0
        candidate.t[5] = 0
        candidate.t[6] = 0
        candidate.t[7] = -h0 + t/2 + af/jMax

        if check!(candidate, UDDU, LIMIT_NONE, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            add_profile!(vpc, RuckigProfile(candidate, pf, vf, af; brake_duration, brake))
            found_any = true
            # Don't return - continue checking other roots
        end
    end

    # ACC0 profile: reaches aMax but not aMin or vMax
    h3_acc0 = (a0_a0 - af_af)/(2*aMax*jMax) + (vf - v0)/aMax
    t_min_acc0 = (aMax - af)/jMax
    t_max_acc0 = (aMax - aMin)/jMax

    h0_acc0 = 3*(af_p4 - a0_p4) + 8*(a0_p3 - af_p3)*aMax + 24*aMax*jMax*(af*vf - a0*v0) -
              6*a0_a0*(aMax^2 - 2*jMax*v0) + 6*af_af*(aMax^2 - 2*jMax*vf) +
              12*jMax*(jMax*(vf_vf - v0_v0 - 2*aMax*pd) - aMax^2*(vf - v0))
    h2_acc0 = -af_af + aMax^2 + 2*jMax*vf

    polynom_acc0_0 = -2*aMax/jMax
    polynom_acc0_1 = h2_acc0 / jMax_jMax
    polynom_acc0_2 = 0.0
    polynom_acc0_3 = h0_acc0 / (12*jMax_jMax*jMax_jMax)

    for t in solve_quartic_real!(roots, 1.0, polynom_acc0_0, polynom_acc0_1, polynom_acc0_2, polynom_acc0_3)
        (t < t_min_acc0 || t > t_max_acc0) && continue

        # Single Newton step
        if t > EPS
            h1 = jMax*t
            orig = h0_acc0/(12*jMax_jMax*t) + t*(h2_acc0 + h1*(h1 - 2*aMax))
            deriv = 2*(h2_acc0 + h1*(2*h1 - 3*aMax))
            t -= orig / deriv
        end

        candidate.t[1] = (-a0 + aMax)/jMax
        candidate.t[2] = h3_acc0 - 2*t + jMax/aMax*t^2
        candidate.t[3] = t
        candidate.t[4] = 0
        candidate.t[5] = 0
        candidate.t[6] = 0
        candidate.t[7] = (af - aMax)/jMax + t

        if check!(candidate, UDDU, LIMIT_ACC0, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            add_profile!(vpc, RuckigProfile(candidate, pf, vf, af; brake_duration, brake))
            found_any = true
        end
    end

    # ACC1 profile: reaches aMin but not aMax or vMax
    h3_acc1 = -(a0_a0 + af_af)/(2*jMax*aMin) + aMin/jMax + (vf - v0)/aMin
    t_min_acc1 = (aMin - a0)/jMax
    t_max_acc1 = (aMax - a0)/jMax

    h0_acc1 = (a0_p4 - af_p4)/4 + 2*(af_p3 - a0_p3)*aMin/3 + (a0_a0 - af_af)*aMin^2/2 +
              jMax*(af_af*vf + a0_a0*v0 + 2*aMin*(jMax*pd - a0*v0 - af*vf) + aMin^2*(v0 + vf) + jMax*(v0_v0 - vf_vf))
    h2_acc1 = a0_a0 - a0*aMin + 2*jMax*v0

    polynom_acc1_0 = 2*(2*a0 - aMin)/jMax
    polynom_acc1_1 = (5*a0_a0 + aMin*(aMin - 6*a0) + 2*jMax*v0) / jMax_jMax
    polynom_acc1_2 = 2*(a0 - aMin)*h2_acc1 / (jMax_jMax*jMax)
    polynom_acc1_3 = h0_acc1 / (jMax_jMax*jMax_jMax)

    for t in solve_quartic_real!(roots, 1.0, polynom_acc1_0, polynom_acc1_1, polynom_acc1_2, polynom_acc1_3)
        (t < t_min_acc1 || t > t_max_acc1) && continue

        # Double/triple Newton step for refinement
        if t > EPS
            h5 = a0_p3 + 2*jMax*a0*v0
            h1 = jMax*t
            orig = -(h0_acc1/2 + h1*(h5 + a0*(aMin - 2*h1)*(aMin - h1) + a0_a0*(5*h1/2 - 2*aMin) + aMin^2*h1/2 + jMax*(h1/2 - aMin)*(h1*t + 2*v0)))/jMax
            deriv = (aMin - a0 - h1)*(h2_acc1 + h1*(4*a0 - aMin + 2*h1))
            t -= min(orig / deriv, t)

            h1 = jMax*t
            orig = -(h0_acc1/2 + h1*(h5 + a0*(aMin - 2*h1)*(aMin - h1) + a0_a0*(5*h1/2 - 2*aMin) + aMin^2*h1/2 + jMax*(h1/2 - aMin)*(h1*t + 2*v0)))/jMax

            if abs(orig) > NEWTON_TOL
                deriv = (aMin - a0 - h1)*(h2_acc1 + h1*(4*a0 - aMin + 2*h1))
                t -= orig / deriv

                h1 = jMax*t
                orig = -(h0_acc1/2 + h1*(h5 + a0*(aMin - 2*h1)*(aMin - h1) + a0_a0*(5*h1/2 - 2*aMin) + aMin^2*h1/2 + jMax*(h1/2 - aMin)*(h1*t + 2*v0)))/jMax

                if abs(orig) > NEWTON_TOL
                    deriv = (aMin - a0 - h1)*(h2_acc1 + h1*(4*a0 - aMin + 2*h1))
                    t -= orig / deriv
                end
            end
        end

        candidate.t[1] = t
        candidate.t[2] = 0
        candidate.t[3] = (a0 - aMin)/jMax + t
        candidate.t[4] = 0
        candidate.t[5] = 0
        candidate.t[6] = h3_acc1 - (2*a0 + jMax*t)*t/aMin
        candidate.t[7] = (af - aMin)/jMax

        if check!(candidate, UDDU, LIMIT_ACC1, jMax, vMax, vMin, aMax, aMin, p0, v0, a0, pf, vf, af)
            add_profile!(vpc, RuckigProfile(candidate, pf, vf, af; brake_duration, brake))
            found_any = true
        end
    end

    return found_any
end

#=============================================================================
 Enhanced Block Calculation
=============================================================================#

"""
    calculate_block_with_collection(lim::JerkLimiter; pf, p0, v0, a0, vf, af) -> Block

Calculate trajectory block with proper collection of ALL valid profiles.
This correctly handles blocked intervals when vf != 0 || af != 0.

The C++ reference (position_third_step1.cpp lines 531-585) distinguishes:
- vf == 0 && af == 0: No blocked intervals possible, return first valid profile
- vf != 0 || af != 0: Must collect ALL valid profiles for block computation
"""
function calculate_block_with_collection(lim::JerkLimiter{T};
                                         pf, p0=zero(T), v0=zero(T), a0=zero(T),
                                         vf=zero(T), af=zero(T)) where T
    (; vmax, vmin, amax, amin, jmax, buffer, candidate, valid_profiles, brake) = lim
    buf = buffer
    clear!(buf)
    clear!(valid_profiles)

    # Validate target constraints
    if vf < vmin || vf > vmax
        error("Target velocity vf=$vf is outside allowed range [$vmin, $vmax]")
    end
    if af < amin || af > amax
        error("Target acceleration af=$af is outside allowed range [$amin, $amax]")
    end

    # Compute brake profile if initial state is outside limits
    get_position_brake_trajectory!(brake, v0, a0, vmax, vmin, amax, amin, jmax)
    ps, vs, as = finalize_brake!(brake, p0, v0, a0)
    brake_duration = brake.duration
    brake_copy = brake_duration > 0 ? deepcopy(brake) : nothing

    # Use post-brake state as effective initial state
    p0_eff, v0_eff, a0_eff = ps, vs, as
    pd = pf - p0_eff

    # Set direction-dependent limits
    if pd >= 0
        jMax1, vMax1, vMin1, aMax1, aMin1 = jmax, vmax, vmin, amax, amin
        jMax2, vMax2, vMin2, aMax2, aMin2 = -jmax, vmin, vmax, amin, amax
    else
        jMax1, vMax1, vMin1, aMax1, aMin1 = -jmax, vmin, vmax, amin, amax
        jMax2, vMax2, vMin2, aMax2, aMin2 = jmax, vmax, vmin, amax, amin
    end

    # Zero-limits special case
    if jmax == 0 || amax == 0 || amin == 0
        if time_all_single_step!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        error("No valid trajectory found for zero-limits case")
    end

    if abs(vf) < EPS && abs(af) < EPS
        # Fast path: no blocked intervals when vf==0 && af==0
        # Return first valid profile found
        if time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        if time_all_none_acc0_acc1!(lim.roots, buf, candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        if time_acc0_acc1!(buf, candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax1, vMax1, vMin1, aMax1, aMin1)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end

        # Try secondary direction
        clear!(buf)
        if time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        if time_all_none_acc0_acc1!(lim.roots, buf, candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
        if time_acc0_acc1!(buf, candidate, p0_eff, v0_eff, a0_eff, pf, vf, af, jMax2, vMax2, vMin2, aMax2, aMin2)
            return Block(RuckigProfile(buf, pf, vf, af; brake_duration, brake=brake_copy))
        end
    else
        # Full collection mode: collect ALL valid profiles for blocked interval computation
        # C++ uses original limits (NOT pd-swapped), see position_third_step1.cpp lines 558-563

        # Collect from all profile types in both directions
        time_all_none_acc0_acc1_collect!(valid_profiles, lim.roots, buf, candidate,
                                         p0_eff, v0_eff, a0_eff, pf, vf, af,
                                         jmax, vmax, vmin, amax, amin;
                                         brake_duration, brake=brake_copy)

        time_all_none_acc0_acc1_collect!(valid_profiles, lim.roots, buf, candidate,
                                         p0_eff, v0_eff, a0_eff, pf, vf, af,
                                         -jmax, vmin, vmax, amin, amax;
                                         brake_duration, brake=brake_copy)

        time_acc0_acc1_collect!(valid_profiles, buf, candidate,
                               p0_eff, v0_eff, a0_eff, pf, vf, af,
                               jmax, vmax, vmin, amax, amin;
                               brake_duration, brake=brake_copy)

        time_acc0_acc1_collect!(valid_profiles, buf, candidate,
                               p0_eff, v0_eff, a0_eff, pf, vf, af,
                               -jmax, vmin, vmax, amin, amax;
                               brake_duration, brake=brake_copy)

        time_all_vel_collect!(valid_profiles, buf,
                             p0_eff, v0_eff, a0_eff, pf, vf, af,
                             jmax, vmax, vmin, amax, amin;
                             brake_duration, brake=brake_copy)

        time_all_vel_collect!(valid_profiles, buf,
                             p0_eff, v0_eff, a0_eff, pf, vf, af,
                             -jmax, vmin, vmax, amin, amax;
                             brake_duration, brake=brake_copy)

        if valid_profiles.count > 0
            return calculate_block!(valid_profiles)
        end
    end

    error("No valid trajectory found from ($p0, $v0, $a0) to ($pf, $vf, $af)")
end
