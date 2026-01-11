# Block Interval Profile Collection for Ruckig
# Enhanced block calculation for blocked interval computation when vf != 0 || af != 0
# Ported from C++ reference: ruckig/src/ruckig/position_third_step1.cpp

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
        time_all_none_acc0_acc1!(lim.roots, buf, candidate,
                                 p0_eff, v0_eff, a0_eff, pf, vf, af,
                                 jmax, vmax, vmin, amax, amin;
                                 vpc=valid_profiles, brake_duration, brake=brake_copy)

        time_all_none_acc0_acc1!(lim.roots, buf, candidate,
                                 p0_eff, v0_eff, a0_eff, pf, vf, af,
                                 -jmax, vmin, vmax, amin, amax;
                                 vpc=valid_profiles, brake_duration, brake=brake_copy)

        time_acc0_acc1!(buf, candidate,
                        p0_eff, v0_eff, a0_eff, pf, vf, af,
                        jmax, vmax, vmin, amax, amin;
                        vpc=valid_profiles, brake_duration, brake=brake_copy)

        time_acc0_acc1!(buf, candidate,
                        p0_eff, v0_eff, a0_eff, pf, vf, af,
                        -jmax, vmin, vmax, amin, amax;
                        vpc=valid_profiles, brake_duration, brake=brake_copy)

        time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af,
                      jmax, vmax, vmin, amax, amin;
                      vpc=valid_profiles, brake_duration, brake=brake_copy)

        time_all_vel!(buf, p0_eff, v0_eff, a0_eff, pf, vf, af,
                      -jmax, vmin, vmax, amin, amax;
                      vpc=valid_profiles, brake_duration, brake=brake_copy)

        if valid_profiles.count > 0
            return calculate_block!(valid_profiles)
        end
    end

    error("No valid trajectory found from ($p0, $v0, $a0) to ($pf, $vf, $af)")
end
