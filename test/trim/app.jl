using TrajectoryLimiters

function (@main)(args::Vector{String})::Cint
    lims = [JerkLimiter(; vmax = 1.0, amax = 1.0, jmax = 10.0), JerkLimiter(; vmax = 1.0, amax = 1.0, jmax = 10.0)]
    for synchronization in (:time, :phase)
        profiles = calculate_trajectory(lims; p0 = [0.0, 0.0], pf = [1.0, 2.0], synchronization)
        println(Core.stdout, duration(profiles[1]))
    end
    return 0
end
