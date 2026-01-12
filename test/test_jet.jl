# JET.jl static analysis tests
# This file should be included LAST in runtests.jl since loading JET may cause issues
# with other tests due to its extensive use of Julia's internals.

using JET
using TrajectoryLimiters
using Test

# Test the package for potential runtime errors
# target_defined_modules=true limits analysis to code defined in this package
@testset "Package-wide analysis" begin
    JET.test_package(TrajectoryLimiters; target_defined_modules=true)
end

# Test specific key functions for type stability and potential errors
@testset "JerkLimiter functions" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    # Test calculate_trajectory
    @test_opt calculate_trajectory(lim; pf=1.0)
    @test_call calculate_trajectory(lim; pf=1.0)

    # Test with more parameters
    @test_call calculate_trajectory(lim; p0=0.0, v0=1.0, a0=0.0, pf=2.0, vf=0.0, af=0.0)

    # Test profile evaluation
    profile = calculate_trajectory(lim; pf=1.0)
    @test_call profile(0.5)
    @test_call duration(profile)
end

@testset "AccelerationLimiter functions" begin
    lim = AccelerationLimiter(; vmax=10.0, amax=50.0)

    @test_opt calculate_trajectory(lim; pf=1.0)
    @test_call calculate_trajectory(lim; pf=1.0)

    profile = calculate_trajectory(lim; pf=1.0)
    @test_call profile(0.5)
end

@testset "VelocityLimiter functions" begin
    lim = VelocityLimiter(; vmax=10.0)

    @test_opt calculate_trajectory(lim; pf=10.0)
    @test_call calculate_trajectory(lim; pf=10.0)

    profile = calculate_trajectory(lim; pf=10.0)
    @test_call profile(0.5)
end

@testset "Velocity control functions" begin
    lim = JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0)

    @test_call calculate_velocity_trajectory(lim; vf=5.0)
    @test_call calculate_velocity_trajectory(lim; v0=1.0, a0=0.0, vf=5.0, af=0.0)
end

@testset "Multi-DOF functions" begin
    lims = [
        JerkLimiter(; vmax=10.0, amax=50.0, jmax=1000.0),
        JerkLimiter(; vmax=5.0, amax=40.0, jmax=500.0),
    ]

    @test_call calculate_trajectory(lims; pf=[1.0, 2.0])
end

@testset "TrajectoryLimiter (filter)" begin
    limiter = TrajectoryLimiter(0.01, 10.0, 50.0)
    R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    @test_call limiter(R)
end
