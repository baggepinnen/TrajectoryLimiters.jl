/**
 * Example C program demonstrating the TrajectoryLimiters shared library.
 *
 * This shows how to use:
 * 1. TrajectoryLimiter - for online trajectory filtering
 * 2. JerkLimiter - for computing time-optimal jerk-limited profiles
 *
 * Build instructions are in the README.md
 */

#include <stdio.h>
#include <stdint.h>

/* Forward declarations for TrajectoryLimiter functions */
void* trajectory_limiter_create(double Ts, double xdotM, double xddotM);
void trajectory_limiter_destroy(void* handle);
double trajectory_limiter_step(void* handle, double* state, double r);

/* Forward declarations for JerkLimiter functions */
void* jerk_limiter_create(double vmax, double amax, double jmax);
void jerk_limiter_destroy(void* handle);

/* Forward declarations for Profile functions */
void* profile_create(void* limiter, double p0, double v0, double a0,
                     double pf, double vf, double af);
void profile_destroy(void* handle);
double profile_duration(void* handle);
void profile_evaluate(void* handle, double t, double* result);

int main() {
    printf("========================================\n");
    printf("TrajectoryLimiters C API Example\n");
    printf("========================================\n\n");

    /*
     * Example 1: TrajectoryLimiter
     *
     * The TrajectoryLimiter filters an incoming reference signal
     * to ensure velocity and acceleration limits are respected.
     */
    printf("=== TrajectoryLimiter Example ===\n");
    printf("Filtering a step reference from 0 to 5\n\n");

    /* Create limiter: sample time = 0.01s, max velocity = 10, max accel = 50 */
    void* tl = trajectory_limiter_create(0.01, 10.0, 50.0);

    /* State array: [x, xdot, r, rdot] */
    double state[4] = {0.0, 0.0, 0.0, 0.0};

    /* Simulate 20 steps with a step reference to 5.0 */
    printf("Step  Reference    Position    Velocity    Acceleration\n");
    printf("----  ---------    --------    --------    ------------\n");

    for (int i = 0; i < 20; i++) {
        double r = 5.0;  /* Step reference */
        double accel = trajectory_limiter_step(tl, state, r);
        printf("%4d  %9.2f    %8.4f    %8.4f    %12.4f\n",
               i, r, state[0], state[1], accel);
    }

    trajectory_limiter_destroy(tl);

    /*
     * Example 2: JerkLimiter
     *
     * The JerkLimiter computes time-optimal trajectories that respect
     * velocity, acceleration, and jerk limits.
     */
    printf("\n=== JerkLimiter Example ===\n");
    printf("Computing trajectory from (0,0,0) to (2,0,0)\n\n");

    /* Create limiter: vmax = 10, amax = 50, jmax = 1000 */
    void* jl = jerk_limiter_create(10.0, 50.0, 1000.0);

    /* Create profile from rest at 0 to rest at 2 */
    void* profile = profile_create(jl,
                                   0.0, 0.0, 0.0,  /* p0, v0, a0 */
                                   2.0, 0.0, 0.0); /* pf, vf, af */

    double dur = profile_duration(profile);
    printf("Profile duration: %.6f seconds\n\n", dur);

    /* Evaluate at 10 evenly spaced points */
    printf("Time        Position    Velocity    Acceleration    Jerk\n");
    printf("----        --------    --------    ------------    ----\n");

    double result[4];  /* [p, v, a, j] */
    int num_points = 10;
    for (int i = 0; i <= num_points; i++) {
        double t = dur * i / num_points;
        profile_evaluate(profile, t, result);
        printf("%.6f    %8.4f    %8.4f    %12.4f    %8.2f\n",
               t, result[0], result[1], result[2], result[3]);
    }

    profile_destroy(profile);

    /*
     * Example 3: Non-zero initial velocity
     */
    printf("\n=== JerkLimiter with Initial Velocity ===\n");
    printf("Computing trajectory from (0, 5, 0) to (3, 0, 0)\n\n");

    /* Create profile starting with velocity 5 */
    void* profile2 = profile_create(jl,
                                    0.0, 5.0, 0.0,  /* p0, v0, a0 */
                                    3.0, 0.0, 0.0); /* pf, vf, af */

    double dur2 = profile_duration(profile2);
    printf("Profile duration: %.6f seconds\n\n", dur2);

    printf("Time        Position    Velocity    Acceleration    Jerk\n");
    printf("----        --------    --------    ------------    ----\n");

    for (int i = 0; i <= num_points; i++) {
        double t = dur2 * i / num_points;
        profile_evaluate(profile2, t, result);
        printf("%.6f    %8.4f    %8.4f    %12.4f    %8.2f\n",
               t, result[0], result[1], result[2], result[3]);
    }

    profile_destroy(profile2);
    jerk_limiter_destroy(jl);

    printf("\n========================================\n");
    printf("Done!\n");
    printf("========================================\n");

    return 0;
}
