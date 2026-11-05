/* Private stagnation clock: allow a complete window after each material
 * improvement. A short nonmonotone excursion is not a stalled window. */
#ifndef SDP_PROGRESS_H
#define SDP_PROGRESS_H
static inline int sdp_progress_stalled(double violation, double *reference,
                                       int *elapsed, int window) {
    if (violation < 0.99 * *reference) {
        *reference = violation;
        *elapsed = 0;
        return 0;
    }
    if (*elapsed < window) ++*elapsed;
    return *elapsed >= window;
}
#endif
