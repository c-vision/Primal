/* SPDX-License-Identifier: Apache-2.0 */
#define PRIMAL_FAULT_IMPLEMENTATION
#include "fault_hooks.h"
#include <errno.h>
int primal_fault_nth, primal_fault_calls, primal_fault_hit;
int primal_fault_threads, primal_fault_probe, primal_fault_sb;
long primal_fault_live;
extern void *probe_worker(void *);
extern void *sb_worker(void *);
static int fail(void) {
    if (primal_fault_nth && ++primal_fault_calls == primal_fault_nth) {
        primal_fault_hit = 1; return 1;
    }
    return 0;
}
void *primal_fault_malloc(size_t n) {
    if (fail()) return NULL;
    void *p = malloc(n); if (p) primal_fault_live++; return p;
}
void *primal_fault_calloc(size_t n, size_t size) {
    if (fail()) return NULL;
    void *p = calloc(n,size); if (p) primal_fault_live++; return p;
}
void *primal_fault_realloc(void *p, size_t n) {
    if (fail()) return NULL;
    int had = p != NULL;
    void *q = realloc(p,n);
    if (q && !had) primal_fault_live++;
    else if (!q && had && n == 0) primal_fault_live--;
    return q;
}
void primal_fault_free(void *p) { if (p) primal_fault_live--; free(p); }
int primal_fault_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
    if (primal_fault_threads) {
        if (fn == probe_worker) primal_fault_probe++;
        if (fn == sb_worker) primal_fault_sb++;
        return EAGAIN;
    }
    return pthread_create(t,a,fn,arg);
}
