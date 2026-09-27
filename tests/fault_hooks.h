/* Test-only allocation and thread-creation fault injection. */
#ifndef _POSIX_C_SOURCE
/* This header is force-included before the source's feature-test macros. */
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef PRIMAL_FAULT_HOOKS_H
#define PRIMAL_FAULT_HOOKS_H
#include <stdlib.h>
#include <pthread.h>
extern int primal_fault_nth, primal_fault_calls, primal_fault_hit;
extern int primal_fault_threads, primal_fault_probe, primal_fault_sb;
extern long primal_fault_live;
void *primal_fault_malloc(size_t n);
void *primal_fault_calloc(size_t n, size_t size);
void *primal_fault_realloc(void *p, size_t n);
void primal_fault_free(void *p);
int primal_fault_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
#ifndef PRIMAL_FAULT_IMPLEMENTATION
#define malloc primal_fault_malloc
#define calloc primal_fault_calloc
#define realloc primal_fault_realloc
#define free primal_fault_free
#define pthread_create primal_fault_create
#endif
#endif
