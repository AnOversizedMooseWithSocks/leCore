#ifndef HOLO_TEST_THREADS_H
#define HOLO_TEST_THREADS_H

typedef int (*holo_test_thread_fn)(void *);

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef struct holo_test_thread {
    HANDLE handle;
    holo_test_thread_fn fn;
    void *arg;
    int result;
} holo_test_thread;

static DWORD WINAPI holo_test_thread_entry(LPVOID opaque)
{
    holo_test_thread *thread = (holo_test_thread *)opaque;
    thread->result = thread->fn(thread->arg);
    return 0;
}

static int holo_test_thread_create(holo_test_thread *thread,
                                   holo_test_thread_fn fn,
                                   void *arg)
{
    thread->fn = fn;
    thread->arg = arg;
    thread->result = 0;
    thread->handle = CreateThread(NULL, 0, holo_test_thread_entry, thread, 0, NULL);
    return thread->handle != NULL ? 0 : -1;
}

static int holo_test_thread_join(holo_test_thread *thread)
{
    DWORD wait_result = WaitForSingleObject(thread->handle, INFINITE);
    int result = thread->result;
    CloseHandle(thread->handle);
    return wait_result == WAIT_OBJECT_0 ? result : -1;
}

#else
#include <pthread.h>

typedef struct holo_test_thread {
    pthread_t handle;
    holo_test_thread_fn fn;
    void *arg;
    int result;
} holo_test_thread;

static void *holo_test_thread_entry(void *opaque)
{
    holo_test_thread *thread = (holo_test_thread *)opaque;
    thread->result = thread->fn(thread->arg);
    return NULL;
}

static int holo_test_thread_create(holo_test_thread *thread,
                                   holo_test_thread_fn fn,
                                   void *arg)
{
    thread->fn = fn;
    thread->arg = arg;
    thread->result = 0;
    return pthread_create(&thread->handle, NULL, holo_test_thread_entry, thread);
}

static int holo_test_thread_join(holo_test_thread *thread)
{
    int rc = pthread_join(thread->handle, NULL);
    return rc == 0 ? thread->result : rc;
}
#endif

#endif
