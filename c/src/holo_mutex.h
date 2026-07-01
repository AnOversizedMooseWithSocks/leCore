#ifndef HOLO_MUTEX_H
#define HOLO_MUTEX_H

#if defined(_MSC_VER)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef CRITICAL_SECTION holo_mutex;

static int holo_mutex_init_recursive(holo_mutex *mutex)
{
    InitializeCriticalSection(mutex);
    return 0;
}

static void holo_mutex_destroy(holo_mutex *mutex)
{
    DeleteCriticalSection(mutex);
}

static int holo_mutex_lock(holo_mutex *mutex)
{
    EnterCriticalSection(mutex);
    return 0;
}

static void holo_mutex_unlock(holo_mutex *mutex)
{
    LeaveCriticalSection(mutex);
}
#elif defined(__APPLE__) || defined(__unix__)
#include <pthread.h>

typedef pthread_mutex_t holo_mutex;

static int holo_mutex_init_recursive(holo_mutex *mutex)
{
    pthread_mutexattr_t attr;
    int rc = pthread_mutexattr_init(&attr);
    if (rc != 0) {
        return rc;
    }
    rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    if (rc == 0) {
        rc = pthread_mutex_init(mutex, &attr);
    }
    pthread_mutexattr_destroy(&attr);
    return rc;
}

static void holo_mutex_destroy(holo_mutex *mutex)
{
    pthread_mutex_destroy(mutex);
}

static int holo_mutex_lock(holo_mutex *mutex)
{
    return pthread_mutex_lock(mutex);
}

static void holo_mutex_unlock(holo_mutex *mutex)
{
    pthread_mutex_unlock(mutex);
}
#else
#error "The C holographic kernel needs pthreads or Windows critical sections for thread safety"
#endif

#endif
