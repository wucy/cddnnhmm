/*
 * barrier.c
 *
 * This file implements the "barrier" synchronization construct.
 *
 * A barrier causes threads to wait until a set of threads has
 * all "reached" the barrier. The number of threads required is
 * set when the barrier is initialized, and cannot be changed
 * except by reinitializing.
 *
 * The barrier_init() and barrier_destroy() functions,
 * respectively, allow you to initialize and destroy the
 * barrier.
 *
 * The barrier_wait() function allows a thread to wait for a
 * barrier to be completed. One thread (the one that happens to
 * arrive last) will return from barrier_wait() with the status
 * -1 on success -- others will return with 0. The special
 * status makes it easy for the calling code to cause one thread
 * to do something in a serial region before entering another
 * parallel section of code.
 */
#include <pthread.h>
#include "Error.h"
#include "Barrier.h"


#define LOCK_MUTEX(ARG) {if(0 != pthread_mutex_lock(&ARG)) { \
 KALDI_ERR << "Cannot lock mutex"; \
}}

#define UNLOCK_MUTEX(ARG) {if(0 != pthread_mutex_unlock(&ARG)) { \
 KALDI_ERR << "Cannot unlock mutex"; \
}}

namespace TNet {

/*
 * Initialize a barrier for use.
 */
Barrier::Barrier(int count)
 : threshold_(count), counter_(count), cycle_(0) {

  if(0 != pthread_mutex_init(&mutex_, NULL))
    KALDI_ERR << "Cannot initialize mutex";
  
  if(0 != pthread_cond_init(&cv_, NULL)) {
    KALDI_ERR << "Cannot initilize condv";
  }
}

/*
 * Destroy a barrier when done using it.
 */
Barrier::~Barrier() {

  /*
   * Check whether any threads are known to be waiting; report
   * "BUSY" if so.
   */
  LOCK_MUTEX(mutex_);
  if(counter_ != threshold_) {
    UNLOCK_MUTEX(mutex_);
    KALDI_ERR << "Cannot destroy barrier with waiting thread";
  }
  UNLOCK_MUTEX(mutex_);

  /*
   * If unable to destroy either 1003.1c synchronization
   * object, halt
   */
  if(0 != pthread_mutex_destroy(&mutex_)) {
    KALDI_ERR << "Cannot destroy mutex";
  }

  if(0 != pthread_cond_destroy(&cv_)) {
    KALDI_ERR << "Cannot destroy condv";
  }
}


void Barrier::SetThreshold(int thr) {
  LOCK_MUTEX(mutex_);

  if(counter_ != threshold_) {
    KALDI_ERR << "Cannot set threshold, while a thread is waiting";
  }

  threshold_ = thr; counter_ = thr;
  
  UNLOCK_MUTEX(mutex_);
}



/*
 * Wait for all members of a barrier to reach the barrier. When
 * the count (of remaining members) reaches 0, broadcast to wake
 * all threads waiting.
 */
int Barrier::Wait() {
  int status, cancel, cycle;

  //make sure the threshold was set
  if(threshold_ == 0) {
    KALDI_ERR << "Cannot wait when Threshold value was not set";
  }

  LOCK_MUTEX(mutex_);

  cycle = cycle_;   /* Remember which cycle we're on */

  if(--counter_ != 0) {
    /*
     * This is not the last thread!
     * Let's wait with cancellation disabled, because 
     * barrier_wait should not be a cancellation point.
     */
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cancel);
    /*
     * Wait until the barrier's cycle changes, which means
     * that it has been broadcast, and we don't want to wait
     * anymore.
     */
    while (cycle == cycle_) {
      status = pthread_cond_wait(&cv_, &mutex_);
      if (status != 0) KALDI_ERR << "Error on pthread_cond_wait";
    }
    /*
     * Restore the cancel states
     */
    pthread_setcancelstate(cancel, NULL);
  } else {
    /*
     * This is the last thread!
     */
    cycle_ = !cycle_;      //flip the cycle
    counter_ = threshold_; //reset the counter
    status = pthread_cond_broadcast(&cv_); //wake the other threads
    if (status != 0) {
      KALDI_ERR << "Error on pthread_cond_broadcast";
    } else {
    /*
     * The last thread into the barrier will return status
     * -1 rather than 0, so that it can be used to perform
     * some special serial code following the barrier.
     */
      status = -1;
    }
  }
  
  UNLOCK_MUTEX(mutex_);
  return status;          /* error, -1 for waker, or 0 */
}


}//namespace TNet
