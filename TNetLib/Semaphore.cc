
#include <climits>
#include "Semaphore.h"
#include "Error.h"


#define LOCK_MUTEX(ARG) {if(0 != pthread_mutex_lock(&ARG)) { \
 KALDI_ERR << "Cannot lock mutex"; \
}}

#define UNLOCK_MUTEX(ARG) {if(0 != pthread_mutex_unlock(&ARG)) { \
 KALDI_ERR << "Cannot unlock mutex"; \
}}

namespace TNet {
  
  Semaphore::
  Semaphore(int initValue) 
  {
    mSemValue = initValue;
    if(0 != pthread_mutex_init(&mMutex, NULL)) {
      KALDI_ERR << "Cannot initialize mutex";
    }
    if(0 != pthread_cond_init(&mCond, NULL)) {
      KALDI_ERR << "Cannot initialize condv";
    }
  }

  Semaphore::
  ~Semaphore()
  {
    if(0 != pthread_mutex_destroy(&mMutex)) {
      KALDI_ERR << "Cannot destroy mutex";
    }
    if(0 != pthread_cond_destroy(&mCond)) {
      KALDI_ERR << "Cannot destroy condv";
    }
  }

  int 
  Semaphore::
  TryWait()
  {
    LOCK_MUTEX(mMutex);
    if(mSemValue > 0) {
      mSemValue--;
      UNLOCK_MUTEX(mMutex);
      return 0;
    }
    UNLOCK_MUTEX(mMutex);
    return -1;
  }

  void 
  Semaphore::
  Wait()
  {
    LOCK_MUTEX(mMutex);
    while(mSemValue <= 0) {
      if(0 != pthread_cond_wait(&mCond, &mMutex)) {
        KALDI_ERR << "Error on pthread_cond_wait";
      }
    }
    mSemValue--;
    UNLOCK_MUTEX(mMutex);
  }

  void
  Semaphore::
  Post()
  {
    LOCK_MUTEX(mMutex);
    if(mSemValue < INT_MAX) {
      mSemValue++;
    } 
    if(0 != pthread_cond_signal(&mCond)) {
      KALDI_ERR << "Error on pthread_cond_signal";
    }
    UNLOCK_MUTEX(mMutex);
  }

  int
  Semaphore::
  GetValue()
  { 
    LOCK_MUTEX(mMutex);
    int val = mSemValue;
    UNLOCK_MUTEX(mMutex);
    return val; 
  }



} //namespace
