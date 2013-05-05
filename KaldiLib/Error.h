//
// C++ Interface: %{MODULE}
//
// Description: 
//
//
// Author: %{AUTHOR} <%{EMAIL}>, (C) %{YEAR}
//
// Copyright: See COPYING file that comes with this distribution
//
//

/** @file Error.h
 *  This header defines several classes relating to logging 
 *  and error handling in the KaldiLib library.
 *
 *  All the prints are done via fprintf, because 
 *  it is quaranteed to be thread-safe.
 */
 
#ifndef TNET_Error_h
#define TNET_Error_h

#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

#include <cstdlib>
#include <cstdio>
#include <execinfo.h>

namespace TNet {
  


  /** MyException
   *  Custom exception class, which is extracting the stacktrace
   */
  class MyException 
    : public std::runtime_error
  {
    public:
      explicit MyException(const std::string& what_arg) throw();

      virtual ~MyException() throw()
      { }

      const char* what() const throw() 
      { return mWhat.c_str(); }

    private:
      std::string mWhat;
  };

  /** 
   * MyException:: implemenatation
   */
  inline
  MyException::
  MyException(const std::string& what_arg) throw()
    : std::runtime_error(what_arg)
  {
    mWhat = what_arg;
    mWhat += "\nTHE STACKTRACE INSIDE MyException OBJECT IS:\n";
    
    void *array[10];
    size_t size;
    char **strings;
    size_t i;

    size = backtrace (array, 10);
    strings = backtrace_symbols (array, size);
    
    //<< 0th string is the MyException constructor, so ignore it and start by 1
    for (i = 1; i < size; i++) { 
      mWhat += strings[i];
      mWhat += "\n";
    }

    free (strings);
  }


/////////////////////////////////////////////////////////
// ERROR HANDLING FUNCTIONS OBJECTS
// we will use fprintf because it is atomic and thread-safe 
// (C++ streams std::cout,std::cerr are not)
/////////////////////////////////////////////////////////

  /**
   * New kaldi error handling:
   *
   * class KaldiErrorMessage is invoked from the KALDI_ERR macro.
   * The destructor throws an exception.
   */
  class KaldiErrorMessage {
   public:
    KaldiErrorMessage(const char *func, const char *file, int line) {
      this->stream() << "ERROR (" 
                     << func << "():"
                     << file << ':' << line << ") ";
    }
    inline std::ostream &stream() { return ss; }
    ~KaldiErrorMessage() { throw MyException(ss.str()); }
   private:
    std::ostringstream ss;
  };
  #define KALDI_ERR TNet::KaldiErrorMessage(__func__, __FILE__, __LINE__).stream() 


  class KaldiWarningMessage {
   public:
    KaldiWarningMessage(const char *func, const char *file, int line) {
      this->stream() << "WARNING (" 
                     << func << "():"
                     << file << ':' << line << ") ";
    }
    inline std::ostream &stream() { return ss; }
    ~KaldiWarningMessage() { fprintf(stdout,"%s\n",ss.str().c_str()); fflush(stdout); }
   private:
    std::ostringstream ss;
  };
  #define KALDI_WARN TNet::KaldiWarningMessage(__func__, __FILE__, __LINE__).stream() 


  class KaldiLogMessage {
   public:
    KaldiLogMessage(const char *func, const char *file, int line) {
      this->stream() << "LOG (" 
                     << func << "():"
                     << file << ':' << line << ") ";
    }
    inline std::ostream &stream() { return ss; }
    ~KaldiLogMessage() { fprintf(stdout,"%s\n",ss.str().c_str()); fflush(stdout); }
   private:
    std::ostringstream ss;
  };
  #define KALDI_LOG TNet::KaldiLogMessage(__func__, __FILE__, __LINE__).stream() 


  class KaldiCout {
   public:
    KaldiCout() { }
    inline std::ostream &stream() { return ss; }
    ~KaldiCout() { fprintf(stdout,"%s",ss.str().c_str()); fflush(stdout); }
   private:
    std::ostringstream ss;
  };
  #define KALDI_COUT TNet::KaldiCout().stream()

  class KaldiCerr {
   public:
    KaldiCerr() { }
    inline std::ostream &stream() { return ss; }
    ~KaldiCerr() { fprintf(stderr,"%s",ss.str().c_str()); fflush(stderr); }
   private:
    std::ostringstream ss;
  };
  #define KALDI_CERR TNet::KaldiCerr().stream()



} // namespace TNet

//#define TNET_Error_h
#endif
