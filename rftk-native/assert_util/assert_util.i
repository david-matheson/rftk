%module assert_util
%{
    #define SWIG_FILE_WITH_INIT
    #include "assert_util.h"
%}

%include <exception.i>
%exception {
   try {
     $action
   }
   catch (const std::out_of_range& e)
   {
     SWIG_exception(SWIG_IndexError, e.what());
   }
   catch (const std::length_error& e)
   {
     SWIG_exception(SWIG_TypeError, e.what());
   }
   catch (const std::exception& e)
   {
     SWIG_exception(SWIG_RuntimeError, e.what());
   }
   catch (...)
   {
     SWIG_exception(SWIG_RuntimeError, "unknown exception");
   }
}

%include "assert_util.h"




