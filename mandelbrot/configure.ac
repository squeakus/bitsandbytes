AC_INIT([mandelSSE], [1.1], [ttsiod@softlab.ntua.gr])

# Don't clutter this dir, store in build-aux
AC_CONFIG_AUX_DIR([build-aux])

AC_ARG_ENABLE(openmp, 
    [AS_HELP_STRING([--enable-openmp],[use OpenMP Multithreading if available (automatic, use --disable-openmp to disable it)])], 
    [enable_openmp="$enableval"], 
    [enable_openmp="yes"])

# Detect the canonical host and target build environment
AC_CANONICAL_HOST
AC_CANONICAL_TARGET

AM_INIT_AUTOMAKE([-Wall -Werror])

if test x"${CFLAGS}" = x ; then
    CFLAGS="-O3 -g -Wall"
fi

AM_PROG_CC_C_O

AC_CHECK_FUNCS([atexit])
AC_CHECK_HEADERS([stdlib.h string.h])
AC_FUNC_MALLOC
AC_FUNC_VPRINTF
AC_HEADER_STDC

AX_CFLAGS_GCC_OPTION(-Wall)
AX_CFLAGS_GCC_OPTION(-Wextra)
AX_CFLAGS_GCC_OPTION(-fomit-frame-pointer)
AX_CFLAGS_GCC_OPTION(-msse)
AX_CFLAGS_GCC_OPTION(-mfpmath=sse)
AX_CFLAGS_GCC_OPTION(-mtune=native)
AX_CFLAGS_GCC_OPTION(-mstackrealign)

# Check for OpenMP (unless not desired)
OPENMP_CFLAGS=""
OPENMP_LIBS=""
HAVE_OPENMP=no
if test x"${enable_openmp}" = xyes ; then
    AX_OPENMP([HAVE_OPENMP=yes])
else
    AC_MSG_NOTICE([Instructed to disable OpenMP. Disabled...])
fi
if test x"${HAVE_OPENMP}" = xyes ; then
    OPENMP_LIBS=${OPENMP_CFLAGS}
fi
if test x"${HAVE_OPENMP}" = xyes ; then
    AC_DEFINE([USE_OPENMP], 1, [Define this to use OpenMP.])
    CFLAGS="$CFLAGS $OPENMP_CFLAGS"
    AC_SUBST(OPENMP_LIBS)
fi

# Check for SDL (minimum: 1.2.0)
SDL_VERSION=1.2.0
AM_PATH_SDL($SDL_VERSION, :,
            AC_MSG_ERROR([*** SDL version $SDL_VERSION or later was not found!]))
AC_SUBST(SDL_CFLAGS)
AC_SUBST(SDL_LIBS)

# Finally create all the generated files
AC_CONFIG_FILES([Makefile src/Makefile]) 
AC_OUTPUT
