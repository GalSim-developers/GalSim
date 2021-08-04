# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
from __future__ import print_function
import sys,os,glob,re
import platform
import ctypes
import ctypes.util
import types
import subprocess
import re
import tempfile
try:
    import urllib2
except ImportError:
    import urllib.request as urllib2
import tarfile
import shutil

try:
    from setuptools import setup, Extension, find_packages
    from setuptools.command.build_ext import build_ext
    from setuptools.command.build_clib import build_clib
    from setuptools.command.install import install
    from setuptools.command.install_scripts import install_scripts
    from setuptools.command.easy_install import easy_install
    from setuptools.command.test import test
    import setuptools
    print("Using setuptools version",setuptools.__version__)
except ImportError:
    print()
    print("****")
    print("    Installation requires setuptools version >= 38.")
    print("    Please upgrade or install with pip install -U setuptools")
    print("****")
    print()
    raise

# Turn this on for more verbose debugging output about compile attempts.
debug = False

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

scripts = ['galsim', 'galsim_download_cosmos']
scripts = [ os.path.join('bin',f) for f in scripts ]

def all_files_from(dir, ext=''):
    files = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(ext) and not filename.startswith( ('.', 'SCons') ):
                files.append(os.path.join(root, filename))
    return files

py_sources = all_files_from('pysrc', '.cpp')
cpp_sources = all_files_from('src', '.cpp')
test_sources = all_files_from('tests', '.cpp')
headers = all_files_from('include')
shared_data = all_files_from('share')

copt =  {
    'gcc' : ['-O2','-msse2','-std=c++11','-fvisibility=hidden','-fopenmp'],
    'icc' : ['-O2','-msse2','-vec-report0','-std=c++11','-openmp'],
    'clang' : ['-O2','-msse2','-std=c++11',
               '-Wno-shorten-64-to-32','-fvisibility=hidden','-stdlib=libc++'],
    'clang w/ OpenMP' : ['-O2','-msse2','-std=c++11','-fopenmp',
                         '-Wno-shorten-64-to-32','-fvisibility=hidden','-stdlib=libc++'],
    'clang w/ Intel OpenMP' : ['-O2','-msse2','-std=c++11','-Xpreprocessor','-fopenmp',
                                '-Wno-shorten-64-to-32','-fvisibility=hidden','-stdlib=libc++'],
    'clang w/ manual OpenMP' : ['-O2','-msse2','-std=c++11','-Xpreprocessor','-fopenmp',
                                '-Wno-shorten-64-to-32','-fvisibility=hidden','-stdlib=libc++'],
    'unknown' : [],
}
lopt =  {
    'gcc' : ['-fopenmp'],
    'icc' : ['-openmp'],
    'clang' : ['-stdlib=libc++'],
    'clang w/ OpenMP' : ['-stdlib=libc++','-fopenmp'],
    'clang w/ Intel OpenMP' : ['-stdlib=libc++','-liomp5'],
    'clang w/ manual OpenMP' : ['-stdlib=libc++','-lomp'],
    'unknown' : [],
}

# If we build with debug, undefine NDEBUG flag
# Note: setuptools stopped allowing --debug, so if we need this, we'll need to find another
# mechanism.
undef_macros = []
if "--debug" in sys.argv:
    undef_macros+=['NDEBUG']
    for name in copt.keys():
        if name != 'unknown':
            copt[name].append('-g')

# Verbose is the default for setuptools logging, but if it's on the command line, we take it
# to mean that we should also be verbose.
if "--debug" in sys.argv or "--verbose" in sys.argv:
    debug = True

local_tmp = 'tmp'

def get_compiler_type(compiler, check_unknown=True, output=False):
    """Try to figure out which kind of compiler this really is.
    In particular, try to distinguish between clang and gcc, either of which may
    be called cc or gcc.
    """
    if debug: output=True
    cc = compiler.compiler_so[0]
    if cc == 'ccache':
        cc = compiler.compiler_so[1]
    cmd = [cc,'--version']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    if output:
        print('compiler version information: ')
        for line in lines:
            print(line.decode().strip())
    # Python3 needs this decode bit.
    # Python2.7 doesn't need it, but it works fine.
    line = lines[0].decode(encoding='UTF-8')
    if line.startswith('Configured'):
        line = lines[1].decode(encoding='UTF-8')

    if 'clang' in line:
        # clang 3.7 is the first with openmp support.  But Apple lies about the version
        # number of clang, so the most reliable thing to do is to just try the compilation
        # with the openmp flag and see if it works.
        if output:
            print('Compiler is Clang.  Checking if it is a version that supports OpenMP.')
        if try_openmp(compiler, 'clang w/ OpenMP'):
            if output:
                print("Yay! This version of clang supports OpenMP!")
            return 'clang w/ OpenMP'
        elif try_openmp(compiler, 'clang w/ Intel OpenMP'):
            if output:
                print("Yay! This version of clang supports OpenMP!")
            return 'clang w/ Intel OpenMP'
        elif try_openmp(compiler, 'clang w/ manual OpenMP'):
            if output:
                print("Yay! This version of clang supports OpenMP!")
            return 'clang w/ manual OpenMP'
        else:
            if output:
                print("\nSorry.  This version of clang doesn't seem to support OpenMP.\n")
                print("If you think it should, you can use `python setup.py build --debug`")
                print("to get more information about the commands that failed.")
                print("You might need to add something to your C_INCLUDE_PATH or LIBRARY_PATH")
                print("(and probabaly LD_LIBRARY_PATH) to get it to work.\n")
            return 'clang'
    elif 'gcc' in line:
        return 'gcc'
    elif 'GCC' in line:
        return 'gcc'
    elif 'clang' in cc:
        return 'clang'
    elif 'gcc' in cc or 'g++' in cc:
        return 'gcc'
    elif 'icc' in cc or 'icpc' in cc:
        return 'icc'
    elif check_unknown:
        # OK, the main thing we need to know is what openmp flag we need for this compiler,
        # so let's just try the various options and see what works.  Don't try icc, since
        # the -openmp flag there gets treated as '-o penmp' by gcc and clang, which is bad.
        # Plus, icc should be detected correctly by the above procedure anyway.
        if output:
            print('Unknown compiler.')
        for cc_type in ['gcc', 'clang w/ OpenMP', 'clang w/ manual OpenMP', 'clang w/ Intel OpenMP',
                        'clang']:
            if output:
                print('Check if the compiler works like ',cc_type)
            if try_openmp(compiler, cc_type):
                return cc_type
        # I guess none of them worked.  Now we really do have to bail.
        if output:
            print("None of these compile options worked.  Not adding any optimization flags.")
        return 'unknown'
    else:
        return 'unknown'

# Check for the fftw3 library in some likely places
def find_fftw_lib(output=False):
    import distutils.sysconfig

    if debug: output = True
    try_libdirs = []

    # Start with the explicit FFTW_DIR, if present.
    if 'FFTW_DIR' in os.environ:
        try_libdirs.append(os.environ['FFTW_DIR'])
        try_libdirs.append(os.path.join(os.environ['FFTW_DIR'],'lib'))

    # Add the python system library directory.
    try_libdirs.append(distutils.sysconfig.get_config_var('LIBDIR'))

    # If using Anaconda, add their lib dir in case fftw is installed there.
    # (With envs, this might be different than the sysconfig LIBDIR.)
    if 'CONDA_PREFIX' in os.environ:
        try_libdirs.append(os.path.join(os.environ['CONDA_PREFIX'],'lib'))

    # Try some standard locations where things get installed
    try_libdirs.extend(['/usr/local/lib', '/usr/lib'])
    if sys.platform == "darwin":
        try_libdirs.extend(['/sw/lib', '/opt/local/lib'])

    # Check the directories in LD_LIBRARY_PATH.  This doesn't work on OSX >= 10.11
    for path in ['LIBRARY_PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH']:
        if path in os.environ:
            for dir in os.environ[path].split(':'):
                try_libdirs.append(dir)

    # The user's home directory is often a good place to check.
    try_libdirs.append(os.path.join(os.path.expanduser("~"),"lib"))

    # If the above don't work, the fftw3 module may have the right directory.
    try:
        import fftw3
        try_libdirs.append(fftw3.lib.libdir)
    except ImportError:
        pass

    if sys.platform == "darwin":
        lib_ext = '.dylib'
    else:
        lib_ext = '.so'
    name = 'libfftw3' + lib_ext
    if output: print("Looking for ",name)
    tried_dirs = set()  # Keep track, so we don't try the same thing twice.
    for dir in try_libdirs:
        if dir == '': continue  # This messes things up if it's in there.
        if dir in tried_dirs: continue
        else: tried_dirs.add(dir)
        if not os.path.isdir(dir): continue
        libpath = os.path.join(dir, name)
        if not os.path.isfile(libpath): continue
        if output: print("  ", dir, end='')
        try:
            lib = ctypes.cdll.LoadLibrary(libpath)
            if output: print("  (yes)")
            return libpath
        except OSError as e:
            if output: print("  (no)")
            # Some places use lib64 rather than/in addition to lib.  Try that as well.
            if dir.endswith('lib') and os.path.isdir(dir + '64'):
                dir += '64'
                try:
                    libpath = os.path.join(dir, name)
                    if not os.path.isfile(libpath): continue
                    lib = ctypes.cdll.LoadLibrary(libpath)
                    if output: print("  ", dir, "  (yes)")
                    return libpath
                except OSError:
                    pass

    # If we didn't find it anywhere, but the user has set FFTW_DIR, trust it.
    if 'FFTW_DIR' in os.environ:
        libpath = os.path.join(os.environ['FFTW_DIR'], name)
        print("WARNING:")
        print("Could not find an installed fftw3 library named %s"%(name))
        print("Trusting the provided FFTW_DIR=%s for the library location."%(libpath))
        print("If this is incorrect, you may have errors later when linking.")
        return libpath

    # Last ditch attempt.  Use ctypes.util.find_library, which sometimes manages to find it
    # when the above attempts fail.
    try:
        libpath = ctypes.util.find_library('fftw3')
        if libpath == None:
            raise OSError
        if os.path.split(libpath)[0] == '':
            # If the above doesn't return a real path, try this instead.
            libpath = ctypes.util._findLib_gcc('fftw3')
            if libpath == None:
                raise OSError
        libpath = os.path.realpath(libpath)
        lib = ctypes.cdll.LoadLibrary(libpath)
    except Exception as e:
        print("Could not find fftw3 library.  Make sure it is installed either in a standard ")
        print("location such as /usr/local/lib, or the installation directory is either in ")
        print("your LIBRARY_PATH or FFTW_DIR environment variable.")
        raise
    else:
        dir, name = os.path.split(libpath)
        if output:
            if dir == '': dir = '[none]'
            print("  ", dir, "  (yes)")
        return libpath


# Check for Eigen in some likely places
def find_eigen_dir(output=False):
    if debug: output = True
    import distutils.sysconfig

    try_dirs = []

    # Start with a user-specified directory.
    if 'EIGEN_DIR' in os.environ:
        try_dirs.append(os.environ['EIGEN_DIR'])
        try_dirs.append(os.path.join(os.environ['EIGEN_DIR'], 'include'))

    # Add the python system include directory.
    try_dirs.append(distutils.sysconfig.get_config_var('INCLUDEDIR'))

    # If using Anaconda, add their lib dir in case fftw is installed there.
    # (With envs, this might be different than the sysconfig LIBDIR.)
    if 'CONDA_PREFIX' in os.environ:
        try_dirs.append(os.path.join(os.environ['CONDA_PREFIX'],'lib'))

    # Some standard install locations:
    try_dirs.extend(['/usr/local/include', '/usr/include'])
    if sys.platform == "darwin":
        try_dirs.extend(['/sw/include', '/opt/local/include'])

    # Also if there is a C_INCLUDE_PATH, check those dirs.
    for path in ['C_INCLUDE_PATH']:
        if path in os.environ:
            for dir in os.environ[path].split(':'):
                try_dirs.append(dir)

    # Finally, (last resort) check our own download of eigen.
    if os.path.isdir('downloaded_eigen'):
        try_dirs.extend(glob.glob(os.path.join('downloaded_eigen','*')))

    if output: print("Looking for Eigen:")
    for dir in try_dirs:
        if dir is None: continue
        if not os.path.isdir(dir): continue
        if output: print("  ", dir, end='')
        if os.path.isfile(os.path.join(dir, 'Eigen/Core')):
            if output: print("  (yes)")
            return dir
        if os.path.isfile(os.path.join(dir, 'eigen3', 'Eigen/Core')):
            dir = os.path.join(dir, 'eigen3')
            if output:
                # Only print this if the eigen3 addition was key to finding it.
                print("\n  ", dir, "  (yes)")
            return dir
        if output: print("  (no)")

    if output:
        print("Could not find Eigen in any of the standard locations.")
        print("Will now try to download it from gitlab.com. This requires an internet")
        print("connection, so it will fail if you are currently offline.")
        print("If Eigen is installed in a non-standard location, and you want to use that")
        print("instead, you should make sure the right directory is either in your")
        print("C_INCLUDE_PATH or specified in an EIGEN_DIR environment variable.")

    try:
        dir = 'downloaded_eigen'
        if os.path.isdir(dir):
            # If this exists, it was tried above and failed.  Something must be wrong with it.
            print("Previous attempt to download eigen found. Deleting and trying again.")
            shutil.rmtree(dir)
        os.mkdir(dir)
        url = 'https://gitlab.com/libeigen/eigen/-/archive/3.3.4/eigen-3.3.4.tar.bz2'
        if output:
            print("Downloading eigen from ",url)
        # Unfortunately, gitlab doesn't allow direct downloads. We need to spoof the request
        # so it thinks we're a web browser.
        # cf. https://stackoverflow.com/questions/42863240/how-to-get-round-the-http-error-403-forbidden-with-urllib-request-using-python
        page=urllib2.Request(url,headers={'User-Agent': 'Mozilla/5.0'})
        data=urllib2.urlopen(page).read()
        fname = 'eigen.tar.bz2'
        with open(fname, 'wb') as f:
            f.write(data)
        if output:
            print("Downloaded %s.  Unpacking tarball."%fname)
        with tarfile.open(fname) as tar:
            tar.extractall(dir)
        os.remove(fname)
        # This actually extracts into a subdirectory with a name eigen-eigen-5a0156e40feb/
        # I'm not sure if that name is reliable, so use glob to get it.
        dir = glob.glob(os.path.join(dir,'*'))[0]
        if os.path.isfile(os.path.join(dir, 'Eigen/Core')):
            return dir
        elif output:
            print("Downloaded eigen, but it didn't have the expected Eigen/Core file.")
    except Exception as e:
        if output:
            print("Error encountered while downloading Eigen from the internet")
            print(e)

    raise OSError("Could not find Eigen")


def try_compile(cpp_code, compiler, cflags=[], lflags=[], prepend=None):
    """Check if compiling some code with the given compiler and flags works properly.
    """
    # Put the temporary files in a local tmp directory, so that they stick around after failures.
    if not os.path.exists(local_tmp): os.makedirs(local_tmp)

    # We delete these manually if successful.  Otherwise, we leave them in the tmp directory
    # so the user can troubleshoot the problem if they were expecting it to work.
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cpp', dir=local_tmp) as cpp_file:
        cpp_file.write(cpp_code.encode())
        cpp_name = cpp_file.name

    # Just get a named temporary file to write to:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.os', dir=local_tmp) as o_file:
        o_name = o_file.name

    # Another named temporary file for the executable
    with tempfile.NamedTemporaryFile(delete=False, suffix='.exe', dir=local_tmp) as exe_file:
        exe_name = exe_file.name

    # Try compiling with the given flags
    cc = [compiler.compiler_so[0]]
    if prepend:
        cc = [prepend] + cc
    cmd = cc + compiler.compiler_so[1:] + cflags + ['-c',cpp_name,'-o',o_name]
    if debug:
        print('cmd = ',' '.join(cmd))
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = p.stdout.readlines()
        p.communicate()
        if debug and p.returncode != 0:
            print('Trying compile command:')
            print(' '.join(cmd))
            print('Output was:')
            print('   ',b'   '.join(lines).decode())
        returncode = p.returncode
    except (IOError,OSError) as e:
        if debug:
            print('Trying compile command:')
            print(cmd)
            print('Caught error: ',repr(e))
        returncode = 1
    if returncode != 0:
        # Don't delete files in case helpful for troubleshooting.
        return False

    # Link
    cc = compiler.linker_so[0]
    cmd = [cc] + compiler.linker_so[1:] + lflags + [o_name,'-o',exe_name]
    if debug:
        print('cmd = ',' '.join(cmd))
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = p.stdout.readlines()
        p.communicate()
        if debug and p.returncode != 0:
            print('Trying link command:')
            print(' '.join(cmd))
            print('Output was:')
            print('   ',b'   '.join(lines).decode())
        returncode = p.returncode
    except (IOError,OSError) as e:
        if debug:
            print('Trying link command:')
            print(' '.join(cmd))
            print('Caught error: ',repr(e))
        returncode = 1

    if returncode:
        # The linker needs to be a c++ linker, which isn't 'cc'.  However, I couldn't figure
        # out how to get setup.py to tell me the actual command to use for linking.  All the
        # executables available from build_ext.compiler.executables are 'cc', not 'c++'.
        # I think this must be related to the bugs about not handling c++ correctly.
        #    http://bugs.python.org/issue9031
        #    http://bugs.python.org/issue1222585
        # So just switch it manually and see if that works.
        if 'clang' in cc:
            cpp = cc.replace('clang', 'clang++')
        elif 'icc' in cc:
            cpp = cc.replace('icc', 'icpc')
        elif 'gcc' in cc:
            cpp = cc.replace('gcc', 'g++')
        elif ' cc' in cc:
            cpp = cc.replace(' cc', ' c++')
        elif cc == 'cc':
            cpp = 'c++'
        else:
            comp_type = get_compiler_type(compiler)
            if comp_type == 'gcc':
                cpp = 'g++'
            elif comp_type == 'clang':
                cpp = 'clang++'
            elif comp_type == 'icc':
                cpp = 'g++'
            else:
                cpp = 'c++'
        # Finally, if GALSIM_CXX is in the environment, let that take precedence.
        # (I don't know if it's safe to use a user's CXX always, so make sure the user really
        # meant to direct GalSim to use some other compiler by requiring the GALSIM prefix.)
        cpp = os.environ('GALSIM_CXX', cpp)
        cmd = [cpp] + compiler.linker_so[1:] + lflags + [o_name,'-o',exe_name]
        if debug:
            print('cmd = ',' '.join(cmd))
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines = p.stdout.readlines()
            p.communicate()
            if debug and p.returncode != 0:
                print('Trying link command:')
                print(' '.join(cmd))
                print('Output was:')
                print('   ',b'   '.join(lines).decode())
            returncode = p.returncode
        except (IOError,OSError) as e:
            if debug:
                print('Trying to link using command:')
                print(' '.join(cmd))
                print('Caught error: ',repr(e))
            returncode = 1

    # Remove the temp files
    if returncode != 0:
        # Don't delete files in case helpful for troubleshooting.
        return False
    else:
        os.remove(cpp_name)
        os.remove(o_name)
        if os.path.exists(exe_name):
            os.remove(exe_name)
        return True

def try_openmp(compiler, cc_type):
    """
    If cc --version is not helpful, the last resort is to try each compiler type and see
    if it works.
    """
    cpp_code = """
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include "omp.h"
#endif

int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int main() {
    int n = 500;
    std::vector<double> x(n,0.);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i=0; i<n; ++i) x[i] = 2*i+1;

    double sum = 0.;
    for (int i=0; i<n; ++i) sum += x[i];
    // Sum should be n^2 = 250000

    std::cout<<get_max_threads()<<"  "<<sum<<std::endl;
    return 0;
}
"""
    extra_cflags = copt[cc_type]
    extra_lflags = lopt[cc_type]
    success = try_compile(cpp_code, compiler, extra_cflags, extra_lflags)
    if not success:
        # In case libc++ doesn't work, try letting the system use the default stdlib
        try:
            extra_cflags.remove('-stdlib=libc++')
            extra_lflags.remove('-stdlib=libc++')
        except (AttributeError, ValueError):
            pass
        else:
            success = try_compile(cpp_code, compiler, extra_cflags, extra_lflags)
    return success


def try_cpp(compiler, cflags=[], lflags=[], prepend=None):
    """Check if compiling a simple bit of c++ code with the given compiler works properly.
    """
    from textwrap import dedent
    cpp_code = dedent("""
    #include <iostream>
    #include <vector>
    int main() {
        int n = 500;
        std::vector<double> x(n,0.);
        for (int i=0; i<n; ++i) x[i] = 2*i+1;
        double sum=0.;
        for (int i=0; i<n; ++i) sum += x[i];
        return sum;
    }
    """)
    return try_compile(cpp_code, compiler, cflags, lflags, prepend=prepend)

def try_cpp11(compiler, cflags=[], lflags=[]):
    """Check if compiling c++11 code with the given compiler works properly.
    """
    from textwrap import dedent
    cpp_code = dedent("""
    #include <iostream>
    #include <forward_list>
    #include <cmath>

    int main(void) {
        std::cout << std::tgamma(1.3) << std::endl;
        return 0;
    }
    """)
    return try_compile(cpp_code, compiler, cflags, lflags)


def cpu_count():
    """Get the number of cpus
    """
    try:
        import psutil
        return psutil.cpu_count()
    except ImportError:
        pass

    if hasattr(os, 'sysconf'):
        if 'SC_NPROCESSORS_ONLN' in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf('SC_NPROCESSORS_ONLN')
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else: # OSX:
            p = subprocess.Popen(['sysctl -n hw.ncpu'],stdout=subprocess.PIPE,shell=True)
            return int(p.stdout.read().strip())
    # Windows:
    if 'NUMBER_OF_PROCESSORS' in os.environ:
        ncpus = int(os.environ['NUMBER_OF_PROCESSORS'])
        if ncpus > 0:
            return ncpus
    return 1 # Default

def parallel_compile(self, sources, output_dir=None, macros=None,
                     include_dirs=None, debug=0, extra_preargs=None,
                     extra_postargs=None, depends=None):
    """New compile function that we monkey patch into the existing compiler instance.
    """
    import multiprocessing.pool

    # Copied from the regular compile function
    macros, objects, extra_postargs, pp_opts, build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                                depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # Set by fix_compiler
    global glob_use_njobs
    if glob_use_njobs == 1:
        # This is equivalent to regular compile function
        for obj in objects:
            _single_compile(obj)
    else:
        # Use ThreadPool, rather than Pool, since the objects are picklable.
        pool = multiprocessing.pool.ThreadPool(glob_use_njobs)
        pool.map(_single_compile, objects)
        pool.close()
        pool.join()

    # Return *all* object filenames, not just the ones we just built.
    return objects


def fix_compiler(compiler, njobs):
    # Remove any -Wstrict-prototypes in the compiler flags (since invalid for C++)
    try:
        compiler.compiler_so.remove("-Wstrict-prototypes")
    except (AttributeError, ValueError):
        pass

    # Figure out what compiler it will use
    comp_type = get_compiler_type(compiler, output=True)
    cc = compiler.compiler_so[0]
    already_have_ccache = False
    if cc == 'ccache':
        already_have_ccache = True
        cc = compiler.compiler_so[1]
    if cc == comp_type:
        print('Using compiler %s'%(cc))
    else:
        print('Using compiler %s, which is %s'%(cc,comp_type))

    # Make sure the compiler works with a simple c++ code
    if not try_cpp(compiler):
        # One failure mode is that sometimes there is a -B /path/to/compiler_compat
        # which can cause problems.  If we get here, try removing that.
        success = False
        if '-B' in compiler.linker_so:
            for i in range(len(compiler.linker_so)):
                if (compiler.linker_so[i] == '-B' and
                    'compiler_compat' in compiler.linker_so[i+1]):
                    print('Removing potentially problematic -B compiler_compat flags')
                    del compiler.linker_so[i:i+2]
                    success = try_cpp(compiler)
                    break
        if not success:
            print("There seems to be something wrong with the compiler or cflags")
            print(str(compiler.compiler_so))
            raise OSError("Compiler does not work for compiling C++ code")

    # Check if we can use ccache to speed up repeated compilation.
    if not already_have_ccache and try_cpp(compiler, prepend='ccache'):
        print('Using ccache')
        compiler.set_executable('compiler_so', ['ccache'] + compiler.compiler_so)

    if njobs > 1:
        # Global variable for tracking the number of jobs to use.
        # We can't pass this to parallel compile, since the signature is fixed.
        # So if using parallel compile, set this value to use within parallel compile.
        global glob_use_njobs
        glob_use_njobs = njobs
        compiler.compile = types.MethodType(parallel_compile, compiler)

    extra_cflags = copt[comp_type]
    extra_lflags = lopt[comp_type]

    success = try_cpp11(compiler, extra_cflags, extra_lflags)
    if not success:
        # In case libc++ doesn't work, try letting the system use the default stdlib
        try:
            extra_cflags.remove('-stdlib=libc++')
            extra_lflags.remove('-stdlib=libc++')
        except (AttributeError, ValueError):
            pass
        else:
            success = try_cpp11(compiler, extra_cflags, extra_lflags)
    if not success:
        print('The compiler %s with flags %s did not successfully compile C++11 code'%
              (cc, ' '.join(extra_cflags)))
        raise OSError("Compiler is not C++-11 compatible")

    # If doing develop installation, it's important for the build directory to be before any
    # other directories.  Particularly ones that might have another version of GalSim installed.
    # Otherwise the wrong library can be linked, which leads to errors.
    # So, make sure that the -Lbuild/... directive happens first among any -L directives in
    # the link flags.
    linker_so = compiler.linker_so
    # Find the first -L flag among the current flags (if any)
    for i, flag in enumerate(linker_so):
        if flag.startswith('-L'):
            print('Found link: ',i,flag)
            break
    else:
        i = len(linker_so)
    # Insert -Llib for any libs that are in build directory, to make sure they are first.
    linker_so[i:i] = ['-L' + l for l in compiler.library_dirs if l.startswith('build')]
    # Copy this list back to the compiler object
    compiler.set_executable('linker_so', linker_so)

    # Return the extra cflags, since those will be added to the build step in a different place.
    print('Using extra flags ',extra_cflags)
    return extra_cflags, extra_lflags

def add_dirs(builder, output=False):
    if debug: output = True
    # We need to do most of this both for build_clib and build_ext, so separate it out here.

    # First some basic ones we always need.
    builder.include_dirs.append('include')
    builder.include_dirs.append('include/galsim')

    # Look for fftw3.
    fftw_lib = find_fftw_lib(output=output)
    fftw_libpath, fftw_libname = os.path.split(fftw_lib)
    if hasattr(builder, 'library_dirs'):
        if fftw_libpath != '':
            builder.library_dirs.append(fftw_libpath)
        builder.libraries.append('galsim')  # Make sure galsim comes before fftw3
        builder.libraries.append(os.path.split(fftw_lib)[1].split('.')[0][3:])
    fftw_include = os.path.join(os.path.split(fftw_libpath)[0], 'include')
    if os.path.isfile(os.path.join(fftw_include, 'fftw3.h')):
        print('Include directory for fftw3 is ',fftw_include)
        # Usually, the fftw3.h file is in an associated include dir, but not always.
        builder.include_dirs.append(fftw_include)
    else:
        # If not, we have our own copy of fftw3.h here.
        print('Using local copy of fftw3.h')
        builder.include_dirs.append('include/fftw3')

    # Look for Eigen/Core
    eigen_dir = find_eigen_dir(output=output)
    builder.include_dirs.append(eigen_dir)

    # Finally, add pybind11's include dir
    import pybind11
    print('PyBind11 is version ',pybind11.__version__)
    print('Looking for pybind11 header files: ')
    locations = [pybind11.get_include(user=True),
                 pybind11.get_include(user=False),
                 '/usr/include',
                 '/usr/local/include',
                 None]
    for try_dir in locations:
        if try_dir is None:
            # Last time through, raise an error.
            print("Could not find pybind11 header files.")
            print("They should have been in one of the following locations:")
            for l in locations:
                if l is not None:
                    print("   ", l)
            raise OSError("Could not find PyBind11")

        print('  ',try_dir,end='')
        if os.path.isfile(os.path.join(try_dir, 'pybind11/pybind11.h')):
            print('  (yes)')
            builder.include_dirs.append(try_dir)
            break
        else:
            print('  (no)')

def parse_njobs(njobs, task=None, command=None, maxn=4):
    """Helper function to parse njobs, which may be None (use ncpu) or an int.
    Returns an int value for njobs
    """
    if njobs is None:
        njobs = cpu_count()
        if maxn != None and njobs > maxn:
            # Usually 4 is plenty.  Testing with too many jobs tends to lead to
            # memory and timeout errors.  The user can bump this up if they want.
            njobs = maxn
        if task is not None:
            if njobs == 1:
                print('Using a single process for %s.'%task)
            else:
                print('Using %d cpus for %s'%(njobs,task))
            print('To override, you may do python setup.py %s -jN'%command)
    else:
        njobs = int(njobs)
        if task is not None:
            if njobs == 1:
                print('Using a single process for %s.'%task)
            else:
                print('Using %d cpus for %s'%(njobs,task))
    return njobs

do_output = True  # Keep track of whether we used output=True in add_dirs yet.
                  # It seems that different installation methods do things in different order,
                  # but we only want to output on the first pass through add_dirs.
                  # (Unless debug = True, then also output in the second pass.)

# Make a subclass of build_ext so we can add to the -I list.
class my_build_clib(build_clib):
    user_options = build_ext.user_options + [('njobs=', 'j', "Number of jobs to use for compiling")]

    def initialize_options(self):
        build_clib.initialize_options(self)
        self.njobs = None

    def finalize_options(self):
        global do_output
        build_clib.finalize_options(self)
        if self.njobs is None and 'glob_njobs' in globals():
            global glob_njobs
            self.njobs = glob_njobs
        add_dirs(self, output=do_output)
        do_output = False

    # Add any extra things based on the compiler being used..
    def build_libraries(self, libraries):

        build_ext = self.distribution.get_command_obj('build_ext')
        njobs = parse_njobs(self.njobs, 'compiling', 'install')

        cflags, lflags = fix_compiler(self.compiler, njobs)

        # Add the appropriate extra flags for that compiler.
        for (lib_name, build_info) in libraries:
            build_info['cflags'] = build_info.get('cflags',[]) + cflags
            build_info['lflags'] = build_info.get('lflags',[]) + lflags

        self.do_build_libraries(libraries)

    def do_build_libraries(self, libraries):
        # This version just calls the setuptools build_libraries function.
        # We'll change this for build_shared_clib below.
        build_clib.build_libraries(self, libraries)

class my_build_shared_clib(my_build_clib):

    def do_build_libraries(self, libraries):
        from distutils.errors import DistutilsSetupError
        from distutils import log
        from setuptools.dep_util import newer_pairwise_group
        from distutils.ccompiler import CCompiler

        builder = self.distribution.get_command_obj('build_ext')
        cflags, lflags = fix_compiler(self.compiler, 1)

        # Most of this is the setuptools version of the build_libraries function.
        # We just change the final link command to build a shared library that can be linked
        # to C++ programs that just want the C++ library.
        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                       "in 'libraries' option (library '%s'), "
                       "'sources' must be present and must be "
                       "a list of source filenames" % lib_name)
            sources = list(sources)

            log.info("building '%s' library", lib_name)

            # Make sure everything is the correct type.
            # obj_deps should be a dictionary of keys as sources
            # and a list/tuple of files that are its dependencies.
            obj_deps = build_info.get('obj_deps', dict())
            if not isinstance(obj_deps, dict):
                raise DistutilsSetupError(
                    "in 'libraries' option (library '%s'), "
                    "'obj_deps' must be a dictionary of "
                    "type 'source: list'" % lib_name)
            dependencies = []

            # Get the global dependencies that are specified by the '' key.
            # These will go into every source's dependency list.
            global_deps = obj_deps.get('', list())
            if not isinstance(global_deps, (list, tuple)):
                raise DistutilsSetupError(
                    "in 'libraries' option (library '%s'), "
                    "'obj_deps' must be a dictionary of "
                    "type 'source: list'" % lib_name)

            # Build the list to be used by newer_pairwise_group
            # each source will be auto-added to its dependencies.
            for source in sources:
                src_deps = [source]
                src_deps.extend(global_deps)
                extra_deps = obj_deps.get(source, list())
                if not isinstance(extra_deps, (list, tuple)):
                    raise DistutilsSetupError(
                        "in 'libraries' option (library '%s'), "
                        "'obj_deps' must be a dictionary of "
                        "type 'source: list'" % lib_name)
                src_deps.extend(extra_deps)
                dependencies.append(src_deps)

            expected_objects = self.compiler.object_filenames(
                sources,
                output_dir=self.build_temp,
            )

            if (
                newer_pairwise_group(dependencies, expected_objects)
                != ([], [])
            ):
                # First, compile the source code to object files in the library
                # directory.  (This should probably change to putting object
                # files in a temporary build directory.)
                macros = build_info.get('macros')
                include_dirs = build_info.get('include_dirs')
                cflags = build_info.get('cflags')
                self.compiler.compile(
                    sources,
                    output_dir=self.build_temp,
                    macros=macros,
                    include_dirs=include_dirs,
                    extra_postargs=cflags,
                    debug=self.debug
                )

            ###
            ###
            ### This is the one bit that is changed from the setuptools version.
            ### The original used self.compiler.create_static_lib
            ###
            ###
            lib_name = self.compiler.library_filename(lib_name, lib_type="shared")
            version_str = '{}.{}'.format(*version_info[:2])
            if sys.platform == "darwin":
                # .so -> .dylib
                lib_name = lib_name.replace('so','dylib')
                full_lib_name = lib_name[:-5] + version_str + '.dylib'
                orig_linker_so = self.compiler.linker_so
                assert orig_linker_so[1] == '-bundle'
                dylib_linker_so = orig_linker_so.copy()
                dylib_linker_so[1] = '-dynamiclib'
                self.compiler.set_executable('linker_so', dylib_linker_so)
            else:
                # Just add the version bit
                full_lib_name = lib_name[:-2] + version_str + '.so'

            libraries = builder.get_libraries(ext)
            print('initial libraries = ',libraries)

            library_dirs = ext.library_dirs
            print('initial library_dirs = ',library_dirs)

            # Link to fftw library
            fftw_lib = find_fftw_lib()
            fftw_libpath, fftw_libname = os.path.split(fftw_lib)
            if fftw_libpath != '':
                library_dirs.append(fftw_libpath)
            libraries.append(fftw_libname.split('.')[0][3:])

            # Check for conda libraries that might host OpenMP
            env = dict(os.environ)
            if 'CONDA_PREFIX' in env:
                library_dirs.append(env['CONDA_PREFIX']+'/lib')

            if sys.platform == 'darwin':
                # Set the compatibility version on macos
                lflags.extend(['-Wl,-compatibility_version,%s.%s'%version_info[:2],
                               '-Wl,-current_version,%s.%s.%s'%version_info ])
                # Also add rpath specification for fftw
                if fftw_libpath != '':
                    lflags.append('-Wl,-rpath,' + fftw_libpath)
                # Set the install_name to the right rpath name.
                lflags.append('-Wl,-install_name,@rpath/%s'%full_lib_name)

            output_dir = os.path.join('build','shared_clib')
            self.compiler.link(CCompiler.SHARED_OBJECT, expected_objects, full_lib_name,
                               libraries=libraries,
                               library_dirs=library_dirs,
                               runtime_library_dirs=ext.runtime_library_dirs,
                               extra_postargs=ext.extra_link_args + lflags,
                               output_dir=output_dir, debug=self.debug)

            # Also make the non-versionful one
            full_lib_name_with_dir = os.path.join(output_dir, full_lib_name)
            lib_name_with_dir = os.path.join(output_dir, lib_name)
            print('Versioned library: ',full_lib_name_with_dir)
            print('Un-versioned library: ',lib_name_with_dir)
            if not os.path.exists(lib_name_with_dir):
                # This is slightly confusing.
                # The target needs the dir, but the source cannot include the dir or it
                # will try to link to build/shared_clib/build/shared_clib/libgalsim...
                os.symlink(full_lib_name, lib_name_with_dir)



# Make a subclass of build_ext so we can add to the -I list.
class my_build_ext(build_ext):
    user_options = build_ext.user_options + [('njobs=', 'j', "Number of jobs to use for compiling")]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.njobs = None

    def finalize_options(self):
        global do_output
        build_ext.finalize_options(self)
        # I couldn't find an easy way to send the user option from my_install to my_buld_ext.
        # So use a global variable. (UGH!)
        if self.njobs is None and 'glob_njobs' in globals():
            global glob_njobs
            self.njobs = glob_njobs
        add_dirs(self, output=do_output)
        do_output = False

    # Add any extra things based on the compiler being used..
    def build_extensions(self):

        njobs = parse_njobs(self.njobs, 'compiling', 'install')
        cflags, lflags = fix_compiler(self.compiler, njobs)

        # Add the appropriate extra flags for that compiler.
        for e in self.extensions:
            e.extra_compile_args = cflags
            for flag in lflags:
                e.extra_link_args.append(flag)

        # Now run the normal build function.
        build_ext.build_extensions(self)

    def run(self):
        # `python setup.py develop` (used implicitly by `pip install -e .`) only calls
        # `build_ext`, unlike `install` which calls `build` and all its related
        # sub-commands. Linking -lgalsim fails, since `build_clib` isn't calld.
        # This override ensures that `build_clib` command is run before the
        # pysrc stuff is linked.
        # cf. https://github.com/pypa/pip/issues/4523

        self.run_command("build_clib")
        build_ext.run(self)

        # If requested, also build the shared library.
        if int(os.environ.get('GALSIM_BUILD_SHARED', 0)):
            self.run_command("build_shared_clib")


class my_install(install):
    user_options = install.user_options + [('njobs=', 'j', "Number of jobs to use for compiling")]

    def initialize_options(self):
        install.initialize_options(self)
        self.njobs = None

    def finalize_options(self):
        install.finalize_options(self)
        global glob_njobs
        glob_njobs = self.njobs


# AFAICT, setuptools doesn't provide any easy access to the final installation location of the
# executable scripts.  This bit is just to save the value of script_dir so I can use it later.
# cf. http://stackoverflow.com/questions/12975540/correct-way-to-find-scripts-directory-from-setup-py-in-python-distutils/
class my_easy_install(easy_install):    # Used when installing via python setup.py install
    # Match the call signature of the easy_install version.
    def write_script(self, script_name, contents, mode="t", *ignored):
        # Run the normal version
        easy_install.write_script(self, script_name, contents, mode, *ignored)
        # Save the script install directory in the distribution object.
        # This is the same thing that is returned by the setup function.
        self.distribution.script_install_dir = self.script_dir

class my_install_scripts(install_scripts):  # Used when pip installing.
    def run(self):
        install_scripts.run(self)
        self.distribution.script_install_dir = self.install_dir

class my_test(test):
    # cf. https://pytest.readthedocs.io/en/2.7.3/goodpractises.html
    user_options = [('njobs=', 'j', "Number of jobs to use in py.test")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = None
        self.njobs = None

    def finalize_options(self):
        test.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_cpp_tests(self):
        builder = self.distribution.get_command_obj('build_ext')
        compiler = builder.compiler
        cflags, lflags = fix_compiler(compiler, 1)

        ext = builder.extensions[0]
        objects = compiler.compile(test_sources,
                output_dir=builder.build_temp,
                macros=ext.define_macros,
                include_dirs=ext.include_dirs,
                extra_postargs=cflags,
                debug=builder.debug,
                depends=ext.depends)

        if ext.extra_objects:
            objects.extend(ext.extra_objects)

        libraries = builder.get_libraries(ext)
        libraries.append('galsim')
        library_dirs = ext.library_dirs

        # Use the shared library when building the c++ executables to make sure it works.
        self.run_command("build_shared_clib")
        library_dirs.append('build/shared_clib')

        # Check for conda libraries that might host OpenMP
        env = dict(os.environ)
        if 'CONDA_PREFIX' in env:
            library_dirs.append(env['CONDA_PREFIX']+'/lib')

        exe_file = os.path.join(builder.build_temp,'cpp_test')
        compiler.link_executable(
                objects, 'cpp_test',
                output_dir=builder.build_temp,
                libraries=libraries,
                library_dirs=library_dirs,
                runtime_library_dirs=ext.runtime_library_dirs,
                debug=builder.debug,
                target_lang='c++')

        # Might need extra dirs in LD_LIBRARY_PATH.  Just add them to make sure.
        for flag in compiler.linker_so:
            if flag.startswith('-L'):
                library_dirs.append(flag[2:])
        if 'LD_LIBRARY_PATH' not in env:
            env['LD_LIBRARY_PATH'] = env.get('LSST_LIBRARY_PATH','')
        if 'DYLD_LIBRARY_PATH' not in env:
            env['DYLD_LIBRARY_PATH'] = env.get('LSST_LIBRARY_PATH','')
        env['LD_LIBRARY_PATH'] += ':' + ':'.join(library_dirs)
        env['DYLD_LIBRARY_PATH'] += ':' + ':'.join(library_dirs)

        # Run the test executable.
        # And pass this env to the execution environment.
        p = subprocess.Popen([exe_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        lines = p.stdout.readlines()
        p.communicate()
        for line in lines:
            print(line.decode().strip())
        if p.returncode != 0:
            raise RuntimeError("C++ tests failed")
        print("All C++ tests passed.")

    def run_tests(self):

        if int(os.environ.get('GALSIM_TEST_PY', 1)):
            njobs = parse_njobs(self.njobs, 'pytest', 'test')
            pytest_args = ['-n=%d'%njobs, '--timeout=60']
            original_dir = os.getcwd()
            os.chdir('tests')
            test_files = glob.glob('test*.py')

            import pytest
            pytest.main(['--version'])
            errno = pytest.main(pytest_args + test_files)
            py_err = errno != 0

            os.chdir(original_dir)

        # Build and run the C++ tests
        if int(os.environ.get('GALSIM_TEST_CPP', 1)):
            self.run_cpp_tests()

        if int(os.environ.get('GALSIM_TEST_PY', 1)):
            if py_err:
                raise RuntimeError("Some Python tests failed")
            else:
                print("All python tests passed.")


lib=("galsim", {'sources' : cpp_sources,
                'depends' : headers,
                'include_dirs' : ['include', 'include/galsim'],
                'undef_macros' : undef_macros })
ext=Extension("galsim._galsim",
              py_sources,
              depends = cpp_sources + headers,
              undef_macros = undef_macros)

build_dep = ['setuptools>=38', 'pybind11>=2.2']
run_dep = ['numpy', 'astropy', 'LSSTDESC.Coord']
test_dep = ['pytest', 'pytest-xdist', 'pytest-timeout', 'nose', 'scipy', 'pyyaml']
# Note: Even though we don't use nosetests, nose is required for some tests to work.
#       cf. https://gist.github.com/dannygoldstein/e18866ebb9c39a2739f7b9f16440e2f5

# If Eigen doesn't exist in the normal places, download it.
find_eigen_dir(output=True)

with open('README.rst') as file:
    long_description = file.read()

# Read in the galsim version from galsim/_version.py
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file=os.path.join('galsim','_version.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    galsim_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('GalSim version is %s'%(galsim_version))

# Write a Version.h file that has this information for people using the C++ library.
vi = re.split('\.|-',galsim_version)
version_info = tuple([int(x) for x in vi if x.isdigit()])
if len(version_info) == 2:
    version_info = version_info + (0,)
version_h_text = """
// This file is auto-generated by SCons.  Do not edit.
#define GALSIM_MAJOR %d
#define GALSIM_MINOR %d
#define GALSIM_REVISION %d

#include <string>
#include <sstream>

#if defined(__GNUC__)
#define PUBLIC_API __attribute__ ((visibility ("default")))
#else
#define PUBLIC_API
#endif

namespace galsim {
    // Compiled versions of the above #define values.
    PUBLIC_API int major_version();
    PUBLIC_API int minor_version();
    PUBLIC_API int revision();

    // Returns string of the form "1.4.2"
    PUBLIC_API std::string version();

    // Checks if the compiled library version matches the #define values in this header file.
    inline bool check_version() {
        // Same code as version(), but inline, so we get the above values to compare
        // to the values compiled into the library.
        std::ostringstream oss;
        oss << GALSIM_MAJOR << '.' << GALSIM_MINOR << '.' << GALSIM_REVISION;
        return oss.str() == version();
    }
}
"""%version_info[:3]
version_h_file = os.path.join('include', 'galsim', 'Version.h')
with open(version_h_file, 'w') as f:
    f.write(version_h_text)

headers.append(version_h_file)

dist = setup(name="GalSim",
    version=galsim_version,
    author="GalSim Developers (point of contact: Mike Jarvis)",
    author_email="michael@jarvis.net",
    description="The modular galaxy image simulation toolkit",
    long_description=long_description,
    license = "BSD License",
    url="https://github.com/rmjarvis/GalSim",
    download_url="https://github.com/GalSim-developers/GalSim/releases/tag/v%s.zip"%galsim_version,
    packages=find_packages(),
    package_data={'galsim' : shared_data + headers},
    libraries=[lib],
    ext_modules=[ext],
    setup_requires=build_dep,
    install_requires=build_dep + run_dep,
    tests_require=test_dep,
    cmdclass = {'build_ext': my_build_ext,
                'build_clib': my_build_clib,
                'build_shared_clib': my_build_shared_clib,
                'install': my_install,
                'install_scripts': my_install_scripts,
                'easy_install': my_easy_install,
                'test': my_test,
                },
    entry_points = {'console_scripts' : [
            'galsim = galsim.__main__:run_main',
            'galsim_download_cosmos = galsim.download_cosmos:run_main'
            ]},
    zip_safe=False,
    )

# Check that the path includes the directory where the scripts are installed.
real_env_path = [os.path.realpath(d) for d in os.environ['PATH'].split(':')]
if hasattr(dist,'script_install_dir'):
    print('scripts installed into ',dist.script_install_dir)
    if (dist.script_install_dir not in os.environ['PATH'].split(':') and
        os.path.realpath(dist.script_install_dir) not in real_env_path):

        print('\nWARNING: The GalSim executables were installed in a directory not in your PATH')
        print('         If you want to use the executables, you should add the directory')
        print('\n             ',dist.script_install_dir,'\n')
        print('         to your path.  The current path is')
        print('\n             ',os.environ['PATH'],'\n')
        print('         Alternatively, you can specify a different prefix with --prefix=PREFIX,')
        print('         in which case the scripts will be installed in PREFIX/bin.')
        print('         If you are installing via pip use --install-option="--prefix=PREFIX"')
