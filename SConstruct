# vim: set filetype=python et ts=4 sw=4:

import os
import subprocess
import sys
import SCons
import platform
import distutils.sysconfig
from sys import stdout,stderr

print 'SCons is version',SCons.__version__,'using python version',platform.python_version()

print "Python is from", distutils.sysconfig.get_python_inc()

# Require SCons version >= 1.1
# (This is the earliest version I could find to test on.  Probably works with 1.0.)
EnsureSConsVersion(1, 1)

# Subdirectories containing SConscript files.  We always process these, but
# there are some other optional ones
subdirs=['src','pysrc','galsim']

# Configurations will be saved here so command line options don't
# have to be sent more than once
config_file = 'gs_scons.conf'

# Default directory for installation.  
# This is the only UNIX specific things I am aware
# of in the script.  On the other hand, these are not required for the
# script to work since prefix can be set on the command line and the
# extra paths are not needed, but I wish I knew how to get the default 
# prefix for the system so I didn't have to set this.

# MJ: Is there a python function that might return this in a more platform-independent way?
default_prefix = '/usr/local'  

# TODO: This is JB's recommendation.  Cool?
default_py_prefix = distutils.sysconfig.get_python_lib() 

# first check for a saved conf file
opts = Variables(config_file)

# Now set up options for the command line
opts.Add('CXX','Name of c++ compiler')
opts.Add('FLAGS','Compile flags to send to the compiler','')
opts.Add('EXTRA_FLAGS','Extra flags to send to the compiler','')
opts.Add(BoolVariable('DEBUG','Turn on debugging statements',True))

opts.Add(PathVariable('PREFIX','prefix for installation','', PathVariable.PathAccept))
opts.Add(PathVariable('PYPREFIX','location of your site-packages directory',
        default_py_prefix,PathVariable.PathAccept))

opts.Add(PathVariable('EXTRA_PATH',
            'Extra paths for executables (separated by : if more than 1)',
            '',PathVariable.PathAccept))
opts.Add(PathVariable('EXTRA_LIB_PATH',
            'Extra paths for linking (separated by : if more than 1)',
            '',PathVariable.PathAccept))
opts.Add(PathVariable('EXTRA_INCLUDE_PATH',
            'Extra paths for header files (separated by : if more than 1)',
            '',PathVariable.PathAccept))
opts.Add(BoolVariable('IMPORT_PATHS',
            'Import PATH, C_INCLUDE_PATH and LIBRARY_PATH/LD_LIBRARY_PATH environment variables',
            False))
opts.Add(BoolVariable('IMPORT_ENV',
            'Import full environment from calling shell',True))
opts.Add(BoolVariable('INCLUDE_PREFIX_PATHS',
            'Add PREFIX/bin, PREFIX/include and PREFIX/lib to corresponding path lists',
            True))

opts.Add('TMV_DIR','Explicitly give the tmv prefix','')
opts.Add('CFITSIO_DIR','Explicitly give the cfitsio prefix','')
opts.Add('FFTW_DIR','Explicitly give the fftw3 prefix','')
opts.Add('BOOST_DIR','Explicitly give the boost prefix','')
#opts.Add('CCFITS_DIR','Explicitly give the ccfits prefix','')

opts.Add('TMV_LINK','File that contains the linking instructions for TMV','')
opts.Add('EXTRA_LIBS','Libraries to send to the linker','')
opts.Add(BoolVariable('CACHE_LIB','Cache the results of the library checks',True))

# None of the code uses openmp yet.  Probably make this default True if we start using it.
opts.Add(BoolVariable('WITH_OPENMP','Look for openmp and use if found.', False))
opts.Add(BoolVariable('MEM_TEST','Test for memory leaks', False))
opts.Add(BoolVariable('WARN','Add warning compiler flags, like -Wall', True))
opts.Add(BoolVariable('TMV_DEBUG','Turn on extra debugging statements within TMV library',False))

#opts.Add(BoolVariable('WITH_UPS',
            #'Create ups/galsim.table.  Install the ups directory under PREFIX/ups',
            #False))
opts.Add(BoolVariable('WITH_PROF',
            'Use the compiler flag -pg to include profiling info for gprof',
            False))

# This helps us determine of openmp is available
openmp_mingcc_vers = 4.1
openmp_minicpc_vers = 9.1  # 9.0 is supposed to work, but has bugs
openmp_minpgcc_vers = 6.0
openmp_mincc_vers = 5.0    # I don't actually know what this should be.

def RunInstall(env, targets, subdir):
    install_dir = os.path.join(env['INSTALL_PREFIX'], subdir)
    env.Alias(target='install',
              source=env.Install(dir=install_dir, source=targets))

def RunUninstall(env, targets, subdir):
    # There is no env.Uninstall method, we must build our own
    install_dir = os.path.join(env['INSTALL_PREFIX'], subdir)
    deltarget = Delete("$TARGET")

    # delete from $prefix/bin/
    files = []
    for t in targets:
        ifile = os.path.join(install_dir, os.path.basename(str(t))) 
        files.append(ifile)

    for f in files:
        env.Alias('uninstall', env.Command(f, None, deltarget))


def BasicCCFlags(env):
    """
    """

    compiler = env['CXXTYPE']
    version = env['CXXVERSION_NUMERICAL']

    # First parse the EXTRA_LIBS option if present
    if env['EXTRA_LIBS'] == '':
        env.Replace(LIBS=[])
    else:
        libs = env['EXTRA_LIBS'].split(' ')
        env.Replace(LIBS=libs)

    if compiler == 'g++' and version >= 4.4:
        # Workaround for a bug in the g++ v4.4 exception handling
        # I don't think 4.5 or 4.6 actually need it, but keep >= for now
        # just to be safe.
        env.AppendUnique(LIBS='pthread')

    if env['FLAGS'] == '':
        if compiler == 'g++':
            env.Replace(CCFLAGS=['-O2'])
            env.Append(CCFLAGS=['-fno-strict-aliasing'])
            if env['WITH_PROF']:
                env.Append(CCFLAGS=['-pg'])
                env.Append(LINKFLAGS=['-pg'])
            if env['WARN']:
                env.Append(CCFLAGS=['-g3','-Wall','-Werror'])
    
        elif compiler == 'clang++':
            env.Replace(CCFLAGS=['-O2'])
            if env['WITH_PROF']:
                env.Append(CCFLAGS=['-pg'])
                env.Append(LINKFLAGS=['-pg'])
            if env['WARN']:
                env.Append(CCFLAGS=['-g3','-Wall','-Werror'])
    
        elif compiler == 'icpc':
            env.Replace(CCFLAGS=['-O2'])
            if version >= 10:
                env.Append(CCFLAGS=['-vec-report0'])
            if env['WITH_PROF']:
                env.Append(CCFLAGS=['-pg'])
                env.Append(LINKFLAGS=['-pg'])
            if env['WARN']:
                env.Append(CCFLAGS=['-g','-Wall','-Werror','-wd279,383,810,981'])
                if version >= 9:
                    env.Append(CCFLAGS=['-wd1572'])
                if version >= 11:
                    env.Append(CCFLAGS=['-wd2259'])

        elif compiler == 'pgCC':
            env.Replace(CCFLAGS=['-O2','-fast','-Mcache_align'])
            if env['WITH_PROF']:
                env.Append(CCFLAGS=['-pg'])
                env.Append(LINKFLAGS=['-pg'])
            if env['WARN']:
                env.Append(CCFLAGS=['-g'])

        elif compiler == 'CC':
            env.Replace(CCFLAGS=['-O2','-fast','-instances=semiexplicit'])
            if env['WARN']:
                env.Append(CCFLAGS=['-g','+w'])

        elif compiler == 'cl':
            env.Replace(CCFLAGS=['/EHsc','/nologo','/O2','/Oi'])
            if env['WARN']:
                env.Append(CCFLAGS=['/W2','/WX'])

        else:
            print '\nWARNING: Unknown compiler.  You should set FLAGS directly.\n'
            env.Replace(CCFLAGS=[])

    else :
        # If flags are specified as an option use them:
        cxx_flags = env['FLAGS'].split(' ')
        env.Replace(CCFLAGS=cxx_flags)

    extra_flags = env['EXTRA_FLAGS'].split(' ')
    env.AppendUnique(CCFLAGS=extra_flags)


def AddOpenMPFlag(env):
    """
    Make sure you do this after you have determined the version of
    the compiler.

    g++ uses -fopemnp
    clang++ doesn't have OpenMP support yet
    icpc uses -openmp
    pgCC uses -mp
    CC uses -xopenmp
    
    Other compilers?
    """
    compiler = env['CXXTYPE']
    version = env['CXXVERSION_NUMERICAL']
    if compiler == 'g++':
        if version < openmp_mingcc_vers: 
            print 'No OpenMP support for g++ versions before ',openmp_mingcc_vers
            env['WITH_OPENMP'] = False
            return
        flag = ['-fopenmp']
        ldflag = ['-fopenmp']
        xlib = ['pthread']
        # Note: gcc_eh is required on MacOs, but not linux
        # Update: Starting with g++4.6, gcc_eh seems to break exception
        # throwing, and so I'm only going to use that for version <= 4.5.
        # Also, I learned how to check if the platform is darwin (aka MacOs)
        if (version <= 4.5) and (sys.platform.find('darwin') != -1):
            xlib += ['gcc_eh']
        env.Append(CCFLAGS=['-fopenmp'])
    elif compiler == 'clang++':
        print 'No OpenMP support for clang++'
        env['WITH_OPENMP'] = False
        return
    elif compiler == 'icpc':
        if version < openmp_minicpc_vers:
            print 'No OpenMP support for icpc versions before ',openmp_minicpc_vers
            env['WITH_OPENMP'] = False
            return
        flag = ['-openmp']
        ldflag = ['-openmp']
        xlib = ['pthread']
        env.Append(CCFLAGS=['-openmp'])
    elif compiler == 'pgCC':
        if version < openmp_minpgcc_vers:
            print 'No OpenMP support for pgCC versions before ',openmp_minpgcc_vers
            env['WITH_OPENMP'] = False
            return
        flag = ['-mp','--exceptions']
        ldflag = ['-mp']
        xlib = ['pthread']
    elif compiler == 'cl':
        #flag = ['/openmp']
        #ldflag = ['/openmp']
        #xlib = []
        # The Express edition, which is the one I have, doesn't come with
        # the file omp.h, which we need.  So I am unable to test TMV's
        # OpenMP with cl.  
        # I believe the Professional edition has full OpenMP support,
        # so if you have that, the above lines might work for you.
        # Just uncomment those, and commend the below three lines.
        print 'No OpenMP support for cl'
        env['WITH_OPENMP'] = False
        return
    else:
        print '\nWARNING: No OpenMP support for compiler ',compiler,'\n'
        env['WITH_OPENMP'] = False
        return

    #print 'Adding openmp support:',flag
    print 'Using OpenMP'
    env.AppendUnique(LINKFLAGS=ldflag)
    env.AppendUnique(LIBS=xlib)

def which(program):
    """
    Mimic functionality of unix which command
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    if sys.platform == "win32" and not program.endswith(".exe"):
        program += ".exe"

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
         for path in os.environ["PATH"].split(os.pathsep):
             exe_file = os.path.join(path, program)
             if is_exe(exe_file):
                 return exe_file

    return None

def GetCompilerVersion(env):
    """
    """
    compiler = which(env['CXX'])
    if compiler is None:
        raise ValueError("Specified compiler not found in path: %s" % env['CXX'])

    print 'Using compiler:',compiler

    compiler_real = os.path.realpath(compiler)
    compiler_base = os.path.basename(compiler)
    # Get the compiler type without suffix or path.  
    # e.g. /sw/bin/g++-4 -> g++
    if 'icpc' in compiler_base :
        compilertype = 'icpc'
        versionflag = '--version'
        linenum=0
    elif 'pgCC' in compiler_base :
        compilertype = 'pgCC'
        versionflag = '--version'
        linenum=1
        # pgCC puts the version number on the second line of output.
    elif 'clang++' in compiler_base :
        compilertype = 'clang++'
        versionflag = '--version'
        linenum=0
    elif 'g++' in compiler_base :
        compilertype = 'g++'
        versionflag = '--version'
        linenum=0
    elif 'CC' in compiler_base :
        compilertype = 'CC'
        versionflag = '-V'
        linenum=0
    elif 'cl' in compiler_base :
        compilertype = 'cl'
        versionflag = ''
        linenum=0
    elif 'c++' in compiler_base :
        compilertype = 'c++'
        versionflag = '--version'
        linenum=0
    else :
        compilertype = 'unknown'
        version = 0
        vnum = 0

    if compilertype != 'unknown':
        cmd = compiler + ' ' + versionflag + ' 2>&1'
        lines = os.popen(cmd).readlines()
    
        # Check if g++ is a symlink for something else:
        if compilertype is 'g++':
            if 'clang' in lines[0]:
                print 'Detected clang++ masquerading as g++'
                compilertype = 'clang++'
            # Any others I should look for?

        # Check if c++ is a symlink for something else:
        if compilertype is 'c++':
            if 'clang' in lines[0]:
                print 'Detected clang++ masquerading as c++'
                compilertype = 'clang++'
            elif 'g++' in lines[0] or 'gcc' in lines[0]:
                print 'Detected g++ masquerading as c++'
                compilertype = 'g++'
            else:
                print 'Cannot determine what kind of compiler c++ really is'
                compilertype = 'unknown'
            # Any others I should look for?

    # redo this check in case was c++ -> unknown
    if compilertype != 'unknown':
        line = lines[linenum]
        import re
        match = re.search(r'[0-9]+(\.[0-9]+)+', line)
    
        if match:
            version = match.group(0)
            # Get the version up to the first decimal
            # e.g. for 4.3.1 we only keep 4.3
            vnum = version[0:version.find('.')+2]
        else:
            version = 0
            vnum = 0

    print 'compiler version:',version

    env['CXXTYPE'] = compilertype
    env['CXXVERSION'] = version
    env['CXXVERSION_NUMERICAL'] = float(vnum)

def ExpandPath(path):
    p=os.path.expanduser(path)
    p=os.path.expandvars(p)
    return p

def AddPath(pathlist, newpath, prepend=False):
    """
    Add path(s) to a list of paths.  Check the path exists and that it is
    not already in the list
    """
    if type(newpath) == list:
        for l in newpath:
            AddPath(pathlist, l)
    else:
        # to deal with ~ and ${var} expansions
        p = ExpandPath(newpath)
        p = os.path.abspath(p)

        if os.path.exists(p):
            if p not in pathlist:
                if prepend:
                    pathlist.insert(0, p)
                else:
                    pathlist.append(p)


def AddDepPaths(bin_paths,cpp_paths,lib_paths):
    """

    Look for paths associated with the dependencies.  E.g. if TMV_DIR is set
    either on the command line or in the environment, add $TMV_DIR/include etc.
    to the paths.  Will add them to the back of the paths.  Also, we don't have
    to worry about dups because AddPath checks for that.

    """

    types = ['TMV','CFITSIO','FFTW','BOOST']

    for t in types:
        dirtag = t+'_DIR'
        tdir = FindPathInEnv(env, dirtag)
        if tdir is None:
            continue

        AddPath(bin_paths, os.path.join(tdir, 'bin'))
        AddPath(lib_paths, os.path.join(tdir, 'lib'))
        AddPath(cpp_paths, os.path.join(tdir, 'include'))


def AddExtraPaths(env):
    """
    Add some include and library paths.
    Also merge in $PATH, $C_INCLUDE_PATH and $LIBRARY_PATH/$LD_LIBRARY_PATH 
    environment variables if requested.
    
    The set itself is created in order of appearance here, but then this 
    whole set is prepended.  The order within this list is:

        local lib and include paths
        paths in FFTW_DIR, TMV_DIR, etc.
        paths in EXTRA_*PATH parameters
        paths in PREFIX directory
        paths from the user's environment

    Only paths that actually exists are kept.
    """
    # local includes and lib paths
    # The # symbol means to interpret these from the top-level scons
    # directory even when we are in a sub-directory (src,test,etc.)
    bin_paths = []
    cpp_paths = ['#include']
    lib_paths = ['#lib']

    # Add directories specified explicitly for our dependencies on the command
    # line or as an environment variable.
    AddDepPaths(bin_paths,cpp_paths,lib_paths)

    # Paths specified in EXTRA_*
    bin_paths += env['EXTRA_PATH'].split(':')
    lib_paths += env['EXTRA_LIB_PATH'].split(':')
    cpp_paths += env['EXTRA_INCLUDE_PATH'].split(':')

    # PREFIX directory
    # If none given, then don't add them to the -L and -I directories.
    # But still use the default /usr/local for installation
    if env['PREFIX'] == '':
        env['INSTALL_PREFIX'] = default_prefix
    else:
        if env['INCLUDE_PREFIX_PATHS']:
            AddPath(bin_paths, os.path.join(env['PREFIX'], 'bin'))
            AddPath(lib_paths, os.path.join(env['PREFIX'], 'lib'))
            AddPath(cpp_paths, os.path.join(env['PREFIX'], 'include'))
        env['INSTALL_PREFIX'] = env['PREFIX']
    
    # Paths found in environment paths
    if env['IMPORT_PATHS'] and os.environ.has_key('PATH'):
        paths=os.environ['PATH']
        paths=paths.split(os.pathsep)
        AddPath(bin_paths, paths)

    if env['IMPORT_PATHS'] and os.environ.has_key('C_INCLUDE_PATH'):
        paths=os.environ['C_INCLUDE_PATH']
        paths=paths.split(os.pathsep)
        AddPath(cpp_paths, paths)

    if env['IMPORT_PATHS'] and os.environ.has_key('LIBRARY_PATH'):
        paths=os.environ['LIBRARY_PATH']
        paths=paths.split(os.pathsep)
        AddPath(lib_paths, paths)

    if env['IMPORT_PATHS'] and os.environ.has_key('LD_LIBRARY_PATH'):
        paths=os.environ['LD_LIBRARY_PATH']
        paths=paths.split(os.pathsep)
        AddPath(lib_paths, paths)

    env.PrependENVPath('PATH', bin_paths)
    env.Prepend(LIBPATH= lib_paths)
    env.Prepend(CPPPATH= cpp_paths)

    # Also make sure PYPREFIX and environ(PYTHONPATH) are in sys.path
    if not os.path.abspath(env['PYPREFIX']) in sys.path:
        sys.path.append(os.path.abspath(env['PYPREFIX']))
    if (env['IMPORT_ENV'] and os.environ.has_key('PYTHONPATH') and
        not os.path.abspath(os.environ['PYTHONPATH']) in sys.path):
        sys.path.append(os.path.abspath(os.environ['PYTHONPATH']))


def ReadFileList(fname):
    """
    This reads a list of whitespace separated values from the input file fname
    and stores it as a list.  We will make this part of the environment so
    other SConscripts can use it
    """
    try:
        files=open(fname).read().split()
    except:
        print 'Could not open file:',fname
        sys.exit(45)
    files = [f.strip() for f in files]
    return files


def CheckLibs(context,try_libs,source_file):
    init_libs = context.env['LIBS']
    context.env.PrependUnique(LIBS=try_libs)
    result = context.TryLink(source_file,'.cpp')
    if not result :
        context.env.Replace(LIBS=init_libs)
    return result
      

def CheckTMV(context):
    tmv_source_file = """
#include "TMV_Sym.h"
int main()
{
  //tmv::SymMatrix<double> S(10,4.);
  tmv::Matrix<double> S(10,10,4.);
  tmv::Matrix<double> m(10,3,2.);
  tmv::Matrix<double> m2 = m / S;
  return 0;
}
"""

    print 'Checking for correct TMV linkage... (this may take a little while)'
    context.Message('Checking for correct TMV linkage... ')

    if context.TryCompile(tmv_source_file,'.cpp'):

        #result = (
            #CheckLibs(context,['tmv_symband','tmv'],tmv_source_file) or
            #CheckLibs(context,['tmv_symband','tmv','irc','imf'],tmv_source_file) )
        result = (
            CheckLibs(context,['tmv'],tmv_source_file) or
            CheckLibs(context,['tmv','irc','imf'],tmv_source_file) )
        
        if not result:
            context.Result(0)
            print 'Error: TMV file failed to link correctly'
            print 'Check that the correct location is specified for TMV_DIR'
            Exit(1)

        context.Result(1)
        return 1

    else:
        context.Result(0)
        print 'Error: TMV file failed to compile.'
        print 'Check that the correct location is specified for TMV_DIR'
        Exit(1)

def CheckPython(context):
    python_source_file = """
#include "Python.h"
int main()
{
  Py_Initialize();
  Py_Finalize();
  return 0;
}
"""
    context.Message('Checking if we can build against Python... ')

    try:
        import distutils.sysconfig
    except ImportError:
        context.Result(0)
        print 'Failed to import distutils.sysconfig.'
        Exit(1)
    context.env.AppendUnique(CPPPATH=distutils.sysconfig.get_python_inc())
    libDir = distutils.sysconfig.get_config_var("LIBDIR")
    context.env.AppendUnique(LIBPATH=libDir)
    libfile = distutils.sysconfig.get_config_var("LIBRARY")
    import re
    match = re.search("(python.*)\.(a|so|dylib)", libfile)
    if match:
        context.env.AppendUnique(LIBS=match.group(1))
    flags = [f for f in " ".join(distutils.sysconfig.get_config_vars("MODLIBS", "SHLIBS")).split()
             if f != "-L"]
    context.env.MergeFlags(" ".join(flags))

    result, output = context.TryRun(python_source_file,'.cpp')

    if not result and sys.platform == 'darwin':
        # Sometimes we need some extra stuff on Mac OS
        frameworkDir = libDir       # search up the libDir tree for the proper home for frameworks
        while frameworkDir and frameworkDir != "/":
            frameworkDir, d2 = os.path.split(frameworkDir)
            if d2 == "Python.framework":
                if not "Python" in os.listdir(os.path.join(frameworkDir, d2)):
                    context.Result(0)
                    print (
                        "Expected to find Python in framework directory %s, but it isn't there"
                        % frameworkDir)
                    Exit(1)
                break
        context.env.AppendUnique(LDFLAGS="-F%s"%frameworkDir)
        result, output = context.TryRun(python_source_file,'.cpp')

    if not result:
        context.Result(0)
        print "Cannot run program built with Python."
        Exit(1)

    context.Result(1)
    return 1

def CheckNumPy(context):
    numpy_source_file = """
#include "Python.h"
#include "numpy/arrayobject.h"
void doImport() {
  import_array();
}
int main()
{
  int result = 0;
  Py_Initialize();
  doImport();
  if (PyErr_Occurred()) {
    result = 1;
  } else {
    npy_intp dims = 2;
    PyObject * a = PyArray_SimpleNew(1, &dims, NPY_INT);
    if (!a) result = 1;
    Py_DECREF(a);
  }
  Py_Finalize();
  return result;
}
"""
    context.Message('Checking if we can build against NumPy... ')
    try:
        import numpy
    except ImportError:
        context.Result(0)
        print 'Failed to import numpy.'
        print 'Things to try:'
        print '1) Check that the python with which you installed numpy,'
        print '   probably the command line python:'
        print '   ',
        sys.stdout.flush()
        subprocess.call('which python',shell=True)
        print '   is the same as the one used by SCons:'
        print '  ',sys.executable
        print '   If not, then you probably need to reinstall numpy with %s.' % sys.executable
        print '   And remember to use that when running python for use with GalSim.'
        print '   Alternatively, you can reinstall SCons with your preferred python.'
        print '2) Check that if you open a python session from the command line,'
        print '   import numpy is successful there.'
        Exit(1)
    context.env.Append(CPPPATH=numpy.get_include())
    result = CheckLibs(context,[''],numpy_source_file)
    if not result:
        context.Result(0)
        print "Cannot build against NumPy."
        Exit(1)
    result, output = context.TryRun(numpy_source_file,'.cpp')
    if not result:
        context.Result(0)
        print "Cannot run program built with NumPy."
        Exit(1)
    context.Result(1)
    return 1

# Note from Barney to Mike: the code below always seems to say (cached) in the message at build
# for me, even after rm .scons.dblite beforehand.  Is that right?  Otherwise, it seems to work...
def CheckPyFITS(context):
    context.Message('Checking for PyFITS... ')
    try:
        import pyfits
    except ImportError:
        context.Result(0)
        print 'Failed to import PyFITS.'
        print 'Things to try:'
        print '1) Check that the python with which you installed PyFITS,'
        print '   probably the command line python:'
        print '   ',
        sys.stdout.flush()
        subprocess.call('which python',shell=True)
        print '   is the same as the one used by SCons:'
        print '  ',sys.executable
        print '   If not, then you probably need to reinstall PyFITS with %s.' % sys.executable
        print '   And remember to use that when running python for use with GalSim.'
        print '   Alternatively, you can reinstall SCons with your preferred python.'
        print '2) Check that if you open a python session from the command line,'
        print '   import pyfits is successful there.'
        Exit(1)
    context.Result(1)
    return 1

def CheckBoostPython(context):
    bp_source_file = """
#include "boost/python.hpp"

class Foo { public: Foo() {} };

int main()
{
  Py_Initialize();
  boost::python::object obj;
  boost::python::class_< Foo >("Foo", boost::python::init<>());
  Py_Finalize();
  return 0;
}
"""
    context.Message('Checking if we can build against Boost.Python... ')

    result = (
        CheckLibs(context,[''],bp_source_file) or
        CheckLibs(context,['boost_python'],bp_source_file) or
        CheckLibs(context,['boost_python-mt'],bp_source_file) )

    if not result:
        context.Result(0)
        print "Cannot build against Boost.Python."
        Exit(1)

    result, output = context.TryRun(bp_source_file,'.cpp')

    if not result:
        context.Result(0)
        print "Cannot run program built with Boost.Python."
        Exit(1)
    context.Result(1)
    return 1

def FindPathInEnv(env, dirtag):
    """
    Find the path tag in the environment and return
    The path must exist
    """
    dir = None
    # first try the local environment (which can be from the command
    # line), then try external environment
    if env[dirtag] != '':
        tmpdir = ExpandPath(env[dirtag])
        if os.path.exists(tmpdir):
            dir = tmpdir

    if dir is None and dirtag in os.environ:
        tmpdir = ExpandPath(os.environ[dirtag])
        if os.path.exists(tmpdir):
            dir=tmpdir
    return dir


def FindTmvLinkFile(config):

    # First check for an explict value for TMV_LINK
    if (config.env['TMV_LINK'] != '') :
        tmv_link = config.env['TMV_LINK']
        if os.path.exists(tmv_link):
            return tmv_link
        else:
            raise ValueError("Specified TMV_LINK does not exist: %s" % tmv_link)

    # Next check in TMV_DIR/share
    tmv_dir = FindPathInEnv(config.env, 'TMV_DIR')
    if tmv_dir is not None:
        tmv_share_dir = os.path.join(tmv_dir,'share')
        tmv_link = os.path.join(tmv_share_dir, 'tmv', 'tmv-link')
        if os.path.exists(tmv_link):
            return tmv_link
        # Older TMV was installed in prefix/share/ rather than prefix/share/tmv/
        # so check that too.  (At least for now.)
        tmv_link = os.path.join(tmv_share_dir, 'tmv-link')
        if os.path.exists(tmv_link):
            return tmv_link

    # If TMV_DIR is not given explicitly, it still probably found TMV.h somewhere,
    # And we want to make sure we use the tmv-link file that correspond with that TMV.h 
    # file, since there could be multiple installations of TMV on the machine, and 
    # we want to use the one that corresponds to the header file we found.
    for dir in config.env['CPPPATH']:
        h_file = os.path.join(ExpandPath(dir),'TMV.h')
        if os.path.exists(h_file):
            tmv_include_dir, junk = os.path.split(h_file)
            tmv_root_dir, incl = os.path.split(tmv_include_dir)
            if incl != 'include':
                # Weird, but possible.
                # If TMV.h is not in d/include/, then don't look in d/share for tmv-link
                continue
            tmv_share_dir = os.path.join(tmv_root_dir,'share')
            tmv_link = os.path.join(tmv_share_dir, 'tmv', 'tmv-link')
            if os.path.exists(tmv_link):
                return tmv_link
            tmv_link = os.path.join(tmv_share_dir, 'tmv-link')
            if os.path.exists(tmv_link):
                return tmv_link

    # Finally try /usr/local and also the install prefix (in case different)
    for prefix in [config.env['INSTALL_PREFIX'] , default_prefix ]:
        tmv_share_dir =  os.path.join(prefix,'share')
        tmv_link = os.path.join(tmv_share_dir, 'tmv','tmv-link')
        if os.path.exists(tmv_link):
            return tmv_link
        tmv_link = os.path.join(tmv_share_dir, 'tmv-link')
        if os.path.exists(tmv_link):
            return tmv_link

    raise ValueError("No tmv-link file could be found")


def DoLibraryAndHeaderChecks(config):
    """
    Check for some headers.  
    """

    # Check for cfitsio
    if not config.CheckLibWithHeader('cfitsio','fitsio.h',language='C++'):
        # Sometimes cfitsio uses include/cfitsio as the location for fitsio.h, so try that.
        config.env.AppendUnique(CPPPATH=os.path.join(config.env['CFITSIO_DIR'],'include','cfitsio'))
        if not config.CheckLibWithHeader('cfitsio','fitsio.h',language='C++'):
            print 'cfitsio not found'
            print 'You should specify the location of cfitsio CFITSIO_DIR=...'
            Exit(1)

    # Check for fftw3
    if not config.CheckLibWithHeader('fftw3','fftw3.h',language='C++'):
        print 'fftw3 not found'
        print 'You should specify the location of fftw3 as FFTW_DIR=...'
        Exit(1)

    # Check for boost
    if not config.CheckHeader('boost/shared_ptr.hpp',language='C++'):
        print 'Boost not found'
        print 'You should specify the location of Boost as BOOST_DIR=...'
        Exit(1)

    # Check for tmv
    # First do a simple check that the library and header are in the path.
    # We check the linking with the BLAS library below.
    if not config.CheckHeader('TMV.h',language='C++'):
        print 'TMV not found'
        print 'You should specify the location of TMV as TMV_DIR=...'
        Exit(1)

    compiler = config.env['CXXTYPE']
    version = config.env['CXXVERSION_NUMERICAL']

    if not (config.env.has_key('LIBS')) :
        config.env['LIBS'] = []

    tmv_link_file = FindTmvLinkFile(config)

    print 'Using TMV_LINK file:',tmv_link_file
    try:
        tmv_link = open(tmv_link_file).read().strip()
    except:
        print 'Could not open TMV link file: ',tmv_link_file
        Exit(1)
    print '    ',tmv_link

    # ParseFlags doesn't know about -fopenmp being a LINKFLAG, so it
    # puts it into CCFLAGS instead.  Move it over to LINKFLAGS before
    # merging everything.
    tmv_link_dict = config.env.ParseFlags(tmv_link)
    config.env.Append(LIBS=tmv_link_dict['LIBS'])
    config.env.AppendUnique(LINKFLAGS=tmv_link_dict['LINKFLAGS'])
    config.env.AppendUnique(LINKFLAGS=tmv_link_dict['CCFLAGS'])
    config.env.AppendUnique(LIBPATH=tmv_link_dict['LIBPATH'])
    
    if compiler == 'g++' and '-openmp' in config.env['LINKFLAGS']:
        config.env['LINKFLAGS'].remove('-openmp')
        config.env.AppendUnique(LINKFLAGS='-fopenmp')

    config.CheckTMV()
    config.CheckPython()
    config.CheckNumPy()
    config.CheckBoostPython()
    config.CheckPyFITS() 


def GetNCPU():
    """
    Detects the number of CPUs on a system. Cribbed from pp.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, 'sysconf'):
        if os.sysconf_names.has_key('SC_NPROCESSORS_ONLN'):
            # Linux & Unix:
            ncpus = os.sysconf('SC_NPROCESSORS_ONLN')
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else: # OSX:
            return int(os.popen2('sysctl -n hw.ncpu')[1].read())
    # Windows:
    if os.environ.has_key('NUMBER_OF_PROCESSORS'):
        ncpus = int(os.environ['NUMBER_OF_PROCESSORS'])
        if ncpus > 0:
            return ncpus
    return 1 # Default


def DoConfig(env):
    """
    Configure the system
    """


    # Add some extra paths 
    AddExtraPaths(env)

    # Figure out what kind of compiler we are dealing with
    GetCompilerVersion(env)
   
    # If not explicit, set number of jobs according to number of CPUs
    if env.GetOption('num_jobs') != 1:
        print "Using specified number of jobs =",env.GetOption('num_jobs')
    else:
        env.SetOption('num_jobs', GetNCPU())
        print "Determined that a good number of jobs =",env.GetOption('num_jobs')

    # The basic flags for this compiler if not explicitly specified
    BasicCCFlags(env)

    # Some extra flags depending on the options:
    if env['WITH_OPENMP']:
        print 'Using OpenMP'
        AddOpenMPFlag(env)
    if not env['DEBUG']:
        print 'Debugging turned off'
        env.Append(CPPDEFINES=['NDEBUG'])
    else:
        if env['TMV_DEBUG']:
            print 'TMV Extra Debugging turned on'
            env.Append(CPPDEFINES=['TMV_EXTRA_DEBUG'])

    import SCons.SConf

    # Don't bother with checks if doing scons -c
    if not env.GetOption('clean'):
        # Sometimes when you are changing around things in other directories, SCons doesn't notice.
        # e.g. You hadn't installed fftw3, so you go and do that.  Now you want SCons to redo
        # the check for it, rather than use the cached result.  
        # To do that set CACHE_LIB=false
        if not env['CACHE_LIB']:
            SCons.SConf.SetCacheMode('force')

        # Add out custom configuration tests
        config = env.Configure(custom_tests = {
            'CheckTMV' : CheckTMV ,
            'CheckPython' : CheckPython ,
            'CheckNumPy' : CheckNumPy ,
            'CheckBoostPython' : CheckBoostPython ,
            'CheckPyFITS' : CheckPyFITS ,
            })
        DoLibraryAndHeaderChecks(config)
        env = config.Finish()

        # Turn the cache back on now, since we always want it for the main compilation steps.
        if not env['CACHE_LIB']:
            SCons.SConf.SetCacheMode('auto')

    # This one should be done after DoLibraryAndHeaderChecks
    # otherwise the TMV link test fails, since TMV wasn't compiled
    # with MEMTEST.  If you do want to test with a TMV library that
    # uses MEMTEST, you might need to move this to before
    # the DoLibraryAndHeaderChecks call.
    if env['MEM_TEST']:
        env.Append(CPPDEFINES=['MEM_TEST'])




#
# main program
#

env = Environment()

opts.Update(env)

if env['IMPORT_ENV']:
    env = Environment(ENV=os.environ)
    opts.Update(env)

# Check for unknown variables in case something is misspelled
unknown = opts.UnknownVariables()
if unknown:
    print "Unknown variables:", unknown.keys()
    Exit(1)

opts.Save(config_file,env)
Help(opts.GenerateHelpText(env))

if not GetOption('help'):

    # Set up the configuration
    DoConfig(env)

    # subdirectory SConscript files can use this function
    env['__readfunc'] = ReadFileList
    env['_InstallProgram'] = RunInstall
    env['_UninstallProgram'] = RunUninstall

    #if env['WITH_UPS']:
        #subdirs += ['ups']

    if 'examples' in COMMAND_LINE_TARGETS:
        subdirs += ['examples']

    if 'tests' in COMMAND_LINE_TARGETS:
        nosetests = which('nosetests')
        if nosetests is None:
            env['RUN_NOSETESTS'] = False
        else:
            env['RUN_NOSETESTS'] = True
        subdirs += ['tests']

    # subdirectores to process.  We process src and pysrc by default
    script_files = []
    for d in subdirs:
        script_files.append(os.path.join(d,'SConscript'))

    SConscript(script_files, exports='env')


