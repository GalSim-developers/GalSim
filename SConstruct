# vim: set filetype=python et ts=4 sw=4:

# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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

import os
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
subdirs=['src', 'pysrc', 'bin', 'galsim', 'share']

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

default_python = '/usr/bin/env python'
default_cxx = 'c++'

# first check for a saved conf file
opts = Variables(config_file)

# Now set up options for the command line
opts.Add('CXX','Name of c++ compiler','g++')
opts.Add('FLAGS','Compiler flags to send to use instead of the automatic ones','')
opts.Add('EXTRA_FLAGS','Extra compiler flags to use in addition to automatic ones','')
opts.Add('LINKFLAGS','Additional flags to use when linking','')
opts.Add(BoolVariable('DEBUG','Turn on debugging statements',True))
opts.Add(BoolVariable('EXTRA_DEBUG','Turn on extra debugging info',False))
opts.Add(BoolVariable('WARN','Add warning compiler flags, like -Wall', False))
opts.Add('PYTHON','Name of python executable','')

opts.Add(BoolVariable('WITH_UPS','Install ups/ directory for use with EUPS', False))

opts.Add(PathVariable('PREFIX','prefix for installation',
         '', PathVariable.PathAccept))
opts.Add(PathVariable('PYPREFIX','location of your site-packages directory',
         '', PathVariable.PathAccept))
opts.Add(PathVariable('FINAL_PREFIX',
         'final installation prefix if different from PREFIX',
         '', PathVariable.PathAccept))

opts.Add('TMV_DIR','Explicitly give the tmv prefix','')
opts.Add('TMV_LINK','File that contains the linking instructions for TMV','')
opts.Add('FFTW_DIR','Explicitly give the fftw3 prefix','')
opts.Add('BOOST_DIR','Explicitly give the boost prefix','')

opts.Add(PathVariable('EXTRA_INCLUDE_PATH',
         'Extra paths for header files (separated by : if more than 1)',
         '', PathVariable.PathAccept))
opts.Add(PathVariable('EXTRA_LIB_PATH',
         'Extra paths for linking (separated by : if more than 1)',
         '', PathVariable.PathAccept))
opts.Add(PathVariable('EXTRA_PATH',
         'Extra paths for executables (separated by : if more than 1)',
         '', PathVariable.PathAccept))
opts.Add(BoolVariable('IMPORT_PATHS',
         'Import PATH, C_INCLUDE_PATH and LIBRARY_PATH/LD_LIBRARY_PATH environment variables',
         False))
opts.Add(BoolVariable('IMPORT_ENV',
         'Import full environment from calling shell',True))
opts.Add('EXTRA_LIBS','Libraries to send to the linker','')
opts.Add(BoolVariable('IMPORT_PREFIX',
         'Use PREFIX/include and PREFIX/lib in search paths', True))

opts.Add('NOSETESTS','Name of nosetests executable','')
opts.Add(BoolVariable('CACHE_LIB','Cache the results of the library checks',True))
opts.Add(BoolVariable('WITH_PROF',
            'Use the compiler flag -pg to include profiling info for gprof', False))
opts.Add(BoolVariable('MEM_TEST','Test for memory leaks', False))
opts.Add(BoolVariable('TMV_DEBUG','Turn on extra debugging statements within TMV library',False))
# None of the code uses openmp yet.  Probably make this default True if we start using it.
opts.Add(BoolVariable('WITH_OPENMP','Look for openmp and use if found.', False))
opts.Add(BoolVariable('USE_UNKNOWN_VARS',
            'Allow other parameters besides the ones listed here.',False))

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

def ClearCache():
    """
    Clear the SCons cache files
    """
    if os.path.exists(".sconsign.dblite"):
        os.remove(".sconsign.dblite")
    import shutil
    if os.path.exists(".sconf_temp"):
        shutil.rmtree(".sconf_temp")

def ErrorExit(*args, **kwargs):
    """
    Whenever we get an error in the initial setup checking for the various
    libraries, compiler, etc., we don't want to cache the result.
    On the other hand, if we delete the .scon* files now, then they aren't
    available to diagnose any problems.
    So we write a file called gs.error that
    a) includes some relevant information to diagnose the problem.
    b) indicates that we should clear the cache the next time we run scons.
    """

    import shutil

    out = open("gs.error","wb")

    # Start with the error message to output both to the screen and to the end of gs.error:
    print
    for s in args:
        print s
        out.write(s + '\n')
    out.write('\n')

    # Write out the current options:
    out.write('Using the following options:\n')
    for opt in opts.options:
        out.write('   %s = %s\n'%(opt.key,env[opt.key]))
    out.write('\n')

    # Write out the current environment:
    out.write('The system environment is:\n')
    for key in os.environ.keys():
        out.write('   %s = %s\n'%(key,os.environ[key]))
    out.write('\n')

    out.write('The SCons environment is:\n')
    for key in env.Dictionary().keys():
        out.write('   %s = %s\n'%(key,env[key]))
    out.write('\n')

    # Next put the full config.log in there.
    out.write('The full config.log file is:\n')
    out.write('==================\n')
    shutil.copyfileobj(open("config.log","rb"),out)
    out.write('==================\n\n')

    # It is sometimes helpful to see the output of the scons executables.
    # SCons just uses >, not >&, so we'll repeat those runs here and get both.
    try:
        import subprocess
        cmd = ("ls -d .sconf_temp/conftest* | grep -v '\.out' | grep -v '\.cpp' "+
               "| grep -v '\.o' | grep -v '\_mod'")
        p = subprocess.Popen([cmd],stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        conftest_list = p.stdout.readlines()
        for conftest in conftest_list:
            conftest = conftest.strip()
            if os.access(conftest, os.X_OK):
                cmd = conftest
            else:
                cmd = env['PYTHON'] + " < " + conftest
            p = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 shell=True)
            conftest_out = p.stdout.readlines()
            out.write('Output of the command %s is:\n'%cmd)
            out.write(''.join(conftest_out) + '\n')
    except:
        out.write("Error trying to get output of conftest executables.\n")
        out.write(sys.exc_info()[0])

    print
    print 'Please fix the above error(s) and rerun scons.'
    print 'Note: you may want to look through the file INSTALL.md for advice.'
    print 'Also, if you are having trouble, please check the INSTALL FAQ at '
    print '   https://github.com/GalSim-developers/GalSim/wiki/Installation%20FAQ'
    print
    Exit(1)



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
            # Unfortunately this next flag requires strict-aliasing, but allowing that
            # opens up a Pandora's box of bugs and warnings, so I don't want to do that.
            #env.Append(CCFLAGS=['-ftree-vectorize'])
            if env['WITH_PROF']:
                env.Append(CCFLAGS=['-pg'])
                env.Append(LINKFLAGS=['-pg'])
            if env['WARN']:
                env.Append(CCFLAGS=['-Wall','-Werror'])
            if env['EXTRA_DEBUG']:
                env.Append(CCFLAGS=['-g3'])

        elif compiler == 'clang++':
            env.Replace(CCFLAGS=['-O2'])
            if env['WITH_PROF']:
                env.Append(CCFLAGS=['-pg'])
                env.Append(LINKFLAGS=['-pg'])
            if env['WARN']:
                env.Append(CCFLAGS=['-Wall','-Werror'])
            if env['EXTRA_DEBUG']:
                env.Append(CCFLAGS=['-g3'])

        elif compiler == 'icpc':
            env.Replace(CCFLAGS=['-O2','-msse2'])
            if version >= 10:
                env.Append(CCFLAGS=['-vec-report0'])
            if env['WITH_PROF']:
                env.Append(CCFLAGS=['-pg'])
                env.Append(LINKFLAGS=['-pg'])
            if env['WARN']:
                env.Append(CCFLAGS=['-Wall','-Werror','-wd279,383,810,981'])
                if version >= 9:
                    env.Append(CCFLAGS=['-wd1572'])
                if version >= 11:
                    env.Append(CCFLAGS=['-wd2259'])
            if env['EXTRA_DEBUG']:
                env.Append(CCFLAGS=['-g'])

        elif compiler == 'pgCC':
            env.Replace(CCFLAGS=['-O2','-fast','-Mcache_align'])
            if env['WITH_PROF']:
                env.Append(CCFLAGS=['-pg'])
                env.Append(LINKFLAGS=['-pg'])
            if env['EXTRA_DEBUG']:
                env.Append(CCFLAGS=['-g'])

        elif compiler == 'CC':
            env.Replace(CCFLAGS=['-O2','-fast','-instances=semiexplicit'])
            if env['WARN']:
                env.Append(CCFLAGS=['+w'])
            if env['EXTRA_DEBUG']:
                env.Append(CCFLAGS=['-g'])

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
        for flag in cxx_flags:
            if flag.startswith('-Wl') or flag.startswith('-m'):
                # Then this also needs to be in LINKFLAGS
                env.AppendUnique(LINKFLAGS=flag)

    extra_flags = env['EXTRA_FLAGS'].split(' ')
    env.AppendUnique(CCFLAGS=extra_flags)
    for flag in extra_flags:
        if flag.startswith('-Wl') or flag.startswith('-m'):
            # Then this also needs to be in LINKFLAGS
            env.AppendUnique(LINKFLAGS=flag)


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
    """Determine the version of the compiler
    """
    if env['CXX'] is None:
        env['CXX'] = default_cxx
    compiler = which(env['CXX'])
    if compiler is None:
        ErrorExit('Specified compiler not found in path: %s' % env['CXX'])

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
        import subprocess
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        lines = p.stdout.readlines()

        # Check if g++ is a symlink for something else:
        if compilertype is 'g++':
            if 'clang' in lines[0]:
                print 'Detected clang++ masquerading as g++'
                compilertype = 'clang++'
                # When it is masquerading, the line with the version is the second line.
                linenum=1

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

def GetNosetestsVersion(env):
    """Determine the version of nosetests
    """
    import subprocess
    cmd = env['NOSETESTS'] + ' --version 2>&1'
    p = subprocess.Popen([cmd],stdout=subprocess.PIPE,shell=True)
    line = p.stdout.readlines()[0]
    version = line.split()[2]
    print 'nosetests version:',version
    env['NOSETESTSVERSION'] = version


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

    types = ['BOOST', 'TMV', 'FFTW']

    for t in types:
        dirtag = t+'_DIR'
        tdir = FindPathInEnv(env, dirtag)
        if tdir is None:
            if env[dirtag] != '':
                print 'WARNING: could not find specified %s = %s'%(dirtag,env[dirtag])
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
        env['FINAL_PREFIX'] = default_prefix
    else:
        if os.path.isfile(env['PREFIX']) and os.path.samefile('.',env['PREFIX']):
            ErrorExit(
                'Using the source directory as the PREFIX value is not allowed.',
                'You should install GalSim to some other directory.  The typical',
                'choice is to use your home directory, which on most machines can',
                'be specified using PREFIX=~')

        env['INSTALL_PREFIX'] = env['PREFIX']

        # FINAL_PREFIX is designed for installations like that done by fink where it installs
        # everything into a temporary directory, and then once it finished successfully, it
        # copies the resulting files to a final location.
        if env['FINAL_PREFIX'] == '':
            env['FINAL_PREFIX'] = env['PREFIX']

        if env['IMPORT_PREFIX']:
            AddPath(bin_paths, os.path.join(env['PREFIX'], 'bin'))
            AddPath(lib_paths, os.path.join(env['PREFIX'], 'lib'))
            AddPath(cpp_paths, os.path.join(env['PREFIX'], 'include'))

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

    if env['IMPORT_PATHS'] and os.environ.has_key('DYLD_LIBRARY_PATH'):
        paths=os.environ['DYLD_LIBRARY_PATH']
        paths=paths.split(os.pathsep)
        AddPath(lib_paths, paths)

    env.PrependENVPath('PATH', bin_paths)
    env.Prepend(LIBPATH= lib_paths)
    env.Prepend(CPPPATH= cpp_paths)

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


def TryRunResult(config,text,name):
    # Check if a particular program (given as text) is compilable, runs, and returns the
    # right value.

    config.sconf.pspawn = config.sconf.env['PSPAWN']
    save_spawn = config.sconf.env['SPAWN']
    config.sconf.env['SPAWN'] = config.sconf.pspawn_wrapper

    # First use the normal TryRun command
    ok, out = config.TryRun(text,'.cpp')

    config.sconf.env['SPAWN'] = save_spawn

    # We have an arbitrary requirement that the executable output the answer 23.
    # So if we didn't get this answer, then something must have gone wrong.
    if out.strip() != '23':
        ok = False

    return ok


def CheckLibsSimple(config,try_libs,source_file,prepend=True):
    init_libs = []
    if 'LIBS' in config.env._dict.keys():
        init_libs = config.env['LIBS']

    if prepend:
        config.env.PrependUnique(LIBS=try_libs)
    else:
        config.env.AppendUnique(LIBS=try_libs)
    result = TryRunResult(config,source_file,'.cpp')

    # If that didn't work, go back to the original LIBS
    if not result :
        config.env.Replace(LIBS=init_libs)
    return result


def AddRPATH(env, rpath, prepend=False):
    """
    Add rpath to the scons environment.
    This is really  a workaround for SCons bug.  The normal command should just be:
        config.env.AppendUnique(RPATH=rpath)
    But this doesn't always work correctly, since RPATH sometimes clashes with LINKFLAGS.
    So if LINKFLAGS is already set, we need this workaround.
    See: http://scons.tigris.org/issues/show_bug.cgi?id=1644
    (Fixed in version 2.1.)
    """
    if prepend:
        env.PrependUnique(RPATH=rpath)
    else:
        env.AppendUnique(RPATH=rpath)
    major , minor , junk = SCons.__version__.split('.',2)
    if int(major) < 2 or (int(major) == 2 and int(minor) == 0):
        env.Append( LINKFLAGS = ["$__RPATH"] )


def CheckLibsFull(config,try_libs,source_file,prepend=True):
    init_libs = []
    if 'LIBS' in config.env._dict.keys():
        init_libs = config.env['LIBS']

    if prepend:
        config.env.PrependUnique(LIBS=try_libs)
    else:
        config.env.AppendUnique(LIBS=try_libs)
    result = TryRunResult(config,source_file,'.cpp')

    if result: return result

    init_rpath = []
    init_link = []
    if 'RPATH' in config.env._dict.keys():
        init_rpath = config.env['RPATH']
    if 'LINKFLAGS' in config.env._dict.keys():
        init_link = config.env['LINKFLAGS']

    # Sometimes we need to add a directory to RPATH, so try each one.
    if not result and 'LIBPATH' in config.env._dict.keys():
        for rpath in config.env['LIBPATH']:
            AddRPATH(config.env,rpath,prepend)
            result = TryRunResult(config,source_file,'.cpp')
            if result:
                break
            else:
                config.env.Replace(RPATH=init_rpath)
                config.env.Replace(LINKFLAGS=init_link)

        # If that doesn't work, also try adding all of them, just in case we need more than one.
        if not result :
            AddRPATH(config.env,config.env['LIBPATH'],prepend)
            result = TryRunResult(config,source_file,'.cpp')
            if not result:
                config.env.Replace(RPATH=init_rpath)
                config.env.Replace(LINKFLAGS=init_link)

    # Next try the LIBRARY_PATH to see if any of these help.
    if not result and 'LIBRARY_PATH' in os.environ.keys():
        library_path=os.environ['LIBRARY_PATH']
        library_path=library_path.split(os.pathsep)
        for rpath in library_path:
            AddRPATH(config.env,rpath,prepend)
            result = TryRunResult(config,source_file,'.cpp')
            if result:
                break
            else:
                config.env.Replace(RPATH=init_rpath)
                config.env.Replace(LINKFLAGS=init_link)

        # If that doesn't work, also try adding all of them, just in case we need more than one.
        if not result :
            AddRPATH(config.env,library_path,prepend)
            result = TryRunResult(config,source_file,'.cpp')
            if not result:
                config.env.Replace(RPATH=init_rpath)
                config.env.Replace(LINKFLAGS=init_link)

    # If nothing worked, go back to the original LIBS
    if not result :
        config.env.Replace(LIBS=init_libs)
    return result


def CheckFFTW(config):
    fftw_source_file = """
#include "fftw3.h"
#include <iostream>
int main()
{
  double* ar = (double*) fftw_malloc(sizeof(double)*64);
  fftw_complex* ac = (fftw_complex*) fftw_malloc(sizeof(double)*2*64);
  fftw_plan plan = fftw_plan_dft_r2c_2d(8,8,ar,ac,FFTW_MEASURE);
  fftw_destroy_plan(plan);
  fftw_free(ar);
  fftw_free(ac);
  std::cout<<"23"<<std::endl;
  return 0;
}
"""
    config.Message('Checking for correct FFTW linkage... ')
    if not config.TryCompile(fftw_source_file,'.cpp'):
        ErrorExit(
            'Error: fftw file failed to compile.',
            'Check that the correct location is specified for FFTW_DIR')

    result = (
        CheckLibsFull(config,[''],fftw_source_file) or
        CheckLibsFull(config,['fftw3'],fftw_source_file) )
    if not result:
        ErrorExit(
            'Error: fftw file failed to link correctly',
            'Check that the correct location is specified for FFTW_DIR')

    config.Result(1)
    return 1


def CheckBoost(config):
    # At the C++ level, we only need boost header files, so no need to check libraries.
    # Use boost/shared_ptr.hpp as a representative choice.

    boost_source_file = """
#define BOOST_NO_CXX11_SMART_PTR
#include "boost/shared_ptr.hpp"
"""
    config.Message('Checking for boost header files... ')
    if not config.TryCompile(boost_source_file, ".cpp"):
        ErrorExit(
            'Boost not found',
            'You should specify the location of Boost as BOOST_DIR=...')
    config.Result(1)

    boost_version_file = """
#include <iostream>
#define BOOST_NO_CXX11_SMART_PTR
#include "boost/version.hpp"
int main() { std::cout<<BOOST_VERSION<<std::endl; return 0; }
"""
    ok, boost_version = config.TryRun(boost_version_file,'.cpp')
    boost_version = int(boost_version.strip())
    print 'Boost version is %d.%d.%d' % (
            boost_version / 100000, boost_version / 100 % 1000, boost_version % 100)
    
    return 1


def CheckTMV(config):
    tmv_source_file = """
#include "TMV_Sym.h"
#include <iostream>
int main()
{
  tmv::SymMatrix<double> S(10,4.);
  //tmv::Matrix<double> S(10,10,4.);
  tmv::Matrix<double> m(10,3,2.);
  S += 50.;
  tmv::Matrix<double> m2 = m / S;
  std::cout<<"23"<<std::endl;
  return 0;
}
"""
    print 'Checking for correct TMV linkage... (this may take a little while)'
    config.Message('Checking for correct TMV linkage... ')

    result = config.TryCompile(tmv_source_file,'.cpp')
    if not result:
        ErrorExit(
            'Error: TMV file failed to compile.',
            'Check that the correct location is specified for TMV_DIR')

    result = (
        CheckLibsSimple(config,[''],tmv_source_file) or
        CheckLibsSimple(config,['tmv_symband','tmv'],tmv_source_file) or
        CheckLibsSimple(config,['tmv_symband','tmv','irc','imf'],tmv_source_file) or
        CheckLibsFull(config,['tmv_symband','tmv'],tmv_source_file) or
        CheckLibsFull(config,['tmv_symband','tmv','irc','imf'],tmv_source_file) )
    if not result:
        ErrorExit(
            'Error: TMV file failed to link correctly',
            'Check that the correct location is specified for TMV_DIR')

    config.Result(1)
    return 1


def TryScript(config,text,executable):
    # Check if a particular script (given as text) is runnable with the
    # executable (given as executable).
    #
    # I couldn't find a way to do this using the existing SCons functions, so this
    # is basically taken from parts of the code for TryBuild and TryRun.

    # First make the file name using the same counter as TryBuild uses:
    from SCons.SConf import _ac_build_counter
    f = "conftest_" + str(SCons.SConf._ac_build_counter)
    SCons.SConf._ac_build_counter = SCons.SConf._ac_build_counter + 1

    config.sconf.pspawn = config.sconf.env['PSPAWN']
    save_spawn = config.sconf.env['SPAWN']
    config.sconf.env['SPAWN'] = config.sconf.pspawn_wrapper

    # Build a file containg the given text
    textFile = config.sconf.confdir.File(f)
    sourcetext = config.env.Value(text)
    textFileNode = config.env.SConfSourceBuilder(target=textFile, source=sourcetext)
    config.sconf.BuildNodes(textFileNode)
    source = textFileNode

    # Run the given executable with the source file we just built
    output = config.sconf.confdir.File(f + '.out')
    node = config.env.Command(output, source, executable + " < $SOURCE > $TARGET 2>&1")
    ok = config.sconf.BuildNodes(node)

    config.sconf.env['SPAWN'] = save_spawn

    if ok:
        # For successful execution, also return the output contents
        outputStr = output.get_contents()
        return 1, outputStr.strip()
    else:
        return 0, ""

def TryModule(config,text,name,pyscript=""):
    # Check if a particular program (given as text) is compilable as a python module.

    config.sconf.pspawn = config.sconf.env['PSPAWN']
    save_spawn = config.sconf.env['SPAWN']
    config.sconf.env['SPAWN'] = config.sconf.pspawn_wrapper

    # First try to build the code as a SharedObject:
    ok = config.TryBuild(config.env.SharedObject,text,'.cpp')
    if not ok: return 0

    # Get the object file as the lastTarget:
    obj = config.sconf.lastTarget

    # Try to build the LoadableModule
    dir = os.path.splitext(os.path.basename(obj.path))[0] + '_mod'
    output = config.sconf.confdir.File(os.path.join(dir,name + '.so'))
    dir = os.path.dirname(output.path)
    mod = config.env.LoadableModule(output, obj)
    ok = config.sconf.BuildNodes(mod)
    if not ok: return 0

    # Finally try to import and run the module in python:
    if pyscript == "":
        pyscript = "import sys\nsys.path.append('%s')\nimport %s\nprint %s.run()"%(dir,name,name)
    else:
        pyscript = "import sys\nsys.path.append('%s')\n"%dir + pyscript
    ok, out = TryScript(config,pyscript,python)

    config.sconf.env['SPAWN'] = save_spawn

    # We have an arbitrary requirement that the run() command output the answer 23.
    # So if we didn't get this answer, then something must have gone wrong.
    if ok and out != '23':
        #print "Script's run() command didn't output '23'."
        ok = False

    return ok


def CheckModuleLibs(config,try_libs,source_file,name,prepend=True):
    init_libs = []
    if 'LIBS' in config.env._dict.keys():
        init_libs = config.env['LIBS']

    if prepend:
        config.env.PrependUnique(LIBS=try_libs)
    else:
        config.env.AppendUnique(LIBS=try_libs)
    result = TryModule(config,source_file,name)

    # Sometimes we need to add a directory to RPATH, so try each one.
    if not result and 'LIBPATH' in config.env._dict.keys():
        init_rpath = []
        if 'RPATH' in config.env._dict.keys():
            init_rpath = config.env['RPATH']

        for rpath in config.env['LIBPATH']:
            if prepend:
                config.env.PrependUnique(RPATH=rpath)
            else:
                config.env.AppendUnique(RPATH=rpath)
            result = TryModule(config,source_file,name)
            if result:
                break
            else:
                config.env.Replace(RPATH=init_rpath)

        # If that doesn't work, also try adding all of them, just in case we need more than one.
        if not result :
            if prepend:
                config.env.PrependUnique(RPATH=config.env['LIBPATH'])
            else:
                config.env.AppendUnique(RPATH=config.env['LIBPATH'])
            result = TryModule(config,source_file,name)
            if not result:
                config.env.Replace(RPATH=init_rpath)

    # If nothing worked, go back to the original LIBS
    if not result :
        config.env.Replace(LIBS=init_libs)
    return result


def CheckPython(config):
    python_source_file = """
#include "Python.h"

static PyObject* run(PyObject* self, PyObject* args)
{ return Py_BuildValue("i", 23); }

static PyMethodDef Methods[] = {
    {"run",  run, METH_VARARGS, "return 23"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcheck_python(void)
{ Py_InitModule("check_python", Methods); }
"""
    config.Message('Checking if we can build against Python... ')

    # First check the python include directory -- see if we can compile the module.
    source_file2 = "import distutils.sysconfig; print distutils.sysconfig.get_python_inc()"
    result, py_inc = TryScript(config,source_file2,python)
    if not result:
        ErrorExit('Unable to get python include path python executable:\n%s'%python)

    config.env.AppendUnique(CPPPATH=py_inc)
    if not config.TryCompile(python_source_file,'.cpp'):
        ErrorExit('Unable to compile a file with #include "Python.h" using the include path:',
                  '%s'%py_inc)

    # Now see if we can build it as a LoadableModule and run it from python.
    # Sometimes (e.g. most linux systems), we don't need the python library to do this.
    # So the first attempt below with [''] for the libs will work.
    if CheckModuleLibs(config,[''],python_source_file,'check_python'):
        config.Result(1)
        return 1

    # Other times (e.g. most Mac systems) we'll need to link the library.
    # It seems that there is not much consistency to where different installations
    # put the library and how they declare its location via distutils.
    # So make a list of possible directories and library names and see which one works.
    py_libfiles = []
    py_libdirs = []

    # Usually, it is the file called LDLIBRARY in the config vars:
    source_file3 = "import distutils.sysconfig; print distutils.sysconfig.get_config_var('LDLIBRARY')"
    result, py_libfile = TryScript(config,source_file3,python)
    py_libfile1 = py_libfile
    if result:
        py_libfiles.append(py_libfile)

    # Sometimes, it is called LIBRARY (e.g. for EPD, LDLIBRARY is a file called Python, which
    # is where the LIBRARY file links to, but that doesn't work for the way with link to the
    # library.  So we need LIBRARY instead.)
    source_file4 = "import distutils.sysconfig; print distutils.sysconfig.get_config_var('LIBRARY')"
    result, py_libfile = TryScript(config,source_file4,python)
    if result:
        py_libfiles.append(py_libfile)

    # If neither of those work, we're probably hosed, but try libpython.a and libpythonx.x.a
    # along with .so or .dylib versions of these, just in case.
    py_libfiles.append('libpython.a')

    # Get the version:
    source_file5 = "import sys; print '%d.%d'%(sys.version_info[:2])"
    result, py_version = TryScript(config,source_file5,python)
    # If that didn't work, try to get it from the file or directory names, since it is usually
    # there:
    if not result:
        if '2.7' in py_inc or '2.7' in python:
            py_version = '2.7'
        elif '2.6' in py_inc or '2.6' in python:
            py_version = '2.6'
        elif '2.5' in py_inc or '2.5' in python:
            py_version = '2.5'
        elif '2.4' in py_inc or '2.4' in python:
            py_version = '2.4'
        else:
            py_version = ''
    py_libfiles.append('libpython'+py_version+'.a')

    # One of these might work as is, so try the list of options now:
    for py_libfile in py_libfiles:
        py_lib = os.path.splitext(py_libfile)[0]
        if py_lib.startswith('lib'):
            py_lib = py_lib[3:]

        result = CheckModuleLibs(config,py_lib,python_source_file,'check_python')
        if result:
            config.Result(1)
            print 'Building for python version '+py_version
            return 1

    # If that didn't work, we'll need to add a directory to LIBPATH.  So let's see if we
    # can find the right path to add.  We'll add two more possibilities to the libfiles
    # to check, depending on which flavor of system we are on.
    if sys.platform.startswith('linux'):
        py_libfiles.append('libpython.so')
        py_libfiles.append('libpython'+py_version+'.so')
    else:
        py_libfiles.append('libpython.dylib')
        py_libfiles.append('libpython'+py_version+'.dylib')

    # Now compile a list of possible locations to look for the library:
    py_libdirs = []

    # There are a number of paths reported by distutils.  We'll try a few of them:
    source_file6 = "import distutils.sysconfig; print distutils.sysconfig.get_config_var('LIBDIR')"
    result, py_libdir = TryScript(config,source_file6,python)
    py_libdir1 = py_libdir
    if result:
        py_libdirs.append(py_libdir)
        py_libdirs.append(os.path.join(py_libdir,'config'))
        py_libdirs.append(os.path.join(py_libdir,'python'+py_version,'config'))
    source_file7 = "import distutils.sysconfig; print distutils.sysconfig.get_config_var('LIBDEST')"
    result, py_libdir = TryScript(config,source_file7,python)
    if result and py_libdir not in py_libdirs: 
        py_libdirs.append(py_libdir)
    source_file8 = "import distutils.sysconfig; print distutils.sysconfig.get_config_var('LIBP')"
    result, py_libdir = TryScript(config,source_file8,python)
    if result and py_libdir not in py_libdirs: 
        py_libdirs.append(py_libdir)
    source_file8 = "import distutils.sysconfig; print distutils.sysconfig.get_config_var('LIBPL')"
    result, py_libdir = TryScript(config,source_file8,python)
    if result and py_libdir not in py_libdirs: 
        py_libdirs.append(py_libdir)

    # We can also try to get the location from the name of the executable.  Typically the 
    # python executable is called PREFIX/bin/python and the corresponding library is
    # PREFIX/lib/python2.7/config/libpython2.7.a.  So try stripping off the bin/python part
    # and add lib/python2.7/config.
    py_root = os.path.split(os.path.split(python)[0])[0]
    py_libdir = os.path.join(py_root,'lib','python'+py_version,'config')
    if py_libdir not in py_libdirs: 
        py_libdirs.append(py_libdir)

    # Look in each of these directories for a valid library file to link to:
    for py_libfile in py_libfiles:
        for py_libdir in py_libdirs:
            if os.path.isfile(os.path.join(py_libdir,py_libfile)):
                config.env.PrependUnique(LIBPATH=py_libdir)
                py_lib = os.path.splitext(py_libfile)[0]
                if py_lib.startswith('lib'):
                    py_lib = py_lib[3:]
                if CheckModuleLibs(config,py_lib,python_source_file,'check_python'):
                    config.Result(1)
                    print 'Building for python version '+py_version
                    print 'Python libdir = ',py_libdir
                    print 'Python libfile = ',py_libfile
                    return 1

    # Oh well, it was worth a shot.
    ErrorExit('Unable to build a python loadable module using the python executable:',
              '%s,'%python,
              'the library name %s,'%py_libfile1,
              'and the libdir %s.'%py_libdir1,
              'Nor did any of the known variations on this work.',
              'If these are not the correct library name or location, you can tell scons',
              'the correct names to use with the flags EXTRA_LIB_PATH and/or EXTRA_LIBS.')

def CheckPyTMV(config):
    tmv_source_file = """
#include "Python.h"
#include "TMV_Sym.h"

static void useTMV() {
    tmv::SymMatrix<double> S(10,4.);
    //tmv::Matrix<double> S(10,10,4.);
    tmv::Matrix<double> m(10,3,2.);
    S += 50.;
    tmv::Matrix<double> m2 = m / S;
}

static PyObject* run(PyObject* self, PyObject* args)
{
    useTMV();
    return Py_BuildValue("i", 23);
}

static PyMethodDef Methods[] = {
    {"run",  run, METH_VARARGS, "return 23"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcheck_tmv(void)
{ Py_InitModule("check_tmv", Methods); }
"""
    config.Message('Checking if we can build module using TMV... ')

    result = config.TryCompile(tmv_source_file,'.cpp')
    if not result:
        ErrorExit('Unable to compile a module using tmv')

    result = CheckModuleLibs(config,[],tmv_source_file,'check_tmv')
    if not result and any(['mkl' in lib for lib in config.env['LIBS']]):
        # TMV finds the mkl libraries necessary for compiling an executable.  But sometimes
        # there are extra libraries required for loading mkl from a python module.
        # So if the above line fails, try adding a few mkl libraries that might make it work.
        result = (
            CheckModuleLibs(config,['mkl_rt'],tmv_source_file,'check_tmv',False) or
            CheckModuleLibs(config,['mkl_base'],tmv_source_file,'check_tmv',False) or
            CheckModuleLibs(config,['mkl_mc3'],tmv_source_file,'check_tmv',False) or
            CheckModuleLibs(config,['mkl_mc3','mkl_def'],tmv_source_file,'check_tmv',False) or
            CheckModuleLibs(config,['mkl_mc'],tmv_source_file,'check_tmv',False) or
            CheckModuleLibs(config,['mkl_mc','mkl_def'],tmv_source_file,'check_tmv'),False)
    if not result:
        ErrorExit('Unable to build a python loadable module that uses tmv')

    config.Result(1)
    return 1


def CheckNumPy(config):
    numpy_source_file = """
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"

static void doImport() {
    import_array();
}

static PyObject* run(PyObject* self, PyObject* args)
{
    doImport();
    int result = 1;
    if (!PyErr_Occurred()) {
        npy_intp dims = 2;
        PyObject* a = PyArray_SimpleNew(1, &dims, NPY_INT);
        if (a) {
            Py_DECREF(a);
            result = 23;
        }
    }
    return Py_BuildValue("i", result);
}

static PyMethodDef Methods[] = {
    {"run",  run, METH_VARARGS, "return 23"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcheck_numpy(void)
{ Py_InitModule("check_numpy", Methods); }
"""
    config.Message('Checking if we can build against NumPy... ')

    result, numpy_inc = TryScript(config,"import numpy; print numpy.get_include()",python)
    if not result:
        ErrorExit("Unable to import numpy using the python executable:\n%s"%python)
    config.env.AppendUnique(CPPPATH=numpy_inc)

    result = config.TryCompile(numpy_source_file,'.cpp')
    if not result:
        ErrorExit('Unable to compile a file with numpy using the include path:\n%s.'%numpy_inc)

    result = TryModule(config,numpy_source_file,'check_numpy')
    if not result:
        ErrorExit('Unable to build a python loadable module that uses numpy')

    config.Result(1)
    return 1

def CheckPyFITS(config):
    config.Message('Checking for PyFITS... ')

    result, output = TryScript(config,"import pyfits",python)
    if not result:
        result, output = TryScript(config,"import astropy.io.fits",python)
    if not result:
        ErrorExit("Unable to import pyfits or astropy.io.fits using the python executable:\n" +
                  python)

    config.Result(1)
    return 1

def CheckBoostPython(config):
    bp_source_file = """

#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif
#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"

int check_bp_run() { return 23; }

BOOST_PYTHON_MODULE(check_bp) {
    boost::python::def("run",&check_bp_run);
}
"""
    config.Message('Checking if we can build against Boost.Python... ')

    result = config.TryCompile(bp_source_file,'.cpp')
    if not result:
        ErrorExit('Unable to compile a file with #include "boost/python.hpp"')

    result = (
        CheckModuleLibs(config,[''],bp_source_file,'check_bp') or
        CheckModuleLibs(config,['boost_python'],bp_source_file,'check_bp') or
        CheckModuleLibs(config,['boost_python-mt'],bp_source_file,'check_bp') )
    if not result:
        ErrorExit('Unable to build a python loadable module with Boost.Python')

    config.Result(1)
    return 1

# If the compiler is incompatible with the compiler that was used to build python,
# then there can be problems with the exception passing between the C++ layer and the
# python layer.  We don't know any solution to this, but it's worth letting the user
# know that C++ exceptions might be a bit uninformative.
def CheckPythonExcept(config):
    cpp_source_file = """
#ifndef __INTEL_COMPILER
#if defined(__GNUC__) && __GNUC__ >= 4 && (__GNUC__ >= 5 || __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#endif
#define BOOST_NO_CXX11_SMART_PTR
#include "boost/python.hpp"
#include <stdexcept>

void run_throw() { throw std::runtime_error("test error handling"); }

BOOST_PYTHON_MODULE(test_throw) {
    boost::python::def("run",&run_throw);
}
"""
    py_source_file = """
import test_throw
try:
    test_throw.run()
    print 0
except RuntimeError, e:
    if str(e) == 'test error handling':
        print 23
    else:
        print 0
except:
    print 0
"""
    config.Message('Checking if C++ exceptions are propagated up to python... ')
    result = TryModule(config,cpp_source_file,"test_throw",py_source_file)
    config.Result(result)

    if not result:
        print """
WARNING: There seems to be a mismatch between this C++ compiler and the one
         that was used to build either python or boost.python (or both).
         This might be ok, but if you get a linking error in the subsequent 
         build, it is possible  that you will need to rebuild boost with the
         same compiler (and sometimes version) that you are using here.
"""
        config.env['final_messages'].append("""
WARNING: There seems to be a mismatch between this C++ compiler and the one
         that was used to build either python or boost.python (or both).
         This should not affect normal usage of GalSim.  However, exceptions
         thrown in the C++ layer are not being correctly propagated to the
         python layer, so the error text for C++ run-time errors  will not
         be very informative.
""")

    return result


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
            ErrorExit('Specified TMV_LINK does not exist: %s' % tmv_link)

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

    ErrorExit('No tmv-link file could be found')


def DoCppChecks(config):
    """
    Check for some headers and libraries.
    """

    #####
    # Check for fftw3:

    # First do a simple check that the library and header are in the path.
    if not config.CheckHeader('fftw3.h',language='C++'):
        ErrorExit(
            'fftw3.h not found',
            'You should specify the location of fftw3 as FFTW_DIR=...')

    config.CheckFFTW()

    #####
    # Check for boost:
    config.CheckBoost()

    #####
    # Check for tmv:

    # First do a simple check that the library and header are in the path.
    # We check the linking with the BLAS library below.
    if not config.CheckHeader('TMV.h',language='C++'):
        ErrorExit(
            'TMV.h not found',
            'You should specify the location of TMV as TMV_DIR=...')

    tmv_version_file = """
#include <iostream>
#include "TMV.h"
int main()
{ std::cout<<tmv::TMV_Version()<<std::endl; return 0; }
"""
    ok, tmv_version = config.TryRun(tmv_version_file,'.cpp')
    print 'TMV version is '+tmv_version.strip()

    compiler = config.env['CXXTYPE']
    version = config.env['CXXVERSION_NUMERICAL']

    if not config.env.has_key('LIBS') :
        config.env['LIBS'] = []

    tmv_link_file = FindTmvLinkFile(config)

    print 'Using TMV_LINK file:',tmv_link_file
    try:
        tmv_link = open(tmv_link_file).read().strip()
    except:
        ErrorExit('Could not open TMV link file: ',tmv_link_file)
    print '    ',tmv_link

    if sys.platform.find('darwin') != -1:
        # The Mac BLAS library is notoriously sketchy.  In particular, we have discovered that it
        # is thread-unsafe for Mac OS 10.7+ prior to XCode 5.1.  Try to give an appropriate warning
        # if we can tell that this is what the TMV library is using.
        # Update: Even after 5.1, it still seems to have problems for some systems.
        import platform
        import subprocess
        print 'Mac version is',platform.mac_ver()[0]
        p = subprocess.Popen(['xcodebuild','-version'], stdout=subprocess.PIPE)
        xcode_version = p.stdout.readlines()[0].split()[1]
        print 'XCode version is',xcode_version
        if (platform.mac_ver()[0] >= '10.7' and #xcode_version < '5.1' and
            '-latlas' not in tmv_link and ('-lblas' in tmv_link or '-lcblas' in tmv_link)):
            print 'WARNING: The Apple BLAS library has been found not to be thread safe on'
            print '         Mac OS versions 10.7+, even across multiple processes (i.e. not'
            print '         just multiple threads in the same process.)  The symptom is that'
            print '         `scons tests` may hang when running nosetests using multiple'
            print '         processes.'
            if xcode_version < '5.1':
                print '         This seems to have been partially fixed with XCode 5.1, so we'
                print '         recommend upgrading to the latest XCode version.  However, even'
                print '         with 5.1, some systems still seem to have problems.'
                env['BAD_BLAS'] = True
            else:
                print '         This seems to have been partially fixed with XCode 5.1, so there'
                print '         is a good chance you will not have any problems.  But there are'
                print '         still occasional systems that fail when using multithreading with'
                print '         programs or modules that link to the BLAS library (such as GalSim).'
                print '         If you do have problems, the solution is to recompile TMV with'
                print '         the SCons option "WITH_BLAS=false".'

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

    # Finally, do the tests for the TMV library linkage:
    config.CheckTMV()

def DoPyChecks(config):
    # These checks are only relevant for the pysrc compilation:

    config.CheckPython()
    config.CheckPyTMV()
    config.CheckNumPy()
    config.CheckPyFITS()
    config.CheckBoostPython()
    config.CheckPythonExcept()


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
            import subprocess
            p = subprocess.Popen(['sysctl','-n','hw.ncpu'],stdout=subprocess.PIPE,shell=True)
            return int(p.stdout.read().strip())
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
        if env.GetOption('num_jobs') != 1:
            print "Determined that a good number of jobs =",env.GetOption('num_jobs')

    # The basic flags for this compiler if not explicitly specified
    BasicCCFlags(env)

    # Some extra flags depending on the options:
    if env['WITH_OPENMP']:
        print 'Using OpenMP'
        AddOpenMPFlag(env)
    if not env['DEBUG']:
        print 'Debugging turned off'
        env.AppendUnique(CPPDEFINES=['NDEBUG'])
    else:
        if env['TMV_DEBUG']:
            print 'TMV Extra Debugging turned on'
            env.AppendUnique(CPPDEFINES=['TMV_EXTRA_DEBUG'])

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
            'CheckFFTW' : CheckFFTW ,
            'CheckBoost' : CheckBoost ,
            })
        DoCppChecks(config)
        env = config.Finish()

        pyenv = env.Clone()
        config = pyenv.Configure(custom_tests = {
            'CheckPython' : CheckPython ,
            'CheckPyTMV' : CheckPyTMV ,
            'CheckNumPy' : CheckNumPy ,
            'CheckPyFITS' : CheckPyFITS ,
            'CheckBoostPython' : CheckBoostPython ,
            'CheckPythonExcept' : CheckPythonExcept ,
            })
        DoPyChecks(config)
        pyenv = config.Finish()
        env['final_messages'] = pyenv['final_messages']

        env['pyenv'] = pyenv

        # Turn the cache back on now, since we always want it for the main compilation steps.
        if not env['CACHE_LIB']:
            SCons.SConf.SetCacheMode('auto')

    # This one should be done after DoLibraryAndHeaderChecks
    # otherwise the TMV link test fails, since TMV wasn't compiled
    # with MEMTEST.  If you do want to test with a TMV library that
    # uses MEMTEST, you might need to move this to before
    # the DoLibraryAndHeaderChecks call.
    if env['MEM_TEST']:
        env.AppendUnique(CPPDEFINES=['MEM_TEST'])

# In both bin and examples, we will need a builder that can take a .py file,
# and add the correct shebang to the top of it, and also make it executable.
# Rather than put this funciton in both SConscript files, we put it here and
# add it as a builder to env.
def BuildExecutableScript(target, source, env):
    for i in range(len(source)):
        f = open(str(target[i]), "w")
        f.write( '#!' + env['PYTHON'] + '\n' )
        f.write(source[i].get_contents())
        f.close()
        os.chmod(str(target[i]),0775)


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
if unknown and not env['USE_UNKNOWN_VARS']:
    print 'Unknown variables:', unknown.keys()
    print 'If you are sure these are right (e.g. you want to set some SCons parameters'
    print 'that are not in the list of GalSim parameters given by scons -h)'
    print 'then you can override this check with USE_UNKNOWN_VARS=true'
    ErrorExit()

if any(opt.default != env[opt.key] for opt in opts.options):
    print 'Using the following (non-default) scons options:'
    for opt in opts.options:
        if (opt.default != env[opt.key]):
            print '   %s = %s'%(opt.key,env[opt.key])
    print 'These can be edited directly in the file %s.'%config_file
    print 'Type scons -h for a full list of available options.'
else:
    print 'Using the default scons options'

opts.Save(config_file,env)
Help(opts.GenerateHelpText(env))

# Keep track of messages to print at the end.
env['final_messages'] = []
# Everything we are going to build so we can have the final message depend on these.
env['all_builds'] = []


if not GetOption('help'):

    # If there is a gs.error file, then this means the last run ended
    # in an error, so we don't want to cache any of the configuration
    # tests from that run in case things in the environment changed.
    # (SCons isn't usually very good at detecting these kinds of changes.)
    if os.path.exists("gs.error"):
        os.remove("gs.error")
        ClearCache()

    if env['PYTHON'] == '':
        python = which('python')
        if python is None:
            python = default_python
    else:
        python = env['PYTHON']
        python = which(python)
        if python == None:
            ErrorExit('Specified python not found in path: %s' % env['PYTHON'])
    print 'Using python = ',python
    env['PYTHON'] = python

    # Set PYPREFIX if not given:
    if env['PYPREFIX'] == '':
        import subprocess
        if sys.platform.startswith('linux') and env['PREFIX'] != '':
            # On linux, we try to match the behavior of distutils
            cmd = "%s -c \"import distutils.sysconfig; "%(python)
            cmd += "print distutils.sysconfig.get_python_lib(prefix='%s')\""%(env['PREFIX'])
            p = subprocess.Popen([cmd],stdout=subprocess.PIPE,shell=True)
            env['PYPREFIX'] = p.stdout.read().strip()
            print 'Using PYPREFIX generated from PREFIX = ',env['PYPREFIX']
        else:
            # On Macs, the regular python lib is usually writable, so it works fine for
            # installing the python modules.
            cmd = "%s -c \"import distutils.sysconfig; "%(python)
            cmd += "print distutils.sysconfig.get_python_lib()\""
            p = subprocess.Popen([cmd],stdout=subprocess.PIPE,shell=True)
            env['PYPREFIX'] = p.stdout.read().strip()
            print 'Using default PYPREFIX = ',env['PYPREFIX']

    # Set up the configuration
    DoConfig(env)

    # subdirectory SConscript files can use this function
    env['_ReadFileList'] = ReadFileList
    env['_RunInstall'] = RunInstall
    env['_RunUninstall'] = RunUninstall
    env['_AddRPATH'] = AddRPATH

    # Both bin and examples use this:
    env['BUILDERS']['ExecScript'] = Builder(action = BuildExecutableScript)

    if 'examples' in COMMAND_LINE_TARGETS:
        subdirs += ['examples']

    if 'tests' in COMMAND_LINE_TARGETS:
        if env['NOSETESTS'] == '':
            nosetests = which('nosetests')
            if nosetests is None:
                env['NOSETESTS'] = None
            else:
                env['NOSETESTS'] = nosetests
        if env['NOSETESTS']:
            GetNosetestsVersion(env)
        subdirs += ['tests']

    if env['WITH_UPS']:
       subdirs += ['ups']

    # subdirectories to process.  We process src and pysrc by default
    script_files = []
    for d in subdirs:
        script_files.append(os.path.join(d,'SConscript'))

    SConscript(script_files, exports='env')

    # Print out anything we've put into the final_messages list.
    def FinalMessages(target, source, env):
        for msg in env['final_messages']:
            print
            print msg

    env['BUILDERS']['FinalMessages'] = Builder(action = FinalMessages)
    final = env.FinalMessages(target='#/final', source=None)
    Depends(final,env['all_builds'])
    AlwaysBuild(final)
    Default(final)
    if 'install' in COMMAND_LINE_TARGETS:
        env.Alias(target='install', source=final)
