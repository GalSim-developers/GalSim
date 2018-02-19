from __future__ import print_function
import sys,os,glob,re
import platform
import ctypes
import types

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib
from setuptools.command.install import install
from setuptools.command.install_scripts import install_scripts
from setuptools.command.easy_install import easy_install
from setuptools.command.test import test
import setuptools
print("Using setuptools version",setuptools.__version__)

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

scripts = ['galsim', 'galsim_download_cosmos']
scripts = [ os.path.join('bin',f) for f in scripts ]

def all_files_from(dir, ext=''):
    files = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(root, filename))
    return files

py_sources = all_files_from('pysrc', '.cpp')
cpp_sources = all_files_from('src', '.cpp')
test_sources = all_files_from('tests', '.cpp')
headers = all_files_from('include')
shared_data = all_files_from('share')

# If we build with debug, undefine NDEBUG flag
undef_macros = []
if "--debug" in sys.argv:
    undef_macros+=['NDEBUG']

copt =  {
    'gcc' : ['-O2','-msse2','-std=c++11','-fvisibility=hidden'],
    'icc' : ['-O2','-msse2','-vec-report0','-std=c++11'],
    'clang' : ['-O2','-msse2','-std=c++11','-Wno-shorten-64-to-32','-fvisibility=hidden'],
    'unknown' : [],
}

if "--debug" in sys.argv:
    copt['gcc'].append('-g')
    copt['icc'].append('-g')
    copt['clang'].append('-g')

def get_compiler(cc):
    """Try to figure out which kind of compiler this really is.
    In particular, try to distinguish between clang and gcc, either of which may
    be called cc or gcc.
    """
    cmd = [cc,'--version']
    import subprocess
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    print('compiler version information: ')
    for line in lines:
        print(line.decode().strip())
    try:
        # Python3 needs this decode bit.
        # Python2.7 doesn't need it, but it works fine.
        line = lines[0].decode(encoding='UTF-8')
        if line.startswith('Configured'):
            line = lines[1].decode(encoding='UTF-8')
    except TypeError:
        # Python2.6 throws a TypeError, so just use the lines as they are.
        line = lines[0]
        if line.startswith('Configured'):
            line = lines[1]

    if 'clang' in line:
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
    else:
        return 'unknown'

# Check for the fftw3 library in some likely places
def find_fftw_lib(output=False):
    try_libdirs = []
    lib_ext = '.so'
    if 'FFTW_DIR' in os.environ:
        try_libdirs.append(os.environ['FFTW_DIR'])
        try_libdirs.append(os.path.join(os.environ['FFTW_DIR'],'lib'))
    if 'posix' in os.name.lower():
        try_libdirs.extend(['/usr/local/lib', '/usr/lib'])
    if 'darwin' in platform.system().lower():
        try_libdirs.extend(['/usr/local/lib', '/usr/lib', '/sw/lib', '/opt/local/lib'])
        lib_ext = '.dylib'
    for path in ['LIBRARY_PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH']:
        if path in os.environ:
            for dir in os.environ[path].split(':'):
                try_libdirs.append(dir)
    # If the above don't work, the fftw3 module may have the right directory.
    try:
        import fftw3
        try_libdirs.append(fftw3.lib.libdir)
    except ImportError:
        pass

    name = 'libfftw3' + lib_ext
    if output:
        print("Looking for ",name)
    for dir in try_libdirs:
        if output:
            print("  ", dir, end='')
        try:
            libpath = os.path.join(dir, name)
            lib = ctypes.cdll.LoadLibrary(libpath)
            if output:
                print("  (yes)")
            return libpath
        except OSError as e:
            if output:
                print("  (no)")
            continue
    try:
        libpath = ctypes.util.find_library('fftw3')
        if libpath == None:
            raise OSError
        if output:
            print("  ", os.path.split(libpath)[0], end='')
        lib = ctypes.cdll.LoadLibrary(libpath)
        if output:
            print("  (yes)")
        return libpath
    except Exception as e:
        if output:
            print("Could not find fftw3 library.  Make sure it is installed either in a standard ")
            print("location such as /usr/local/lib, or the installation directory is either in ")
            print("your LIBRARY_PATH or FFTW_DIR environment variable.")
        raise

# Check for Eigen in some likely places
def find_eigen_dir(output=False):
    import distutils.sysconfig

    try_dirs = []
    if 'EIGEN_DIR' in os.environ:
        try_dirs.append(os.environ['EIGEN_DIR'])
        try_dirs.append(os.path.join(os.environ['EIGEN_DIR']))
    # This is where conda will install it.
    try_dirs.append(distutils.sysconfig.get_config_var('INCLUDEDIR'))
    if 'posix' in os.name.lower():
        try_dirs.extend(['/usr/local/include', '/usr/include'])
    if 'darwin' in platform.system().lower():
        try_dirs.extend(['/usr/local/include', '/usr/include', '/sw/include',
                            '/opt/local/include'])
    for path in ['C_INCLUDE_PATH']:
        if path in os.environ:
            for dir in os.environ[path].split(':'):
                try_dirs.append(dir)
    # eigency is a python package that bundles the Eigen header files, so if that's there,
    # can use that.
    try:
        import eigency
        try_dirs.append(eigency.get_includes()[2])
    except ImportError:
        pass

    if output:
        print("Looking for Eigen:")
    for dir in try_dirs:
        if output:
            print("  ", dir, end='')
        if os.path.isfile(os.path.join(dir, 'Eigen/Core')):
            if output:
                print("  (yes)")
            return dir
        if os.path.isfile(os.path.join(dir, 'eigen3', 'Eigen/Core')):
            dir = os.path.join(dir, 'eigen3')
            if output:
                # Only print this if the eigen3 addition was key to finding it.
                print("\n  ", dir, "  (yes)")
            return dir
        if output:
            print("  (no)")
    if output:
        print("Could not find Eigen.  Make sure it is installed either in a standard ")
        print("location such as /usr/local/include, or the installation directory is either in ")
        print("your C_INCLUDE_PATH or EIGEN_DIR environment variable.")
    raise OSError("Could not find Eigen")


def try_cc(cc, cflags=[], lflags=[]):
    """Check if compiling a simple bit of c++ code with the given compiler works properly.
    """
    import subprocess
    import tempfile
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
    cpp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.cpp')
    cpp_file.write(cpp_code.encode())
    cpp_file.close();
    os_file = tempfile.NamedTemporaryFile(delete=False, suffix='.os')
    os_file.close()
    exe_file = tempfile.NamedTemporaryFile(delete=False, suffix='.exe')
    exe_file.close()

    # Compile
    cmd = cc + ' ' + ' '.join(cflags + ['-c',cpp_file.name,'-o',os_file.name])
    #print('cmd = ',cmd)
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        lines = p.stdout.readlines()
        #print('output = ',lines)
        p.communicate()
    except (IOError,OSError) as e:
        p.returncode = 1
    if p.returncode != 0:
        os.remove(cpp_file.name)
        if os.path.exists(os_file.name):
            os.remove(os_file.name)
        return False

    # Link
    cmd = cc + ' ' + ' '.join(lflags + [os_file.name,'-o',exe_file.name])
    #print('cmd = ',cmd)
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        lines = p.stdout.readlines()
        #print('output = ',lines)
        p.communicate()
    except (IOError,OSError) as e:
        p.returncode = 1

    if p.returncode and cc.endswith('cc'):
        # The linker needs to be a c++ linker, which isn't 'cc'.  However, I couldn't figure
        # out how to get setup.py to tell me the actual command to use for linking.  All the
        # executables available from build_ext.compiler.executables are 'cc', not 'c++'.
        # I think this must be related to the bugs about not handling c++ correctly.
        #    http://bugs.python.org/issue9031
        #    http://bugs.python.org/issue1222585
        # So just switch it manually and see if that works.
        cmd = 'c++ ' + ' '.join(lflags + [os_file.name,'-o',exe_file.name])
        #print('cmd = ',cmd)
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            lines = p.stdout.readlines()
            #print('output = ',lines)
            p.communicate()
        except (IOError,OSError) as e:
            p.returncode = 1

    # Remove the temp files
    os.remove(cpp_file.name)
    os.remove(os_file.name)
    if os.path.exists(exe_file.name):
        os.remove(exe_file.name)
    return p.returncode == 0

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
    ncpu = cpu_count()
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

    if ncpu == 1:
        # This is equivalent to regular compile function
        for obj in objects:
            _single_compile(obj)
    else:
        # This next bit is taken from here:
        # https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
        # convert to list, imap is evaluated on-demand
        list(multiprocessing.pool.ThreadPool(ncpu).imap(_single_compile,objects))

    # Return *all* object filenames, not just the ones we just built.
    return objects


def fix_compiler(compiler, parallel):
    # Remove any -Wstrict-prototypes in the compiler flags (since invalid for C++)
    try:
        compiler.compiler_so.remove("-Wstrict-prototypes")
    except (AttributeError, ValueError):
        pass

    # Figure out what compiler it will use
    #print('compiler = ',compiler.compiler)
    cc = compiler.compiler_so[0]
    cflags = compiler.compiler_so[1:]
    comp_type = get_compiler(cc)
    if cc == comp_type:
        print('Using compiler %s'%(cc))
    else:
        print('Using compiler %s, which is %s'%(cc,comp_type))

    # Check if we can use ccache to speed up repeated compilation.
    if try_cc('ccache ' + cc, cflags):
        print('Using ccache')
        compiler.set_executable('compiler_so', ['ccache',cc] + cflags)

    if parallel is None or parallel is True:
        ncpu = cpu_count()
    elif parallel: # is an integer
        ncpu = parallel
    else:
        ncpu = 1
    if ncpu > 1:
        print('Using %d cpus for compiling'%ncpu)
        if parallel is None:
            print('To override, you may do python setup.py build -j1')
        compiler.compile = types.MethodType(parallel_compile, compiler)

    extra_cflags = copt[comp_type]
    print('Using extra flags ',extra_cflags)

    # Return the extra cflags, since those will be added to the build step in a different place.
    return extra_cflags

def add_dirs(builder, output=False):
    # We need to do most of this both for build_clib and build_ext, so separate it out here.

    # First some basic ones we always need.
    builder.include_dirs.append('include')
    builder.include_dirs.append('include/galsim')

    # Look for fftw3.
    fftw_lib = find_fftw_lib(output=output)
    fftw_libpath, fftw_libname = os.path.split(fftw_lib)
    if hasattr(builder, 'library_dirs'):
        builder.library_dirs.append(os.path.split(fftw_lib)[0])
        builder.libraries.append(os.path.split(fftw_lib)[1].split('.')[0][3:])
    fftw_include = os.path.join(os.path.split(fftw_libpath)[0], 'include')
    if os.path.isfile(os.path.join(fftw_include, 'fftw3.h')):
        # Usually, the fftw3.h file is in an associated include dir, but not always.
        builder.include_dirs.append(fftw_include)
    else:
        # If not, we have our own copy of fftw3.h here.
        builder.include_dirs.append('include/fftw3')

    # Look for Eigen/Core
    eigen_dir = find_eigen_dir(output=output)
    builder.include_dirs.append(eigen_dir)

    # Finally, add pybind11's include dir
    import pybind11
    print('PyBind11 is version ',pybind11.__version__)
    # Include both the standard location and the --user location, since it's hard to tell
    # which one is the right choice.
    builder.include_dirs.append(pybind11.get_include(user=True))
    builder.include_dirs.append(pybind11.get_include(user=False))
    print('Include files for pybind11 are ',builder.include_dirs[-2:])


# Make a subclass of build_ext so we can add to the -I list.
class my_build_clib(build_clib):
    def finalize_options(self):
        build_clib.finalize_options(self)
        add_dirs(self, output=True)  # This happens first, so only output for this call.

    # Add any extra things based on the compiler being used..
    def build_libraries(self, libraries):

        # They didn't put the parallel option into build_clib like they did with build_ext, so
        # look for the parallel option there instead.
        build_ext = self.distribution.get_command_obj('build_ext')
        parallel = getattr(build_ext, 'parallel', True)

        cflags = fix_compiler(self.compiler, parallel)

        # Add the appropriate extra flags for that compiler.
        for (lib_name, build_info) in libraries:
            build_info['cflags'] = build_info.get('cflags',[]) + cflags

        # Now run the normal build function.
        build_clib.build_libraries(self, libraries)


# Make a subclass of build_ext so we can add to the -I list.
class my_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        add_dirs(self)

    # Add any extra things based on the compiler being used..
    def build_extensions(self):

        # The -jN option was new in distutils version 3.5.
        # If user has older version, just set parallel to True and move on.
        parallel = getattr(self, 'parallel', True)

        cflags = fix_compiler(self.compiler, parallel)

        # Add the appropriate extra flags for that compiler.
        for e in self.extensions:
            e.extra_compile_args = cflags

        # Now run the normal build function.
        build_ext.build_extensions(self)


def make_meta_data(install_dir):
    print('install_dir = ',install_dir)
    meta_data_file = os.path.join('galsim','meta_data.py')
    share_dir = os.path.join(install_dir,'galsim','share')
    try:
        f = open(meta_data_file,'w')
    except IOError:
        # Not sure if this is still relevant in setup.py world, but if user ran this under
        # sudo and now is not using sudo, then the file might exist, but not be writable.
        # However, it should still be removable, since the directory should be owned
        # by the user.  So remove it and then retry opening it.
        os.remove(meta_data_file)
        f = open(meta_data_file,'w')

    f.write('# This file is automatically generated by setup.py when building GalSim.\n')
    f.write('# Do not edit.  Any edits will be lost the next time setpu.py is run.\n')
    f.write('\n')
    f.write('install_dir = "%s"\n'%install_dir)
    f.write('share_dir = "%s"\n'%share_dir)
    f.close()
    return meta_data_file

class my_install(install):
    def run(self):
        # Make the meta_data.py file based on the actual installation directory.
        meta_data_file = make_meta_data(self.install_lib)
        install.run(self)

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
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = None

    def finalize_options(self):
        test.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_cpp_tests(self):
        import subprocess

        builder = self.distribution.get_command_obj('build_ext')
        compiler = builder.compiler
        ext = builder.extensions[0]
        objects = compiler.compile(test_sources,
                output_dir=builder.build_temp,
                macros=ext.define_macros,
                include_dirs=ext.include_dirs,
                debug=builder.debug,
                extra_postargs=ext.extra_compile_args,
                depends=ext.depends)

        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []

        libraries = builder.get_libraries(ext)
        library_dirs = ext.library_dirs
        fftw_lib = find_fftw_lib()
        fftw_libpath, fftw_libname = os.path.split(fftw_lib)
        library_dirs.append(os.path.split(fftw_lib)[0])
        libraries.append(os.path.split(fftw_lib)[1].split('.')[0][3:])
        libraries.append('galsim')

        exe_file = os.path.join(builder.build_temp,'cpp_test')
        compiler.link_executable(
                objects, 'cpp_test',
                output_dir=builder.build_temp,
                libraries=libraries,
                library_dirs=library_dirs,
                runtime_library_dirs=ext.runtime_library_dirs,
                extra_postargs=extra_args,
                debug=builder.debug,
                target_lang='c++')

        p = subprocess.Popen([exe_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = p.stdout.readlines()
        p.communicate()
        for line in lines:
            print(line.decode().strip())
        if p.returncode == 0:
            print("All C++ tests passed.")
        else:
            raise RuntimeError("C++ tests failed")

    def run_tests(self):
        import pytest

        # Build and run the C++ tests
        self.run_cpp_tests()

        ncpu = cpu_count()
        if self.pytest_args is None:
            self.pytest_args = ['-n=%d'%ncpu, '--timeout=60']
        else:
            self.pytest_args = self.pytest_args.split()

        #print('Using pytest args: ',self.pytest_args,' (can update with -a pytest_args)')
        original_dir = os.getcwd()
        os.chdir('tests')
        test_files = glob.glob('test*.py')

        errno = pytest.main(self.pytest_args + test_files)
        errno = 0
        if errno != 0:
            sys.exit(errno)
        os.chdir(original_dir)


lib=("galsim", {'sources' : cpp_sources,
                'depends' : headers,
                'include_dirs' : ['include', 'include/galsim'],
                'undef_macros' : undef_macros })
ext=Extension("galsim._galsim",
              py_sources,
              undef_macros = undef_macros)

build_dep = ['pybind11>=2.2']
run_dep = ['numpy', 'future', 'astropy', 'LSSTDESC.Coord']
test_dep = ['pytest', 'pytest-xdist', 'pytest-timeout', 'scipy']

# If Eigen doesn't exist in the normal places, add eigency ad a build dependency.
try:
    find_eigen_dir()
except OSError:
    print('Adding eigency to build_dep')
    build_dep += ['eigency>=1.77']


with open('README.md') as file:
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
version_info = tuple(map(int, galsim_version.split('.')))
if len(version_info) == 2:
    version_info = version_info + (0,)
version_h_text = """
// This file is auto-generated by SCons.  Do not edit.
#define GALSIM_MAJOR %d
#define GALSIM_MINOR %d
#define GALSIM_REVISION %d

#include <string>
#include <sstream>

namespace galsim {
    // Compiled versions of the above #define values.
    extern int major_version();
    extern int minor_version();
    extern int revision();

    // Returns string of the form "1.4.2"
    extern std::string version();

    // Checks if the compiled library version matches the #define values in this header file.
    inline bool check_version() {
        // Same code as version(), but inline, so we get the above values to compare
        // to the values compiled into the library.
        std::ostringstream oss;
        oss << GALSIM_MAJOR << '.' << GALSIM_MINOR << '.' << GALSIM_REVISION;
        return oss.str() == version();
    }
}
"""%version_info
version_h_file = os.path.join('include', 'galsim', 'Version.h')
with open(version_h_file, 'w') as f:
    f.write(version_h_text)

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
    package_data={'galsim' : shared_data},
    #include_package_data=True,
    libraries=[lib],
    ext_modules=[ext],
    setup_requires=build_dep,
    install_requires=run_dep,
    tests_require=test_dep,
    cmdclass = {'build_ext': my_build_ext,
                'build_clib': my_build_clib,
                'install': my_install,
                'install_scripts': my_install_scripts,
                'easy_install': my_easy_install,
                'test': my_test,
                },
    entry_points = {'console_scripts' : [
            'galsim = galsim.__main__:main',
            'galsim_download_cosmos = galsim.download_cosmos:main'
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

