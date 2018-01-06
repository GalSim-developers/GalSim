from __future__ import print_function
import sys,os,glob,re
import platform
import ctypes
import types

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.install_scripts import install_scripts
from setuptools.command.easy_install import easy_install
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

sources = all_files_from('src', '.cpp') + all_files_from('pysrc', '.cpp')
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
def find_fftw_lib():
    try_libdirs = []
    lib_ext = '.so'
    if 'FFTW_PATH' in os.environ:
        try_libdirs.append(os.environ['FFTW_PATH'])
        try_libdirs.append(os.path.join(os.environ['FFTW_PATH'],'lib'))
    if 'posix' in os.name.lower():
        try_libdirs.extend(['/usr/local/lib', '/usr/lib'])
    if 'darwin' in platform.system().lower():
        try_libdirs.extend(['/sw/lib', '/opt/local/lib'])
        lib_ext = '.dylib'
    for path in ['LIBRARY_PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH']:
        if path in os.environ:
            for dir in os.environ[path].split(':'):
                try_libdirs.append(dir)

    name = 'libfftw3' + lib_ext
    for dir in try_libdirs:
        try:
            libpath = os.path.join(dir, name)
            lib = ctypes.cdll.LoadLibrary(libpath)
            print("found %s at %s" %(name, libpath))
            return libpath
        except OSError as e:
            print("Did not find %s in %s" %(name, libpath))
            continue
    print("Could not find %s in any of the normal locations"%name)
    print("Trying ctypes.util.find_library")
    try:
        libpath = ctypes.util.find_library('fftw3')
        if libpath == None:
            raise OSError
        lib = ctypes.cdll.LoadLibrary(libpath)
        print("found %s at %s" %(name, libpath))
        return libpath
    except Exception as e:
        print("Could not find fftw3 library.  Make sure it is installed either in a standard ")
        print("location such as /usr/local/lib, or the installation directory is either in ")
        print("your LIBRARY_PATH or FFTW_PATH environment variable.")
        raise

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


# Make a subclass of build_ext so we can add to the -I list.
class my_builder( build_ext ):
    # Adding the libraries and include_dirs here rather than when declaring the Extension
    # means that the setup_requires modules should already be installed, so pybind11, eigency,
    # and fftw3 should all import properly.
    def finalize_options(self):
        build_ext.finalize_options(self)
        self.include_dirs.append('include')
        self.include_dirs.append('include/galsim')

        import pybind11
        # Include both the standard location and the --user location, since it's hard to tell
        # which one is the right choice.
        self.include_dirs.append(pybind11.get_include(user=False))
        self.include_dirs.append(pybind11.get_include(user=True))

        self.include_dirs.append('include/fftw3')
        fftw_lib = find_fftw_lib()
        fftw_libpath, fftw_libname = os.path.split(fftw_lib)
        self.library_dirs.append(os.path.split(fftw_lib)[0])
        self.libraries.append(os.path.split(fftw_lib)[1].split('.')[0][3:])

        import eigency
        self.include_dirs.append(eigency.get_includes()[2])

    # Add any extra things based on the compiler being used..
    def build_extensions(self):
        # Remove any -Wstrict-prototypes in the compiler flags (since invalid for C++)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass

        print('Platform is ',self.plat_name)

        # Figure out what compiler it will use
        #print('compiler_so = ',self.compiler.compiler_so)
        cc = self.compiler.compiler_so[0]
        cflags = self.compiler.compiler_so[1:]
        comp_type = get_compiler(cc)
        if cc == comp_type:
            print('Using compiler %s'%(cc))
        else:
            print('Using compiler %s, which is %s'%(cc,comp_type))

        # Add the appropriate extra flags for that compiler.
        print('Using extra args ',copt[comp_type])
        cflags += copt[comp_type]

        # Check if we can use ccache to speed up repeated compilation.
        if try_cc('ccache ' + cc, cflags):
            print('Using ccache')
            self.compiler.set_executable('compiler_so', ['ccache',cc] + cflags)
        #print('compiler_so => ',self.compiler.compiler_so)

        # Try to compile in parallel
        if self.parallel is None or self.parallel is True:
            ncpu = cpu_count()
        elif self.parallel: # is an integer
            ncpu = self.parallel
        else:
            ncpu = 1
        if ncpu > 1:
            print('Using %d cpus for compiling'%ncpu)
            if self.parallel is None:
                print('To override, you may do python setup.py build -j1')
            self.compiler.compile = types.MethodType(parallel_compile, self.compiler)

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

ext=Extension("galsim._galsim",
              sources,
              depends=headers,
              undef_macros = undef_macros)

# Note: We don't actually need cython or setuptools_scm, but eigency depends on them at build time,
# and their setup.py is broken such that if they're not already installed it fails catastrophically.
build_dep = ['pybind11', 'setuptools_scm', 'cython', 'eigency']
run_dep = ['numpy', 'future', 'astropy', 'pyyaml', 'LSSTDESC.Coord', 'pandas']

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
    ext_modules=[ext],
    setup_requires=build_dep,
    install_requires=build_dep + run_dep,
    cmdclass = {'build_ext': my_builder,
                'install': my_install,
                'install_scripts': my_install_scripts,
                'easy_install': my_easy_install,
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

