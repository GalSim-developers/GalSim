from __future__ import print_function
import sys,os,glob,re
import platform
import ctypes


from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
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
print('sources = ',sources)
print('headers = ',headers)
print('shared = ',shared_data)

# If we build with debug, undefine NDEBUG flag
undef_macros = []
if "--debug" in sys.argv:
    undef_macros+=['NDEBUG']

copt =  {
    'gcc' : ['-O3','-ffast-math','-std=c++11'],
    'icc' : ['-O3','-std=c++11'],
    'clang' : ['-O3','-ffast-math','-std=c++11','-Wno-shorten-64-to-32'],
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

# Make a subclass of build_ext so we can add to the -I list.
class my_builder( build_ext ):
    # Adding the libraries and include_dirs here rather than when declaring the Extension
    # means that the setup_requires modules should already be installed, so pybind11, eigency,
    # and fftw3 should all import properly.
    def finalize_options(self):
        print('finalize_options:')
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
        print('include_dirs = ',self.include_dirs)
        print('library_dirs = ',self.library_dirs)
        print('libraries = ',self.libraries)

    # Add any extra things based on the compiler being used..
    def build_extensions(self):
        print('Platform is ',self.plat_name)

        # Figure out what compiler it will use
        cc = self.compiler.executables['compiler_cxx'][0]
        print('Using compiler %s'%(cc))
        # Figure out what compiler it will use
        cc = self.compiler.executables['compiler_cxx'][0]
        comp_type = get_compiler(cc)
        if cc == comp_type:
            print('Using compiler %s'%(cc))
        else:
            print('Using compiler %s, which is %s'%(cc,comp_type))

        # Add the appropriate extra flags for that compiler.
        for e in self.extensions:
            e.extra_compile_args = copt[ comp_type ]
            #e.extra_link_args = lopt[ comp_type ]

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

# AFAICT, setuptools doesn't provide any easy access to the final installation location of the
# executable scripts.  This bit is just to save the value of script_dir so I can use it later.
# cf. http://stackoverflow.com/questions/12975540/correct-way-to-find-scripts-directory-from-setup-py-in-python-distutils/
class my_easy_install( easy_install ):
    def finalize_options(self):
        easy_install.finalize_options(self)
        # Make the meta_data.py file based on the actual installation directory.
        make_meta_data(self.install_dir)

    # Match the call signature of the easy_install version.
    def write_script(self, script_name, contents, mode="t", *ignored):
        # Run the normal version
        easy_install.write_script(self, script_name, contents, mode, *ignored)
        # Save the script install directory in the distribution object.
        # This is the same thing that is returned by the setup function.
        self.distribution.script_install_dir = self.script_dir

ext=Extension("galsim._galsim",
              sources,
              undef_macros = undef_macros)

# Note: We don't actually need cython, but eigency depends on it at build time, and their
# setup.py is broken such that if it's not already installed it fails catastrophically.
build_dep = ['pybind11', 'cython', 'eigency']
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

dist = setup(name="GalSim", 
    version=galsim_version,
    author="GalSim Developers (point of contact: Mike Jarvis)",
    author_email="michael@jarvis.net",
    description="The modular galaxy image simulation toolkit",
    long_description=long_description,
    license = "BSD License",
    url="https://github.com/rmjarvis/GalSim",
    download_url="https://github.com/GalSim-developers/GalSim/releases/tag/v%s.zip"%galsim_version,
    packages=['galsim'],
    include_package_data=True,
    ext_modules=[ext],
    setup_requires=build_dep,
    install_requires=build_dep + run_dep,
    cmdclass = {'build_ext': my_builder,
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
if (hasattr(dist,'script_install_dir') and
    dist.script_install_dir not in os.environ['PATH'].split(':') and
    os.path.realpath(dist.script_install_dir) not in real_env_path):

    print('\nWARNING: The GalSim executables were installed in a directory not in your PATH')
    print('         If you want to use the executables, you should add the directory')
    print('\n             ',dist.script_install_dir,'\n')
    print('         to your path.  The current path is')
    print('\n             ',os.environ['PATH'],'\n')
    print('         Alternatively, you can specify a different prefix with --prefix=PREFIX,')
    print('         in which case the scripts will be installed in PREFIX/bin.')
    print('         If you are installing via pip use --install-option="--prefix=PREFIX"')
