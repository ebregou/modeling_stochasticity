import numpy as np
import os
import sys
import subprocess as sbp
import os.path as osp
from Cython.Build import cythonize
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.sysconfig import get_python_lib

import site
path_install = site.getusersitepackages()
if os.path.exists(path_install):
  pass
else:
  path_install = site.getsitepackages()[0]
path_install = os.path.join(path_install,"class_public")

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

#with open("class_public/README.md", 'r') as f:
#    long_description = f.read()

# Recover the gcc compiler
GCCPATH_STRING = sbp.Popen(
    ['gcc', '-print-libgcc-file-name'],
    stdout=sbp.PIPE).communicate()[0]
GCCPATH = osp.normpath(osp.dirname(GCCPATH_STRING)).decode()

liblist = ["class"]
MVEC_STRING = sbp.Popen(
    ['gcc', '-lmvec'],
    stderr=sbp.PIPE).communicate()[1]
if b"mvec" not in MVEC_STRING:
    liblist += ["mvec","m"]

# define absolute paths
root_folder = "class_public"
include_folder = os.path.join(root_folder, "include")
classy_folder = os.path.join(root_folder, "python")
heat_folder = os.path.join(os.path.join(root_folder, "external"),"heating")
recfast_folder = os.path.join(os.path.join(root_folder, "external"),"RecfastCLASS")
hyrec_folder = os.path.join(os.path.join(root_folder, "external"),"HyRec2020")

# Recover the CLASS version
with open(os.path.join(include_folder, 'common.h'), 'r') as v_file:
    for line in v_file:
        if line.find("_VERSION_") != -1:
            # get rid of the " and the v
            VERSION = line.split()[-1][2:-1]+".2"
            break

# Define cython extension and fix Python version
classy_ext = Extension("classy", [os.path.join(classy_folder, "classy.pyx")],
                                  include_dirs=[np.get_include(), include_folder, heat_folder, recfast_folder, hyrec_folder],
                           libraries=liblist,
                           library_dirs=[root_folder, GCCPATH],
                           #extra_link_args=['-lgomp']
                           language="c++",
                           extra_compile_args=["-std=c++11"]
                       )

classy_ext.cython_directives = {'language_level': "3" if sys.version_info.major>=3 else "2"}

class ClassyBuildExt(build_ext):
  def run(self):
    import subprocess
    #run_env = dict(CLASSDIR='\"' + path_install + '\"',**(os.environ.copy()))
    run_env = dict(CLASSDIR=path_install,**(os.environ.copy()))
    subprocess.Popen("make libclass.a -j",shell=True,cwd=os.path.join(os.getcwd(),"class_public"),env=run_env).wait()
    build_ext.run(self)

pck_files = package_files('class_public')
print(pck_files)

long_description = "The official repository for the classy code ('http://www.class-code.net')."
setup(
    name='classy',
    version=VERSION,
    author="Julien Lesgourgues & Thomas Tram",
    author_email="Julien.Lesgourgues@physik.rwth-aachen.de",
    description='Python interface to the Cosmological Boltzmann code CLASS',
    long_description=long_description,
    #long_description_content_type="text/plain",
    url='http://www.class-code.net',
    cmdclass={'build_ext': ClassyBuildExt},
    ext_modules=[classy_ext],
    packages=['class_public'],
    package_data={'class_public': pck_files},
    include_package_data=True
)

