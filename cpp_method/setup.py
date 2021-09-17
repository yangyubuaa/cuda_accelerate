from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension


include_dirs = os.path.dirname(os.path.abspath(__file__))

source_cpu = glob.glob(os.path.join(include_dirs, 'cpu', '*.cpp'))

setup(
    name='cppex',
    version="0.1",
    ext_modules=[
        CppExtension('cppex', sources=source_cpu, include_dirs=[include_dirs]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)