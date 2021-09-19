from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


include_dirs = os.path.dirname(os.path.abspath(__file__))

setup(
    name='cppexm',
    version="0.2",
    ext_modules=[
        CUDAExtension('cppexm', sources=["gpu/matmul.cpp", "gpu/matmul_cuda.cu"], include_dirs=[include_dirs]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)