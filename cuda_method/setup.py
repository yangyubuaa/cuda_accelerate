from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


include_dirs = os.path.dirname(os.path.abspath(__file__))

source_cpu = glob.glob(os.path.join(include_dirs, 'gpu', '*.cpp', "*.cu"))
setup(
    name='cppex_gpu',
    version="0.1",
    ext_modules=[
        CUDAExtension('cppex_gpu', sources=["gpu/add.cpp", "gpu/add_cuda.cu"], include_dirs=[include_dirs]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)