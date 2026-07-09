import sys
from setuptools import setup, Extension
import numpy as np

if sys.platform == 'win32':
    # MSVC: /TC = compile as C, /O2 = optimize; -std=c99 is not valid
    compile_args = ['/O2', '/TC']
    link_args = []
else:
    # GCC / Clang (Linux, macOS)
    compile_args = ['-std=c99', '-O2']
    link_args = ['-lm']

ext = Extension(
    'eiko_skelfm._msfm',
    sources=[
        'src/msfm_module.c',
        'src/common.c',
        'src/msfm2d.c',
        'src/msfm3d.c',
    ],
    include_dirs=[np.get_include(), 'src/'],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(ext_modules=[ext])
