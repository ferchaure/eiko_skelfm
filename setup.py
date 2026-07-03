import sys
from setuptools import setup, Extension
import numpy as np

ext = Extension(
    'eiko_skelfm._msfm',
    sources=[
        'src/msfm_module.c',
        'src/common.c',
        'src/msfm2d.c',
        'src/msfm3d.c',
    ],
    include_dirs=[np.get_include(), 'src/'],
    extra_compile_args=['-std=c99', '-O2'],
    extra_link_args=['-lm'] if sys.platform != 'win32' else [],
)

setup(ext_modules=[ext])
