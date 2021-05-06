# -*- coding: utf-8 -*-
import numpy
from setuptools import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

#
setup(
    ext_modules = cythonize("algo.pyx", annotate=False), include_dirs=[numpy.get_include()]
)
