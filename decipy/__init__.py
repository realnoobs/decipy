# Copyright (c) 2017-2018 Rizki Sasri Dwitama
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
A collection of Multi Criteria Decision Making (MCDM) functions
for python.  The function names appear below.

Disclaimers :  This modu is obviously incomplete and, worse, the
functions are not optimized.  All functions have been tested (some more
so than others), but they are far from bulletproof.  Thus, as with any
free software, no warranty or guarantee is expressed or implied.

"""

from .utils.version import get_version

VERSION = (0, 0, 2, 'final', 1)

__author__ = "Rizki Sasri Dwitama <sasri.project@gmail.org>"
__version__ = get_version(VERSION)
