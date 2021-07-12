LLVM Command Guide
------------------

The following documents are command descriptions for all of the LLVM tools.
These pages describe how to use the LLVM commands and what their options are.
Note that these pages do not describe all of the options available for all
tools. To get a complete listing, pass the ``--help`` (general options) or
``--help-hidden`` (general and debugging options) arguments to the tool you are
interested in.

Basic Commands
~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   llvm-as
   llvm-dis
   opt
   llc
   lli
   llvm-link
   llvm-lib
   llvm-lipo
   llvm-config
   llvm-cxxmap
   llvm-diff
   llvm-cov
   llvm-profdata
   llvm-stress
   llvm-symbolizer
   llvm-dwarfdump
   dsymutil
   llvm-mca
   llvm-readobj

GNU binutils replacements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   llvm-addr2line
   llvm-ar
   llvm-cxxfilt
   llvm-nm
   llvm-objcopy
   llvm-objdump
   llvm-ranlib
   llvm-readelf
   llvm-size
   llvm-strings
   llvm-strip

Debugging Tools
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   bugpoint
   llvm-extract
   llvm-bcanalyzer

Developer Tools
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   FileCheck
   tblgen
   lit
   llvm-build
   llvm-exegesis
   llvm-pdbutil
   llvm-locstats
