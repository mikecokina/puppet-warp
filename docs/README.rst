Build API docs
~~~~~~~~~~~~~~

1. Clone puppet-warp
2. Change directory to ``puppet-warp/docs``
3. Install requirements
4. Use terminal and change directory to docs
5. Create directory `build`
6. Run command::

    sphinx-build -W -b html -c ./source -d ./build/doctrees ./source ./build

:warning: make sure that path separator is valid for your OS


Generate apidoc - developers only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ``cd /path/to/docs``
2. activate virtual environment
3. remove old api docs folder
4. ``sphinx-apidoc  ../src/pwarp  -o ./source/api -f``
5. make sure ``source/api/pwarp.rst`` contain at the end of file

::

    .. automodule:: pwarp
       :members:
       :undoc-members:
       :show-inheritance:
       :noindex:

6. build (``sphinx-build -W -b html -c ./source -d ./build/doctrees ./source ./build -v``)
