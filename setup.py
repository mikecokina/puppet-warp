"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

Pre-release steps
-----------------

1.  make sure all newly added data files are listed in `setup.py` in variable
    `package_data`as well as in `MANIFEST.in` file

2.  make sure version information in following destinations is up to date:

    - `version.py` in function `get_version()`
    - `README.rst` in yellow version badge https://img.shields.io/badge/version-<VERSION>-yellow.svg
    - in `src/pwarp/__init__.py`, variable `__version__`

3.  make sure that `CHANGELOG.rst` is up to date; content as well as release date and valid version

4.  make sure all newly added dependencies are listed in `requirements.txt` as well as in
    `setup.py` in variable `install_requires`

5.  make sure a latest docstsring documentation is generated and there is no error during Sphinx HTML build
    (for more comprehensive information take a look into `docs/README.rst`)

6.  make sure setup.cfg contains all supported Python versions

7.  commit changes

8.  create release branch with name `release/<VERSION>` and push changes

9.  make sure all unittests are running

10. create a tag::

    >> git tag -a v<VERSION> -m "version <VERSION>"
    >> git push origin --tags

11. build package with command::

    >> python setup.py sdist bdist_wheel

12. configure pypi repositories if necessary (use following configuration template with valid credentials)::

        # ~/.pypirc
        [distutils] # this tells distutils what package indexes you can push to
        index-servers =
            pypi

        [pypi]
        repository = https://upload.pypi.org/legacy/
        username = <username>
        password = <password>


13. release packages with following command (install twine if necessary with `pip install twine`)::

    >> twine upload dist/* -r pypi
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from version import get_version


here = path.dirname(__file__)

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = "For more information visit https://github.com/mikecokina/puppet-warp/blob/master/README.rst"

setup(
    name='puppet-warp',
    src_root='src',
    version=get_version(),

    description='Puppet Warp',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/mikecokina/puppet-warp',

    # Author details
    author='Michal Cokina',
    author_email='mikecokina@gmail.com',

    # Choose your license
    license='GPLv3',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Image Processing',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],

    # What does your project relate to?
    keywords='image',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=find_packages(where='src', exclude=["single_system"]),
    packages=find_packages(where='src'),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy>=1.21.5',
        'opencv-contrib-python>=4.5.4.60,<=4.7.0.72',
        'opencv-python>=4.5.4.60,<=4.7.0.72',
        'scikit-image>=0.19.2,<=0.21.0',
        'scikit-learn>=1.0.2,<=1.2.2',
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [],
        'jrs': ['triangle>=20200424'],
        'test': ['parameterized>=0.8.1'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'pwarp': [
            'data/*',
            'conf/logging_schemas/*'
        ],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
        ],
    },
)
