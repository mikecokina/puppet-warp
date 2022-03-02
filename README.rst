|GitHub version|  |Licence GPLv3| |Python version| |OS|

.. |GitHub version| image:: https://img.shields.io/badge/version-0.0.0.dev0-yellow.svg
   :target: https://github.com/Naereen/StrapDown.js

.. |Python version| image:: https://img.shields.io/badge/python-3.7|3.8|3.9-orange.svg
   :target: https://github.com/Naereen/StrapDown.js

.. |Licence GPLv3| image:: https://img.shields.io/badge/license-GNU/GPLv3-blue.svg
   :target: https://github.com/Naereen/StrapDown.js

.. |OS| image:: https://img.shields.io/badge/os-Linux|Windows-magenta.svg
   :target: https://github.com/Naereen/StrapDown.js

Puppet Warp
-----------

The goal of the package `puppet-warp` is provide plug and play solution for image
transformation similar to Adobe Photoshop `Puppet Warp` tool. Since Photoshop
solution is proprietary, hence any scripting might be a big issues especially in
enviroments where Photoshop is not supported, we decided to create this package based
on Python in which Puppet Warp is programmatically manageable and used in automatation
processes where advaned transformation method is required.

Features
--------

- As-Rigid-as-Possible Shape Manipulation of triangular mesh
- Image transfer from triangualar mesh at rest to mesh defined by ARAP transformation

Installation
~~~~~~~~~~~~
TBD


Usage
~~~~~
TBD


References
----------

::

[1] https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf
[2] https://github.com/deliagander/ARAPShapeManipulation.git
[3] https://learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/


Cite:
-----

::

    @article{journals/jgtools/IgarashiI09,
        author = {Igarashi, Takeo and Igarashi, Yuki},
        ee = {http://dx.doi.org/10.1080/2151237X.2009.10129273},
        journal = {J. Graphics, GPU, & Game Tools},
        number = 1,
        pages = {17-30},
        title = {Implementing As-Rigid-As-Possible Shape Manipulation and Surface Flattening.},
        url = {http://dblp.uni-trier.de/db/journals/jgtools/jgtools14.html#IgarashiI09},
        volume = 14,
        year = 2009
    }

or

::

    @article{10.1145/1073204.1073323,
        author = {Igarashi, Takeo and Moscovich, Tomer and Hughes, John F.},
        title = {As-Rigid-as-Possible Shape Manipulation},
        year = {2005},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {24},
        number = {3},
        doi = {10.1145/1073204.1073323},
        journal = {ACM Trans. Graph.},
        month = {jul},
        pages = {1134–1141},
        numpages = {8}
    }

