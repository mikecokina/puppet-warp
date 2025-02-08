Change Log
==========
|

v0.4.dev0
---------
.. _v0.4.dev0: https://github.com/mikecokina/puppet-warp/tree/dev

**Release date:** YYYY-MM-DD

**Enhancements**

- Update dependencies to fulfill compatibility with Python up to v3.12.


v0.3
----
.. _v0.3: https://github.com/mikecokina/puppet-warp/tree/release/0.3

**Release date:** 2024-03-10

**Features**

- Demo supports control points remove
- Triangle is not mandatory requirements anymore, triangulation is by default provided via `scipy.spatial.Delaunay`


v0.2
----

v0.2_
-----
.. _v0.2: https://github.com/mikecokina/puppet-warp/tree/release/0.2

**Release date:** 2023-06-02

**Features**

- Wavefront loader supports following forms of `face` definitions:

    - f v1 v2 v3 ....
    - f v1/vt1 v2/vt2 v3/vt3 ...
    - f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
    - f v1//vn1 v2//vn2 v3//vn3 ...

**Fixes**

- `Demo` loader supports scaling factor as `FLOAT` instead of `INT` what also allows to decrease size of supplied triangular mesh
- Since there was issue with some Python versions and `triangle` package availability, requirements for library are decreased to `20200424`
- Numpy is not fixed to the single version anymore


v0.1_
-----
.. _v0.1: https://github.com/mikecokina/puppet-warp/tree/release/0.1
.. _Takeo_Igarashi_2009: https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf
.. _ARAPShapeManipulation: https://github.com/deliagander/ARAPShapeManipulation.git

**Release date:** 2022-03-05

**Features**


* **pwarp modul**

    - Implementiion As-Rigid-As-Possible Shape Manipulation based on paper Takeo_Igarashi_2009_
    - Implementation of Demo based on script ARAPShapeManipulation_ by `deliagander`
    - Implementation of image transformation based on transformation of triangular mesh over image.
