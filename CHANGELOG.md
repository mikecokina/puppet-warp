# Changelog

## v0.4.dev0
[View on GitHub](https://github.com/mikecokina/puppet-warp/tree/dev)

**Release date:** YYYY-MM-DD

### Enhancements

- Update dependencies to ensure compatibility up to Python 3.12.
- Drop support for Python versions lower than 3.10.
- Update `graph_defined_warp` to accept background color via the `bg_fill` parameter.
- Migrate packaging to `pyproject.toml` (replacing deprecated `setup.py`).
- Apply speed optimizations across the codebase.

---

## v0.3
[View on GitHub](https://github.com/mikecokina/puppet-warp/tree/release/0.3)

**Release date:** 2024-03-10

### Features

- Add support for removing control points in the demo.
- Make Triangle optional; default triangulation now uses `scipy.spatial.Delaunay`.

---

## v0.2
[View on GitHub](https://github.com/mikecokina/puppet-warp/tree/release/0.2)

**Release date:** 2023-06-02

### Features

- Extend Wavefront OBJ loader to support multiple face definition formats:
  - `f v1 v2 v3 ...`
  - `f v1/vt1 v2/vt2 v3/vt3 ...`
  - `f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...`
  - `f v1//vn1 v2//vn2 v3//vn3 ...`

### Fixes

- Allow `Demo` loader to use floating-point scaling factors.
- Relax Triangle dependency version to `20200424` for broader compatibility.
- Remove strict NumPy version pinning.

---

## v0.1
[View on GitHub](https://github.com/mikecokina/puppet-warp/tree/release/0.1)

**Release date:** 2022-03-05

### Features

- Implement As-Rigid-As-Possible shape manipulation based on Igarashi et al. (2009).
- Provide an interactive demo inspired by `ARAPShapeManipulation` by *deliagander*.
- Support image deformation via triangular mesh warping.

### References

- [Igarashi et al., 2009](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf)
- [ARAPShapeManipulation](https://github.com/deliagander/ARAPShapeManipulation.git)