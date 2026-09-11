# Third-party notices

The authors' material is available under the [MIT License](../LICENSE),
with the scope stated in [NOTICE.md](../NOTICE.md). Third-party dependencies
and fonts retain their own licenses. The project license does not replace
those terms.

## Project source and data

`cmfg_cce/` is a byte-exact copy of 105 scientific files, identified by
`metadata/source-copy-manifest.json`. `policy_cce_repro/` contains the
reproduction adapters. The copy manifest records provenance and integrity;
it is not evidence of a third party's permission to redistribute material.
The authors have authorized the public release and MIT license.

The bounded source review found no separately vendored dependency tree in
the scientific/adaptor source directories. This is not a complete legal
provenance audit. Historical algorithm names and citations do not, on their
own, establish whether every implementation was independently written.
Any copied implementation must retain its original notices and permission
terms once identified.

The separate data archive retains original simulation records and provenance.
The project license covers the authors' generated data and result figures.
It does not grant rights to external industry reports, third-party materials,
the manuscript or its structural diagrams.

## Direct dependencies

The following versions are pinned by `pyproject.toml`. License descriptions
below were checked against the installed distributions' metadata and
license files during preparation. The upstream license texts govern;
this summary does not replace them.

| Dependency | Version | Upstream terms and location of notices |
| --- | --- | --- |
| NumPy | 2.1.3 | BSD-style project terms; see `numpy-2.1.3.dist-info/LICENSE.txt`, which also identifies bundled components |
| SciPy | 1.15.3 | BSD-style project terms; see `scipy-1.15.3.dist-info/LICENSE.txt` and component notices |
| PyYAML | 6.0.2 | MIT; see `PyYAML-6.0.2.dist-info/LICENSE` |
| pandas | 2.2.3 | BSD 3-Clause project terms, with additional included notices in `pandas-2.2.3.dist-info/LICENSE` |
| Matplotlib | 3.10.0 | Matplotlib's license agreement; see `matplotlib-3.10.0.dist-info/LICENSE` |
| pytest, optional test dependency | 8.3.4 | MIT; see `pytest-8.3.4.dist-info/LICENSE` |

Dependency wheels can contain additional software. For example, the checked
NumPy/SciPy wheel notices cover BLAS/LAPACK and compiler-runtime components;
the precise bundled components depend on platform and wheel build. Do not
describe an entire binary dependency bundle as covered only by the parent
project's short license name. If redistributing wheels, a container, or an
installed environment, retain the notices from those exact distributions
and review their applicable redistribution terms.

The project wheel declares dependencies; it is not a bundle of their
installed directories. Python, packaging/build tools, transitive
dependencies, and any future container base image have their own terms.
`metadata/validated-environment.json` records the tested environment;
`metadata/historical-requirements.lock` is a historical dependency record,
not a project-wide license or an instruction to redistribute every listed
package.

## Fonts, plotting, and TeX

Plotting selects available fonts and records the selected family in the
paper report. Fonts are not copied into the project package. Matplotlib's
installed font assets have their own notices, including DejaVu and STIX
notices in its distribution. Installing or redistributing other fonts
requires their applicable permissions.

Compiling the optional LaTeX proof uses an external TeX installation and
packages such as `newtxtext` and `newtxmath`. Those distributions and fonts
are not supplied by this repository. Preserve their licenses if they are
later included in a container or offline environment bundle.

## Credentials and personal information

A scoped text review of the scientific source, adapters, configuration
metadata, tests, and documentation found no matching private-key or access-
token payload and no embedded user-specific absolute home path. A
credential-pattern match in the tests is synthetic test material. Generic
`gs://` strings and optional credential-loading code in the historical cloud
modules are interfaces, not included credentials.

The data archive was scanned separately; see [DATA.md](DATA.md). Original
records can contain hostnames, local paths, project identifiers and executor
metadata, which are preserved as provenance. A targeted scan is not a
guarantee that all possible sensitive information has been detected. Do not
add `.venv`, local caches, credential directories or authentication files to
future releases.

## Historical cloud and compatibility modules

The frozen scientific tree includes optional Google Cloud storage and
worker code. Cloud dependencies and credentials are not required by the
supported saved-data commands. They are not implicitly authorized by
installing this package. If cloud deployment is added later, document its
dependencies, account permissions, costs, and provenance separately.

`metadata/compatibility/publication_core_numeric_repair_sitecustomize_v9.py`
is retained as an inert historical provenance file. It is not automatically
installed as a global Python startup hook or activated during reanalysis.
Its presence does not change the project's licensing decision.
