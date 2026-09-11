# Third-party notices and release boundaries

This package is prepared for a private GitHub repository and versioned
GitHub releases. No Zenodo archive or DOI is planned. This notice does not
grant any additional right to reuse the project's source, data, manuscript,
or figures. No MIT, Creative Commons, or other project-wide license has
been assigned. A future license requires the rights holders' decision.

## Project source and data

`cmfg_cce/` is a byte-exact copy of 105 scientific files, identified by
`metadata/source-copy-manifest.json`. `policy_cce_repro/` contains the
reproduction adapters. The copy manifest records provenance and integrity;
it is not evidence of a third party's permission to redistribute material.
The authors must confirm the applicable ownership and institutional terms
before changing repository visibility or granting a reuse license.

The bounded source review found no separately vendored dependency tree in
the scientific/adaptor source directories. This is not a complete legal
provenance audit. Historical algorithm names and citations do not, on their
own, establish whether every implementation was independently written.
Any copied implementation must retain its original notices and permission
terms once identified.

The separate data archive retains original job records and provenance.
Source-code permission does not automatically cover data or figure reuse.
Any third-party industrial inputs, reports, copied illustrations, or
confidential records require their own review. Structural manuscript
diagrams and the manuscript are outside the numerical reproduction package.

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

This scan does not clear the separate data archive, local build/test logs,
or every validation record for public release. These can contain hostnames,
absolute paths, project names, and original executor metadata. Review the
exact files selected for upload; do not include `.venv`, local caches,
credential directories, authentication files, or machine-specific logs by
copying a whole working directory. Keep the repository private while that
review is incomplete. Preserve any original scientific records; make a
separately documented shareable copy if redaction is needed.

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
