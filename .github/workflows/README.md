# GitHub Actions workflows

This directory contains the workflows for use with [GitHub Actions](https://docs.github.com/actions).  They specify what standards should be expected for development of this software, including pull requests.

## Structure

Most workflows here are _reusable_: they only have a `workflow_call` trigger and are driven by a small number of callers that own the triggers and the `concurrency` settings.

- **`config.yml`** centralizes shared values (supported Python versions, runner images, the current release branch).  Every caller reads them as `needs.config.outputs.<name>`, so changing e.g. the supported Python versions is a single edit there.
- **`branch-protection.yml`** runs on pull requests (and the merge queue, once enabled) and calls every required check.  Its `Finalize` job depends on all of them, so the repository ruleset only needs to require that one check.  Adding, removing or renaming a CI job is therefore a pure code change.
- **`on-nightly.yml`** runs the same test suites once per day.  This is the main post-merge safety net and is what catches breakage from new upstream releases or commits when no pull request is open.

Most jobs are guarded by `github.repository_owner == 'Qiskit'`, so they skip on forks.

## Required checks (called by `branch-protection.yml`)

### Lint check (`lint.yml`)

Checks that the code is formatted properly and follows the style guide by running the [lint environment](/tests/#lint-environment) (`tox -e lint`).

### Documentation (`docs.yml`)

Ensures that the [Sphinx](https://www.sphinx-doc.org/) documentation builds successfully (`tox -e docs`), and uploads the result as the `qiskit-addon-slc-htmldocs` artifact so that it can be downloaded and browsed.  It only builds; publishing is done by `docs_deploy.yml`.

### Latest version tests (`test_latest_versions.yml`)

Runs [the current repository's tests](/tests/#test-py-environments), doctests and notebook tests against the latest version of each dependency, under each supported Python version on Linux and under the newest one on macOS.  This is the primary testing workflow.

### Development version tests (`test_development_versions.yml`)

Modifies `pyproject.toml` to use the _development_ versions of certain Qiskit packages, using [extremal-python-dependencies](https://github.com/IBM/extremal-python-dependencies).  For all other packages, the latest version is installed.  This runs on the oldest and the newest supported Python version.  Its purpose is to identify as soon as possible (i.e., before a Qiskit release) when changes upstream will break the current repository.

### Minimum version tests (`test_minimum_versions.yml`)

Installs the minimum supported tox version (the `minversion` specified in [`tox.ini`](/tox.ini)) and then the _minimum_ compatible version of each package listed in `pyproject.toml`, using [extremal-python-dependencies](https://github.com/IBM/extremal-python-dependencies).  The purpose of this workflow is to make sure the minimum version specifiers are accurate, i.e., that the tests actually pass with these versions.  It uses the oldest supported Python version, as the minimum supported versions of each package may not be compatible with the most recent Python release.

Under the hood, this uses a regular expression to change each `>=` and `~=` specifier in the dependencies to instead be `==`, as pip [does not support](https://github.com/pypa/pip/issues/8085) resolving the minimum versions of packages directly.  Unfortunately, this means that the workflow will only install the minimum version of a package if it is _explicitly_ listed with a minimum version.  For instance, if the only listed dependency is `qiskit>=1.0`, this workflow will install `qiskit==1.0` along with the latest version of each transitive dependency, such as `rustworkx`.

## Other workflows

### Code coverage (`coverage.yml`)

Runs the [coverage environment](/tests/#coverage-environment) (`tox -e coverage`) and uploads the report to [Coveralls](https://coveralls.io/).  It runs on pull requests and on every push to `main` and `stable/**`.  It is deliberately not a required check: coverage is a metric rather than a correctness property, and a Coveralls outage should not block merging.

### Documentation deployment (`docs_deploy.yml`)

Runs on pushes to `main`, `stable/**` and release tags.  It builds the documentation via `docs.yml` and, for the current release branch (`stable-branch` in `config.yml`), publishes it to [GitHub Pages](https://pages.github.com/).

### Citation preview (`citation.yml`)

Only triggered when the `CITATION.bib` file is changed.  It ensures that the file contains only ASCII characters ([escaped codes](https://en.wikibooks.org/wiki/LaTeX/Special_Characters#Escaped_codes) are preferred, as then the `bib` file will work even when `inputenc` is not used).  It also compiles a sample LaTeX document which includes the citation in its bibliography and uploads the resulting PDF as an artifact so it can be previewed.

### Backport metadata (`backport.yml`)

Backports are opened by [Mergify](https://mergify.com/) for pull requests labeled `stable backport potential` (see [`.mergify.yml`](/.mergify.yml)).  This workflow copies the labels and milestone of the original pull request onto the backport.

### Release (`release.yml`)

Triggered by a maintainer pushing a version-shaped tag (e.g. `0.2.0` or `0.2.0rc1`).  It builds the sdist and wheel once, attaches them to a new GitHub release (marked as a pre-release when the tag has a pre-release suffix) and publishes them to [PyPI](https://pypi.org/).
