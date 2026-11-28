# Training client releases

`packages/training-client` produces the `nlp-lab-training` wheel. The root
`nlp-lab` project is a workspace, not a second wheel. Its release version must
match the client version. Update both `pyproject.toml` files and their lockfiles
in one PR. Do not reuse an existing tag or package version.

Require the **Training client / verify** job on `main` changes. It verifies the
locks, runs the tests, and inspects metadata from the actual wheel. It also
installs the wheel in an isolated Python 3.10 environment and checks the client
imports and runtime profile. A root `0.1.7` paired with a client `0.1.6` fails
before any publish job can run.

To publish, dispatch **Training client** on `main` with the committed release
version. Set up the `training-release` environment with `TRAINING_PUBLISH_URL`
(the private Python repository upload URL) and `TRAINING_PUBLISH_TOKEN`. Require
review and restrict the environment to `main`. Restrict release tag creation
and registry writes to the release workflow identity. The publish job consumes
the verified wheel and checksum rather than rebuilding from source.

If the private index upload succeeds but GitHub release creation fails, recover
the release using the original workflow artifact and its checksum. Do not rebuild
or overwrite the published wheel. Use the exact published version when updating
the Jupyter consumer lock.

Wheel verification covers the lightweight client. For DGX validation also retain
the submitted bundle's `git_base`, `git_status`, and `bundle_sha256`, since the
worker can use trainer code from a separate checkout.
