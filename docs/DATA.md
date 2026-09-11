# Versioned offline data

The core data asset is `policy-cce-data-v1.tar.gz`. It belongs in the private GitHub Release assets, separately from the code and ordinary Git history. The archive contains one directory, `offline-inputs-v1/`.

The dataset contains 588 selected formal jobs and 24 calibration jobs. It supports saved-result verification and the paper's offline statistical and graphical reproduction. The selection is described in the repository README. Historical reference summaries can include additional cases; the dataset manifest defines the selected job scope.

## Exact contents

| Item | Count or size |
|---|---:|
| Manifest-listed payload files | 6,161 |
| Payload bytes | 838,732,517 |
| Manifest bytes | 6,998,562 |
| Archive files, including the manifest | 6,162 |
| Uncompressed file bytes, including the manifest | 845,731,079 |
| JSON files, including the manifest | 5,232 |
| Numeric NPZ files | 930 |

The dataset manifest SHA-256 is:

```text
a0b5b46789c1a879a55ad82d57e4ccb07dc2a5c66b73438ab2d27127b4406ff7
```

The release's data descriptor supplies the compressed archive size and SHA-256. Verify both the archive and its manifest-listed files before using them. Keep the downloaded data separate from output directories; the offline commands never overwrite their inputs or existing reports.

The archive includes the original results, completion records, required statistical samples, calibration selection and associated provenance. Each file retains its original bytes. It excludes the source tree, cloud credentials, deployment files and unrelated workspace files. The optional original formal campaign manifest is a separate asset for formal rerun instructions; it is not part of this core archive.

## Privacy and provenance

This asset is for the private repository. It retains original project identifiers, storage-object locations and generations, node/executor metadata, and a few historical local paths. These records document where the results came from. They are not changed or relabeled during packaging. Logical node IDs must not be treated as proof of the physical executor; missing executor provenance remains missing.

A targeted scan covered every JSON file and the keys and headers of every NPZ member. It found no credential-like filenames, nonempty credential fields, supported secret-token signatures, or arbitrary NPZ metadata. The 10,190 inspected arrays have numeric types: `float64`, `int16`, or `int64`. A scan report records counts and affected filenames without reproducing private metadata values.

Private provenance metadata was detected in 3,958 files. This includes 3,356 files with project identifiers, 1,496 with executor/node metadata, 2,464 with storage metadata, and three with local-path metadata; these categories overlap. A targeted scan is not a guarantee that a dataset contains no sensitive information. Public redistribution or a change in repository visibility requires a separate review.

## Packaging checks

Packaging accepts only the pinned manifest and its exact file whitelist. The archive stores regular files only, with no symlinks, hardlinks, devices or explicit directory entries. Paths, ownership fields, permissions and timestamps are fixed. Every archived member is streamed back and checked against its expected size and SHA-256. A second independently generated archive stream must match the first archive byte for byte.

These checks validate the release package. They do not rerun experiments, alter frozen policy distributions, replace missing results, or certify claims beyond the selected data.
