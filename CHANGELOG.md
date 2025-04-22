The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Fixed rare infinite loop occuring when optimising very symmetric instances.


## [0.2.0] - 2025-04-13
### Fixed
- `price_of_hierarchy` now actually returns the price-of-hierarchy, instead of sometimes returning price-of-greedy.

### Added
- `Cluster` now implements `Default`.

### Removed
- Cost-structs no longer implement `PartialEq` due to ambiguity


## [0.1.0] - 2025-03-30
### Added
- Weighted and Unweighted variants for Discrete KMedian, Discrete KMeans, Continuous KMeans
- Solvers for clusterings, greedy-hierarchical-clusterings, optimal-hierarchical-clusterings