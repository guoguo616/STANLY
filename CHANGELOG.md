# Changelog
All notable changes will be documented in this file

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2] - 2023-10-30

### Added
- CHANGELOG.md added to keep track of updates
- `annotateDigitalSpots`, function that annotates digital spots based on Allen CCF coordinates
### Fixed
- Updated `processVisiumData`, `processMerfishData`, `runANTsToAllenRegistration`, `runANTsInterSampleRegistration`, and `applyAntsTransformations` to use the ANTsPyX built-in function `apply_transforms_to_points` instead of using a system call to run it from ANTs. This should clear up some dependency issues
- began organizing functions based on purpose, i.e. importation, registration, etc.
- added `displayImage=False` option to functions that generate images to make figure windows optional. Changing to `displayImage=True` will display figures