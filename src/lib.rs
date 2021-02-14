//! Allows parsing and writing MagicaVoxel files, including manipulating
//! them using a [semantic](`semantic`) representation.
//! # Reading a model
//! ```ignore
//! use vox::semantic::VoxFile;
//!
//! fn main() {
//!     let file = VoxFile::read("test.vox");
//!     for model in file.root.iter_models() {
//!         
//!     }
//! }
//! ```
#![feature(array_map, or_patterns)]

pub mod syntax;
pub mod semantic;
