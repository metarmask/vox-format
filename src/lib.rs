//! Allows parsing and writing MagicaVoxel files, including manipulating
//! them using a [semantic](`semantic`) representation.
//! # Reading a model
//! ```
//! let file = vox::semantic::parse_file("tests/pyramid.vox").unwrap();
//! for node in file.root.children().unwrap() {
//!     println!("{:?}", node);
//! }
//! ```
#![feature(array_map, or_patterns)]

pub mod syntax;
pub mod semantic;
