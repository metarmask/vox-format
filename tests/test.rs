use std::convert::TryInto;
use semantic::{Node, Voxel};
use vox::{semantic::{self, Material, Model}, syntax};
use pretty_assertions::assert_eq;
use anyhow::Result;

#[test]
fn semantic_pyramid() -> Result<()> {
    let parsed = syntax::VoxFile::parse_flat(include_bytes!("pyramid.vox"))?;
    let _semantized: semantic::VoxFile = parsed.1.try_into()?;
    Ok(())
}

#[test]
fn palette_from_and_to() -> Result<()> {
    let original_bytes = include_bytes!("table.vox");
    let original = semantic::parse_bytes(original_bytes)?;
    let mut reconstructed_bytes = Vec::new();
    original.clone().write(&mut reconstructed_bytes)?;
    let reconstructed = semantic::parse_bytes(&reconstructed_bytes)?;
    assert_eq!(original.palette()[168].rgba, reconstructed.palette()[168].rgba);
    assert_eq!(original.palette()[169].rgba, reconstructed.palette()[169].rgba);
    Ok(())
}

#[test]
fn semantic_from_scratch() -> Result<()> {
    let mut semantized = semantic::VoxFile::new();
    semantized.root.add(Node::new([0, 0, 0], Model::new([5, 5, 5], vec![
        Voxel { pos: [2, 3, 4], index: 1 }
    ])));
    semantized.root.add(Node::new([0, 0, 0], Model::new([10, 10, 10], vec![
        Voxel { pos: [3, 3, 4], index: 0 },
        Voxel { pos: [3, 3, 5], index: 1 },
        Voxel { pos: [3, 3, 6], index: 255 }
    ])));
    semantized.set_palette(&[Material::default(), Material::new_color([0, 0, 0, 255]), Material {
        rgba: [255, 0, 255, 255],
        .. Default::default()
    }])?;
    Ok(())
}
