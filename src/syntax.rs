//! Syntaxical representation of MagicaVoxel file - a series of chunks.
//!
//! In newer versions of MagicaVoxel, only the main chunk has children
use std::{convert::{TryFrom, TryInto}, fmt::{self, Formatter}};

use nom::{IResult, call, eof, combinator::{success, rest}, complete, count, do_parse, error::{VerboseError}, exact, length_count, length_value, many0, map, map_res, named, number::complete::{le_i32, le_u32, le_u8}, tag, take, value, verify};
use indexmap::IndexMap;

const MAGIC_PREFIX: &'static str = "VOX ";

/// A voxel represented by a position and index into the palette array.
#[derive(Clone, Copy, PartialEq)]
pub struct Voxel {
    pub pos: [u8; 3],
    pub index: u8
}

impl Voxel {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        x: le_u8 >>
        y: le_u8 >>
        z: le_u8 >>
        index: le_u8 >>
        ( Self { pos: [x, y, z], index } )
    )}

    fn bytes(self) -> [u8; 4] {
        [self.pos[0], self.pos[1], self.pos[2], self.index]
    }
}

impl fmt::Debug for Voxel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:03},{:03},{:03}:{:03}", self.pos[0], self.pos[1], self.pos[2], self.index)
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Color {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8
}

impl Color {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        red: le_u8 >>
        green: le_u8 >>
        blue: le_u8 >>
        alpha: le_u8 >>
        ( Self { red, green, blue, alpha } )
    )}


    fn bytes(self) -> [u8; 4] {
        [self.red, self.green, self.blue, self.alpha]
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02x}{:02x}{:02x}{:02x}", self.red, self.green, self.blue, self.alpha)
    }
}

pub type Dict = IndexMap<String, String>;

named!{parse_str<&[u8], String, VerboseError<&[u8]>>, length_value!(le_u32, map_res!(call!(rest), |i: &[u8]| {
    String::from_utf8(i.to_vec())
}))}

named!{parse_dict<&[u8], Dict, VerboseError<&[u8]>>, do_parse!(
    entries: map!(
            length_count!(le_u32, count!(parse_str, 2)),
            |vec_of_key_value_vecs| {
                let mut map = Dict::new();
                for key_value_vec in vec_of_key_value_vecs {
                    map.insert(key_value_vec[0].to_owned(), key_value_vec[1].to_owned());
                }
                map
            }
        ) >>
    ( entries )
)}

fn string_bytes(string: String) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(&i32::try_from(string.len()).expect("String too large").to_le_bytes());
    bytes.append(&mut string.into_bytes());
    bytes
}

fn dict_bytes(dict: Dict) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(&i32::try_from(dict.len()).expect("Dict too large").to_le_bytes());
    for (k, v) in dict {
        bytes.append(&mut string_bytes(k));
        bytes.append(&mut string_bytes(v));
    }
    bytes
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub id: i32,
    pub attrs: Dict,
    pub _unknown: i32
}

impl Layer {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        id: le_i32 >>
        attrs: parse_dict >>
        unknown: le_i32 >>
        ( Self { id, attrs, _unknown: unknown } )
    )}

    fn bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(&self.id.to_le_bytes());
        bytes.append(&mut dict_bytes(self.attrs));
        bytes.extend(&self._unknown.to_le_bytes());
        bytes
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeTransform {
    pub id: u32,
    pub attrs: Dict,
    pub child: u32,
    pub reserved: i32,
    pub layer: i32,
    pub frames: Vec<Dict>
}

impl NodeTransform {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        id: le_u32 >>
        attrs: parse_dict >>
        child: le_u32 >>
        reserved: le_i32 >>
        layer: le_i32 >>
        frames: length_count!(le_u32, parse_dict) >>
        ( Self { id, attrs, child, reserved, layer, frames } )
    )}

    fn bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(&self.id.to_le_bytes());
        bytes.append(&mut dict_bytes(self.attrs));
        bytes.extend(&self.child.to_le_bytes());
        bytes.extend(&self.reserved.to_le_bytes());
        bytes.extend(&self.layer.to_le_bytes());
        bytes.extend(&u32::try_from(self.frames.len()).expect("Too many node transform frames").to_le_bytes());
        for frame in self.frames {
            bytes.append(&mut dict_bytes(frame));
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeGroup {
    pub id: u32,
    pub attrs: Dict,
    pub children: Vec<u32>
}

impl NodeGroup {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        id: le_u32 >>
        attrs: parse_dict >>
        children: length_count!(le_u32, le_u32) >>
        ( Self { id, attrs, children } )
    )}

    fn bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(&self.id.to_le_bytes());
        bytes.append(&mut dict_bytes(self.attrs));
        bytes.extend(&u32::try_from(self.children.len()).expect("Too many node group children").to_le_bytes());
        for child in self.children {
            bytes.extend(&child.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub id: u32,
    pub attrs: Dict
}

impl Model {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        id: le_u32 >>
        attrs: parse_dict >>
        ( Self { id, attrs } )
    )}

    fn bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(&self.id.to_le_bytes());
        bytes.append(&mut dict_bytes(self.attrs));
        bytes
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeShape {
    pub id: u32,
    pub attrs: Dict,
    pub models: Vec<Model>
}

impl NodeShape {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        id: le_u32 >>
        attrs: parse_dict >>
        models: length_count!(le_u32, Model::parse) >>
        ( Self { id, attrs, models } )
    )}

    fn bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(&self.id.to_le_bytes());
        bytes.append(&mut dict_bytes(self.attrs));
        bytes.extend(&u32::try_from(self.models.len()).expect("Too many shape node models").to_le_bytes());
        for model in self.models {
            bytes.append(&mut model.bytes());
        }
        bytes
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct KindString(pub [u8; 4]);
impl fmt::Debug for KindString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        String::from_utf8_lossy(&self.0).fmt(f)
    }
}

impl KindString {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, map!(take!(4), |bytes| {
        KindString(bytes.try_into().unwrap())
    })}
}

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    pub id: i32,
    pub props: Dict
}

impl Material {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        id: le_i32 >>
        props: parse_dict >>
        ( Self { id, props } )
    )}

    fn bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(&self.id.to_le_bytes());
        bytes.append(&mut dict_bytes(self.props));
        bytes
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub id: i32,
    pub props: Dict
}

impl Camera {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        id: le_i32 >>
        props: parse_dict >>
        ( Self { id, props } )
    )}

    fn bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(&self.id.to_le_bytes());
        bytes.append(&mut dict_bytes(self.props));
        bytes
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkKind {
    Main,
    Size([u32; 3]),
    Voxels(Vec<Voxel>),
    Layer(Layer),
    Colors([Color; 256]),
    NodeTransform(NodeTransform),
    NodeGroup(NodeGroup),
    NodeShape(NodeShape),
    Material(Material),
    #[deprecated]
    OldMaterial(Vec<u8>),
    ColorIndexMapping([u8; 256]),
    RenderingObject(Dict),
    RenderingCamera(Camera),
    PaletteNotes(Vec<String>),
    Unknown(KindString, Vec<u8>)
}

impl ChunkKind {
    fn parse<'a>(kind_string: &KindString, i: &'a [u8]) -> IResult<&'a [u8], ChunkKind, VerboseError<&'a [u8]>> {
        match &kind_string.0 {
            b"MAIN" => value!(i, ChunkKind::Main),
            b"SIZE" => map!(i, count!(le_u32, 3), |what| {
                ChunkKind::Size(what.try_into().unwrap())
            }),
            b"XYZI" => map!(i, 
                exact!(length_count!(le_u32, Voxel::parse)),
                |voxels| ChunkKind::Voxels(voxels)),
            b"LAYR" => map!(i, Layer::parse, |ok| ChunkKind::Layer(ok)),
            b"RGBA" => map!(i, 
                count!(Color::parse, 256),
                |colors| ChunkKind::Colors(colors.try_into().unwrap())),
            b"nTRN" => map!(i, NodeTransform::parse, |ok| ChunkKind::NodeTransform(ok)),
            b"nGRP" => map!(i, NodeGroup::parse, |ok| ChunkKind::NodeGroup(ok)),
            b"nSHP" => map!(i, NodeShape::parse, |ok| ChunkKind::NodeShape(ok)),
            b"MATL" => map!(i, Material::parse, |ok| ChunkKind::Material(ok)),
            #[allow(deprecated)]
            b"MATT" => map!(i, rest, |ok| ChunkKind::OldMaterial(ok.to_owned())),
            b"IMAP" => map!(i, take!(256), |ok| ChunkKind::ColorIndexMapping(ok.try_into().unwrap())),
            b"rOBJ" => map!(i, parse_dict, |ok| ChunkKind::RenderingObject(ok)),
            b"rCAM" => map!(i, Camera::parse, |ok| ChunkKind::RenderingCamera(ok)),
            b"NOTE" => map!(i, 
                exact!(length_count!(le_u32, parse_str)),
                |notes| ChunkKind::PaletteNotes(notes.into_iter().collect::<Vec<_>>())),
            _ => map!(i, rest, |bytes| ChunkKind::Unknown(kind_string.to_owned(), bytes.to_owned()))
        }
    }

    fn bytes(self) -> ([u8; 4], Vec<u8>) {
        let mut bytes = Vec::new();
        let kind_string = match self {
            ChunkKind::Main => {
                b"MAIN"
            },
            ChunkKind::Size(xyz) => {
                for dim in xyz.iter() {
                    bytes.extend(&dim.to_le_bytes());
                }
                b"SIZE"
            },
            ChunkKind::Voxels(voxels) => {
                bytes.extend(&u32::try_from(voxels.len()).expect("Too many voxels").to_le_bytes());
                for voxel in voxels {
                    bytes.extend(&voxel.bytes());
                }
                b"XYZI"
            }
            ChunkKind::Layer(layer) => {
                bytes.append(&mut layer.bytes());
                b"LAYR"
            }
            ChunkKind::Colors(colors) => {
                // Really?
                let colors = colors.map(|hmm| hmm.bytes());
                for color in &colors {
                    bytes.extend(color);
                }
                b"RGBA"
            }
            ChunkKind::NodeTransform(node_transform) => {
                bytes.append(&mut node_transform.bytes());
                b"nTRN"
            }
            ChunkKind::NodeGroup(node_group) => {
                bytes.append(&mut node_group.bytes());
                b"nGRP"
            }
            ChunkKind::Material(material) => {
                bytes.append(&mut material.bytes());
                b"MATL"
            }
            ChunkKind::ColorIndexMapping(color_indices) => {
                bytes.extend(&color_indices);
                b"IMAP"
            }
            ChunkKind::NodeShape(shape) => {
                bytes.append(&mut shape.bytes());
                b"nSHP"
            }
            #[allow(deprecated)]
            ChunkKind::OldMaterial(material_bytes) => {
                bytes.extend(material_bytes);
                b"MATT"
            }
            ChunkKind::RenderingObject(object) => {
                bytes.append(&mut dict_bytes(object));
                b"rOBJ"
            }
            ChunkKind::RenderingCamera(camera) => {
                bytes.append(&mut camera.bytes());
                b"rCAM"
            }
            ChunkKind::PaletteNotes(notes) => {
                bytes.extend(&u32::try_from(notes.len()).expect("Too many palette notes").to_le_bytes());
                for note in notes {
                    bytes.append(&mut string_bytes(note));
                }
                b"NOTE"
            }
            ChunkKind::Unknown(ref kind_string, content_bytes) => {
                bytes.extend(content_bytes);
                &kind_string.0
            }
        };
        (*kind_string, bytes)
    }
}

#[derive(Clone, PartialEq)]
pub struct Chunk {
    pub kind: ChunkKind,
    pub children: Vec<Chunk>
}

impl Chunk {
    named!{parse<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        kind_string: call!(KindString::parse) >>
        content_size: le_u32 >>
        size_all_children: le_u32 >>
        kind: length_value!(
            success(content_size),
            exact!(|i| ChunkKind::parse(&kind_string, i))
        ) >>
        children: length_value!(
            success(size_all_children),
            exact!(many0!(complete!(Chunk::parse)))) >>
        (Chunk {
            kind,
            children
        })
    )}

    fn bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut children_bytes = Vec::new();
        for child in self.children {
            children_bytes.append(&mut child.bytes());
        }
        let (kind_string, mut content) = self.kind.bytes();
        bytes.extend(&kind_string);
        bytes.extend(&u32::try_from(content.len()).expect("Content too large").to_le_bytes());
        bytes.extend(&u32::try_from(children_bytes.len()).expect("Children too large").to_le_bytes());
        bytes.append(&mut content);
        bytes.append(&mut children_bytes);
        bytes
    }
}

impl fmt::Debug for Chunk {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind)?;
        // self.kind.fmt(f)?;
        if !self.children.is_empty() {
            f.write_str(":\n")?;
            f.debug_list().entries(self.children.iter()).finish()?;
        }
        Ok(())
    }
}

/// A parsed contents of a MagicaVoxel .vox file
#[derive(Debug, Clone, PartialEq)]
pub struct VoxFile {
    pub version: u32,
    pub main_chunk: Chunk
}

/*pub struct SimpleModel {
    size: [u32; 3],
    voxels: Vec<[u8; 4]>,
    
}*/

impl VoxFile {
    named!{pub parse_flat<&[u8], Self, VerboseError<&[u8]>>, do_parse!(
        magic: tag!(MAGIC_PREFIX) >>
        version: le_u32 >>
        main_chunk: verify!(Chunk::parse, |Chunk { kind, .. }| kind == &ChunkKind::Main) >>
        eof: eof!() >>
        ( VoxFile { version, main_chunk } )
    )}

    pub fn bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(MAGIC_PREFIX.as_bytes());
        bytes.extend(&self.version.to_le_bytes());
        bytes.append(&mut self.main_chunk.bytes());
        bytes
    }

    /*pub fn add_models(&mut self, size: [u8; 3], ) {
        for chunk in self.main_chunk.children.iter_mut() {
            if let Some(Chunk { kind: ChunkKind::NodeGroup(NodeGroup { id: 1, children: , .. }), .. }) = chunk {

            }
        }
    }*/
}