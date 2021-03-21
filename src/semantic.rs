//! Semantic representation of MagicaVoxel file - nodes of groups and models as well as materials and layers.
//!
//! In newer versions of MagicaVoxel, only the main chunk has children

use std::{collections::HashMap, convert::{TryFrom, TryInto}, io::Write, iter::{self, Peekable}, ops::Deref, path::Path, result::Result as StdResult, str::FromStr, sync::Arc};
use crate::syntax::{self, Chunk, ChunkKind, NodeTransform};
use anyhow::{Error, Result, bail};
use indexmap::IndexMap;
use thiserror::Error;

const MAX_DIMENSION: u32 = 256;
const PALETTE_LEN: usize = 256;

#[derive(Error, Debug)]
pub enum SemanticError {
    #[error("Model size {size:?} is too small to fit all voxels")]
    ModelSizeTooSmall { size: [u32; 3] },
    #[error("Model size is larger than maximum voxel dimension size, {MAX_DIMENSION}: {size:?}")]
    ModelSizeInvalid { size: [u32; 3] },
    #[error("Expected model voxels, found {found:?}")]
    NoModelVoxels { found: Option<Chunk> },
    #[error("Shape has {instead} models, expected 1")]
    NotSingleShapeModel { instead: usize },
    #[error("Invalid child {child} for node {parent}")]
    InvalidChildReference { child: u32, parent: u32 },
    #[error("Invalid reference to size+XYZI pair {id} in chunk {chunk:?}")]
    InvalidVoxelsReference { id: u32, chunk: ChunkKind },
    #[error("No root node for scene graph")]
    NoRoot,
    #[error("Expected node {node:?} to be transform")]
    ExpectedTransform { node: BuildingNode },
    #[error("Node {node:?} unexpectedly a transform")]
    UnexpectedTransform { node: NodeTransform },
    #[error("Invalid value {value} for attribute {attribute} for chunk {chunk:?}; {expected}")]
    InvalidAttributeValue { attribute: &'static str, value: String, chunk: ChunkKind, expected: &'static str },
    #[error("Chunk {chunk:?} cannot have children")]
    ChunkChildren { chunk: Chunk },
    #[error("Transform chunk had no frames: {transform:?}")]
    NoTransformFrame { transform: NodeTransform },
    #[error("Transform chunk had no transform in frame: {transform:?}")]
    NoTransformFrameTransform { transform: NodeTransform },
    #[error("Transform chunk had transform which did not have exactly three dimensions")]
    TransformWrongDimensionNumber,
    #[error("Set {n} palette materials, maximum is {PALETTE_LEN}")]
    PaletteTooLarge { n: usize }
}

pub use syntax::Voxel;

#[derive(Debug, Clone)]
pub struct Model {
    size: [u32; 3],
    pub voxels: Vec<Voxel>,
    unused_attrs: syntax::Dict,
    shape_attrs: syntax::Dict
}

impl Model {
    pub fn new(size: [u32; 3], voxels: Vec<Voxel>) -> Self {
        Model {
            size,
            voxels,
            unused_attrs: syntax::Dict::new(),
            shape_attrs: syntax::Dict::new(),
        }
    }

    pub fn validate(&self) -> bool {
        if self.size.iter().any(|&dim| dim > MAX_DIMENSION) { return false }
        self.voxels.iter().all(|voxel| {
            voxel.pos.iter().zip(self.size.iter()).all(|(&voxel, &size)| (voxel as u32) < size)
            && voxel.index != 0
        })
    }
}

impl From<Model> for syntax::Model {
    fn from(model: Model) -> Self {
        syntax::Model {
            attrs: model.shape_attrs,
            id: 5
        }
    }
}

impl Model {
    pub fn set_size(&mut self, size: [u32; 3]) -> Result<()> {
        if self.voxels.iter().all(|voxel| {
            voxel.pos.iter().zip(size.iter()).all(|(&voxel, &size)| (voxel as u32) < size)
        }) {
            self.size = size;
            Ok(())
        } else {
            Err(SemanticError::ModelSizeTooSmall { size }.into())
        }
    }

    pub fn size(&self) -> [u32; 3] {
        self.size
    }
}

#[derive(Debug)]
pub enum NodeKind {
    Model(Model),
    Group(Vec<Node>)
}

impl From<Model> for NodeKind {
    fn from(model: Model) -> Self {
        NodeKind::Model(model)
    }
}

impl From<Vec<Node>> for NodeKind {
    fn from(nodes: Vec<Node>) -> Self {
        NodeKind::Group(nodes)
    }
}

impl From<ChunkKind> for Chunk {
    fn from(kind: ChunkKind) -> Self {
        Chunk { children: Vec::new(), kind }
    }
}

impl Into<Arc<NodeKind>> for Model {
    fn into(self) -> Arc<NodeKind> {
        Arc::new(NodeKind::from(self))
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub name: Option<String>,
    pub hidden: bool,
    pub transform: [i32; 3],
    pub kind: Arc<NodeKind>
}

impl Node {
    pub fn new_group(transform: [i32; 3], children: Vec<Node>) -> Self {
        Node {
            name: None,
            hidden: false,
            transform,
            kind: Arc::new(NodeKind::Group(children))
        }
    }

    pub fn new<T: Into<Arc<NodeKind>>>(transform: [i32; 3], kind: T) -> Self {
        Node {
            name: None,
            hidden: false,
            transform,
            kind: kind.into()
        }
    }

    pub fn is_model(&self) -> bool {
        match self.kind.deref() {
            NodeKind::Model(_) => true,
            _ => false
        }
    }

    pub fn is_group(&self) -> bool {
        match self.kind.deref() {
            NodeKind::Group(_) => true,
            _ => false
        }
    }

    pub fn children(&self) -> Option<&Vec<Node>> {
        if let NodeKind::Group(ref group) = self.kind.deref() {
            Some(group)
        } else { None }
    }

    pub fn children_mut(&mut self) -> Option<&mut Vec<Node>> {
        if self.is_group() {
            Arc::get_mut(&mut self.kind).map(|what| {
                if let &mut NodeKind::Group(ref mut group) = what {
                    group
                } else { panic!("checked with is_group") }
            })
        } else { None }
    }

    /// Adds a node into this group. The node turns into a group with itself if not a group already.
    pub fn add(&mut self, node: Node) {
        if !self.is_group() {
            *self = Node::new_group([0, 0, 0], vec![self.clone()])
        }
        let children = self.children_mut().unwrap();
        children.push(node);
    }
}

#[derive(Default)]
struct NodeChunks {
    models: Vec<ChunkKind>,
    nodes: Vec<BuildingNode>,
}

impl Into<Vec<Chunk>> for NodeChunks {
    fn into(self) -> Vec<Chunk> {
               self.models.into_iter()
        .chain(self.nodes.into_iter().map(Into::into))
        .map(|kind| Chunk { kind, children: Vec::new() }).collect()
    }
}

impl Node {
    fn into_chunks(&self, nodes: &mut Vec<Arc<NodeKind>>, chunks: &mut NodeChunks) {
        let node_transform_index = nodes.len() as u32 * 2;
        nodes.push(self.kind.clone());
        chunks.nodes.push(BuildingNode::Transform(syntax::NodeTransform {
            id: node_transform_index,
            frames: {
                vec![
                    {
                        let mut attrs = IndexMap::new();
                        attrs.insert(
                            "_t".to_string(),
                            self.transform.iter().map(ToString::to_string).collect::<Vec<_>>().join(" ").to_string());
                        attrs
                    }
                ]
            },
            attrs: self.name.iter()
                .map(|name| ("_name".to_string(), name.to_string())).collect::<IndexMap<_, _>>(),
            child: node_transform_index + 1,
            reserved: -1,
            layer: 0
        }));
        match &*self.kind {
            NodeKind::Group(group) => {
                let index = chunks.nodes.len();
                chunks.nodes.push(BuildingNode::Group(syntax::NodeGroup {
                    attrs: IndexMap::new(),
                    id: node_transform_index + 1,
                    children: Vec::new() 
                }));
                for node in group.iter() {
                    node.into_chunks(nodes, chunks);
                }
                let children = group.iter().map(|child_node| {
                    nodes.iter().enumerate().find_map(|(i, existing_node)| {
                        if Arc::ptr_eq(&existing_node, &child_node.kind) {
                            Some(i * 2)
                        } else {
                            None
                        }
                    }).unwrap() as u32
                }).collect::<Vec<_>>();
                if let BuildingNode::Group(ref mut group) = chunks.nodes[index] {
                    group.children = children;
                } else { unreachable!() }
            }
            NodeKind::Model(model) => {
                chunks.nodes.push(BuildingNode::Shape(syntax::NodeShape {
                    attrs: IndexMap::new(),
                    id: node_transform_index + 1,
                    models: vec![syntax::Model {
                        attrs: IndexMap::new(),
                        id: chunks.models.len() as u32 / 2
                    }]
                }));
                chunks.models.push(ChunkKind::Size([model.size[0] as u32, model.size[1] as u32, model.size[2] as u32]));
                chunks.models.push(ChunkKind::Voxels(model.voxels.clone()));
            }
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    pub name: Option<String>,
    pub hidden: bool
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaterialKind {
    Diffuse,
    Metal,
    Glass,
    Emit,
    Plastic
}

impl Default for MaterialKind {
    fn default() -> Self {
        MaterialKind::Diffuse
    }
}

impl From<&str> for MaterialKind {
    fn from(s: &str) -> Self {
        match s {
            "_diffuse" => MaterialKind::Diffuse,
            "_metal" => MaterialKind::Metal,
            "_glass" => MaterialKind::Glass,
            "_emit" => MaterialKind::Emit,
            "_plastic" => MaterialKind::Plastic,
            _ => panic!("oh no")
        }
    }
}

impl Into<&str> for MaterialKind {
    fn into(self) -> &'static str {
        match self {
            MaterialKind::Diffuse => "_diffuse",
            MaterialKind::Metal => "_metal",
            MaterialKind::Glass => "_glass",
            MaterialKind::Emit => "_emit",
            MaterialKind::Plastic => "_plastic"
        }
    }
}

impl Into<String> for MaterialKind {
    fn into(self) -> String {
        let str: &'static str = self.into();
        str.to_string()
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Material {
    pub rgba: [u8; 4],
    pub kind: MaterialKind,
    pub weight: Option<f32>,
    /// Roughness
    pub rough: Option<f32>,
    /// Specularity
    pub spec: Option<f32>,
    pub ior: Option<f32>,
    /// Attenuation
    pub att: Option<f32>,
    pub flux: Option<f32>,
    pub plastic: Option<f32>,
    pub g: Option<f32>,
    pub g0: Option<f32>,
    pub g1: Option<f32>,
    pub gw: Option<f32>,
    pub metal: Option<f32>,
    pub spec_p: Option<f32>,
    pub ldr: Option<f32>,
    pub alpha: Option<f32>
}

impl Material {
    pub fn new_color(rgba: [u8; 4]) -> Self {
        Material { rgba, ..Default::default() }
    }
}

impl TryFrom<(syntax::Material, syntax::Color)> for Material {
    type Error = Error;

    fn try_from((syntax, color): (syntax::Material, syntax::Color)) -> Result<Self, Self::Error> {
        let props = &syntax.props;
        let get = |key: &str| -> Result<Option<f32>> {
            Ok(match props.get(key) {
                Some(key) => Some(key.parse::<f32>()?),
                None => None
            })
        };
        Ok(Self {
            rgba: [color.red, color.green, color.blue, color.alpha],
            kind: MaterialKind::from(props.get("_type")
                .ok_or(SemanticError::InvalidAttributeValue { attribute: "_type", value: "(((()()()())))".to_owned(), expected: "s", chunk: syntax::ChunkKind::Material(syntax.to_owned()) })?.as_ref()),
            weight: get("_weight")?,
            rough: get("_rough")?,
            spec: get("_spec")?,
            att: get("_att")?,
            flux: get("_flux")?,
            g: get("_g")?,
            ior: get("_ior")?,
            plastic: get("_ior")?,
            metal: get("_metal")?,
            g0: get("_g0")?,
            g1: get("_g1")?,
            gw: get("_gw")?,
            spec_p: get("_spec_p")?,
            ldr: get("_ldr")?,
            alpha: get("_alpha")?,
        })
    }
}

impl Material {
    fn set_props(&self, props: &mut syntax::Dict) {
        props.insert("_type".to_string(), self.kind.into());
        let mut set = |key: &str, value: Option<f32>| {
            if let Some(value) = value {
                props.insert(key.to_string(), value.to_string());
            }
        };
        set("_weight", self.weight);
        set("_rough", self.rough);
        set("_spec", self.spec);
        set("_att", self.att);
        set("_flux", self.flux);
        set("_g", self.g);
        set("_ior", self.ior);
        set("_plastic", self.ior);
        set("_metal", self.metal);
        set("_g1", self.g1);
        set("_gw", self.gw);
        set("_spec_p", self.spec_p);
        set("_ldr", self.ldr);
        set("_alpha", self.alpha);
    }
}

#[derive(Debug, Clone)]
pub struct VoxFile {
    pub root: Node,
    palette: Vec<Material>,
    pub layers: Vec<Arc<Layer>>
}

impl VoxFile {
    pub fn new() -> Self {
        VoxFile {
            root: Node {
                kind: Arc::new(NodeKind::Group(Vec::new())),
                name: None,
                hidden: false,
                transform: [0, 0, 0],
            },
            palette: Vec::new(),
            layers: Vec::new()
        }
    }
    
    pub fn set_palette(&mut self, palette: &[Material]) -> Result<()> {
        if palette.len() > PALETTE_LEN {
            return Err(SemanticError::PaletteTooLarge { n: palette.len() }.into())
        }
        self.palette = palette.to_vec();
        Ok(())
    }

    pub fn palette(&self) -> &[Material] {
        &self.palette
    }

    fn collect_models<I: Iterator<Item = Chunk>>(chunks: &mut Peekable<I>) -> Result<Vec<([u32; 3], Vec<syntax::Voxel>)>> {
        let mut vec = Vec::new();
        loop {
            match chunks.peek() {
                Some(Chunk { kind: ChunkKind::Size(_), .. }) => {},
                _ => return Ok(vec)
            };
            let size = if let Some(Chunk { kind: ChunkKind::Size(size), .. }) = chunks.next() { size } else { unreachable!() };
            if size.iter().any(|&dim| dim > MAX_DIMENSION) { return Err(SemanticError::ModelSizeInvalid { size }.into()) }
            let next = chunks.next();
            if let Some(Chunk { kind: ChunkKind::Voxels(voxels), .. }) = next {
                vec.push((size, voxels))
            } else {
                return Err(SemanticError::NoModelVoxels { found: next }.into())
            }
        }
    }

    fn palette_to_chunks(&self) -> Vec<Chunk> {
        let mut colors = Vec::new();
        let mut materials = Vec::new();
        for (i, material) in self.palette.iter().enumerate() {
            colors.push({
                let [red, green, blue, alpha] = material.rgba;
                syntax::Color { red, green, blue, alpha }
            });
            let mut props = syntax::Dict::new();
            material.set_props(&mut props);
            materials.push(syntax::Material {
                id: i as i32 + 1,
                props
            });
        }
        for _ in 0..PALETTE_LEN-materials.len() {
            colors.push(syntax::Color::default())
        }
        iter::once(ChunkKind::Colors(colors.try_into().unwrap()))
        .chain(materials.into_iter().map(Into::into))
        .map(Into::into)
        .collect()
    }

    pub fn write<W: Write>(self, writer: &mut W) -> Result<()> {
        let syntaxical = syntax::VoxFile::try_from(self)?;
        writer.write_all(&mut syntaxical.bytes())?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum BuildingNode {
    Group(syntax::NodeGroup),
    Shape(syntax::NodeShape),
    Transform(syntax::NodeTransform)
}

impl TryFrom<Chunk> for BuildingNode {
    type Error = anyhow::Error;

    fn try_from(chunk: Chunk) -> Result<Self, Self::Error> {
        debug_assert!(BuildingNode::is_node_chunk(&chunk));
        if chunk.children.len() != 0 { return Err(SemanticError::ChunkChildren { chunk } .into()) }
        Ok(match chunk.kind {
            ChunkKind::NodeGroup(node) => BuildingNode::Group(node),
            ChunkKind::NodeShape(node) => BuildingNode::Shape(node),
            ChunkKind::NodeTransform(node) => BuildingNode::Transform(node),
            _ => bail!("Chunk {:?} is not a kind of node", chunk.kind)
        })
    }
}

impl From<BuildingNode> for ChunkKind {
    fn from(building: BuildingNode) -> Self {
        match building {
            BuildingNode::Group(x) => ChunkKind::NodeGroup(x),
            BuildingNode::Shape(x) => ChunkKind::NodeShape(x),
            BuildingNode::Transform(x) => ChunkKind::NodeTransform(x)
        }
    }
}

impl From<syntax::Material> for ChunkKind {
    fn from(material: syntax::Material) -> Self {
        ChunkKind::Material(material)
    }
}

impl BuildingNode {
    /// Is it a node kind of chunk?
    fn is_node_chunk(chunk: &Chunk) -> bool {
        if let ChunkKind::NodeGroup(_) | ChunkKind::NodeShape(_) | ChunkKind::NodeTransform(_) = chunk.kind { true } else { false }
    }

    fn id(&self) -> u32 {
        match self {
            BuildingNode::Group(group) => group.id,
            BuildingNode::Shape(shape) => shape.id,
            BuildingNode::Transform(transform) => transform.id
        }
    }

    fn into_transform(self) -> Result<syntax::NodeTransform> {
        match self {
            BuildingNode::Transform(transform) => Ok(transform),
            _ => Err(SemanticError::ExpectedTransform { node: self } .into())
        }
    }
}

fn build_graph(mut transform: syntax::NodeTransform, build_nodes: &mut HashMap<u32, BuildingNode>, built: &mut HashMap<u32, Arc<NodeKind>>, models: &Vec<([u32; 3], Vec<syntax::Voxel>)>) -> Result<Node> {
    let kind = if let Some(existing) = built.get(&transform.child) {
        existing.clone()
    } else {
        let child = build_nodes.remove(&transform.child)
            .ok_or(SemanticError::InvalidChildReference { child: transform.child, parent: transform.id })?;
        match child {
            BuildingNode::Group(group) => {
                let mut children = Vec::new();
                for group_child in group.children {
                    let build_node = build_nodes.get(&group_child).ok_or(
                        SemanticError::InvalidChildReference { child: group_child, parent: transform.id }
                    )?;
                    children.push(build_graph(build_node.to_owned().into_transform()?, build_nodes, built, models)?);
                }
                Arc::new(NodeKind::Group(children))
            }
            BuildingNode::Shape(mut shape) => {
                if shape.models.len() != 1 {
                    return Err(SemanticError::NotSingleShapeModel { instead: shape.models.len() }.into())
                }
                let syntax_model = shape.models.remove(0);
                let (size, voxels) = models.get(syntax_model.id as usize)
                    .ok_or(SemanticError::InvalidVoxelsReference { id: syntax_model.id, chunk: ChunkKind::NodeTransform(transform.to_owned()) })?
                    .to_owned();
                Arc::new(NodeKind::Model(Model {
                    unused_attrs: syntax_model.attrs,
                    shape_attrs: shape.attrs,
                    size,
                    voxels,
                }))
            }
            BuildingNode::Transform(child_transform) => {
                return Err(SemanticError::UnexpectedTransform { node: child_transform }.into());
            }
        }
    };
    built.insert(transform.child, kind.clone());
    // let default_transform = [("_t".to_string(), )].iter().cloned().collect::<IndexMap<_, _>>();
    Ok(
        Node {
            hidden: match transform.attrs.remove("_hidden") {
                Some(v) => match v.as_str() {
                    "1" => true,
                    "0" => false,
                    _ => return Err(SemanticError::InvalidAttributeValue { attribute: "_hidden", value: v.to_owned(), expected: "0 or 1", chunk: ChunkKind::NodeTransform(transform.to_owned()) }.into())
                },
                None => false
            },
            name: transform.attrs.remove("_name"),
            kind,
            transform: transform.frames
                .get(0)
                    .ok_or_else(|| SemanticError::NoTransformFrame { transform: transform.clone() })?
                .get("_t")
                    .map(|s| s.as_ref())
                    .or(if transform.id == 0 { Some("0 0 0") } else { None })
                    .ok_or_else(|| SemanticError::NoTransformFrameTransform { transform: transform.clone() })?
                .split(" ").map(FromStr::from_str).collect::<Result<Vec<i32>, <i32 as FromStr>::Err>>()?
                    .try_into().map_err(|_| SemanticError::TransformWrongDimensionNumber)?
        }
    )
}

impl TryFrom<syntax::VoxFile> for VoxFile {
    type Error = anyhow::Error;

    fn try_from(syntax: syntax::VoxFile) -> Result<Self> {
        // PACK isn't recognized, but should it be?
        let mut chunk_iter = syntax.main_chunk.children.into_iter().peekable();
        let models = VoxFile::collect_models(&mut chunk_iter)?;
        let mut build_nodes = HashMap::new();
        while let Some(chunk) = chunk_iter.peek() {
            if BuildingNode::is_node_chunk(chunk) {
                let node = BuildingNode::try_from(chunk_iter.next().unwrap())?;
                build_nodes.insert(node.id(), node);
            } else {
                break
            }
        }
        let root = build_graph(build_nodes.remove(&0).ok_or(SemanticError::NoRoot)?.into_transform()?, &mut build_nodes, &mut HashMap::new(), &models)?;
        let mut syntax_layers = HashMap::new();
        loop {
            if let Some(Chunk { kind: ChunkKind::Layer(_), ..}) = chunk_iter.peek() {} else { break }
            if let Some(Chunk { kind: ChunkKind::Layer(layer), ..}) = chunk_iter.next() {
                syntax_layers.insert(layer.id, layer);
            } else {
                unreachable!()
            }
        }
        let mut syntax_colors = None;
        let mut syntax_materials = HashMap::new();
        for chunk in chunk_iter {
            match chunk.kind {
                ChunkKind::Colors(colors) => {
                    syntax_colors = Some(colors);
                },
                ChunkKind::Material(material) => {
                    syntax_materials.insert(material.id as usize, material);
                },
                _ => {}
            }
        }
        let mut palette = Vec::new();
        if let Some(colors) = syntax_colors {
            for (i, color) in colors.iter().enumerate() {
                if let Some(syntax_material) = syntax_materials.remove(&i) {
                    palette.push((syntax_material, *color).try_into()?);
                } else {
                    palette.push(Default::default());
                }
            }
        }
        Ok(VoxFile { root, layers: Vec::new(), palette })
    }
}

impl TryFrom<VoxFile> for syntax::VoxFile {
    type Error = anyhow::Error;

    fn try_from(value: VoxFile) -> StdResult<Self, Self::Error> {
        let mut node_chunks = NodeChunks::default();
        value.root.into_chunks(&mut Vec::new(), &mut node_chunks);
        let mut chunks: Vec<Chunk> = node_chunks.into();
        chunks.append(&mut value.palette_to_chunks());
        Ok(syntax::VoxFile {
            version: 150,
            main_chunk: Chunk {
                kind: ChunkKind::Main,
                children: chunks
            }
        })
    }
}

pub fn parse_bytes<'a>(bytes: &'a [u8]) -> Result<VoxFile> {
    syntax::parse_bytes(bytes)?.try_into()
}

pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<VoxFile> {
    syntax::parse_file(path)?.try_into()
}
