//! Vertex array objects for GL without [`VERTEX_BUFFER_LAYOUT`] (WebGL2, GLES 3.0).
//!
//! There a vertex buffer and its offset exist only inside `glVertexAttrib*Pointer`, so
//! a draw that moves a buffer or an offset re-specifies every attribute it reads: a
//! buffer bind, a pointer and often an enable and a divisor per attribute, each one a
//! JS call on WebGL2. A vertex array object captures all of it together with the
//! element buffer, and a draw whose state was seen before binds it with one call.
//!
//! A state becomes an object only when a later submit draws it again. Instance ranges
//! that follow label placement, or per-draw offsets, would otherwise fill the cache with
//! objects no draw binds twice. Until then, and past the capacity, one spare object is
//! re-specified in place, which costs what every draw paid before the cache.
//!
//! The encoder records which attributes and index buffer a draw reads; the queue
//! resolves them to an object right before the draw. Outside render passes the main
//! object is bound again, so code that binds `ELEMENT_ARRAY_BUFFER` or points
//! attribute 0 there (buffer creation, copies, barriers) never alters a cached one.
//!
//! [`VERTEX_BUFFER_LAYOUT`]: super::PrivateCapabilities::VERTEX_BUFFER_LAYOUT

use alloc::boxed::Box;

use arrayvec::ArrayVec;
use glow::HasContext;
use naga::FastHashMap;

use super::{VertexAttribKind, VertexFormatDesc, MAX_VERTEX_ATTRIBUTES};

/// Past this many cached objects a draw with a new state re-specifies the spare one.
const CAPACITY: usize = 2048;
/// A state drawn once is forgotten unless a draw within this many submits repeats it.
const SEEN_WINDOW: u64 = 8;
/// Cached objects no draw has bound for this many submits are deleted.
const MAX_AGE: u64 = 256;
/// Submits between sweeps for objects older than [`MAX_AGE`].
const SWEEP_PERIOD: u64 = 64;

/// What `glVertexAttrib*Pointer` captures for one location.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct AttributePointer {
    pub buffer: glow::Buffer,
    pub format: VertexFormatDesc,
    pub stride: u32,
    pub offset: u32,
}

impl AttributePointer {
    /// Points `location` here; the buffer must be bound to `ARRAY_BUFFER`.
    pub(super) unsafe fn specify(&self, gl: &glow::Context, location: u32) {
        let (size, data_type) = (self.format.element_count, self.format.element_format);
        let (stride, offset) = (self.stride as i32, self.offset as i32);
        match self.format.attrib_kind {
            VertexAttribKind::Float => unsafe {
                // always normalized
                gl.vertex_attrib_pointer_f32(location, size, data_type, true, stride, offset)
            },
            VertexAttribKind::Integer => unsafe {
                gl.vertex_attrib_pointer_i32(location, size, data_type, stride, offset)
            },
        }
    }
}

/// One enabled attribute array a draw reads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct VertexAttribute {
    pub location: u32,
    pub pointer: AttributePointer,
    pub divisor: u32,
}

type Attributes = ArrayVec<VertexAttribute, MAX_VERTEX_ATTRIBUTES>;

/// The state of a vertex array object as far as this module has set it.
#[derive(Debug, Default)]
struct ObjectState {
    enabled: u32,
    /// `None` where unknown: never specified, or its buffer was destroyed.
    pointers: [Option<AttributePointer>; MAX_VERTEX_ATTRIBUTES],
    divisors: [u32; MAX_VERTEX_ATTRIBUTES],
    /// `None` also stands for unknown: an indexed draw binds its buffer either way.
    index_buffer: Option<glow::Buffer>,
}

impl ObjectState {
    /// Makes the bound object hold `attributes`, calling GL only for what differs.
    unsafe fn apply(&mut self, gl: &glow::Context, attributes: &[VertexAttribute]) {
        let mut wanted = 0u32;
        let mut array_buffer = None;
        for attribute in attributes {
            let location = attribute.location as usize;
            let bit = 1 << attribute.location;
            wanted |= bit;
            if self.enabled & bit == 0 {
                unsafe { gl.enable_vertex_attrib_array(attribute.location) };
            }
            if self.pointers[location].as_ref() != Some(&attribute.pointer) {
                let buffer = attribute.pointer.buffer;
                if array_buffer != Some(buffer) {
                    unsafe { gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer)) };
                    array_buffer = Some(buffer);
                }
                unsafe { attribute.pointer.specify(gl, attribute.location) };
                self.pointers[location] = Some(attribute.pointer.clone());
            }
            if self.divisors[location] != attribute.divisor {
                unsafe { gl.vertex_attrib_divisor(attribute.location, attribute.divisor) };
                self.divisors[location] = attribute.divisor;
            }
        }
        let mut stale = self.enabled & !wanted;
        while stale != 0 {
            unsafe { gl.disable_vertex_attrib_array(stale.trailing_zeros()) };
            stale &= stale - 1;
        }
        self.enabled = wanted;
    }

    fn refers_to(&self, raw: glow::Buffer) -> bool {
        self.index_buffer == Some(raw)
            || self
                .pointers
                .iter()
                .flatten()
                .any(|pointer| pointer.buffer == raw)
    }

    fn forget_buffer(&mut self, raw: glow::Buffer) {
        for pointer in &mut self.pointers {
            if pointer
                .as_ref()
                .is_some_and(|pointer| pointer.buffer == raw)
            {
                *pointer = None;
            }
        }
        if self.index_buffer == Some(raw) {
            self.index_buffer = None;
        }
    }
}

#[derive(Debug)]
struct Cached {
    raw: glow::VertexArray,
    index_buffer: Option<glow::Buffer>,
    /// The last submit a draw bound it in.
    used: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Bound {
    /// The start of a submit: code outside wgpu may have bound anything since the last.
    Unknown,
    Main,
    Cached(glow::VertexArray),
    Spare,
}

/// Vertex array objects of one device, shared by its queue, which resolves draws to
/// them, and the device itself, which forgets a buffer's objects when it destroys it.
#[derive(Debug)]
pub(super) struct VertexArrays {
    /// `false` with `VERTEX_BUFFER_LAYOUT`: an offset there is one `glBindVertexBuffer`,
    /// and the main object serves every draw.
    enabled: bool,
    main: glow::VertexArray,
    /// Takes the states without an object: first sightings and those past the capacity.
    spare: glow::VertexArray,
    spare_state: ObjectState,
    cached: FastHashMap<Box<[VertexAttribute]>, Cached>,
    /// States drawn once, with the submit that drew them.
    seen: FastHashMap<Box<[VertexAttribute]>, u64>,
    submit: u64,
    /// What the next draw reads, sorted by location.
    attributes: Attributes,
    index_buffer: Option<glow::Buffer>,
    bound: Bound,
    /// Attributes the bound object holds when it is cached or the spare.
    bound_attributes: Attributes,
    bound_index_buffer: Option<glow::Buffer>,
    /// `attributes` may differ from what the bound object holds.
    dirty: bool,
}

impl VertexArrays {
    /// `main` must be the bound object.
    pub(super) fn new(enabled: bool, main: glow::VertexArray, spare: glow::VertexArray) -> Self {
        Self {
            enabled,
            main,
            spare,
            spare_state: ObjectState::default(),
            cached: FastHashMap::default(),
            seen: FastHashMap::default(),
            submit: 0,
            attributes: Attributes::new(),
            index_buffer: None,
            bound: Bound::Main,
            bound_attributes: Attributes::new(),
            bound_index_buffer: None,
            dirty: true,
        }
    }

    pub(super) fn begin_submit(&mut self) {
        self.submit += 1;
        self.bound = Bound::Unknown;
        self.dirty = true;
    }

    /// Forgets states not drawn again within [`SEEN_WINDOW`] and, every [`SWEEP_PERIOD`],
    /// deletes cached objects past [`MAX_AGE`]; the main object is bound.
    pub(super) unsafe fn end_submit(&mut self, gl: &glow::Context) {
        let submit = self.submit;
        self.seen.retain(|_, first| submit - *first < SEEN_WINDOW);
        if !submit.is_multiple_of(SWEEP_PERIOD) {
            return;
        }
        self.cached.retain(|_, cached| {
            let fresh = submit - cached.used <= MAX_AGE;
            if !fresh {
                unsafe { gl.delete_vertex_array(cached.raw) };
            }
            fresh
        });
    }

    pub(super) fn set_attribute(&mut self, attribute: &VertexAttribute) {
        match self
            .attributes
            .binary_search_by_key(&attribute.location, |current| current.location)
        {
            Ok(index) if self.attributes[index] == *attribute => {}
            Ok(index) => {
                self.attributes[index] = attribute.clone();
                self.dirty = true;
            }
            Err(index) => {
                self.attributes.insert(index, attribute.clone());
                self.dirty = true;
            }
        }
    }

    pub(super) fn unset_attribute(&mut self, location: u32) {
        if let Ok(index) = self
            .attributes
            .binary_search_by_key(&location, |current| current.location)
        {
            self.attributes.remove(index);
            self.dirty = true;
        }
    }

    pub(super) fn set_index_buffer(&mut self, buffer: glow::Buffer) {
        self.index_buffer = Some(buffer);
    }

    /// The object bound outside render passes.
    #[cfg(webgl)]
    pub(super) const fn main(&self) -> glow::VertexArray {
        self.main
    }

    /// Ends a render pass: binds the main object back for the code outside passes.
    pub(super) unsafe fn reset(&mut self, gl: &glow::Context) {
        self.attributes.clear();
        self.index_buffer = None;
        if self.bound != Bound::Main {
            unsafe { gl.bind_vertex_array(Some(self.main)) };
            self.bound = Bound::Main;
        }
        self.dirty = true;
    }

    /// Binds an object that holds the draw's attributes and, when `indexed`, its index
    /// buffer.
    pub(super) unsafe fn prepare_draw(&mut self, gl: &glow::Context, indexed: bool) {
        if !self.enabled {
            return;
        }
        if self.dirty {
            unsafe { self.bind_attributes(gl) };
        }
        if indexed && self.bound_index_buffer != self.index_buffer {
            unsafe { gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, self.index_buffer) };
            self.bound_index_buffer = self.index_buffer;
            match self.bound {
                Bound::Cached(_) => {
                    if let Some(cached) = self.cached.get_mut(self.bound_attributes.as_slice()) {
                        cached.index_buffer = self.index_buffer;
                    }
                }
                Bound::Spare => self.spare_state.index_buffer = self.index_buffer,
                Bound::Unknown | Bound::Main => unreachable!(),
            }
        }
    }

    unsafe fn bind_attributes(&mut self, gl: &glow::Context) {
        self.dirty = false;
        if matches!(self.bound, Bound::Cached(_) | Bound::Spare)
            && self.bound_attributes == self.attributes
        {
            return;
        }
        self.bound_attributes.clone_from(&self.attributes);
        if let Some(cached) = self.cached.get_mut(self.attributes.as_slice()) {
            cached.used = self.submit;
            unsafe { gl.bind_vertex_array(Some(cached.raw)) };
            self.bound = Bound::Cached(cached.raw);
            self.bound_index_buffer = cached.index_buffer;
            return;
        }
        if self.admits() {
            if let Ok(raw) = unsafe { gl.create_vertex_array() } {
                self.seen.remove(self.attributes.as_slice());
                unsafe { gl.bind_vertex_array(Some(raw)) };
                unsafe { ObjectState::default().apply(gl, &self.attributes) };
                let cached = Cached {
                    raw,
                    index_buffer: None,
                    used: self.submit,
                };
                self.cached
                    .insert(self.attributes.as_slice().into(), cached);
                self.bound = Bound::Cached(raw);
                self.bound_index_buffer = None;
                return;
            }
        }
        if self.bound != Bound::Spare {
            unsafe { gl.bind_vertex_array(Some(self.spare)) };
        }
        unsafe { self.spare_state.apply(gl, &self.attributes) };
        self.bound = Bound::Spare;
        self.bound_index_buffer = self.spare_state.index_buffer;
    }

    /// Whether the draw's state earns an object: an earlier submit drew it and the cache has
    /// room. A first sighting is remembered instead.
    fn admits(&mut self) -> bool {
        if self.cached.len() >= CAPACITY {
            return false;
        }
        match self.seen.get(self.attributes.as_slice()) {
            Some(&first) => first < self.submit,
            None => {
                if self.seen.len() < CAPACITY {
                    self.seen
                        .insert(self.attributes.as_slice().into(), self.submit);
                }
                false
            }
        }
    }

    /// Drops every object that refers to a buffer being destroyed: GL may hand its name
    /// out again, and an object still holding it would draw the old storage.
    pub(super) unsafe fn forget_buffer(&mut self, gl: &glow::Context, raw: glow::Buffer) {
        self.cached.retain(|attributes, cached| {
            let refers = cached.index_buffer == Some(raw)
                || attributes
                    .iter()
                    .any(|attribute| attribute.pointer.buffer == raw);
            if refers {
                unsafe { gl.delete_vertex_array(cached.raw) };
            }
            !refers
        });
        if self.spare_state.refers_to(raw) {
            // A buffer deleted while another object is bound stays attached to the spare,
            // storage and all; a fresh spare lets it go.
            if let Ok(fresh) = unsafe { gl.create_vertex_array() } {
                unsafe { gl.delete_vertex_array(self.spare) };
                self.spare = fresh;
                self.spare_state = ObjectState::default();
            } else {
                self.spare_state.forget_buffer(raw);
            }
        }
    }

    pub(super) unsafe fn delete(&mut self, gl: &glow::Context) {
        for (_, cached) in self.cached.drain() {
            unsafe { gl.delete_vertex_array(cached.raw) };
        }
        unsafe { gl.delete_vertex_array(self.spare) };
        unsafe { gl.delete_vertex_array(self.main) };
    }
}
