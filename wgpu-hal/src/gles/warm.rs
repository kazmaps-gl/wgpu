//! First draw of a new render pipeline outside any frame (WebGL).
//!
//! ANGLE on Metal (Safari; Chrome and Edge on macOS) builds the Metal pipeline state of a
//! program at its first draw with a given set of target formats, sample count, blend, write
//! mask, depth write and vertex layout. Safari presents a WebGL canvas with a synchronous
//! call into its GPU process, so the frame that first draws new programs blocks the page
//! for ~15 ms per program: six at once stalled an iPhone 16 Pro Max for 100 ms right as a
//! map appeared. A warm-up draw issued as soon as the program links moves that build out of
//! the presented frame: one draw into a 1x1 target with the pipeline's formats, reading
//! zeroed vertex and uniform buffers.

use arrayvec::ArrayVec;
use glow::HasContext;
use naga::FastHashMap;

use super::{conv, vertex_array::AttributePointer};

const COLORS: usize = crate::MAX_COLOR_ATTACHMENTS;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TargetKey {
    colors: ArrayVec<Option<wgt::TextureFormat>, COLORS>,
    depth_stencil: Option<wgt::TextureFormat>,
    samples: u32,
}

#[derive(Debug)]
struct Target {
    framebuffer: glow::Framebuffer,
    renderbuffers: ArrayVec<glow::Renderbuffer, { COLORS + 1 }>,
    draw_buffers: ArrayVec<u32, COLORS>,
}

#[derive(Debug, Default)]
pub(super) struct Warm {
    targets: FastHashMap<TargetKey, Target>,
    /// Backs every attribute and uniform block of a warm-up draw; sized for the largest
    /// uniform binding, which WebGL checks against the block before it draws.
    zeroes: Option<(glow::Buffer, i32)>,
}

impl Warm {
    /// Best effort: an object that fails to create skips the warm-up and leaves the build
    /// to the first frame, as before. Leaves the main vertex array bound, as code outside
    /// render passes expects.
    pub(super) unsafe fn draw(
        &mut self,
        gl: &glow::Context,
        shared: &super::AdapterShared,
        desc: &crate::RenderPipelineDescriptor<
            super::PipelineLayout,
            super::ShaderModule,
            super::PipelineCache,
        >,
        pipeline: &super::RenderPipeline,
        main_vertex_array: glow::VertexArray,
    ) {
        let Some((zeroes, size)) = (unsafe { self.zeroes(gl, shared) }) else {
            return;
        };
        let key = TargetKey {
            colors: desc
                .color_targets
                .iter()
                .map(|target| target.as_ref().map(|target| target.format))
                .collect(),
            depth_stencil: desc.depth_stencil.as_ref().map(|ds| ds.format),
            samples: desc.multisample.count,
        };
        if !self.targets.contains_key(&key) {
            let Some(target) = (unsafe { create_target(gl, shared, &key) }) else {
                return;
            };
            self.targets.insert(key.clone(), target);
        }
        let target = &self.targets[&key];
        let Ok(vertex_array) = (unsafe { gl.create_vertex_array() }) else {
            return;
        };
        unsafe {
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(target.framebuffer));
            gl.draw_buffers(&target.draw_buffers);
            gl.viewport(0, 0, 1, 1);
            gl.disable(glow::SCISSOR_TEST);
            gl.use_program(Some(pipeline.inner.program));
            set_output_state(gl, pipeline);
            bind_zeroes(gl, desc.layout, zeroes, size);
            gl.bind_vertex_array(Some(vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(zeroes));
            for attribute in pipeline.vertex_attributes.iter() {
                let Some(Some(buffer)) =
                    pipeline.vertex_buffers.get(attribute.buffer_index as usize)
                else {
                    continue;
                };
                gl.enable_vertex_attrib_array(attribute.location);
                AttributePointer {
                    buffer: zeroes,
                    format: attribute.format_desc.clone(),
                    stride: buffer.stride,
                    offset: attribute.offset,
                }
                .specify(gl, attribute.location);
                gl.vertex_attrib_divisor(attribute.location, buffer.step as u32);
            }
            gl.draw_arrays_instanced(
                conv::map_primitive_topology(pipeline.primitive.topology),
                0,
                3,
                1,
            );
            gl.bind_vertex_array(Some(main_vertex_array));
            gl.delete_vertex_array(vertex_array);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            gl.use_program(None);
            gl.disable(glow::BLEND);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::STENCIL_TEST);
            gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            gl.color_mask(true, true, true, true);
            gl.depth_mask(true);
        }
    }

    unsafe fn zeroes(
        &mut self,
        gl: &glow::Context,
        shared: &super::AdapterShared,
    ) -> Option<(glow::Buffer, i32)> {
        if self.zeroes.is_none() {
            let uniforms = shared.limits.max_uniform_buffer_binding_size;
            // Three vertices or one instance past the farthest attribute offset.
            let vertices = 3 * u64::from(shared.limits.max_vertex_buffer_array_stride) + 4096;
            let size = i32::try_from(uniforms.max(vertices)).unwrap_or(i32::MAX);
            let buffer = unsafe { gl.create_buffer() }.ok()?;
            unsafe { gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer)) };
            unsafe { gl.buffer_data_size(glow::ARRAY_BUFFER, size, glow::STATIC_DRAW) };
            unsafe { gl.bind_buffer(glow::ARRAY_BUFFER, None) };
            self.zeroes = Some((buffer, size));
        }
        self.zeroes
    }

    pub(super) unsafe fn delete(&mut self, gl: &glow::Context) {
        for (_, target) in self.targets.drain() {
            unsafe { gl.delete_framebuffer(target.framebuffer) };
            for renderbuffer in target.renderbuffers {
                unsafe { gl.delete_renderbuffer(renderbuffer) };
            }
        }
        if let Some((buffer, _)) = self.zeroes.take() {
            unsafe { gl.delete_buffer(buffer) };
        }
    }
}

unsafe fn create_target(
    gl: &glow::Context,
    shared: &super::AdapterShared,
    key: &TargetKey,
) -> Option<Target> {
    let framebuffer = unsafe { gl.create_framebuffer() }.ok()?;
    let mut target = Target {
        framebuffer,
        renderbuffers: ArrayVec::new(),
        draw_buffers: ArrayVec::new(),
    };
    unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(framebuffer)) };
    for (index, format) in key.colors.iter().enumerate() {
        let attachment = glow::COLOR_ATTACHMENT0 + index as u32;
        let Some(format) = *format else {
            target.draw_buffers.push(glow::NONE);
            continue;
        };
        let renderbuffer = unsafe { create_renderbuffer(gl, shared, format, key.samples) }?;
        unsafe {
            gl.framebuffer_renderbuffer(
                glow::DRAW_FRAMEBUFFER,
                attachment,
                glow::RENDERBUFFER,
                Some(renderbuffer),
            )
        };
        target.renderbuffers.push(renderbuffer);
        target.draw_buffers.push(attachment);
    }
    if let Some(format) = key.depth_stencil {
        let aspects = crate::FormatAspects::from(format);
        let attachment = if aspects.contains(crate::FormatAspects::DEPTH_STENCIL) {
            glow::DEPTH_STENCIL_ATTACHMENT
        } else if aspects.contains(crate::FormatAspects::DEPTH) {
            glow::DEPTH_ATTACHMENT
        } else {
            glow::STENCIL_ATTACHMENT
        };
        let renderbuffer = unsafe { create_renderbuffer(gl, shared, format, key.samples) }?;
        unsafe {
            gl.framebuffer_renderbuffer(
                glow::DRAW_FRAMEBUFFER,
                attachment,
                glow::RENDERBUFFER,
                Some(renderbuffer),
            )
        };
        target.renderbuffers.push(renderbuffer);
    }
    unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None) };
    Some(target)
}

unsafe fn create_renderbuffer(
    gl: &glow::Context,
    shared: &super::AdapterShared,
    format: wgt::TextureFormat,
    samples: u32,
) -> Option<glow::Renderbuffer> {
    let renderbuffer = unsafe { gl.create_renderbuffer() }.ok()?;
    let internal = shared.describe_texture_format(format).internal;
    unsafe {
        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(renderbuffer));
        if samples > 1 {
            gl.renderbuffer_storage_multisample(glow::RENDERBUFFER, samples as i32, internal, 1, 1);
        } else {
            gl.renderbuffer_storage(glow::RENDERBUFFER, internal, 1, 1);
        }
        gl.bind_renderbuffer(glow::RENDERBUFFER, None);
    }
    Some(renderbuffer)
}

/// The state the queue sets for the pipeline that keys Metal's pipeline state: blend and
/// write masks as `set_render_pipeline` records them, depth write, alpha to coverage.
unsafe fn set_output_state(gl: &glow::Context, pipeline: &super::RenderPipeline) {
    let targets = &pipeline.color_targets;
    if targets.iter().skip(1).any(|target| *target != targets[0]) {
        for (index, target) in targets.iter().enumerate() {
            unsafe { super::queue::set_color_target(gl, Some(index as u32), target) };
        }
    } else {
        let target = targets.first().cloned().unwrap_or_default();
        unsafe { super::queue::set_color_target(gl, None, &target) };
    }
    if let Some(ref depth) = pipeline.depth {
        unsafe { gl.enable(glow::DEPTH_TEST) };
        unsafe { gl.depth_func(depth.function) };
        unsafe { gl.depth_mask(depth.mask) };
    } else {
        unsafe { gl.disable(glow::DEPTH_TEST) };
    }
    if pipeline.stencil.is_some() {
        unsafe { gl.enable(glow::STENCIL_TEST) };
    } else {
        unsafe { gl.disable(glow::STENCIL_TEST) };
    }
    if pipeline.alpha_to_coverage_enabled {
        unsafe { gl.enable(glow::SAMPLE_ALPHA_TO_COVERAGE) };
    } else {
        unsafe { gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE) };
    }
}

/// Every uniform block reads the zero buffer; every texture unit of the layout is left
/// empty, which WebGL samples as an incomplete texture of the sampler's own kind. A texture
/// a frame left there could mismatch the sampler type and fail the draw.
unsafe fn bind_zeroes(
    gl: &glow::Context,
    layout: &super::PipelineLayout,
    zeroes: glow::Buffer,
    size: i32,
) {
    for info in layout.group_infos.iter().flatten() {
        for entry in info.entries.iter() {
            let slot = info.binding_to_slot[entry.binding as usize];
            if slot == !0 {
                continue;
            }
            let slot = u32::from(slot);
            match entry.ty {
                wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    ..
                } => unsafe {
                    gl.bind_buffer_range(glow::UNIFORM_BUFFER, slot, Some(zeroes), 0, size)
                },
                wgt::BindingType::Texture { .. } => unsafe {
                    gl.active_texture(glow::TEXTURE0 + slot);
                    for target in [
                        glow::TEXTURE_2D,
                        glow::TEXTURE_2D_ARRAY,
                        glow::TEXTURE_3D,
                        glow::TEXTURE_CUBE_MAP,
                    ] {
                        gl.bind_texture(target, None);
                    }
                    gl.bind_sampler(slot, None);
                },
                _ => {}
            }
        }
    }
}
