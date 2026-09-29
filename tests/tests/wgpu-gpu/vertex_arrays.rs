//! Draws whose vertex state repeats, moves or outlives its buffers. On GL without
//! `VERTEX_BUFFER_LAYOUT` (WebGL2, GLES 3.0) a state drawn in two submits becomes a cached
//! vertex array object, and a state drawn once re-specifies a spare one; each test fails
//! when either hands a draw another draw's state.

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array, BufferUsages, PollType,
};
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        A_BUFFER_THAT_TAKES_A_DESTROYED_BUFFERS_PLACE_DRAWS_ITS_OWN_DATA,
        AN_INDEX_BUFFER_THAT_TAKES_A_DESTROYED_ONES_PLACE_DRAWS_ITS_OWN_INDICES,
        ONE_VERTEX_STATE_DRAWS_WITH_THE_INDEX_BUFFER_OF_EACH_DRAW,
        MORE_VERTEX_STATES_THAN_THE_CACHE_HOLDS_DRAW_THEIR_OWN_DATA,
        A_CACHED_VERTEX_ARRAY_SURVIVES_ASYNC_PIPELINE_COMPLETION,
    ]);
}

/// The canvas is `SIZE` × `SIZE`; draw `i` colors pixel `i` through a 1×1 viewport.
const SIZE: u32 = 64;
/// A triangle that covers the viewport.
const COVER: [[f32; 2]; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];

const SHADER: &str = "
    struct Out {
        @builtin(position) position: vec4f,
        @location(0) color: vec4f,
    }

    @vertex fn interleaved(@location(0) position: vec2f, @location(1) color: vec4f) -> Out {
        return Out(vec4f(position, 0.0, 1.0), color);
    }

    @vertex fn per_instance(@location(0) position: vec2f, @location(2) color: vec4f) -> Out {
        return Out(vec4f(position, 0.0, 1.0), color);
    }

    @fragment fn fs(input: Out) -> @location(0) vec4f {
        return input.color;
    }
";

/// A vertex of the `interleaved` entry point: position, then an `Unorm8x4` color.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [u8; 4],
}

fn cover(color: [u8; 4]) -> [Vertex; 3] {
    COVER.map(|position| Vertex { position, color })
}

/// The color draw `i` writes, distinct for every pixel of the canvas.
fn color_of(i: u32) -> [u8; 4] {
    [(i & 0xff) as u8, (i >> 8) as u8, 0x5a, 0xff]
}

struct Canvas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    interleaved: wgpu::RenderPipeline,
    per_instance: wgpu::RenderPipeline,
}

impl Canvas {
    fn new(ctx: &TestingContext) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(SIZE * SIZE * 4),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = |entry_point, buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some(entry_point),
                        compilation_options: Default::default(),
                        buffers,
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let interleaved = pipeline(
            "interleaved",
            &[Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2, 1 => Unorm8x4],
            })],
        );
        let per_instance = pipeline(
            "per_instance",
            &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<[f32; 2]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x2],
                }),
                Some(wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &vertex_attr_array![2 => Unorm8x4],
                }),
            ],
        );
        Self {
            texture,
            view,
            readback,
            interleaved,
            per_instance,
        }
    }

    /// One submit with one pass over the canvas.
    fn submit(
        &self,
        ctx: &TestingContext,
        load: wgpu::LoadOp<wgpu::Color>,
        record: impl FnOnce(&mut wgpu::RenderPass<'_>),
    ) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            record(&mut pass);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    async fn pixels(&self, ctx: &TestingContext) -> Vec<[u8; 4]> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 4),
                    rows_per_image: None,
                },
            },
            self.texture.size(),
        );
        ctx.queue.submit(Some(encoder.finish()));
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
        let pixels =
            bytemuck::cast_slice(&self.readback.slice(..).get_mapped_range().unwrap()).to_vec();
        self.readback.unmap();
        pixels
    }
}

fn at_pixel(pass: &mut wgpu::RenderPass<'_>, i: u32) {
    pass.set_viewport((i % SIZE) as f32, (i / SIZE) as f32, 1.0, 1.0, 0.0, 1.0);
}

const CLEAR: wgpu::LoadOp<wgpu::Color> = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
const LOAD: wgpu::LoadOp<wgpu::Color> = wgpu::LoadOp::Load;

/// Buffers created right after `destroyed` ones: GL hands the freed names out again,
/// lowest first (ANGLE, Mesa), so the first of them carries the lowest destroyed name.
fn successors(ctx: &TestingContext, usage: BufferUsages, contents: &[u8]) -> Vec<wgpu::Buffer> {
    let buffers: Vec<wgpu::Buffer> = (0..4)
        .map(|_| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: contents.len() as u64,
                usage: usage | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();
    for buffer in &buffers {
        ctx.queue.write_buffer(buffer, 0, contents);
    }
    buffers
}

/// A buffer that takes a destroyed one's name must not draw through what referred to the
/// old one: the cached object of a state drawn twice, nor the spare that drew it once.
#[gpu_test]
static A_BUFFER_THAT_TAKES_A_DESTROYED_BUFFERS_PLACE_DRAWS_ITS_OWN_DATA: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_async(a_buffer_that_takes_a_destroyed_buffers_place_draws_its_own_data);

async fn a_buffer_that_takes_a_destroyed_buffers_place_draws_its_own_data(ctx: TestingContext) {
    let canvas = Canvas::new(&ctx);
    let old = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&cover(color_of(1))),
        usage: BufferUsages::VERTEX,
    });
    let draw = |buffers: &[&wgpu::Buffer], first_pixel: u32, load| {
        canvas.submit(&ctx, load, |pass| {
            pass.set_pipeline(&canvas.interleaved);
            for (i, buffer) in (first_pixel..).zip(buffers) {
                at_pixel(pass, i);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..3, 0..1);
            }
        });
    };
    // Once on the spare, then as a cached object; the spare still points at it.
    draw(&[&old], 0, CLEAR);
    draw(&[&old], 1, LOAD);
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    old.destroy();
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

    let new = successors(
        &ctx,
        BufferUsages::VERTEX,
        bytemuck::cast_slice(&cover(color_of(2))),
    );
    draw(&new.iter().collect::<Vec<_>>(), 2, LOAD);

    let pixels = canvas.pixels(&ctx).await;
    assert_eq!(pixels[..2], [color_of(1); 2]);
    assert_eq!(pixels[2..6], [color_of(2); 4]);
}

/// The same for the element buffer, which an object keeps apart from its attributes.
#[gpu_test]
static AN_INDEX_BUFFER_THAT_TAKES_A_DESTROYED_ONES_PLACE_DRAWS_ITS_OWN_INDICES:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(an_index_buffer_that_takes_a_destroyed_ones_place_draws_its_own_indices);

async fn an_index_buffer_that_takes_a_destroyed_ones_place_draws_its_own_indices(
    ctx: TestingContext,
) {
    let canvas = Canvas::new(&ctx);
    let mut contents = cover(color_of(1)).to_vec();
    contents.extend(cover(color_of(2)));
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&contents),
        usage: BufferUsages::VERTEX,
    });
    let old = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0u32, 1, 2]),
        usage: BufferUsages::INDEX,
    });
    let draw = |index_buffers: &[&wgpu::Buffer], first_pixel: u32, load| {
        canvas.submit(&ctx, load, |pass| {
            pass.set_pipeline(&canvas.interleaved);
            pass.set_vertex_buffer(0, vertices.slice(..));
            for (i, index_buffer) in (first_pixel..).zip(index_buffers) {
                at_pixel(pass, i);
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..3, 0, 0..1);
            }
        });
    };
    draw(&[&old], 0, CLEAR);
    draw(&[&old], 1, LOAD);
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    old.destroy();
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

    let new = successors(
        &ctx,
        BufferUsages::INDEX,
        bytemuck::cast_slice(&[3u32, 4, 5]),
    );
    draw(&new.iter().collect::<Vec<_>>(), 2, LOAD);

    let pixels = canvas.pixels(&ctx).await;
    assert_eq!(pixels[..2], [color_of(1); 2]);
    assert_eq!(pixels[2..6], [color_of(2); 4]);
}

/// A vertex array object also holds the element buffer. A later pass that reuses it
/// must bind its own index buffer unless the object still holds exactly that one, and
/// a submit, which starts by unbinding the element buffer of whatever object is bound,
/// must not alter a cached object.
#[gpu_test]
static ONE_VERTEX_STATE_DRAWS_WITH_THE_INDEX_BUFFER_OF_EACH_DRAW: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_async(one_vertex_state_draws_with_the_index_buffer_of_each_draw);

async fn one_vertex_state_draws_with_the_index_buffer_of_each_draw(ctx: TestingContext) {
    let canvas = Canvas::new(&ctx);
    let (first, second) = (color_of(1), color_of(2));
    let mut contents = cover(first).to_vec();
    contents.extend(cover(second));
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&contents),
        usage: BufferUsages::VERTEX,
    });
    let indices = |range: [u32; 3]| {
        ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&range),
            usage: BufferUsages::INDEX,
        })
    };
    let (to_first, to_second) = (indices([0, 1, 2]), indices([3, 4, 5]));
    let draw = |index_buffers: &[&wgpu::Buffer], first_pixel: u32, load| {
        canvas.submit(&ctx, load, |pass| {
            pass.set_pipeline(&canvas.interleaved);
            pass.set_vertex_buffer(0, vertices.slice(..));
            for (i, index_buffer) in (first_pixel..).zip(index_buffers) {
                at_pixel(pass, i);
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..3, 0, 0..1);
            }
        });
    };

    // The spare, then the object from its first draw on. The third submit starts with the
    // buffer the object holds, the fourth with another.
    draw(&[&to_first, &to_second, &to_first], 0, CLEAR);
    draw(&[&to_first, &to_second], 3, LOAD);
    draw(&[&to_second, &to_first], 5, LOAD);
    draw(&[&to_second], 7, LOAD);

    let pixels = canvas.pixels(&ctx).await;
    assert_eq!(
        pixels[..8],
        [first, second, first, first, second, second, first, second]
    );
}

/// Draws that switch buffers, offsets, enabled arrays and divisors on the spare, and the
/// same draws again, as cached objects up to the capacity and on the spare past it.
#[gpu_test]
static MORE_VERTEX_STATES_THAN_THE_CACHE_HOLDS_DRAW_THEIR_OWN_DATA: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_async(more_vertex_states_than_the_cache_holds_draw_their_own_data);

async fn more_vertex_states_than_the_cache_holds_draw_their_own_data(ctx: TestingContext) {
    // Must stay above the GL backend's `CAPACITY` for the second submit to reach the spare.
    const DRAWS: u32 = 2560;
    let canvas = Canvas::new(&ctx);
    let interleaved: Vec<Vertex> = (0..DRAWS).flat_map(|i| cover(color_of(i))).collect();
    let interleaved = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&interleaved),
        usage: BufferUsages::VERTEX,
    });
    let positions = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&COVER),
        usage: BufferUsages::VERTEX,
    });
    let colors: Vec<[u8; 4]> = (0..DRAWS).map(color_of).collect();
    let colors = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&colors),
        usage: BufferUsages::VERTEX,
    });
    let draw_all = |pass: &mut wgpu::RenderPass<'_>| {
        for i in 0..DRAWS {
            at_pixel(pass, i);
            if i % 2 == 0 {
                let start = u64::from(i) * size_of::<[Vertex; 3]>() as u64;
                pass.set_pipeline(&canvas.interleaved);
                pass.set_vertex_buffer(0, interleaved.slice(start..));
                pass.draw(0..3, 0..1);
            } else {
                // The color comes through the first instance, which GL without base
                // instance emulates with the instance buffer's offset.
                pass.set_pipeline(&canvas.per_instance);
                pass.set_vertex_buffer(0, positions.slice(..));
                pass.set_vertex_buffer(1, colors.slice(..));
                pass.draw(0..3, i..i + 1);
            }
        }
    };
    let wrong = |pixels: &[[u8; 4]]| -> Vec<u32> {
        (0..DRAWS)
            .filter(|&i| pixels[i as usize] != color_of(i))
            .collect()
    };

    canvas.submit(&ctx, CLEAR, draw_all);
    let first = wrong(&canvas.pixels(&ctx).await);
    canvas.submit(&ctx, CLEAR, draw_all);
    let second = wrong(&canvas.pixels(&ctx).await);
    assert!(
        first.is_empty() && second.is_empty(),
        "draws with wrong colors: {first:?} on first sight, {second:?} drawn again"
    );
}

#[gpu_test]
static A_CACHED_VERTEX_ARRAY_SURVIVES_ASYNC_PIPELINE_COMPLETION: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_async(a_cached_vertex_array_survives_async_pipeline_completion);

async fn a_cached_vertex_array_survives_async_pipeline_completion(ctx: TestingContext) {
    let canvas = Canvas::new(&ctx);
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&cover([255, 0, 0, 255])),
        usage: BufferUsages::VERTEX,
    });
    let indices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0u32, 1, 2]),
        usage: BufferUsages::INDEX,
    });
    let draw = |pipeline: &wgpu::RenderPipeline, pixel, load| {
        canvas.submit(&ctx, load, |pass| {
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            at_pixel(pass, pixel);
            pass.draw_indexed(0..3, 0, 0..1);
        });
    };
    draw(&canvas.interleaved, 0, CLEAR);
    draw(&canvas.interleaved, 1, LOAD);

    let inverted = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                SHADER
                    .replace(
                        "return input.color;",
                        "return vec4f(vec3f(1.0) - input.color.rgb, input.color.a);",
                    )
                    .into(),
            ),
        });
    let ready = ctx
        .device
        .create_render_pipeline_async(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &inverted,
                entry_point: Some("interleaved"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x2, 1 => Unorm8x4],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &inverted,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            multiview_mask: None,
            cache: None,
        })
        .await
        .unwrap();
    draw(&canvas.interleaved, 2, LOAD);
    draw(&ready, 3, LOAD);
    draw(&canvas.interleaved, 4, LOAD);

    let pixels = canvas.pixels(&ctx).await;
    assert_eq!(
        pixels[..5],
        [
            [255, 0, 0, 255],
            [255, 0, 0, 255],
            [255, 0, 0, 255],
            [0, 255, 255, 255],
            [255, 0, 0, 255],
        ]
    );
}
