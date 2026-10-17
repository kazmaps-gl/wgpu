use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        COMPLETION_PRECEDES_REFLECTION,
        CANCELLATION_RELEASES_PROGRAM_AND_SHADERS,
        LOST_CONTEXT_SETTLES_COMPILATION,
        MISSING_EXTENSION_USES_SYNCHRONOUS_FALLBACK,
    ]);
}

fn parameters() -> TestParameters {
    TestParameters::default().skip(FailureCase::backend(!wgpu::Backends::GL))
}

#[gpu_test]
static COMPLETION_PRECEDES_REFLECTION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(|ctx| async move {
        #[cfg(target_arch = "wasm32")]
        browser::completion_precedes_reflection(&ctx).await;
        let _ = ctx;
    });

#[gpu_test]
static CANCELLATION_RELEASES_PROGRAM_AND_SHADERS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(parameters())
        .run_async(|ctx| async move {
            #[cfg(target_arch = "wasm32")]
            browser::cancellation_releases_program_and_shaders(&ctx).await;
            let _ = ctx;
        });

#[gpu_test]
static LOST_CONTEXT_SETTLES_COMPILATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(|ctx| async move {
        #[cfg(target_arch = "wasm32")]
        browser::lost_context_settles_compilation(&ctx).await;
        let _ = ctx;
    });

#[gpu_test]
static MISSING_EXTENSION_USES_SYNCHRONOUS_FALLBACK: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(parameters())
        .run_async(|ctx| async move {
            #[cfg(target_arch = "wasm32")]
            browser::missing_extension_uses_synchronous_fallback(&ctx).await;
            let _ = ctx;
        });

#[cfg(target_arch = "wasm32")]
mod browser {
    use std::{future::Future, pin::pin};
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen(module = "/tests/wgpu-gpu/webgl_async_probe.js")]
    extern "C" {
        #[wasm_bindgen(js_name = installProbe)]
        fn install_probe(extension_missing: bool);
        #[wasm_bindgen(js_name = allowCompletion)]
        fn allow_completion();
        #[wasm_bindgen(js_name = simulateContextLoss)]
        fn simulate_context_loss();
        fn metric(name: &str) -> u32;
        #[wasm_bindgen(js_name = removeProbe)]
        fn remove_probe();
    }

    struct Probe;
    impl Probe {
        fn new(extension_missing: bool) -> Self {
            install_probe(extension_missing);
            Self
        }
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            remove_probe();
        }
    }

    fn shader(device: &wgpu::Device) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("async pipeline contract"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                r#"
            @group(0) @binding(0) var<uniform> color: vec4<f32>;
            @vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0, 0, 0, 1); }
            @fragment fn fs() -> @location(0) vec4<f32> { return color; }
        "#,
            )),
        })
    }

    fn pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
    ) -> impl Future<Output = Result<wgpu::RenderPipeline, wgpu::Error>> + 'static {
        device.create_render_pipeline_async(&wgpu::RenderPipelineDescriptor {
            label: Some("async pipeline contract"),
            layout: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    pub(super) async fn completion_precedes_reflection(ctx: &wgpu_test::TestingContext) {
        let _probe = Probe::new(false);
        let shader = shader(&ctx.device);
        let mut pending = pin!(pipeline(&ctx.device, &shader));
        assert!(futures_lite::future::poll_once(pending.as_mut())
            .await
            .is_none());
        assert_eq!(metric("polls"), 1);
        assert_eq!(metric("premature"), 0);
        allow_completion();
        let ready = pending.await.unwrap();
        let created = metric("created");
        let cached = pipeline(&ctx.device, &shader).await.unwrap();
        assert_eq!(metric("created"), created, "cached program compiled twice");
        assert_eq!(metric("premature"), 0);
        assert_eq!(metric("shaderCreated"), metric("shaderDeleted"));
        ready.get_bind_group_layout(0);
        cached.get_bind_group_layout(0);
    }

    pub(super) async fn cancellation_releases_program_and_shaders(ctx: &wgpu_test::TestingContext) {
        let _probe = Probe::new(false);
        let shader = shader(&ctx.device);
        let mut pending = Box::pin(pipeline(&ctx.device, &shader));
        assert!(futures_lite::future::poll_once(pending.as_mut())
            .await
            .is_none());
        drop(pending);
        assert_eq!(metric("created"), metric("deleted"));
        assert_eq!(metric("shaderCreated"), metric("shaderDeleted"));
        allow_completion();
        let pending = pipeline(&ctx.device, &shader);
        drop(shader);
        pending.await.unwrap().get_bind_group_layout(0);
        assert_eq!(metric("premature"), 0);
    }

    pub(super) async fn lost_context_settles_compilation(ctx: &wgpu_test::TestingContext) {
        let _probe = Probe::new(false);
        ctx.device.set_device_lost_callback(|_, _| {});
        let shader = shader(&ctx.device);
        let mut pending = pin!(pipeline(&ctx.device, &shader));
        assert!(futures_lite::future::poll_once(pending.as_mut())
            .await
            .is_none());
        simulate_context_loss();
        assert!(pending.await.is_err());
        assert_eq!(metric("created"), metric("deleted"));
        assert_eq!(metric("shaderCreated"), metric("shaderDeleted"));
        assert_eq!(metric("premature"), 0);
    }

    pub(super) async fn missing_extension_uses_synchronous_fallback(
        _ctx: &wgpu_test::TestingContext,
    ) {
        let _probe = Probe::new(true);
        let canvas = wgpu_test::initialize_html_canvas();
        let instance = wgpu_test::initialize_instance(wgpu::Backends::GL, &super::parameters());
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            })
            .await
            .unwrap();
        let shader = shader(&device);
        pipeline(&device, &shader)
            .await
            .unwrap()
            .get_bind_group_layout(0);
        assert_eq!(metric("polls"), 0);
        assert_eq!(metric("premature"), 0);
    }
}
