//! Extension bindings for WebGPU.
//!
//! These contain ideomatic Rust extension traits for various parts of the WebGPU
//! bindings that are missing, need to be improved, or otherwise need to be different
//! from the generated web_sys bindings.

use crate::backend::webgpu::webgpu_sys;
use wasm_bindgen::prelude::*;

/// Extension trait for [`web_sys::Navigator`] and [`web_sys::WorkerNavigator`] to
/// access the `gpu` property.
pub trait NavigatorGpu {
    /// Get the `gpu` property.
    ///
    /// This is intentionally a free function, to prevent overload conflicts with
    /// the method if it is enabled in web-sys itself.
    fn gpu(navigator: &Self) -> webgpu_sys::Gpu;
}

// --- Bindings for `Navigator` ---
#[wasm_bindgen]
extern "C" {
    /// Create a fake class which we tell wasm-bindgen has access to the `gpu` property.
    #[wasm_bindgen]
    type NavigatorWithGpu;

    #[wasm_bindgen(method, getter)]
    fn gpu(ext: &NavigatorWithGpu) -> webgpu_sys::Gpu;
}

impl NavigatorGpu for web_sys::Navigator {
    fn gpu(navigator: &Self) -> webgpu_sys::Gpu {
        // Must be an unchecked ref as this class does not exist at runtime.
        let extension: &NavigatorWithGpu = navigator.unchecked_ref();
        extension.gpu()
    }
}

impl NavigatorGpu for web_sys::WorkerNavigator {
    fn gpu(navigator: &Self) -> webgpu_sys::Gpu {
        // Must be an unchecked ref as this class does not exist at runtime.
        let extension: &NavigatorWithGpu = navigator.unchecked_ref();
        extension.gpu()
    }
}

// --- Bindings for `GPUPipelineError` ---

/// The rejection value of `GPUDevice.createRenderPipelineAsync()`.
///
/// Not part of the `webgpu_sys` bindings vendored from `web-sys`, hence the hand-written
/// binding here. It extends `js_sys::Object` rather than `web_sys::DomException`, which it
/// derives from in the specification, because wgpu does not enable the `DomException`
/// feature of `web-sys`.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(
        extends = ::js_sys::Object,
        js_name = "GPUPipelineError",
        typescript_type = "GPUPipelineError"
    )]
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub type GpuPipelineError;

    /// Getter for the `message` field of this object.
    #[wasm_bindgen(method, getter, js_class = "GPUPipelineError", js_name = "message")]
    pub fn message(this: &GpuPipelineError) -> ::alloc::string::String;

    /// Getter for the `reason` field of this object: `"validation"` or `"internal"`.
    #[wasm_bindgen(method, getter, js_class = "GPUPipelineError", js_name = "reason")]
    pub fn reason(this: &GpuPipelineError) -> ::alloc::string::String;
}
