let probe;
const parallelExtension = "KHR_parallel_shader_compile";

export function installProbe(extensionMissing) {
  if (probe) throw new Error("shader probe already installed");
  const proto = WebGL2RenderingContext.prototype;
  probe = { originals: new Map(), programs: new Map(), shaders: new Set(),
    created: 0, deleted: 0, shaderCreated: 0, shaderDeleted: 0, polls: 0,
    premature: 0, pending: true, lost: false, extensionMissing };
  function wrap(name, callback) {
    const original = proto[name];
    probe.originals.set(name, original);
    proto[name] = function (...args) { return callback.call(this, original, args); };
  }
  wrap("getSupportedExtensions", function (original, args) {
    const extensions = original.apply(this, args);
    return extensions;
  });
  wrap("getExtension", function (original, args) {
    return extensionMissing && args[0] === parallelExtension ? null : original.apply(this, args);
  });
  wrap("createProgram", function (original, args) {
    const program = original.apply(this, args);
    if (program) {
      if (probe.context && probe.context !== this) throw new Error("probe used by multiple contexts");
      probe.context = this;
      probe.programs.set(program, extensionMissing);
      probe.created++;
    }
    return program;
  });
  wrap("deleteProgram", function (original, args) {
    if (probe.programs.delete(args[0])) probe.deleted++;
    return original.apply(this, args);
  });
  wrap("createShader", function (original, args) {
    const shader = original.apply(this, args);
    if (shader) { probe.shaders.add(shader); probe.shaderCreated++; }
    return shader;
  });
  wrap("deleteShader", function (original, args) {
    if (probe.shaders.delete(args[0])) probe.shaderDeleted++;
    return original.apply(this, args);
  });
  wrap("getProgramParameter", function (original, args) {
    const [program, parameter] = args;
    if (probe.programs.has(program)) {
      if (parameter === 0x91B1) {
        probe.polls++;
        // Preserve the browser's completion query and link bookkeeping while
        // forcing the Rust future to suspend until the test releases it.
        const done = original.apply(this, args);
        if (probe.pending) return false;
        probe.programs.set(program, !!done);
        return done;
      }
      if (!probe.programs.get(program)) probe.premature++;
    }
    return original.apply(this, args);
  });
  for (const name of ["getProgramInfoLog", "getUniformLocation", "getUniformBlockIndex"]) {
    wrap(name, function (original, args) {
      if (probe.programs.has(args[0]) && !probe.programs.get(args[0])) probe.premature++;
      return original.apply(this, args);
    });
  }
  wrap("getShaderParameter", function (original, args) {
    if (!extensionMissing && probe.shaders.has(args[0])) probe.premature++;
    return original.apply(this, args);
  });
}
export function allowCompletion() { probe.pending = false; }
export function simulateContextLoss() {
  const gl = probe.context;
  if (!gl || gl.isContextLost()) throw new Error("expected a live GL context before loss");
  const extension = gl.getExtension("WEBGL_lose_context");
  if (!extension) throw new Error("WEBGL_lose_context is required by this contract");
  extension.loseContext();
  probe.lost = gl.isContextLost();
  if (!probe.lost) throw new Error("WEBGL_lose_context did not lose the actual context");
  // The harness treats every pending GL error as a validation failure. Consume
  // only the one mandated loss notification; any other value fails this test.
  probe.lossError = gl.getError();
  if (probe.lossError !== gl.CONTEXT_LOST_WEBGL) {
    throw new Error(`expected CONTEXT_LOST_WEBGL, got 0x${probe.lossError.toString(16)}`);
  }
}
export function metric(name) { return probe[name]; }
export function removeProbe() {
  if (probe.lost) {
    const error = probe.context.getError();
    if (error !== probe.context.NO_ERROR) {
      throw new Error(`unexpected GL error after loss cleanup: 0x${error.toString(16)}`);
    }
  }
  for (const [name, original] of probe.originals) WebGL2RenderingContext.prototype[name] = original;
  probe = undefined;
}
