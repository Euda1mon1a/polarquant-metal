# Zig Evaluation — Metal Kernel Build Tooling

## Why This Exists

PolarQuant's custom Metal kernels are currently authored as **MSL embedded in Python
string literals** via `mx.fast.metal_kernel()`. This works but has real costs:
- No IDE support for the shader code (no syntax highlighting, no autocomplete)
- Errors surface at runtime, not build time
- Diffs are unreadable (Metal shader inside a Python string literal)

Zig is a candidate replacement for the build/glue layer — not for the shader logic
itself, but for the infrastructure around it.

## What to Test

**Scope:** Replace the `mx.fast.metal_kernel()` Python string approach for ONE
existing kernel with a proper `.metal` file + Zig build. Don't rewrite kernel logic.

**Target kernel:** Pick the simplest existing fused op (see `polarquant_metal/kernels.py`).

**What Zig provides:**
- Build system that handles `xcrun -sdk macosx metal` + `xcrun metallib` natively
- C-ABI bridge to MLX's internal kernel registration without ctypes
- `comptime` can generate dispatch boilerplate for kernel variants at build time
- `zig build` replaces any CMake/Makefile glue

**What stays the same:**
- The `.metal` shader source (identical MSL, just in its own file)
- The Python API surface (still callable from Python the same way)
- The numerical results

## Success Criteria

| Criteria | Pass |
|----------|------|
| `zig build` produces a callable Python extension | Build works cleanly |
| Same numerical output as current `mx.fast.metal_kernel` version | Yes |
| MSL shader errors caught at build time, not runtime | Yes |
| Would I start a new kernel this way? | Subjective — the real output |

## If Evaluation Passes

Future new kernels start as `.metal` files. The `mx.fast.metal_kernel()` path
stays for quick prototyping (it's not wrong, just ergonomically rough for anything
over ~50 lines of shader code).

## References

- Zig C interop: ziglang.org/learn/overview/#integration-with-c-libraries
- Metal compilation: developer.apple.com/documentation/metal/mtldevice/makedefaultlibrary
- MLX custom kernels: ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
