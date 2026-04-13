# Zig Evaluation — Metal Kernel Build Tooling

## Why This Exists

PolarQuant's custom Metal kernels are currently authored as **MSL embedded in Python
string literals** via `mx.fast.metal_kernel()`. This works but has real costs:
- No IDE support for the shader code (no syntax highlighting, no autocomplete)
- Errors surface at runtime, not build time
- Diffs are unreadable (Metal shader inside a Python string literal)

Zig is a candidate replacement for the build/glue layer — not for the shader logic
itself, but for the infrastructure around it.

## What Was Tested

**Environment:** Mac Mini M4 Pro, Zig 0.15.2, Xcode 26.4, Metal Toolchain 17E188

**Scope:** Added `kernels/polarquant_qk.metal` (the simplest fused op) plus
`build.zig` that auto-discovers and compiles all `.metal` files in `kernels/`.

**What was NOT changed:**
- The Python API surface (`mx.fast.metal_kernel()` still drives runtime JIT)
- The kernel logic (identical MSL, just in its own file)
- The numerical results (unchanged by construction)

## Results

| Criteria | Result |
|----------|--------|
| `zig build` produces clean output for valid shader | **PASS** |
| MSL syntax errors caught at build time with line/col | **PASS** — exact error location in `xcrun` output |
| Same numerical output as current approach | **Yes** (same MSL, same JIT) |
| Auto-discovers new `.metal` files without registration | **PASS** |
| Would I start Phase 3 kernels this way? | **Yes — for anything over ~40 lines** |

## What the Zig Layer Actually Does

```
zig build
  └─ xcrun -sdk macosx metal -c kernels/*.metal -o kernels/*.air
  └─ xcrun metallib kernels/*.air -o kernels/*.metallib
```

The build artifacts (`.air`, `.metallib`) are ephemeral — `.gitignore`d.
At Python runtime, `mx.fast.metal_kernel()` still compiles the same MSL via
`MTLDevice.makeLibrary(source:)`. There's no runtime loading of the `.metallib`
(MLX doesn't expose that path cleanly).

## Key Friction Points Found

1. **Metal Toolchain is not in Xcode by default on macOS 26.** Requires:
   `xcodebuild -downloadComponent MetalToolchain` (~688 MB one-time download).
   Once installed, `xcrun metal` works from any shell.

2. **Buffer signature diverges from MLX-generated signatures.** `mx.fast.metal_kernel()`
   generates the kernel function signature (buffer bindings, thread indexing) from
   `input_names`/`output_names`. The standalone `.metal` file needs a hand-written
   signature. These can drift if MLX's code-gen changes. Kept in sync by: the runtime
   path still uses the Python string source, the `.metal` file is for validation only.

3. **Template instantiation gap.** The Python code calls `_build_qk_kernel(bits=4)`
   for each bit width. The standalone `.metal` file contains a concrete 4-bit
   instantiation for compilation checking. Testing all bit widths would require
   3 separate concrete functions or Metal template specialization.

## Decision

**Use Zig for future new kernels. Not worth retrofitting existing ones.**

**Why yes for new kernels:**
- Phase 3 sparse SV kernel (580 lines of MSL across two shaders) would have
  caught several issues earlier in development if there were proper IDE support.
- The build.zig auto-discovery means no registration cost — drop a `.metal` file
  and `zig build` validates it.
- `zig build` is fast (<2s for a single kernel) and can be a pre-commit hook.

**Why not retrofit existing kernels:**
- Phase 1-3 kernels are complete and tested. Extracting to `.metal` files adds
  churn without changing behavior.
- The 40-line kernels (QK, SV) are short enough that the string-literal ergonomics
  are tolerable.
- The signature divergence issue (MLX-generated vs hand-written) means the `.metal`
  file can't be the single source of truth for existing kernels — you'd maintain two
  representations.

## If Starting Phase 4

Start with `.metal` files. The workflow:
1. Write shader in `kernels/phase4_name.metal` (real MSL file, IDE support)
2. `zig build` validates before any Python is written
3. Python reads the source from the file: `Path("kernels/phase4_name.metal").read_text()`
4. Pass source to `mx.fast.metal_kernel()` (still JIT at runtime, but source is a file)
5. `zig build` as pre-commit hook catches shader regressions

This gives IDE support + build-time validation without requiring any change to how
MLX dispatches kernels.

## References

- Zig build system: ziglang.org/learn/overview
- Metal offline compilation: developer.apple.com/documentation/metal (makeLibrary)
- MLX custom kernels: ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
