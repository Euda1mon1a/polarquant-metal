//! build.zig — PolarQuant Metal kernel build validation
//!
//! Purpose: compile all .metal files at `zig build` time so MSL syntax errors
//! surface here rather than at Python runtime.
//!
//! Usage:
//!   zig build          — validate all Metal shaders
//!   zig build validate — same (explicit step)
//!
//! Runtime: mx.fast.metal_kernel() still does JIT compilation at runtime.
//! This step validates correctness at build time, not for artifact production.
//! Build artifacts (.air, .metallib) are in .gitignore — not committed.
//!
//! Adding a new kernel: drop a .metal file in kernels/ — auto-discovered.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const validate = b.step("validate", "Compile all Metal shaders (MSL validation)");
    b.default_step.dependOn(validate);

    var kernels_dir = std.fs.cwd().openDir("kernels", .{ .iterate = true }) catch {
        std.debug.print("No kernels/ directory — skipping Metal validation\n", .{});
        return;
    };
    defer kernels_dir.close();

    var iter = kernels_dir.iterate();
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    while (iter.next() catch null) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".metal")) continue;

        const metal_path = std.fmt.allocPrint(alloc, "kernels/{s}", .{entry.name}) catch unreachable;
        const air_path   = std.mem.replaceOwned(u8, alloc, metal_path, ".metal", ".air") catch unreachable;
        const lib_path   = std.mem.replaceOwned(u8, alloc, metal_path, ".metal", ".metallib") catch unreachable;

        // Step 1: MSL → AIR (catches syntax/type/logic errors)
        const compile = b.addSystemCommand(&[_][]const u8{
            "xcrun", "-sdk", "macosx", "metal",
            "-c", metal_path,
            "-o", air_path,
            "-gline-tables-only",  // source locations in error messages
        });
        validate.dependOn(&compile.step);

        // Step 2: AIR → metallib (catches linker-level Metal issues)
        const link = b.addSystemCommand(&[_][]const u8{
            "xcrun", "metallib",
            air_path,
            "-o", lib_path,
        });
        link.step.dependOn(&compile.step);
        validate.dependOn(&link.step);
    }
}
