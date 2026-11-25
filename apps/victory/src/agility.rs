//! The D3D12 Agility SDK: the exports by which the system's `d3d12.dll`
//! loads a newer runtime (`D3D12Core.dll`) from beside this executable
//! instead of its own. Windows 10's own runtime has no shader model 6.6,
//! which every translated pipeline needs (descriptor heaps indexed from
//! shaders). `build.rs` has the linker export both symbols.
//!
//! `SDK_VERSION` must be the minor version of the redistributable that
//! `cargo xtask agility` installs (`xtask/src/agility.rs`, package
//! 1.620.2 → 620): the loader refuses a runtime of another version and
//! falls back to the system's.

#![allow(non_upper_case_globals)] // the names the D3D12 loader looks up

/// The Agility SDK version this executable asks for: 614, the release whose
/// `D3D12Core.dll` this build is run with (`D3D12\` beside the executable).
/// `cargo xtask agility` installs 1.620.2, which also serves a request for
/// 614 (a runtime serves its own version and older ones).
pub const SDK_VERSION: u32 = 614;

#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKVersion: u32 = SDK_VERSION;

/// A `const char*` as the loader reads it.
#[repr(transparent)]
pub struct SdkPath(*const u8);

// SAFETY: points at a string literal, never written.
unsafe impl Sync for SdkPath {}

/// Relative to the executable's folder.
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKPath: SdkPath = SdkPath(c".\\D3D12\\".as_ptr().cast());
