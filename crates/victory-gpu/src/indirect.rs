//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#ifdef ISOLATED
#define RS "RootConstants(num32BitConstants=6, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
cbuffer Record : register(b0) {
    uint kind; uint stride; uint records; uint out_stride; uint count_offset; uint has_count;
};
RWByteAddressBuffer source_args : register(u2);
RWByteAddressBuffer source_count : register(u3);
#define INPUT source_args
#else
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
#define INPUT args
#endif
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint from = id.x * stride;
#ifdef ISOLATED
    uint at = id.x * out_stride;
    // Copy only the command fields, never the padding in a title record.
    if (kind == 0) args.Store2(at + 8, INPUT.Load2(from + 8));
    if (kind == 1) args.Store3(at + 8, INPUT.Load3(from + 8));
    if (id.x == 0 && has_count != 0) {
        args.Store(count_offset, min(source_count.Load(0), records));
    }
#else
    uint at = from;
#endif
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(INPUT.Load(from), 1u << 24);
        uint instances = min(INPUT.Load(from + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(INPUT.Load3(from), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = INPUT.Load3(from);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    // Reserve without overflow. Keep retries bounded and the loop exit
    // independent of UAV data for the legacy cs_5_1 compiler.
    if (cost == 0) return;
    uint before;
    budget.InterlockedAdd(0, 0, before);
    bool reserved = false;
    [loop]
    for (uint attempt = 0; attempt < 128; ++attempt) {
        if (!reserved && before < BUDGET && cost <= BUDGET - before) {
            uint observed;
            budget.InterlockedCompareExchange(0, before, before + cost, observed);
            reserved = (observed == before);
            before = observed;
        }
    }
    // Contention or a full budget: fail closed, never admit excess work.
    if (!reserved) args.Store(at, 0);
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    make_variant(false)
}

fn make_variant(isolated: bool) -> windows::core::Result<Clamp> {
    let source = if isolated { format!("#define ISOLATED 1\n{SOURCE}") } else { SOURCE.to_owned() };
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
            #[cfg(test)]
            eprintln!("FXC argument clamp diagnostic: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).map_err(|e| {
            #[cfg(test)]
            eprintln!("Clamp root signature creation failed: {e}");
            e
        })?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        }).map_err(|e| {
            #[cfg(test)]
            eprintln!("Clamp PSO creation failed: {e}");
            e
        })?;
        Ok(Clamp { root, pso })
    }
}

fn isolated_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP.get_or_init(|| match make_variant(true) {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(error = %e, "isolated argument clamp unavailable");
            None
        }
    }).as_ref()
}

fn command_size(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Compact scratch records followed by a separate, optional count word.
/// The limit bounds temporary allocation per indirect call.
fn scratch_layout(kind: u32, stride: u32, records: u32) -> Option<(u32, u32, u64)> {
    if !valid_layout(kind, stride, records) { return None; }
    let size = command_size(kind)?;
    let count_offset = size.checked_mul(records)?;
    let bytes = u64::from(count_offset) + 4;
    (bytes <= 16 * 1024 * 1024).then_some((size, count_offset, bytes))
}

/// A private argument allocation. The source buffers are read by the
/// clamp as UAVs as before, but never mutated or transitioned to INDIRECT.
pub(crate) struct IsolatedArgs {
    pub buffer: ID3D12Resource,
    pub stride: u32,
    pub count_offset: u64,
    #[cfg(test)]
    data: *mut u8,
}

/// GPU-side copy and clamp, retaining the original arguments and count.
/// The caller retains source allocations; this list retains the result.
/// The scratch buffer is used for exactly one ExecuteIndirect.
pub(crate) fn bound_isolated(
    list: &HostList,
    source_gpu: u64,
    count_gpu: Option<u64>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<IsolatedArgs> {
    let (out_stride, count_offset, bytes) = scratch_layout(kind, stride, records)?;
    if !source_gpu.is_multiple_of(4) || count_gpu.is_some_and(|g| !g.is_multiple_of(4)) {
        return None;
    }
    let c = isolated_clamp()?;
    let budget = list.budget()?;
    let (buffer, data) = host()?.shared_buffer(bytes)?;
    list.keep(&buffer);
    let constants = [kind, stride, records, out_stride, count_offset, u32::from(count_gpu.is_some())];
    let l = &list.list;
    unsafe {
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, buffer.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, source_gpu);
        l.SetComputeRootUnorderedAccessView(4, count_gpu.unwrap_or(source_gpu));
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    let _ = data;
    Some(IsolatedArgs {
        buffer, stride: out_stride, count_offset: u64::from(count_offset),
        #[cfg(test)]
        data,
    })
}

/// Reject layouts that cannot be addressed by the clamp's uint byte offsets
/// or exceed D3D12's one-dimensional dispatch limit.
fn valid_layout(kind: u32, stride: u32, records: u32) -> bool {
    let Some(minimum) = command_size(kind) else { return false; };
    records != 0
        && stride >= minimum
        && stride.is_multiple_of(4)
        && records.div_ceil(64) <= 65_535
        && records.checked_mul(stride).is_some()
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    if !gpu.is_multiple_of(4) || !valid_layout(kind, stride, records) {
        return false;
    }
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn indirect_layout_validation() {
        assert!(super::valid_layout(0, 16, 1));
        assert!(super::valid_layout(1, 20, 1));
        assert!(super::valid_layout(2, 12, 65_535 * 64));
        assert!(super::valid_layout(10, 16, 128));
        assert!(!super::valid_layout(0, 12, 1));
        assert!(!super::valid_layout(1, 16, 1));
        assert!(!super::valid_layout(2, 13, 1));
        assert!(!super::valid_layout(2, 12, 0));
        assert!(!super::valid_layout(2, 12, 65_535 * 64 + 1));
        assert!(!super::valid_layout(2, 0xffff_fffc, 2));
        assert!(!super::valid_layout(3, 16, 1));
    }

    #[test]
    fn isolated_layout_is_compact_and_bounded() {
        assert_eq!(super::scratch_layout(1, 32, 3), Some((20, 60, 64)));
        assert_eq!(super::scratch_layout(2, 16, 8), Some((12, 96, 100)));
        assert!(super::scratch_layout(0, 16, 2_000_000).is_none());
        assert!(super::scratch_layout(1, 16, 1).is_none());
    }

    #[test]
    fn isolated_clamp_preserves_source_and_copies_count_on_gpu() {
        let Some(host) = super::host() else {
            eprintln!("no host GPU; isolated execution test skipped");
            return;
        };
        let list = host.new_list().expect("isolated test list");
        let (source, data) = host.shared_buffer(8 * 32 + 4).expect("isolated source");
        unsafe {
            for i in 0..8 {
                let record = data.cast::<u32>().add(i * 8);
                record.write(1 << 24);
                record.add(1).write(4);
                record.add(2).write(17);
                record.add(3).write(23);
                for j in 4..8 { record.add(j).write(0xfeedbeef); }
            }
            data.cast::<u32>().add(64).write(99);
        }
        let gpu = unsafe { source.GetGPUVirtualAddress() };
        let result = super::bound_isolated(&list, gpu, Some(gpu + 256), (0, 32, 8))
            .expect("isolated clamp records");
        assert_eq!(result.stride, 16);
        assert_eq!(result.count_offset, 128);
        list.close();
        host.execute(&[&list]);
        let mut accepted = 0;
        unsafe {
            for i in 0..8 {
                let original = data.cast::<u32>().add(i * 8);
                assert_eq!(original.read_volatile(), 1 << 24);
                assert_eq!(original.add(1).read_volatile(), 4);
                assert_eq!(original.add(4).read_volatile(), 0xfeedbeef);
                let record = result.data.cast::<u32>().add(i * 4);
                let count = record.read_volatile();
                assert!(count == 0 || count == 1 << 24);
                accepted += usize::from(count != 0);
                assert_eq!(record.add(1).read_volatile(), 4);
                assert_eq!(record.add(2).read_volatile(), 17);
                assert_eq!(record.add(3).read_volatile(), 23);
            }
            assert_eq!(data.cast::<u32>().add(64).read_volatile(), 99);
            assert_eq!(result.data.cast::<u32>().add(32).read_volatile(), 8);
        }
        assert_eq!(accepted, 2);
        eprintln!("isolated GPU clamp: 2/8 accepted; source unchanged; count clamped to 8");
    }

    /// Execute only the clamp (never the clamped draw arguments). Check
    /// that contention cannot exceed the budget, and that rejected records
    /// retain the other fields of their layout.
    #[test]
    fn argument_clamp_executes_budget_on_gpu() {
        let Some(host) = super::host() else {
            eprintln!("no host GPU; execution test skipped");
            return;
        };
        let list = host.new_list().expect("clamp test list");
        let (buffer, data) = host.shared_buffer(128).expect("clamp test buffer");
        // Eight records costing 2^26 each; budget 2^27.
        unsafe {
            for i in 0..8 {
                let record = data.cast::<u32>().add(i * 4);
                record.write(1 << 24);
                record.add(1).write(4);
                record.add(2).write(17);
                record.add(3).write(23);
            }
        }
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        assert!(super::bound(&list, &buffer, gpu, (0, 16, 8)));
        super::done(&list, &buffer);
        list.close();
        host.execute(&[&list]);
        let mut accepted = 0;
        unsafe {
            for i in 0..8 {
                let record = data.cast::<u32>().add(i * 4);
                let count = record.read_volatile();
                assert!(count == 0 || count == 1 << 24, "unexpected count {count}");
                accepted += usize::from(count != 0);
                assert_eq!(record.add(1).read_volatile(), 4);
                assert_eq!(record.add(2).read_volatile(), 17);
                assert_eq!(record.add(3).read_volatile(), 23);
            }
        }
        assert_eq!(accepted, 2, "GPU budget must admit exactly two records");
        eprintln!("GPU clamp executed: {accepted}/8 records accepted within budget");
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        super::make().unwrap_or_else(|e| panic!("argument clamp build failed: {e}"));
    }
}
