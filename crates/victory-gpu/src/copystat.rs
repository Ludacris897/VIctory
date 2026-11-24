//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Where a submission's preparation goes (microseconds): flushing
    /// refreshes, snapshots for shader writes, texture checks, recording
    /// the steps; and of copy-back, the bottom-of-pipe writes.
    FlushUs,
    GpuWriteUs,
    TextureUs,
    StepsUs,
    BottomUs,
    /// Ranges shaders were declared to write, and their bytes (each
    /// compared at copy-back).
    BackRanges,
    BackBytes,
    /// Dynamic view slots taken, and requests refused.
    DynTaken,
    DynFailed,
    /// Rectangle draws run through a pipeline's rectangle geometry stage,
    /// direct and indirect.
    RectGsDraws,
    RectGsIndirect,
    /// Lists run again after their dynamic slots were given back.
    Resubmits,
}
const COUNTERS: usize = 34;

/// Pipelines made with a rectangle geometry stage, so far.
pub(crate) static RECT_GS_PIPELINES: AtomicU64 = AtomicU64::new(0);

/// The most dynamic view slots in use at once since the last log, of how
/// many.
static DYN_PEAK: AtomicU64 = AtomicU64::new(0);
static DYN_CAP: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dynamic_in_use(in_use: u32, of: u32) {
    DYN_PEAK.fetch_max(u64::from(in_use), Relaxed);
    DYN_CAP.store(u64::from(of), Relaxed);
}
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
    );
    tracing::info!(
        "preparing per frame: steps {:.1} ms (flushing refreshes {:.1} ms, snapshots for shader \
         writes {:.1} ms, textures {:.1} ms); bottom-of-pipe writes {:.2} ms; copy-back compared \
         {:.0} ranges {:.1} MB",
        ms(c[C::StepsUs as usize]),
        ms(c[C::FlushUs as usize]),
        ms(c[C::GpuWriteUs as usize]),
        ms(c[C::TextureUs as usize]),
        ms(c[C::BottomUs as usize]),
        c[C::BackRanges as usize],
        mb(c[C::BackBytes as usize]),
    );
    tracing::info!(
        "dynamic view slots per frame: {:.0} taken, {:.0} requests refused; at most {} of {} in use",
        c[C::DynTaken as usize],
        c[C::DynFailed as usize],
        DYN_PEAK.swap(0, Relaxed),
        DYN_CAP.load(Relaxed),
    );
    for (capacity, free, largest) in crate::descriptor::dynamic_report() {
        tracing::info!(capacity, free, largest, "dynamic view slots now: free, and the largest free run");
    }
    tracing::info!(
        "rectangles per frame: {:.0} draws and {:.0} indirect draws through a geometry stage; {} \
         such pipelines so far; {:.1} lists run again after their slots were given back",
        c[C::RectGsDraws as usize],
        c[C::RectGsIndirect as usize],
        RECT_GS_PIPELINES.load(Relaxed),
        c[C::Resubmits as usize],
    );
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}
