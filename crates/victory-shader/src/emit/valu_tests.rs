//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and the exact
//! rounding of `v_cvt_pknorm_*`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// DS `ds_swizzle_b32 vdst, vsrc offset:offset`.
const fn ds_swizzle_b32(vdst: u8, vsrc: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | 0x35 << 18 | offset as u32,
        (vdst as u32) << 24 | vsrc as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, voff, s[srsrc:srsrc+3], 0 offen`.
const fn buffer_store_dword_offen(vdata: u8, voff: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | voff as u32,
    ]
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOPC `op vcc, src0, vsrc1`.
const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// SOP1 `op sdst, src0`.
const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

/// DS `ds_bpermute_b32 vdst, vaddr, vdata0 offset:off`.
const fn ds_bpermute_b32(vdst: u8, vaddr: u8, vdata0: u8, offset: u16) -> [u32; 2] {
    [
        0xD800_0000 | (0x0B3 << 18) | offset as u32,
        (vdst as u32) << 24 | (vdata0 as u32) << 8 | vaddr as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen`.
const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);
    build_cs(&words, 1)
}

/// Lowers a synthetic RDNA2 compute shader of `threads` threads in one row
/// (thread id in `v0`): 4 dwords of root constants in `s0..s3`, the
/// descriptor table address in `s4`, a raw UAV at heap index 0.
fn build_cs(words: &[u32], threads: u32) -> HostShader {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        console_cf: false,
        per_lane_only: false,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let host = match Host::new(warp) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("SM 6.6 {label} initialization failed: {e}");
                continue;
            }
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

/// One wave of 32 lanes holding 1..=32 (the input of the ISA's examples)
/// through `ds_swizzle_b32` with `offset`: what each lane reads.
fn swizzle_32(offset: u16) -> Vec<Vec<u32>> {
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.push(vop2(0x25, 1, 129, 0)); // v_add_nc_u32 v1, 1, v0
    words.extend_from_slice(&ds_swizzle_b32(2, 1, offset));
    words.push(vop2(0x1A, 3, 130, 0)); // v_lshlrev_b32 v3, 2, v0
    words.extend_from_slice(&buffer_store_dword_offen(2, 3, 8));
    words.push(S_ENDPGM);
    let shader = build_cs(&words, 32);

    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // A host without 32-lane waves cannot run it.
            Err(e) if warp => {
                eprintln!("{label}: no pipeline ({e}); skipped");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push(uavs[0].clone());
    }
    if out.is_empty() {
        eprintln!("no SM 6.6 device with 32-lane waves; skipped");
    }
    out
}

/// The basic modes: a quad permutation and a 32-lane and/or/xor mapping.
#[test]
fn ds_swizzle_basic_modes() {
    // offset[15]: lanes of each quad read lanes 3, 2, 1, 0 of it.
    let reversed: Vec<u32> = (0..32).map(|i| (i ^ 3) + 1).collect();
    // and 0x1f, or 0, xor 0x10: the halves swap.
    let swapped: Vec<u32> = (0..32).map(|i| (i ^ 16) + 1).collect();
    for (offset, want) in [(0x801B, reversed), (0x401F, swapped)] {
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
}

/// Offsets from 0xE000: the FFT orders, as the ISA lists them for inputs
/// 1..=0x20 (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_fft_mode() {
    let cases: [(u16, [u32; 32]); 3] = [
        (
            0xE000,
            [
                0x01, 0x11, 0x09, 0x19, 0x05, 0x15, 0x0d, 0x1d, 0x03, 0x13, 0x0b, 0x1b, 0x07, 0x17,
                0x0f, 0x1f, 0x02, 0x12, 0x0a, 0x1a, 0x06, 0x16, 0x0e, 0x1e, 0x04, 0x14, 0x0c, 0x1c,
                0x08, 0x18, 0x10, 0x20,
            ],
        ),
        (
            0xE010,
            [
                0x01, 0x09, 0x05, 0x0d, 0x03, 0x0b, 0x07, 0x0f, 0x02, 0x0a, 0x06, 0x0e, 0x04, 0x0c,
                0x08, 0x10, 0x11, 0x19, 0x15, 0x1d, 0x13, 0x1b, 0x17, 0x1f, 0x12, 0x1a, 0x16, 0x1e,
                0x14, 0x1c, 0x18, 0x20,
            ],
        ),
        // 0x1f: no swizzle.
        (0xE01F, std::array::from_fn(|i| i as u32 + 1)),
    ];
    for (offset, want) in cases {
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
    // The other bits of an FFT offset do not matter.
    for got in swizzle_32(0xFFF0) {
        assert_eq!(got, cases[1].1, "offset 0xfff0");
    }
}

/// Offsets from 0xC000 to 0xDFFF: the wave's 32 lanes rotate left by
/// offset[9:5], or right with offset[10] (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_rotate_mode() {
    let rotated = |by: u32| -> Vec<u32> { (0..32).map(|i| (i + by) % 32 + 1).collect() };
    for (offset, by) in [
        (0xC000 | 1 << 5, 1),            // the ISA's example: {2, 3, ... 0x20, 1}
        (0xC000 | 1 << 10 | 1 << 5, 31), // and {0x20, 1, ... 0x1f}
        (0xC000, 0),
        (0xC000 | 5 << 5, 5),
        (0xC000 | 1 << 10 | 5 << 5, 27),
        (0xD800 | 31 << 5, 31), // bits 11 and 12 are not read
    ] {
        for got in swizzle_32(offset) {
            assert_eq!(got, rotated(by), "offset {offset:#06x}");
        }
    }
}

/// Builds, validates and runs a single-wave compute shader (`32` threads in
/// Wave32, `64` in Wave64, with zero LDS allocated) where `body` leaves each
/// lane's result in `v3`, stored to dword `threadIdInGroup.x` of a raw UAV.
fn run_wave_cs(wave64: bool, body: &[u32], mut check: impl FnMut(&str, &[u32])) {
    let wave = if wave64 { 64 } else { 32 };
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    // v7 = v0 << 2 (per-lane byte offset in the output UAV).
    words.push(vop2(0x1A, 7, 0x80 + 2, 0)); // v_lshlrev_b32 v7, 2, v0
    words.extend_from_slice(&buffer_store_dword_offen(3, 7, 8));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 12,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs, lds_bytes = 0
            threads: [wave, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // WARP does not support fixed WaveSize, and 32-lane GPUs reject WaveSize(64).
            Err(e) if warp || wave64 => {
                eprintln!("{label}: skipped fixed wave size {wave} ({e})");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(wave)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        check(label, &out[0]);
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no SM 6.6 Wave{wave} device available; DXIL validation only");
    }
}

#[test]
fn ds_bpermute_b32_gathers_with_offset_and_wrap() {
    // Lane t (v0 = 0..31):
    //   v1 = 0x1000 + t          (source payload)
    //   v2 = ((31 - t) << 2) | (t & 3)  (reversed byte addr + low sub-dword bits)
    //   ds_bpermute_b32 v3, v2, v1 offset:12  (adds +3 lanes and wraps mod 32)
    let mut body = vec![
        vop2(0x25, 1, 0xFF, 0),      // v_add_nc_u32 v1, 0x1000, v0
        0x1000,                      // literal
        vop2(0x26, 2, 0x80 + 31, 0), // v_sub_nc_u32 v2, 31, v0
        vop2(0x1A, 2, 0x80 + 2, 2),  // v_lshlrev_b32 v2, 2, v2
        vop2(0x1B, 4, 0x80 + 3, 0),  // v_and_b32 v4, 3, v0
        vop2(0x1C, 2, vgpr(4), 2),   // v_or_b32 v2, v4, v2
    ];
    body.extend_from_slice(&ds_bpermute_b32(3, 2, 1, 12));

    run_wave_cs(false, &body, |backend, got| {
        for (t, &val) in got.iter().enumerate() {
            let src = ((31 - t as u32) + 3) & 31;
            let want = 0x1000 + src;
            assert_eq!(
                val, want,
                "{backend}: lane {t} got {val:#x}, want {want:#x} (from lane {src})",
            );
        }
    });
}

#[test]
fn ds_bpermute_b32_zeroes_inactive_source_and_preserves_inactive_dst() {
    // Lane t (v0 = 0..31):
    //   v1 = 0x2000 + t          (source payload)
    //   v3 = 0xDEAD_0000 + t     (pre-existing destination value)
    //   v2 = (t + 8) << 2        (pulls from lane (t + 8) & 31)
    // Narrow EXEC to t < 16:
    //   - lanes 0..7  pull from active lanes 8..15   -> 0x2000 + t + 8
    //   - lanes 8..15 pull from inactive lanes 16..23 -> 0
    //   - lanes 16..31 are inactive during bpermute  -> preserve 0xDEAD_0000 + t
    let mut body = vec![
        vop2(0x25, 1, 0xFF, 0),     // v_add_nc_u32 v1, 0x2000, v0
        0x2000,                     // literal
        vop2(0x25, 3, 0xFF, 0),     // v_add_nc_u32 v3, 0xDEAD_0000, v0
        0xDEAD_0000,                // literal
        vop2(0x25, 2, 0x80 + 8, 0), // v_add_nc_u32 v2, 8, v0
        vop2(0x1A, 2, 0x80 + 2, 2), // v_lshlrev_b32 v2, 2, v2
        vop1(0x001, 5, 0x80 + 16),  // v_mov_b32 v5, 16
        vopc(0xC1, vgpr(0), 5),     // v_cmp_lt_u32 vcc, v0, v5
        sop1(0x3C, 6, 0x6A),        // s_and_saveexec_b32 s6, vcc_lo
    ];
    body.extend_from_slice(&ds_bpermute_b32(3, 2, 1, 0));
    body.push(sop1(0x03, 0x7E, 6)); // s_mov_b32 exec_lo, s6

    run_wave_cs(false, &body, |backend, got| {
        for (t, &val) in got.iter().enumerate() {
            let t = t as u32;
            let want = if t < 8 {
                0x2000 + t + 8
            } else if t < 16 {
                0
            } else {
                0xDEAD_0000 + t
            };
            assert_eq!(
                val, want,
                "{backend}: lane {t} got {val:#010x}, want {want:#010x}",
            );
        }
    });
}

#[test]
fn ds_bpermute_b32_divergent_branch_zeroes_skipped_lanes() {
    // Same mask as above, but with `s_cbranch_execz` around `ds_bpermute_b32`
    // so structurisation emits a per-lane CFG branch rather than a guard.
    let mut body = vec![
        vop2(0x25, 1, 0xFF, 0),     // v_add_nc_u32 v1, 0x3000, v0
        0x3000,                     // literal
        vop2(0x25, 3, 0xFF, 0),     // v_add_nc_u32 v3, 0xBEEF_0000, v0
        0xBEEF_0000,                // literal
        vop2(0x25, 2, 0x80 + 8, 0), // v_add_nc_u32 v2, 8, v0
        vop2(0x1A, 2, 0x80 + 2, 2), // v_lshlrev_b32 v2, 2, v2
        vop1(0x001, 5, 0x80 + 16),  // v_mov_b32 v5, 16
        vopc(0xC1, vgpr(0), 5),     // v_cmp_lt_u32 vcc, v0, v5
        sop1(0x3C, 6, 0x6A),        // s_and_saveexec_b32 s6, vcc_lo
        0xBF88_0002,                // s_cbranch_execz +2 (skip the 2-dword DS op)
    ];
    body.extend_from_slice(&ds_bpermute_b32(3, 2, 1, 0));
    body.push(sop1(0x03, 0x7E, 6)); // s_mov_b32 exec_lo, s6

    run_wave_cs(false, &body, |backend, got| {
        for (t, &val) in got.iter().enumerate() {
            let t = t as u32;
            let want = if t < 8 {
                0x3000 + t + 8
            } else if t < 16 {
                0
            } else {
                0xBEEF_0000 + t
            };
            assert_eq!(
                val, want,
                "{backend}: lane {t} got {val:#010x}, want {want:#010x}",
            );
        }
    });
}

#[test]
fn ds_bpermute_b32_wave64_stays_within_each_32_lane_half() {
    // RDNA 2 wave64 permutes lanes 0..31 and 32..63 independently (section 10.4):
    //   v1 = 0x4000 + t
    //   v2 = (t & 31) << 2    (lane 32 passes byte addr 0)
    //   ds_bpermute_b32 v3, v2, v1 offset:12
    // Lower half wraps 29..31 -> 0..2; upper half gathers 35..63 and wraps 61..63 -> 32..34.
    let mut body = vec![
        vop2(0x25, 1, 0xFF, 0),      // v_add_nc_u32 v1, 0x4000, v0
        0x4000,                      // literal
        vop2(0x1B, 2, 0x80 + 31, 0), // v_and_b32 v2, 31, v0
        vop2(0x1A, 2, 0x80 + 2, 2),  // v_lshlrev_b32 v2, 2, v2
    ];
    body.extend_from_slice(&ds_bpermute_b32(3, 2, 1, 12));

    run_wave_cs(true, &body, |backend, got| {
        for (t, &val) in got.iter().enumerate() {
            let t = t as u32;
            let src = (t & !31) | ((t + 3) & 31);
            let want = 0x4000 + src;
            assert_eq!(
                val, want,
                "{backend}: wave64 lane {t} got {val:#x}, want {want:#x} (from lane {src})",
            );
        }
    });
}

#[test]
fn ds_bpermute_b32_wave64_exec_mask_per_half() {
    // Active lanes 0..39 in wave64:
    //   v1 = 0x5000 + t, v3 = 0xCAFE_0000 + t, v2 = ((t + 8) & 31) << 2
    //   - lanes 0..31:  all active, gather from lower half (t + 8) & 31
    //   - lanes 32..39: active, pull from upper-half lanes 40..47 (inactive -> 0,
    //                   even though lower-half lanes 8..15 are active)
    //   - lanes 40..63: inactive -> preserve 0xCAFE_0000 + t
    let mut body = vec![
        vop2(0x25, 1, 0xFF, 0),      // v_add_nc_u32 v1, 0x5000, v0
        0x5000,                      // literal
        vop2(0x25, 3, 0xFF, 0),      // v_add_nc_u32 v3, 0xCAFE_0000, v0
        0xCAFE_0000,                 // literal
        vop2(0x25, 2, 0x80 + 8, 0),  // v_add_nc_u32 v2, 8, v0
        vop2(0x1B, 2, 0x80 + 31, 2), // v_and_b32 v2, 31, v2
        vop2(0x1A, 2, 0x80 + 2, 2),  // v_lshlrev_b32 v2, 2, v2
        vop1(0x001, 5, 0x80 + 40),   // v_mov_b32 v5, 40
        vopc(0xC1, vgpr(0), 5),      // v_cmp_lt_u32 vcc, v0, v5
        sop1(0x24, 6, 0x6A),         // s_and_saveexec_b64 s[6:7], vcc
    ];
    body.extend_from_slice(&ds_bpermute_b32(3, 2, 1, 0));
    body.push(sop1(0x04, 0x7E, 6)); // s_mov_b64 exec, s[6:7]

    run_wave_cs(true, &body, |backend, got| {
        for (t, &val) in got.iter().enumerate() {
            let t = t as u32;
            let want = if t < 32 {
                0x5000 + ((t + 8) & 31)
            } else if t < 40 {
                0
            } else {
                0xCAFE_0000 + t
            };
            assert_eq!(
                val, want,
                "{backend}: wave64 lane {t} got {val:#010x}, want {want:#010x}",
            );
        }
    });
}

/// Runs VOP1 `op` on each input and compares the result dword.
fn check_cvt(name: &str, op: u16, cases: &[(u32, u32)]) {
    let shader = build_valu_cs(&[vop1(op, 0, vgpr(1))]);
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [c.0, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got,
            want,
            "{backend}: {name}({:#010x} = {:e}) = {got:#010x}, want {want:#010x}",
            inp[0],
            f32::from_bits(inp[0]),
        );
    });
}

#[test]
fn v_cvt_rpi_i32_f32_rounds_to_nearest() {
    // floor(x + 0.5) of the exact sum: an f32 sum rounds to even first, so
    // it is 1 too high just below 0.5 and for odd integers from 2^23 up.
    check_cvt(
        "v_cvt_rpi_i32_f32",
        0x00C,
        &[
            (0x0000_0000, 0),
            (0x3EFF_FFFF, 0), // 0.5 - 2^-25
            (0x3F00_0000, 1), // 0.5: ties go up
            (0xBF00_0000, 0), // -0.5
            (0xBF00_0001, 0xFFFF_FFFF),
            (0x3FC0_0000, 2),           // 1.5
            (0x4020_0000, 3),           // 2.5
            (0xBFC0_0000, 0xFFFF_FFFF), // -1.5
            (0xC020_0000, 0xFFFF_FFFE), // -2.5
            (0x0000_0001, 0),           // denormals, flushed or not
            (0x8000_0001, 0),
            (0x4A80_0001, 0x0040_0001), // 2^22 + 0.5
            (0x4B00_0001, 0x0080_0001), // 2^23 + 1
            (0x4B7F_FFFF, 0x00FF_FFFF), // 2^24 - 1
            (0xCB00_0001, 0xFF7F_FFFF), // -(2^23 + 1)
            (0xCB7F_FFFF, 0xFF00_0001),
            (0x4EFF_FFFF, 0x7FFF_FF80),
            (0x4F00_0000, 0x7FFF_FFFF), // 2^31 and up saturate
            (0x7F80_0000, 0x7FFF_FFFF),
            (0xCF00_0000, 0x8000_0000),
            (0xCF00_0001, 0x8000_0000),
            (0xFF80_0000, 0x8000_0000),
        ],
    );
}

#[test]
fn v_cvt_flr_i32_f32_rounds_down() {
    check_cvt(
        "v_cvt_flr_i32_f32",
        0x00D,
        &[
            (0x0000_0000, 0),
            (0x8000_0000, 0),
            (0x3FC0_0000, 1),           // 1.5
            (0xBFC0_0000, 0xFFFF_FFFE), // -1.5
            (0x3F7F_FFFF, 0),
            (0xBF80_0000, 0xFFFF_FFFF), // -1
            (0xB300_0000, 0xFFFF_FFFF), // -2^-25
            (0x4B7F_FFFF, 0x00FF_FFFF),
            (0x4F00_0000, 0x7FFF_FFFF),
            (0x7F80_0000, 0x7FFF_FFFF),
            (0xCF00_0001, 0x8000_0000),
            (0xFF80_0000, 0x8000_0000),
        ],
    );
}

#[test]
fn v_cvt_flr_and_rpi_i32_f32_saturate_nan_by_sign() {
    // Unlike v_cvt_i32_f32 (NaN gives 0), these saturate whatever has an
    // exponent above 30, a NaN too: to the largest or the smallest integer
    // by its sign.
    let cases = [
        (0x7FC0_0000, 0x7FFF_FFFF),
        (0x7F80_0001, 0x7FFF_FFFF),
        (0x7FFF_FFFF, 0x7FFF_FFFF),
        (0xFFC0_0000, 0x8000_0000),
        (0xFF80_0001, 0x8000_0000),
        (0xFFFF_FFFF, 0x8000_0000),
    ];
    check_cvt("v_cvt_rpi_i32_f32", 0x00C, &cases);
    check_cvt("v_cvt_flr_i32_f32", 0x00D, &cases);
    // v_cvt_i32_f32 keeps giving 0.
    check_cvt(
        "v_cvt_i32_f32",
        0x008,
        &[(0x7FC0_0000, 0), (0xFFC0_0000, 0), (0x7F80_0001, 0)],
    );
}

/// A 16-bit snorm (`signed`) or unorm of `x`, as the RDNA2 ISA reference
/// gives `v_cvt_pknorm_*`: clamped, scaled, rounded to nearest even from the
/// exact product (exact in f64), NaN as 0.
fn norm16_ref(x: f32, signed: bool) -> u16 {
    if x.is_nan() {
        return 0;
    }
    let (lo, k) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    let n = (f64::from(x.clamp(lo, 1.0)) * k).round_ties_even() as i32;
    n as u16
}

/// Inputs in [0.5, 1) and near 0.001 whose f32 product with the scale lands
/// exactly on a tie although the exact product does not: a translation that
/// multiplies in f32 rounds them the wrong way.
fn f32_tie_traps(signed: bool) -> Vec<f32> {
    let k: f32 = if signed { 32767.0 } else { 65535.0 };
    let mut traps = Vec::new();
    for start in [0.5f32, 0.001] {
        let first = start.to_bits();
        for b in first..first + 4_000_000 {
            let x = f32::from_bits(b);
            let p = x * k;
            if p.fract() == 0.5 && f64::from(x) * f64::from(k) != f64::from(p) {
                traps.push(x);
                if traps.len() % 3 == 0 {
                    break;
                }
            }
        }
    }
    traps
}

fn pknorm_f32_inputs(signed: bool) -> Vec<f32> {
    let mut xs = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        0.5,
        -0.5,
        0.25,
        1.0 / 3.0,
        -0.7,
        0.999_99,
        f32::MIN_POSITIVE,
        1e-6,
        -1e-6,
        f32::from_bits(1), // denormal
        1.0 / 65535.0,
        0.5 / 65535.0,
        1.5 / 32767.0,
        f32::from_bits(0x3F7F_FFFF), // just below 1
    ];
    xs.extend(f32_tie_traps(signed));
    xs
}

#[test]
fn v_cvt_pknorm_f32_rounds_the_exact_product() {
    for (op, signed, what) in [
        (0x368u16, true, "v_cvt_pknorm_i16_f32"),
        (0x369, false, "v_cvt_pknorm_u16_f32"),
    ] {
        // The traps must tell an f32 product apart from the exact one.
        let k: f32 = if signed { 32767.0 } else { 65535.0 };
        let f32_product = |x: f32| (x * k).round_ties_even() as i32 as u16;
        let traps = f32_tie_traps(signed);
        assert!(
            traps
                .iter()
                .any(|&x| f32_product(x) != norm16_ref(x, signed)),
            "{what}: no input separates an f32 product from the exact one"
        );
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), 0));
        let xs = pknorm_f32_inputs(signed);
        // Each input once in each half, paired with its neighbour.
        let cases: Vec<[u32; 4]> = (0..xs.len())
            .map(|i| [xs[i].to_bits(), xs[(i + 1) % xs.len()].to_bits(), 0, 0])
            .collect();
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let (a, b) = (f32::from_bits(inp[0]), f32::from_bits(inp[1]));
            let want = u32::from(norm16_ref(a, signed)) | u32::from(norm16_ref(b, signed)) << 16;
            assert_eq!(
                got, want,
                "{backend}: {what}({a:e} [{:#010x}], {b:e} [{:#010x}]) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_cvt_pknorm_f16_rounds_the_exact_product() {
    let halves: Vec<u16> = vec![
        0x0000, 0x8000, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x7C00, 0xFC00, 0x7E00, 0xFE00,
        0x3800, // 0.5: a tie in both scales
        0xB800, 0x3555, 0x0001, // smallest denormal
        0x8001, 0x0400, // smallest normal
        0x3BFF, // just below 1
        0x1001, 0x2E66, 0xB266, 0x3A00, 0x3401,
    ];
    for (op, signed, what) in [
        (0x312u16, true, "v_cvt_pknorm_i16_f16"),
        (0x313, false, "v_cvt_pknorm_u16_f16"),
    ] {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), 0));
        let cases: Vec<[u32; 4]> = (0..halves.len())
            .map(|i| {
                let next = halves[(i + 1) % halves.len()];
                // Junk in the high halves: only the low halves are read.
                [
                    u32::from(halves[i]) | 0xABCD_0000,
                    u32::from(next) | 0x1234_0000,
                    0,
                    0,
                ]
            })
            .collect();
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let (a, b) = (inp[0] as u16, inp[1] as u16);
            let (fa, fb) = (f16_to_f32(a), f16_to_f32(b));
            let want = u32::from(norm16_ref(fa, signed)) | u32::from(norm16_ref(fb, signed)) << 16;
            assert_eq!(
                got, want,
                "{backend}: {what}({a:#06x}, {b:#06x}) = {got:#010x}, want {want:#010x}",
            );
        });
    }
}
