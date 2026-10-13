// lowlight-dn  --  GPU low-light video denoiser
//
//   decode (ffmpeg, 16-bit YCbCr 4:4:4) -> GPU:
//     1. noise profile   (motion-compensated temporal residuals, per channel/level/brightness)
//     2. motion search   (1/4-res block -> full-res refine -> 1/4 pel)
//     3. temporal fusion (patch-similarity weighted, 2N+1 frames)
//     4. spatial wavelet shrinkage (a-trous B3, thresholds from the profile)
//   -> encode (ffmpeg: ProRes / H.265 / H.264) + original audio
//
// Algorithm is identical to reference/ref_denoise.py (validated there).

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

const NB: usize = 16; // brightness bins
const LV: usize = 6; // profile levels: 0 = pixel domain, 1..5 = wavelet details
const MAX_NEIGH: usize = 16;
const ENT: u64 = 256; // uniform dynamic-offset stride
const LOAD_ENT: u32 = 120; // uniform slots used by frame-load submits

// pipeline indices (order must match ENTRY_POINTS)
const PIPE_UNPACK: usize = 0;
const PIPE_DOWN4: usize = 1;
const PIPE_ME_COARSE: usize = 2;
const PIPE_ME_REFINE: usize = 3;
const PIPE_ME_SUBPEL: usize = 4;
const PIPE_TEMPORAL: usize = 5;
const PIPE_ATROUS_H: usize = 6;
const PIPE_ATROUS_V: usize = 7;
const PIPE_SHRINK: usize = 8;
const PIPE_PACK: usize = 9;
const PIPE_RESIDUAL: usize = 10;
const ENTRY_POINTS: [&str; 11] = [
    "k_unpack",
    "k_down4",
    "k_me_coarse",
    "k_me_refine",
    "k_me_subpel",
    "k_temporal",
    "k_atrous_h",
    "k_atrous_v",
    "k_shrink",
    "k_pack",
    "k_residual",
];

#[derive(Parser, Debug)]
#[command(
    name = "lowlight-dn",
    version,
    about = "GPU low-light video denoiser: automatic noise profile + motion-compensated temporal + wavelet spatial"
)]
struct Args {
    /// input video
    #[arg(short, long)]
    input: Option<String>,
    /// output file (default: <input>_dn.mov)
    #[arg(short, long)]
    output: Option<String>,
    /// frames on each side used for temporal denoising (0 = off, max 8)
    #[arg(long, default_value_t = 3)]
    temporal_frames: usize,
    /// temporal tolerance: higher = averages more aggressively (risk of ghosting)
    #[arg(long, default_value_t = 2.0)]
    tau: f32,
    /// spatial strength multiplier (0 = off)
    #[arg(long, default_value_t = 1.0)]
    spatial: f32,
    /// extra multiplier for chroma (colour) noise
    #[arg(long, default_value_t = 1.0)]
    chroma: f32,
    /// multiplier on the measured noise level (use >1 if noise remains, <1 if too smooth)
    #[arg(long, default_value_t = 1.0)]
    noise_scale: f32,
    /// number of frame pairs used for the noise profile
    #[arg(long, default_value_t = 4)]
    profile_pairs: usize,
    /// GPU index (see --list-gpus). Default: the most powerful GPU
    #[arg(long)]
    gpu: Option<usize>,
    /// list available GPUs and exit
    #[arg(long)]
    list_gpus: bool,
    /// prores | prores444 | h265 | h264
    #[arg(long, default_value = "prores")]
    codec: String,
}

// ------------------------------------------------------------------ uniform block
#[derive(Clone, Copy, Default)]
struct P {
    w: u32,
    h: u32,
    bw: u32,
    bh: u32,
    w4: u32,
    h4: u32,
    vec_off: u32,
    cslot: u32,
    nslot: u32,
    level: u32,
    nn: u32,
    a: f32,
    b: f32,
    ns: [u32; MAX_NEIGH],
}

impl P {
    fn words(&self) -> [u32; 32] {
        let mut o = [0u32; 32];
        o[0] = self.w;
        o[1] = self.h;
        o[2] = self.bw;
        o[3] = self.bh;
        o[4] = self.w4;
        o[5] = self.h4;
        o[6] = self.vec_off;
        o[7] = self.cslot;
        o[8] = self.nslot;
        o[9] = self.level;
        o[10] = self.nn;
        o[11] = 0;
        o[12] = self.a.to_bits();
        o[13] = self.b.to_bits();
        o[14] = 0f32.to_bits();
        o[15] = 0f32.to_bits();
        for i in 0..MAX_NEIGH {
            o[16 + i] = self.ns[i];
        }
        o
    }
}

#[derive(Default)]
struct Batch {
    items: Vec<(usize, P, [u32; 3])>,
}
impl Batch {
    fn add(&mut self, pipe: usize, p: P, dims: [u32; 3]) {
        self.items.push((pipe, p, dims));
    }
}

fn ceil_div(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
fn dims_1d(count: u32) -> [u32; 3] {
    let groups = ceil_div(count, 256);
    let gx = groups.min(32768).max(1);
    let gy = ceil_div(groups, gx).max(1);
    [gx, gy, 1]
}

// ------------------------------------------------------------------ GPU context
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    raw_in: wgpu::Buffer,
    raw_out: wgpu::Buffer,
    work: wgpu::Buffer,
    sigma: wgpu::Buffer,
    readback: wgpu::Buffer,
    w: u32,
    h: u32,
    bw: u32,
    bh: u32,
    w4: u32,
    h4: u32,
    slots: usize,
}

impl Gpu {
    fn new(args: &Args, w: u32, h: u32, slots: usize) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = if let Some(i) = args.gpu {
            instance
                .enumerate_adapters(wgpu::Backends::PRIMARY)
                .into_iter()
                .nth(i)
                .context("no GPU with that index (use --list-gpus)")?
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .context("no GPU found (Vulkan / DX12 / Metal driver missing?)")?
        };
        let info = adapter.get_info();
        eprintln!("GPU: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("lowlight-dn"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .context("could not open the GPU device")?;

        let n = w as u64 * h as u64;
        let bw = ceil_div(w, 16);
        let bh = ceil_div(h, 16);
        let w4 = ceil_div(w, 4);
        let h4 = ceil_div(h, 4);

        let frames_bytes = slots as u64 * 3 * n * 4;
        let work_bytes = 25 * n * 4;
        let max_bind = device.limits().max_storage_buffer_binding_size as u64;
        let max_buf = device.limits().max_buffer_size;
        let biggest = frames_bytes.max(work_bytes);
        if biggest > max_bind || biggest > max_buf {
            bail!(
                "resolution too large for this GPU: needs a {} MB buffer, limit is {} MB. Lower --temporal-frames or downscale.",
                biggest / 1_048_576,
                max_bind.min(max_buf) / 1_048_576
            );
        }

        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let st = wgpu::BufferUsages::STORAGE;
        let cd = wgpu::BufferUsages::COPY_DST;
        let cs = wgpu::BufferUsages::COPY_SRC;
        let params = mk("params", 128 * ENT, wgpu::BufferUsages::UNIFORM | cd);
        let raw_in = mk("raw_in", 6 * n, st | cd);
        let frames = mk("frames", frames_bytes, st);
        let luma4 = mk("luma4", slots as u64 * w4 as u64 * h4 as u64 * 4, st);
        let vecs = mk("vecs", MAX_NEIGH as u64 * bw as u64 * bh as u64 * 8, st);
        let work = mk("work", work_bytes, st | cs | cd);
        let sigma = mk("sigma", (3 * LV * NB * 4) as u64, st | cd);
        let raw_out = mk("raw_out", 6 * n, st | cs);
        let readback = mk(
            "readback",
            12 * n,
            wgpu::BufferUsages::MAP_READ | cd,
        );

        // ---- bind group layout (shared by every kernel)
        let sto = |binding: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(128),
                    },
                    count: None,
                },
                sto(1, true),
                sto(2, false),
                sto(3, false),
                sto(4, false),
                sto(5, false),
                sto(6, true),
                sto(7, false),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: wgpu::BufferSize::new(128),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: raw_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: frames.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: luma4.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: vecs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: work.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: sigma.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: raw_out.as_entire_binding() },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });
        let mut pipelines = Vec::new();
        for &name in ENTRY_POINTS.iter() {
            pipelines.push(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: name,
                compilation_options: Default::default(),
                cache: None,
            }));
        }

        Ok(Gpu {
            device,
            queue,
            pipelines,
            bind_group,
            params,
            raw_in,
            raw_out,
            work,
            sigma,
            readback,
            w,
            h,
            bw,
            bh,
            w4,
            h4,
            slots,
        })
    }

    fn p0(&self) -> P {
        P {
            w: self.w,
            h: self.h,
            bw: self.bw,
            bh: self.bh,
            w4: self.w4,
            h4: self.h4,
            ..Default::default()
        }
    }

    fn n(&self) -> u32 {
        self.w * self.h
    }

    fn submit(&self, b: &Batch, ent_base: u32, copy: Option<(&wgpu::Buffer, u64, u64)>) {
        for (i, (_, p, _)) in b.items.iter().enumerate() {
            let words = p.words();
            self.queue.write_buffer(
                &self.params,
                (ent_base as u64 + i as u64) * ENT,
                bytemuck::cast_slice(&words[..]),
            );
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for (i, (pipe, _, d)) in b.items.iter().enumerate() {
                pass.set_pipeline(&self.pipelines[*pipe]);
                let off = ((ent_base as u64 + i as u64) * ENT) as u32;
                pass.set_bind_group(0, &self.bind_group, &[off]);
                pass.dispatch_workgroups(d[0], d[1], d[2]);
            }
        }
        if let Some((src, off, size)) = copy {
            enc.copy_buffer_to_buffer(src, off, &self.readback, 0, size);
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn read_back(&self, size: u64) -> Vec<u8> {
        let slice = self.readback.slice(0..size);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("GPU buffer map failed");
        let data = slice.get_mapped_range().to_vec();
        self.readback.unmap();
        data
    }

    fn load_frame(&self, bytes: &[u8], slot: usize) {
        self.queue.write_buffer(&self.raw_in, 0, bytes);
        let mut b = Batch::default();
        let mut p = self.p0();
        p.cslot = slot as u32;
        b.add(PIPE_UNPACK, p, dims_1d(3 * self.n()));
        b.add(PIPE_DOWN4, p, [ceil_div(self.w4, 8), ceil_div(self.h4, 8), 1]);
        self.submit(&b, LOAD_ENT, None);
    }

    fn set_sigma(&self, lut: &[f32]) {
        self.queue.write_buffer(&self.sigma, 0, bytemuck::cast_slice(lut));
    }

    fn atrous_passes(&self, b: &mut Batch) {
        let dims = [ceil_div(self.w, 16), ceil_div(self.h, 16), 3];
        for j in 1..=5u32 {
            let mut p = self.p0();
            p.level = j;
            b.add(PIPE_ATROUS_H, p, dims);
            b.add(PIPE_ATROUS_V, p, dims);
        }
    }

    fn motion_passes(&self, b: &mut Batch, cslot: u32, nslot: u32, k: u32) {
        let mut p = self.p0();
        p.cslot = cslot;
        p.nslot = nslot;
        p.vec_off = k * self.bw * self.bh;
        let d = [ceil_div(self.bw, 8), ceil_div(self.bh, 8), 1];
        b.add(PIPE_ME_COARSE, p, d);
        b.add(PIPE_ME_REFINE, p, d);
        b.add(PIPE_ME_SUBPEL, p, d);
    }

    /// full denoise of centre frame `c` (frame indices 0..=last are on the GPU ring)
    fn process(&self, c: usize, last: usize, nwin: usize, a: &Args) -> Vec<u8> {
        let mut ns: Vec<u32> = Vec::new();
        let r = nwin as isize;
        for d in -r..=r {
            if d == 0 {
                continue;
            }
            let idx = c as isize + d;
            if idx >= 0 && (idx as usize) <= last {
                ns.push((idx as usize % self.slots) as u32);
            }
        }
        let cs = (c % self.slots) as u32;
        let mut b = Batch::default();
        for (k, s) in ns.iter().enumerate() {
            self.motion_passes(&mut b, cs, *s, k as u32);
        }
        let mut p = self.p0();
        p.cslot = cs;
        p.nn = ns.len() as u32;
        p.a = 1.0; // noise_scale is already applied to the sigma LUT
        p.b = a.tau.max(0.05);
        for (i, s) in ns.iter().enumerate() {
            p.ns[i] = *s;
        }
        b.add(PIPE_TEMPORAL, p, [ceil_div(self.w, 16), ceil_div(self.h, 16), 1]);
        self.atrous_passes(&mut b);
        let mut p = self.p0();
        p.a = a.spatial;
        p.b = a.chroma;
        b.add(PIPE_SHRINK, p, [ceil_div(self.w, 16), ceil_div(self.h, 16), 3]);
        b.add(PIPE_PACK, self.p0(), dims_1d(3 * self.n() / 2));
        let size = 6 * self.n() as u64;
        self.submit(&b, 0, Some((&self.raw_out, 0, size)));
        self.read_back(size)
    }
}

// ------------------------------------------------------------------ noise profile
fn quantile(v: &[f32], q: f32) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[((s.len() - 1) as f32 * q) as usize]
}

fn sm_base(n: u64, j: u64) -> u64 {
    if j == 0 {
        0
    } else {
        4 * n + (j - 1) * 3 * n
    }
}

fn accumulate(
    acc: &mut [Vec<f32>],
    plane: &[f32],
    ch: usize,
    lvl: usize,
    w: usize,
    bins: &[usize],
    bwc: usize,
    bhc: usize,
) {
    let mut buf = vec![0f32; 1024];
    for by in 0..bhc {
        for bx in 0..bwc {
            let mut k = 0;
            for y in 0..32 {
                let row = (by * 32 + y) * w + bx * 32;
                for x in 0..32 {
                    buf[k] = plane[row + x].abs();
                    k += 1;
                }
            }
            let (_, m, _) = buf.select_nth_unstable_by(512, |a, b| a.total_cmp(b));
            let sigma = 1.4826 * *m;
            if sigma > 1e-5 {
                acc[(ch * LV + lvl) * NB + bins[by * bwc + bx]].push(sigma);
            }
        }
    }
}

fn finalize_profile(acc: &[Vec<f32>], corr: f32) -> Vec<f32> {
    let mut lut = vec![0f32; 3 * LV * NB];
    for c in 0..3 {
        for l in 0..LV {
            let mut vals = [f32::NAN; NB];
            let mut pooled: Vec<f32> = Vec::new();
            for b in 0..NB {
                let v = &acc[(c * LV + l) * NB + b];
                pooled.extend_from_slice(v);
                if v.len() >= 6 {
                    vals[b] = quantile(v, 0.25) * corr;
                }
            }
            let valid: Vec<usize> = (0..NB).filter(|&b| !vals[b].is_nan()).collect();
            if valid.is_empty() {
                let fb = if pooled.is_empty() { 0.02 } else { quantile(&pooled, 0.25) * corr };
                for b in 0..NB {
                    vals[b] = fb;
                }
            } else {
                let mut filled = [0f32; NB];
                for b in 0..NB {
                    if !vals[b].is_nan() {
                        filled[b] = vals[b];
                        continue;
                    }
                    let lo = valid.iter().rev().find(|&&v| v < b).copied();
                    let hi = valid.iter().find(|&&v| v > b).copied();
                    filled[b] = match (lo, hi) {
                        (Some(l0), Some(h0)) => {
                            let t = (b - l0) as f32 / (h0 - l0) as f32;
                            vals[l0] * (1.0 - t) + vals[h0] * t
                        }
                        (Some(l0), None) => vals[l0],
                        (None, Some(h0)) => vals[h0],
                        _ => 0.02,
                    };
                }
                vals = filled;
            }
            for b in 0..NB {
                lut[(c * LV + l) * NB + b] = vals[b];
            }
        }
    }
    lut
}

fn run_profile(gpu: &Gpu, frames: &[Vec<u8>], pairs: usize) -> Result<Vec<f32>> {
    let w = gpu.w as usize;
    let h = gpu.h as usize;
    let n = w * h;
    let bwc = w / 32;
    let bhc = h / 32;
    if bwc == 0 || bhc == 0 {
        bail!("video is too small to profile the noise");
    }
    let npairs = pairs.min(frames.len().saturating_sub(1)).max(1);
    let mut acc: Vec<Vec<f32>> = vec![Vec::new(); 3 * LV * NB];

    for pi in 0..npairs {
        let (a, b) = (pi, pi + 1);
        gpu.load_frame(&frames[a], 0);
        gpu.load_frame(&frames[b], 1);

        // block-mean luma of the centre frame -> brightness bin per 32x32 block
        let ybytes = &frames[a][..2 * n];
        let mut bins = vec![0usize; bwc * bhc];
        for by in 0..bhc {
            for bx in 0..bwc {
                let mut s = 0f64;
                for y in 0..32 {
                    let row = (by * 32 + y) * w + bx * 32;
                    for x in 0..32 {
                        let i = (row + x) * 2;
                        s += u16::from_le_bytes([ybytes[i], ybytes[i + 1]]) as f64;
                    }
                }
                let mean = (s / 1024.0 / 65535.0) as f32;
                bins[by * bwc + bx] = ((mean * NB as f32) as usize).min(NB - 1);
            }
        }

        let mut batch = Batch::default();
        gpu.motion_passes(&mut batch, 0, 1, 0);
        let mut p = gpu.p0();
        p.cslot = 0;
        p.nslot = 1;
        p.vec_off = 0;
        batch.add(
            PIPE_RESIDUAL,
            p,
            [ceil_div(gpu.w, 16), ceil_div(gpu.h, 16), 1],
        );
        gpu.atrous_passes(&mut batch);
        gpu.submit(&batch, 0, None);

        let mut prev: Vec<f32> = Vec::new();
        for j in 0..LV as u64 {
            let off = sm_base(n as u64, j) * 4;
            let size = 3 * n as u64 * 4;
            let empty = Batch::default(); // no dispatches: just copy this smooth level to the readback buffer
            gpu.submit(&empty, 0, Some((&gpu.work, off, size)));
            let bytes = gpu.read_back(size);
            let cur: Vec<f32> = bytemuck::pod_collect_to_vec::<u8, f32>(&bytes);
            for ch in 0..3 {
                let pl = &cur[ch * n..(ch + 1) * n];
                if j == 0 {
                    accumulate(&mut acc, pl, ch, 0, w, &bins, bwc, bhc);
                } else {
                    let pv = &prev[ch * n..(ch + 1) * n];
                    let det: Vec<f32> = pv.iter().zip(pl.iter()).map(|(a, b)| a - b).collect();
                    accumulate(&mut acc, &det, ch, j as usize, w, &bins, bwc, bhc);
                }
            }
            prev = cur;
        }
        eprintln!("  noise profile: pair {}/{}", pi + 1, npairs);
    }
    Ok(finalize_profile(&acc, 1.4))
}

// ------------------------------------------------------------------ ffmpeg helpers
struct VideoInfo {
    w: u32,
    h: u32,
    fps: String,
    fps_f: f64,
    frames: Option<u64>,
}

fn probe(input: &str) -> Result<VideoInfo> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:stream_side_data=rotation:stream_tags=rotate",
            "-of",
            "json",
            input,
        ])
        .output()
        .context("ffprobe not found - install ffmpeg and add it to PATH")?;
    if !out.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let s = &v["streams"][0];
    let mut w = s["width"].as_u64().context("no video stream")? as u32;
    let mut h = s["height"].as_u64().context("no video stream")? as u32;
    let mut rot: i64 = 0;
    if let Some(r) = s["tags"]["rotate"].as_str() {
        rot = r.parse().unwrap_or(0);
    }
    if let Some(list) = s["side_data_list"].as_array() {
        for sd in list {
            if let Some(r) = sd["rotation"].as_i64() {
                rot = r;
            }
        }
    }
    if rot.abs() % 180 == 90 {
        std::mem::swap(&mut w, &mut h);
    }
    let mut fps = s["avg_frame_rate"].as_str().unwrap_or("0/0").to_string();
    if fps.starts_with("0/") || fps == "0" {
        fps = s["r_frame_rate"].as_str().unwrap_or("30/1").to_string();
    }
    let parts: Vec<&str> = fps.split('/').collect();
    let num: f64 = parts[0].parse().unwrap_or(30.0);
    let den: f64 = if parts.len() > 1 { parts[1].parse().unwrap_or(1.0) } else { 1.0 };
    let fps_f = if den > 0.0 { num / den } else { 30.0 };
    let frames = s["nb_frames"].as_str().and_then(|x| x.parse::<u64>().ok());
    Ok(VideoInfo { w, h, fps, fps_f, frames })
}

fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut got = 0;
    while got < buf.len() {
        let k = r.read(&mut buf[got..])?;
        if k == 0 {
            return Ok(false);
        }
        got += k;
    }
    Ok(true)
}

fn codec_args(codec: &str) -> Result<Vec<String>> {
    let v: Vec<&str> = match codec {
        "prores" => vec!["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"],
        "prores444" => vec!["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le", "-vendor", "apl0"],
        "h265" => vec!["-c:v", "libx265", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"],
        "h264" => vec!["-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p"],
        other => bail!("unknown codec '{}': use prores | prores444 | h265 | h264", other),
    };
    Ok(v.into_iter().map(|s| s.to_string()).collect())
}

// ------------------------------------------------------------------ main
fn main() -> Result<()> {
    let args = Args::parse();

    if args.list_gpus {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        for (i, a) in instance.enumerate_adapters(wgpu::Backends::PRIMARY).iter().enumerate() {
            let info = a.get_info();
            println!("[{}] {}  ({:?}, {:?})", i, info.name, info.device_type, info.backend);
        }
        return Ok(());
    }

    let input = args.input.clone().context("missing -i <input video>")?;
    let output = args.output.clone().unwrap_or_else(|| {
        let stem = std::path::Path::new(&input)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "out".into());
        let ext = if args.codec.starts_with("prores") { "mov" } else { "mp4" };
        format!("{}_dn.{}", stem, ext)
    });
    let nwin = args.temporal_frames.min(8);
    let info = probe(&input)?;
    if (info.w * info.h) % 2 != 0 {
        bail!("frame width x height must be even");
    }
    eprintln!("video: {}x{} @ {} fps", info.w, info.h, info.fps);

    let slots = 2 * nwin + 1;
    let gpu = Gpu::new(&args, info.w, info.h, slots.max(2))?;
    let frame_bytes = (6 * info.w as u64 * info.h as u64) as usize;

    // ---- decoder thread
    let mut dec = Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", &input, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "yuv444p16le", "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("ffmpeg not found - install it and add it to PATH")?;
    let mut dec_out = dec.stdout.take().unwrap();
    let (dtx, drx) = mpsc::sync_channel::<Vec<u8>>(4);
    let dec_thread = thread::spawn(move || {
        loop {
            let mut buf = vec![0u8; frame_bytes];
            match read_full(&mut dec_out, &mut buf) {
                Ok(true) => {
                    if dtx.send(buf).is_err() {
                        break;
                    }
                }
                _ => break,
            }
        }
        let _ = dec.wait();
    });

    // ---- buffer the first frames for the noise profile
    let want = args.profile_pairs.max(1) + 1;
    let mut pending: VecDeque<Vec<u8>> = VecDeque::new();
    while pending.len() < want {
        match drx.recv() {
            Ok(f) => pending.push_back(f),
            Err(_) => break,
        }
    }
    if pending.len() < 2 {
        bail!("video has fewer than 2 frames");
    }
    eprintln!("measuring noise profile ...");
    let t_prof = Instant::now();
    let pf: Vec<Vec<u8>> = pending.iter().cloned().collect();
    let mut lut = run_profile(&gpu, &pf, args.profile_pairs)?;
    for v in lut.iter_mut() {
        *v *= args.noise_scale;
    }
    {
        let b = NB / 3;
        let l: Vec<String> = (0..LV).map(|k| format!("{:.4}", lut[k * NB + b])).collect();
        let cb: Vec<String> = (0..LV).map(|k| format!("{:.4}", lut[(LV + k) * NB + b])).collect();
        eprintln!("  luma   sigma [pixel, L1..L5] = {}", l.join("  "));
        eprintln!("  chroma sigma [pixel, L1..L5] = {}   ({:.1}s)", cb.join("  "), t_prof.elapsed().as_secs_f32());
    }
    gpu.set_sigma(&lut);

    // ---- encoder thread
    let mut eargs: Vec<String> = vec![
        "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv444p16le",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();
    eargs.push("-s".into());
    eargs.push(format!("{}x{}", info.w, info.h));
    eargs.push("-framerate".into());
    eargs.push(info.fps.clone());
    for s in ["-i", "-", "-i"] {
        eargs.push(s.to_string());
    }
    eargs.push(input.clone());
    for s in ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"] {
        eargs.push(s.to_string());
    }
    eargs.extend(codec_args(&args.codec)?);
    eargs.push(output.clone());
    let mut enc = Command::new("ffmpeg")
        .args(&eargs)
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("could not start the ffmpeg encoder")?;
    let enc_in = enc.stdin.take().unwrap();
    let (etx, erx) = mpsc::sync_channel::<Vec<u8>>(4);
    let enc_thread = thread::spawn(move || {
        let mut sin = enc_in;
        for f in erx {
            if sin.write_all(&f).is_err() {
                break;
            }
        }
        drop(sin);
        enc.wait()
    });

    // ---- main loop
    let t0 = Instant::now();
    let mut loaded: usize = 0;
    let mut done: usize = 0;
    let total = info.frames;
    let mut next_frame = || -> Option<Vec<u8>> {
        if let Some(f) = pending.pop_front() {
            Some(f)
        } else {
            drx.recv().ok()
        }
    };
    let mut emit = |bytes: Vec<u8>, done: &mut usize| {
        let _ = etx.send(bytes);
        *done += 1;
        if *done % 10 == 0 {
            let fps = *done as f64 / t0.elapsed().as_secs_f64();
            match total {
                Some(t) => eprint!("\r  frame {}/{}   {:.1} fps   ", done, t, fps),
                None => eprint!("\r  frame {}   {:.1} fps   ", done, fps),
            }
        }
    };

    while let Some(bytes) = next_frame() {
        gpu.load_frame(&bytes, loaded % slots.max(2));
        loaded += 1;
        let last = loaded - 1;
        if last >= nwin {
            let c = last - nwin;
            let out = gpu.process(c, last, nwin, &args);
            emit(out, &mut done);
        }
    }
    if loaded == 0 {
        bail!("no frames decoded");
    }
    let last = loaded - 1;
    let first_left = if loaded > nwin { loaded - nwin } else { 0 };
    for c in first_left..loaded {
        let out = gpu.process(c, last, nwin, &args);
        emit(out, &mut done);
    }
    eprintln!();

    drop(etx);
    let st = enc_thread.join().map_err(|_| anyhow::anyhow!("encoder thread panicked"))??;
    let _ = dec_thread.join();
    if !st.success() {
        bail!("ffmpeg encoder reported an error");
    }
    eprintln!(
        "done: {} frames in {:.1}s ({:.1} fps, source {:.2} fps) -> {}",
        done,
        t0.elapsed().as_secs_f32(),
        done as f64 / t0.elapsed().as_secs_f64(),
        info.fps_f,
        output
    );
    Ok(())
}
