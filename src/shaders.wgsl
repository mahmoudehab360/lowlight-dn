// ============================================================================
//  lowlight-dn : GPU kernels
//  Planar float frames (Y, Cb, Cr in 0..1). Same algorithm as reference/ref_denoise.py
// ============================================================================

struct P {
    w: u32, h: u32, bw: u32, bh: u32,
    w4: u32, h4: u32, vec_off: u32, cslot: u32,
    nslot: u32, level: u32, nn: u32, pad0: u32,
    a: f32, b: f32, c: f32, d: f32,
    ns: array<vec4<u32>, 4>,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> raw_in: array<u32>;          // packed u16 pairs, planar Y|Cb|Cr
@group(0) @binding(2) var<storage, read_write> frames: array<f32>;    // [slot][ch][pixel]
@group(0) @binding(3) var<storage, read_write> luma4: array<f32>;     // [slot][w4*h4]
@group(0) @binding(4) var<storage, read_write> vecs: array<f32>;      // [k][block][2]
@group(0) @binding(5) var<storage, read_write> work: array<f32>;      // see layout below
@group(0) @binding(6) var<storage, read> sigma: array<f32>;           // [ch 3][level 6][bin 16]
@group(0) @binding(7) var<storage, read_write> raw_out: array<u32>;

// work layout (n = w*h floats per plane):
//   [0   , 3n )  fused image  == a-trous smooth level 0
//   [3n  , 4n )  neff (temporal residual-noise factor)
//   [4n  , 19n)  smooth levels 1..5 (3n each)
//   [19n , 22n)  tmp (separable pass)
//   [22n , 25n)  final output

const NB: u32 = 16u;

fn npx() -> u32 { return p.w * p.h; }
fn fidx(slot: u32, ch: u32, x: u32, y: u32) -> u32 { return (slot * 3u + ch) * npx() + y * p.w + x; }
fn cl(v: i32, hi: u32) -> u32 { return u32(clamp(v, 0, i32(hi) - 1)); }
fn sm_base(j: u32) -> u32 {
    let n = npx();
    if (j == 0u) { return 0u; }
    return 4u * n + (j - 1u) * 3u * n;
}
fn tmp_base() -> u32 { return 19u * npx(); }
fn out_base() -> u32 { return 22u * npx(); }

fn get_ns(k: u32) -> u32 {
    let q = p.ns[k >> 2u];
    let r = k & 3u;
    if (r == 0u) { return q.x; }
    if (r == 1u) { return q.y; }
    if (r == 2u) { return q.z; }
    return q.w;
}

fn bl(slot: u32, ch: u32, fx: f32, fy: f32) -> f32 {
    let sx = clamp(fx, 0.0, f32(p.w) - 1.0);
    let sy = clamp(fy, 0.0, f32(p.h) - 1.0);
    let x0 = u32(floor(sx));
    let y0 = u32(floor(sy));
    let x1 = min(x0 + 1u, p.w - 1u);
    let y1 = min(y0 + 1u, p.h - 1u);
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);
    let base = (slot * 3u + ch) * npx();
    let v00 = frames[base + y0 * p.w + x0];
    let v10 = frames[base + y0 * p.w + x1];
    let v01 = frames[base + y1 * p.w + x0];
    let v11 = frames[base + y1 * p.w + x1];
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

fn sig(ch: u32, lvl: u32, bright: f32) -> f32 {
    let pos = clamp(bright * f32(NB) - 0.5, 0.0, f32(NB) - 1.0);
    let i0 = u32(floor(pos));
    let i1 = min(i0 + 1u, NB - 1u);
    let t = pos - f32(i0);
    let base = (ch * 6u + lvl) * NB;
    return mix(sigma[base + i0], sigma[base + i1], t);
}

// ---------------------------------------------------------------- load
@compute @workgroup_size(256)
fn k_unpack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    let total = 3u * npx();
    if (i >= total) { return; }
    let word = raw_in[i >> 1u];
    var v: u32;
    if ((i & 1u) == 0u) { v = word & 0xFFFFu; } else { v = word >> 16u; }
    frames[p.cslot * total + i] = f32(v) / 65535.0;
}

@compute @workgroup_size(8, 8)
fn k_down4(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w4 || gid.y >= p.h4) { return; }
    var s = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let x = min(gid.x * 4u + i, p.w - 1u);
            let y = min(gid.y * 4u + j, p.h - 1u);
            s = s + frames[fidx(p.cslot, 0u, x, y)];
        }
    }
    luma4[p.cslot * p.w4 * p.h4 + gid.y * p.w4 + gid.x] = s / 16.0;
}

// ---------------------------------------------------------------- motion
// 1) coarse: 1/4-res, 8x8 window per 16x16 block, +-6 px (= +-24 px full-res)
@compute @workgroup_size(8, 8)
fn k_me_coarse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    if (bx >= p.bw || by >= p.bh) { return; }
    let base_c = p.cslot * p.w4 * p.h4;
    let base_n = p.nslot * p.w4 * p.h4;
    var best = 1.0e30;
    var bvx = 0;
    var bvy = 0;
    for (var dy = -6; dy <= 6; dy = dy + 1) {
        for (var dx = -6; dx <= 6; dx = dx + 1) {
            var sad = 0.0;
            for (var j = -2; j < 6; j = j + 1) {
                let y = clamp(i32(by) * 4 + j, 0, i32(p.h4) - 1);
                let yn = clamp(y + dy, 0, i32(p.h4) - 1);
                for (var i = -2; i < 6; i = i + 1) {
                    let x = clamp(i32(bx) * 4 + i, 0, i32(p.w4) - 1);
                    let xn = clamp(x + dx, 0, i32(p.w4) - 1);
                    sad = sad + abs(luma4[base_c + u32(y) * p.w4 + u32(x)] - luma4[base_n + u32(yn) * p.w4 + u32(xn)]);
                }
            }
            sad = sad + 1.0e-4 * f32(dx * dx + dy * dy);
            if (sad < best) {
                best = sad;
                bvx = dx;
                bvy = dy;
            }
        }
    }
    let o = (p.vec_off + by * p.bw + bx) * 2u;
    vecs[o] = f32(bvx * 4);
    vecs[o + 1u] = f32(bvy * 4);
}

fn block_sad_int(bx: u32, by: u32, vx: i32, vy: i32) -> f32 {
    var s = 0.0;
    for (var j = 0u; j < 16u; j = j + 1u) {
        let y = by * 16u + j;
        if (y >= p.h) { break; }
        for (var i = 0u; i < 16u; i = i + 1u) {
            let x = bx * 16u + i;
            if (x >= p.w) { break; }
            let xn = cl(i32(x) + vx, p.w);
            let yn = cl(i32(y) + vy, p.h);
            s = s + abs(frames[fidx(p.cslot, 0u, x, y)] - frames[fidx(p.nslot, 0u, xn, yn)]);
        }
    }
    return s;
}

// 2) refine at full-res: +-3 px around the coarse vector
@compute @workgroup_size(8, 8)
fn k_me_refine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    if (bx >= p.bw || by >= p.bh) { return; }
    let o = (p.vec_off + by * p.bw + bx) * 2u;
    let cx = i32(vecs[o]);
    let cy = i32(vecs[o + 1u]);
    var best = 1.0e30;
    var rx = cx;
    var ry = cy;
    for (var oy = -3; oy <= 3; oy = oy + 1) {
        for (var ox = -3; ox <= 3; ox = ox + 1) {
            let s = block_sad_int(bx, by, cx + ox, cy + oy) + 1.0e-3 * f32(ox * ox + oy * oy);
            if (s < best) {
                best = s;
                rx = cx + ox;
                ry = cy + oy;
            }
        }
    }
    vecs[o] = f32(rx);
    vecs[o + 1u] = f32(ry);
}

fn block_sad_f(bx: u32, by: u32, vx: f32, vy: f32) -> f32 {
    var s = 0.0;
    for (var j = 0u; j < 16u; j = j + 1u) {
        let y = by * 16u + j;
        if (y >= p.h) { break; }
        for (var i = 0u; i < 16u; i = i + 1u) {
            let x = bx * 16u + i;
            if (x >= p.w) { break; }
            s = s + abs(frames[fidx(p.cslot, 0u, x, y)] - bl(p.nslot, 0u, f32(x) + vx, f32(y) + vy));
        }
    }
    return s;
}

// 3) sub-pixel: 1/2 then 1/4 pel, 8 neighbours each
@compute @workgroup_size(8, 8)
fn k_me_subpel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    if (bx >= p.bw || by >= p.bh) { return; }
    let o = (p.vec_off + by * p.bw + bx) * 2u;
    var fx = vecs[o];
    var fy = vecs[o + 1u];
    var best = block_sad_f(bx, by, fx, fy);
    var stp = 0.5;
    for (var it = 0; it < 2; it = it + 1) {
        var bdx = 0.0;
        var bdy = 0.0;
        var bc = best;
        for (var oy = -1; oy <= 1; oy = oy + 1) {
            for (var ox = -1; ox <= 1; ox = ox + 1) {
                if (ox == 0 && oy == 0) { continue; }
                let s = block_sad_f(bx, by, fx + f32(ox) * stp, fy + f32(oy) * stp);
                if (s < bc) {
                    bc = s;
                    bdx = f32(ox) * stp;
                    bdy = f32(oy) * stp;
                }
            }
        }
        best = bc;
        fx = fx + bdx;
        fy = fy + bdy;
        stp = stp * 0.5;
    }
    vecs[o] = fx;
    vecs[o + 1u] = fy;
}

// ---------------------------------------------------------------- profiling residual
// work[0..3n) = (center - motion-compensated neighbour) / sqrt(2)
@compute @workgroup_size(16, 16)
fn k_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let n = npx();
    let idx = gid.y * p.w + gid.x;
    let o = (p.vec_off + (gid.y / 16u) * p.bw + (gid.x / 16u)) * 2u;
    let vx = vecs[o];
    let vy = vecs[o + 1u];
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let c = frames[fidx(p.cslot, ch, gid.x, gid.y)];
        let m = bl(p.nslot, ch, f32(gid.x) + vx, f32(gid.y) + vy);
        work[ch * n + idx] = (c - m) * 0.70710678;
    }
}

// ---------------------------------------------------------------- temporal fusion
// p.a = noise scale, p.b = tau, p.nn = neighbour count, p.ns = neighbour slots
@compute @workgroup_size(16, 16)
fn k_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let n = npx();
    let x = i32(gid.x);
    let y = i32(gid.y);
    let idx = gid.y * p.w + gid.x;

    let c0 = frames[fidx(p.cslot, 0u, gid.x, gid.y)];
    let c1 = frames[fidx(p.cslot, 1u, gid.x, gid.y)];
    let c2 = frames[fidx(p.cslot, 2u, gid.x, gid.y)];

    var bs = 0.0;
    for (var j = -2; j <= 2; j = j + 1) {
        for (var i = -2; i <= 2; i = i + 1) {
            bs = bs + frames[fidx(p.cslot, 0u, cl(x + i, p.w), cl(y + j, p.h))];
        }
    }
    let bright = bs / 25.0;
    let s0 = sig(0u, 0u, bright) * p.a;
    let s1 = sig(1u, 0u, bright) * p.a;
    let s2 = sig(2u, 0u, bright) * p.a;
    let inv0 = 1.0 / (s0 * s0 + 1.0e-8);
    let inv1 = 1.0 / (s1 * s1 + 1.0e-8);
    let inv2 = 1.0 / (s2 * s2 + 1.0e-8);

    var a0 = c0;
    var a1 = c1;
    var a2 = c2;
    var wsum = 1.0;
    var w2 = 1.0;
    let bx = gid.x / 16u;
    let by = gid.y / 16u;

    for (var k = 0u; k < p.nn; k = k + 1u) {
        let slot = get_ns(k);
        let o = (k * p.bw * p.bh + by * p.bw + bx) * 2u;
        let vx = vecs[o];
        let vy = vecs[o + 1u];
        var d = 0.0;
        for (var j = -2; j <= 2; j = j + 1) {
            for (var i = -2; i <= 2; i = i + 1) {
                let px = cl(x + i, p.w);
                let py = cl(y + j, p.h);
                let fxx = f32(px) + vx;
                let fyy = f32(py) + vy;
                let e0 = frames[fidx(p.cslot, 0u, px, py)] - bl(slot, 0u, fxx, fyy);
                let e1 = frames[fidx(p.cslot, 1u, px, py)] - bl(slot, 1u, fxx, fyy);
                let e2 = frames[fidx(p.cslot, 2u, px, py)] - bl(slot, 2u, fxx, fyy);
                d = d + e0 * e0 * inv0 + e1 * e1 * inv1 + e2 * e2 * inv2;
            }
        }
        let dn = d / 75.0;
        let wgt = exp(-max(dn - 2.0, 0.0) / p.b);
        let fx0 = f32(x) + vx;
        let fy0 = f32(y) + vy;
        a0 = a0 + wgt * bl(slot, 0u, fx0, fy0);
        a1 = a1 + wgt * bl(slot, 1u, fx0, fy0);
        a2 = a2 + wgt * bl(slot, 2u, fx0, fy0);
        wsum = wsum + wgt;
        w2 = w2 + wgt * wgt;
    }
    work[idx] = a0 / wsum;
    work[n + idx] = a1 / wsum;
    work[2u * n + idx] = a2 / wsum;
    work[3u * n + idx] = w2 / (wsum * wsum);
}

// ---------------------------------------------------------------- a-trous (B3 spline), level p.level in 1..5
fn mir(i: i32, n: i32) -> i32 {
    var v = i;
    if (n <= 1) { return 0; }
    for (var t = 0; t < 4; t = t + 1) {
        if (v < 0) { v = -v; }
        if (v >= n) { v = 2 * (n - 1) - v; }
    }
    return v;
}

@compute @workgroup_size(16, 16, 1)
fn k_atrous_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= 3u) { return; }
    let n = npx();
    let dil = i32(1u << (p.level - 1u));
    let src = sm_base(p.level - 1u) + gid.z * n + gid.y * p.w;
    let x = i32(gid.x);
    let w = i32(p.w);
    let s = work[src + u32(mir(x - 2 * dil, w))]
          + 4.0 * work[src + u32(mir(x - dil, w))]
          + 6.0 * work[src + gid.x]
          + 4.0 * work[src + u32(mir(x + dil, w))]
          + work[src + u32(mir(x + 2 * dil, w))];
    work[tmp_base() + gid.z * n + gid.y * p.w + gid.x] = s / 16.0;
}

@compute @workgroup_size(16, 16, 1)
fn k_atrous_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= 3u) { return; }
    let n = npx();
    let dil = i32(1u << (p.level - 1u));
    let src = tmp_base() + gid.z * n + gid.x;
    let y = i32(gid.y);
    let h = i32(p.h);
    let s = work[src + u32(mir(y - 2 * dil, h)) * p.w]
          + 4.0 * work[src + u32(mir(y - dil, h)) * p.w]
          + 6.0 * work[src + gid.y * p.w]
          + 4.0 * work[src + u32(mir(y + dil, h)) * p.w]
          + work[src + u32(mir(y + 2 * dil, h)) * p.w];
    work[sm_base(p.level) + gid.z * n + gid.y * p.w + gid.x] = s / 16.0;
}

// ---------------------------------------------------------------- spatial shrinkage
// p.a = strength, p.b = chroma multiplier
fn kfac(ch: u32, j: u32) -> f32 {
    var kl = array<f32, 5>(1.6, 1.5, 1.4, 1.3, 1.2);
    var kc = array<f32, 5>(2.0, 1.9, 1.8, 1.7, 1.6);
    if (ch == 0u) { return kl[j]; }
    return kc[j];
}

@compute @workgroup_size(16, 16, 1)
fn k_shrink(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= 3u) { return; }
    let n = npx();
    let idx = gid.y * p.w + gid.x;
    let ch = gid.z;
    let bright = work[sm_base(5u) + idx];
    let sn = sqrt(work[3u * n + idx]);
    var cm = p.a;
    if (ch != 0u) { cm = p.a * p.b; }
    var acc = work[sm_base(5u) + ch * n + idx];
    for (var j = 1u; j <= 5u; j = j + 1u) {
        let d = work[sm_base(j - 1u) + ch * n + idx] - work[sm_base(j) + ch * n + idx];
        let t = kfac(ch, j - 1u) * sig(ch, j, bright) * sn * cm;
        acc = acc + d * max(0.0, 1.0 - (t * t) / (d * d + 1.0e-12));
    }
    work[out_base() + ch * n + idx] = acc;
}

// ---------------------------------------------------------------- output
@compute @workgroup_size(256)
fn k_pack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let wi = gid.x + gid.y * nwg.x * 256u;
    let total = 3u * npx();
    if (wi * 2u >= total) { return; }
    let ob = out_base();
    let lo = u32(clamp(work[ob + wi * 2u], 0.0, 1.0) * 65535.0 + 0.5);
    let hi = u32(clamp(work[ob + wi * 2u + 1u], 0.0, 1.0) * 65535.0 + 0.5);
    raw_out[wi] = lo | (hi << 16u);
}
