#version 330
// The display. Every quantity is in shape units, so nothing depends on the
// render resolution except anti-aliasing.
//
// Structure, outermost first:
//   1. rho: distance from the shape's core, 0 at the core, 1 at the edge.
//   2. The BODY: a core blob of radius R0 (the audio-driven reach, with only a
//      gentle wobble), plus up to three NOISE PULSES. A pulse is the same
//      blob profile centred on a point that starts at the core and drifts
//      outward, spreading a little and fading as it goes (see motion.py).
//      Core and pulses are combined as METABALLS: their smooth fields are
//      added and the profile is read from the sum, so the noise map stays one
//      rounded mass that swells out and melts back, with no seams.
//   3. A gentle domain warp bends the whole body (a continuous deformation,
//      so it can't tear it into islands either).
//   4. Inside the body: core heat, kick pulse and snare waves, all scaled by
//      the body profile.
//   5. A soft knee before shading, so the top end compresses smoothly into
//      white instead of clipping into a flat plateau.
//   6. The edge fade is an OVERLAY, like a diffuser or bezel lip over the
//      panel: it blends the finished colour toward the surrounding material
//      (u_frame) near the edge, leaving intensity untouched.

#include "noise.glsl"
#include "color.glsl"

#define MAX_RIPPLES 8
#define MAX_PULSES 3

out vec4 frag;

uniform vec2 u_resolution;
uniform vec2 u_view_min;
uniform vec2 u_view_size;
uniform sampler2D u_coords;   // (depth: 0 at the edge .. 1 at the core, signed edge distance)
uniform vec2 u_origin;
uniform vec2 u_size;
uniform float u_thickness;    // the shape's depth at its core, shape units
uniform vec2 u_core;
uniform float u_edge_margin;
uniform vec3 u_frame;         // OKLab colour outside the shape

uniform float u_time;
uniform float u_scale;
uniform float u_branch_freq;
uniform float u_balance;      // 0 = mostly low intensity, 1 = mostly high
uniform float u_branching;    // noise-pulse frequency (wobble scales with it)
uniform vec4 u_npulses[MAX_PULSES];  // (centre x, centre y, radius, strength); in blob radii
uniform float u_level;
uniform float u_reach;
uniform float u_warp;
uniform float u_pulse;
uniform float u_ripple_radius[MAX_RIPPLES];
uniform float u_ripple_amp[MAX_RIPPLES];
uniform int u_ripple_count;
uniform float u_ripple_width;

vec2 coords_at(vec2 pos) { return texture(u_coords, (pos - u_origin) / u_size).rg; }

// Direction from the core as a vector of length <= 1, regularised so that
// near the core it shrinks smoothly to zero instead of spinning through every
// angle (a raw angle is singular at the core).
vec2 core_dir(vec2 pos) {
    vec2 v = pos - u_core;
    return v / max(length(v), 0.35 * u_thickness);
}

// Body profile across one blob: a plateau out to x0, then a smoothstep
// descent to zero at x = x1 (zero slope at both ends: no cliff anywhere).
// x1 pulls in slightly as the plateau grows: a steeper descent lifts the
// faint outer edge, and this keeps the visible footprint the same.
float blob(float x, float x0, float gamma) {
    float x1 = 1.0 - 0.12 * x0 / 0.65;
    return pow(1.0 - smoothstep(x0, x1, x), gamma);
}

// Metaball kernel: 1 at the centre, 0 at d = 1, smooth everywhere. Fields
// from several blobs are ADDED, and the outline comes from the sum, so two
// blobs merge into one shape with a smooth neck and no seam between them.
float kernel(float d) {
    float t = clamp(1.0 - d * d, 0.0, 1.0);
    return t * t;
}

// Soft knee: identity below k, then eases asymptotically toward 1.
float soft_knee(float I, float k) {
    if (I <= k) return I;
    return k + (1.0 - k) * tanh((I - k) / (1.0 - k));
}

// The fade uses the SMOOTH depth, not the exact edge distance: exact distance
// has creases along corner diagonals, which would show as lines in the fade.
// The exact distance is still used for the outline itself.
float edge_factor(vec2 g) {
    if (u_edge_margin <= 0.0) return 1.0;
    return smoothstep(0.0, u_edge_margin, g.r * u_thickness);
}

// Snare waves: rings at rho == radius, travelling out through the body.
float ripple_field(float rho) {
    float acc = 0.0;
    for (int k = 0; k < MAX_RIPPLES; ++k) {
        if (k >= u_ripple_count) break;
        float x = (rho - u_ripple_radius[k]) / u_ripple_width;
        acc += u_ripple_amp[k] * exp(-x * x);
    }
    return acc;
}

void main() {
    vec2 f = gl_FragCoord.xy / u_resolution;
    vec2 pos = u_view_min + vec2(f.x, 1.0 - f.y) * u_view_size;  // SVG is y-down
    vec2 g = coords_at(pos);
    float d = g.g;
    float coverage = clamp(d / max(fwidth(d), 1e-6) + 0.5, 0.0, 1.0);
    if (coverage <= 0.0) { frag = vec4(linear_to_srgb(oklab_to_linear(u_frame)), 1.0); return; }

    float T = u_time;
    vec3 q0 = vec3(pos, 0.0) / u_scale;

    // 3. organic bend: sample the geometry at a slightly displaced point.
    // The warp is part of the body's irregularity, so branching scales it too.
    vec2 disp = vec2(fbm(vec4(q0, T), 2), fbm(vec4(q0 + vec3(5.2, 1.3, 2.8), T), 2));
    float warp = 0.3 * u_warp * mix(0.2, 1.0, u_branching);
    vec2 pos_w = pos + warp * u_scale * disp;
    vec2 gw = coords_at(pos_w);

    // 1. distance from the core
    float rho = 1.0 - gw.r;

    // 2. the body: core blob + noise pulses
    vec2 dir = core_dir(pos_w);
    float wob = snoise(vec4(dir * u_branch_freq, 0.0, 0.4 * T));
    float R0 = u_reach * (1.0 + 0.05 * mix(0.3, 1.0, u_branching) * wob);

    // Balance fills the body with heat WITHOUT moving its edge. The profile
    // always reaches zero at the body's radius; balance only widens the hot
    // plateau inside it (x0), so the descent gets shorter and steeper while
    // the footprint stays put. The curve's exponent is fixed, because lowering
    // it (as an earlier version did) lifted the faint tail and made the glow
    // spread across the whole shape.
    float x0 = mix(0.0, 0.1625, u_balance);
    float gamma = 1.4;
    // Core + noise pulses as metaballs. F is the summed field; x_eff turns it
    // back into a "normalised distance" through the inverse of the kernel, so
    // with no pulses x_eff == rho / R0 exactly.
    float F = (R0 > 1e-3) ? kernel(rho / R0) : 0.0;
    // q_core: this pixel in the same normalised space as the core.
    vec2 q_core = rho * dir / max(length(dir), 1e-6);
    for (int k = 0; k < MAX_PULSES; ++k) {
        vec4 np_ = u_npulses[k];
        if (np_.w <= 0.0 || R0 <= 1e-3) continue;
        F += np_.w * kernel(length(q_core - np_.xy * R0) / (np_.z * R0));
    }
    float x_eff = sqrt(max(0.0, 1.0 - sqrt(min(F, 1.0))));
    float profile = (R0 > 1e-3) ? blob(x_eff, x0, gamma) : 0.0;

    // 4. inside the body
    float ripple = ripple_field(rho);
    float gain = mix(0.95, 1.1, u_balance);
    // Core heat: intensity peaks at the core point and eases off outward, so
    // the hottest (white) point is the middle of the body. It is measured from
    // the core point alone, not from the summed field: pulses add body in the
    // colour band without each becoming a white disc. On a thin shape the
    // spot stretches along the arm: across it rho limits it (colour at the
    // sides), along it it reaches ~2.5 depths from the core. The two combine
    // as a 2-norm so the contours are round (a max() gives a boxy white
    // rectangle on wide shapes). For a disc along == rho / 2.5, and the
    // constant divides that back out: exactly rho / R0.
    // Balance (mostly high intensity) widens the hot spot along with the plateau.
    float hot_r = mix(0.45, 0.9, u_balance);
    float along = length(pos_w - u_core) / max(2.5 * u_thickness, 1e-3);
    float x_core = length(vec2(rho, along)) / (1.077 * max(R0, 1e-3));
    float core_hot = 1.0 - smoothstep(0.0, hot_r, x_core);
    float heat = 0.65 + 0.7 * core_hot;
    float I = gain * (u_level * heat + u_pulse + ripple) * profile;

    // 5. soft knee into white
    I = soft_knee(I, 0.55);
    vec3 screen = oklab_to_linear(shade(I));

    // 6. the overlay: blend toward the surrounding material in linear light
    // (that is how light through a physical diffuser mixes), using the true,
    // unwarped edge distance. Anti-aliasing of the outline is the same blend.
    float clear = coverage * edge_factor(g);
    vec3 rgb = mix(max(oklab_to_linear(u_frame), 0.0), max(screen, 0.0), clear);
    frag = vec4(linear_to_srgb(rgb), 1.0);
}
