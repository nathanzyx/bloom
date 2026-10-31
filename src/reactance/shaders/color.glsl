// OKLab <-> sRGB, and the intensity -> colour rule.
// Mirrors colour.py::shade. See that module for the reasoning.

vec3 oklab_to_linear(vec3 c) {
    float l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    float m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    float s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return vec3(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

uniform vec3 u_background;     // OKLab
uniform vec3 u_base;           // OKLab: the colour
// The colour band is i_full .. white_lo; white_lo .. white_hi is where the
// colour gives way to white, and everything above white_hi is pure white.
// All four come from ShadeParams in colour.py.
uniform float u_i_full;
uniform float u_white_hi;
uniform float u_white_band;
uniform float u_white_gamma;

float max_chroma(float L) { return 0.37 * clamp((1.0 - L) / 0.3, 0.0, 1.0); }

vec3 shade(float I) {
    I = clamp(I, 0.0, 1.0);
    float white_lo = u_white_hi - u_white_band;
    // Lightness is a function of intensity alone above i_full: hotter = brighter,
    // reaching 1 at white_hi.
    float k = clamp((I - u_i_full) / max(u_white_hi - u_i_full, 1e-3), 0.0, 1.0);
    float L = mix(u_base.x, 1.0, pow(k, u_white_gamma));
    // Saturation plateau: across white_lo .. white_hi the chroma drops to 0, so
    // the hottest part of the body is a solid white rather than a pale tint.
    vec2 ab = u_base.yz * (1.0 - smoothstep(white_lo, u_white_hi, I));
    float C = length(ab);
    ab *= min(1.0, max_chroma(L) / max(C, 1e-5));
    return mix(u_background, vec3(L, ab), smoothstep(0.0, u_i_full, I));
}
