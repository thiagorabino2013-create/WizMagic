#!/usr/bin/env python3
"""Generates every original asset of the mod: particle textures, wand textures + 3D models, GUI icons,
sounds (synthesised), lang files, recipes. Run from the project root:  python3 tools/generate_assets.py
Requires numpy, scipy, Pillow and ffmpeg (with libvorbis) for the .ogg sounds."""
import json, math, os, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage, signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "src/main/resources")
ASSETS = os.path.join(RES, "assets/wizmagic")
DATA = os.path.join(RES, "data/wizmagic")
MOD = "wizmagic"
rng = np.random.default_rng(1337)

def mk(*p):
    os.makedirs(os.path.join(*p), exist_ok=True)

for d in ["textures/particle", "textures/item", "textures/gui", "particles", "models/item", "items", "lang", "sounds"]:
    mk(ASSETS, d)
mk(DATA, "recipe")

# ------------------------------------------------------------------ texture helpers
def grid(n):
    y, x = np.mgrid[0:n, 0:n]
    return (x + 0.5) / n * 2 - 1, (y + 0.5) / n * 2 - 1

def sstep(a, lo, hi):
    t = np.clip((a - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)

def noise(n, scale, seed):
    r = np.random.default_rng(seed)
    small = r.random((max(2, int(n / scale)) + 2, max(2, int(n / scale)) + 2))
    return ndimage.zoom(small, n / (small.shape[0] - 2) if small.shape[0] > 2 else 1, order=3)[:n, :n]

def fbm(n, seed, octaves=4):
    out = np.zeros((n, n))
    amp = 1.0
    tot = 0
    for o in range(octaves):
        sc = max(2, n // (2 ** (o + 1)))
        out += amp * noise(n, sc, seed + o * 17)
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-9)

def save_rgba(path, rgb, a):
    """rgb: HxWx3 floats 0..1 (or 0..255), a: HxW 0..1"""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    img = np.dstack([np.clip(rgb, 0, 1), np.clip(a, 0, 1)])
    Image.fromarray((img * 255 + 0.5).astype(np.uint8), "RGBA").save(path)

def tex(name, rgb, a):
    save_rgba(os.path.join(ASSETS, "textures/particle", name + ".png"), rgb, a)

def col(c, n):
    return np.ones((n, n, 3)) * (np.array(c, dtype=float) / 255.0)

def ramp(v, stops):
    """stops: list of (pos, (r,g,b)) -> HxWx3"""
    out = np.zeros(v.shape + (3,))
    for ch in range(3):
        out[..., ch] = np.interp(v, [s[0] for s in stops], [s[1][ch] / 255.0 for s in stops])
    return out

def blur_pil(arr, radius):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im, dtype=float) / 255.0

# ------------------------------------------------------------------ particle textures
def make_particles():
    n = 32
    x, y = grid(n)
    r = np.sqrt(x * x + y * y)
    glow = np.clip(1 - r, 0, 1) ** 2.0

    # wand glow & spark (white, tinted at runtime)
    tex("wand_glow", col((255, 255, 255), n), np.clip(glow * 1.15, 0, 1))
    star = np.exp(-np.abs(x * y) * 70) * np.clip(1 - r * 0.95, 0, 1)
    diag = np.exp(-np.abs((x - y) * (x + y)) * 120) * np.clip(1 - r * 1.1, 0, 1) * 0.5
    tex("wand_spark", col((255, 255, 255), n), np.clip(star + diag + glow * 0.5, 0, 1))

    # ---- aqua
    lx, ly = -0.45, -0.5
    z = np.sqrt(np.clip(1 - (r / 0.85) ** 2, 0, 1))
    diffuse = np.clip((x / 0.85 * lx + y / 0.85 * ly + z * 0.75), 0, 1)
    base = ramp(diffuse, [(0, (30, 110, 220)), (0.5, (70, 170, 255)), (1, (200, 240, 255))])
    spec = np.exp(-(((x + 0.32) ** 2 + (y + 0.36) ** 2) / 0.012))
    rim = sstep(r, 0.55, 0.85) * 0.25
    rgb = np.clip(base + spec[..., None] * 0.9 + rim[..., None] * np.array([0.3, 0.5, 0.6]), 0, 1)
    body = sstep(0.88 - r, 0, 0.12)
    tex("aqua_orb", rgb, np.clip(body * 0.92 + glow * 0.28, 0, 1))

    for i in range(4):
        nz = fbm(n, 100 + i * 7)
        wob = r + (nz - 0.5) * 0.35
        a = sstep(0.86 - wob, 0, 0.3) * 0.85
        c = ramp(np.clip(wob / 0.9, 0, 1), [(0, (205, 238, 255)), (0.6, (110, 195, 255)), (1, (60, 150, 250))])
        hl = sstep(nz, 0.68, 0.9)[..., None] * 0.7
        tex("aqua_stream_%d" % i, np.clip(c + hl, 0, 1), a)

    # droplet
    hw = np.where(y < 0.3, 0.5 * np.clip((y + 0.9) / 1.2, 0, 1) ** 1.6,
                  np.sqrt(np.clip(0.25 - (y - 0.3) ** 2, 0, 1)))
    mask = sstep(hw - np.abs(x), -0.02, 0.06) * (y > -0.92)
    dshade = np.clip(0.6 + 0.4 * (-x * 0.6 - y * 0.2), 0, 1)
    c = ramp(dshade, [(0, (60, 150, 240)), (0.6, (130, 205, 255)), (1, (225, 248, 255))])
    dh = np.exp(-(((x + 0.15) ** 2 + (y - 0.15) ** 2) / 0.006))[..., None]
    tex("aqua_droplet", np.clip(c + dh, 0, 1), mask * 0.9)

    core = np.exp(-r * r * 22)
    c = col((90, 190, 255), n) * (1 - core[..., None]) + col((255, 255, 255), n) * core[..., None]
    tex("aqua_spiral", c, np.clip(glow * 1.2, 0, 1))

    # splash crown, 64px, 6 frames
    m = 64
    for i in range(6):
        p = (i + 1) / 6.0
        im = Image.new("L", (m, m), 0)
        d = ImageDraw.Draw(im)
        cx, cy = m / 2, m * 0.62
        for k in range(14):
            ang = math.pi * (0.08 + 0.84 * k / 13) + (k % 2) * 0.05
            r0 = m * 0.08
            r1 = m * (0.1 + 0.34 * p) * (0.8 + 0.3 * ((k * 7) % 5) / 4)
            ex, ey = cx + math.cos(ang) * r1 * 1.1, cy - math.sin(ang) * r1 * 1.25 + p * p * m * 0.12
            d.line([(cx + math.cos(ang) * r0, cy - math.sin(ang) * r0), (ex, ey)], fill=255, width=2)
            d.ellipse([ex - 2.2, ey - 2.2, ex + 2.2, ey + 2.2], fill=255)
        d.ellipse([cx - m * 0.2 * (1.1 - p * 0.4), cy - m * 0.06, cx + m * 0.2 * (1.1 - p * 0.4), cy + m * 0.07], fill=210)
        a = np.asarray(im, dtype=float) / 255.0
        a = np.clip(blur_pil(a, 0.8) * 1.2, 0, 1) * (1.0 - 0.45 * p ** 2)
        xx, yy = grid(m)
        c = ramp(np.clip(0.5 + 0.5 * yy * -1 + 0.2, 0, 1), [(0, (120, 195, 255)), (1, (235, 250, 255))])
        tex("aqua_splash_%d" % i, c, a)

    nz = fbm(n, 300)
    tex("aqua_mist", col((225, 244, 255), n), np.clip(sstep(1 - r, 0, 0.9) * (0.35 + nz * 0.9) * 0.6, 0, 1))

    m = 64
    x2, y2 = grid(m)
    r2 = np.sqrt(x2 * x2 + y2 * y2)
    band = np.exp(-((r2 - 0.8) / 0.075) ** 2)
    inner = sstep(r2, 0.2, 0.8) * 0.12 * sstep(0.95 - r2, 0, 0.1)
    c = ramp(np.clip(band, 0, 1), [(0, (90, 180, 255)), (0.6, (170, 225, 255)), (1, (255, 255, 255))])
    tex("water_ring", c, np.clip(band * 0.95 + inner, 0, 1))

    # ---- lightning
    core = np.exp(-r * r * 14)
    tex("bolt_core", ramp(np.clip(core, 0, 1), [(0, (200, 225, 255)), (1, (255, 255, 255))]), np.clip(glow * 1.5, 0, 1))
    tex("bolt_glow", col((79, 160, 255), n), np.clip(glow * 0.95, 0, 1))
    for i in range(4):
        rr = np.random.default_rng(500 + i)
        im = Image.new("L", (n * 2, n * 2), 0)
        d = ImageDraw.Draw(im)
        def bolt(x0, y0, ang, length, width, depth):
            pts = [(x0, y0)]
            for s in range(5):
                ang += rr.uniform(-0.9, 0.9)
                x0 += math.cos(ang) * length / 5
                y0 += math.sin(ang) * length / 5
                pts.append((x0, y0))
            d.line(pts, fill=255, width=width)
            if depth > 0:
                for _ in range(2):
                    k = rr.integers(1, 4)
                    bolt(pts[k][0], pts[k][1], ang + rr.uniform(-1.2, 1.2), length * 0.55, 1, depth - 1)
        for a0 in range(3):
            bolt(n, n, a0 * 2.1 + rr.uniform(0, 1), n * 0.85, 2, 1)
        a = np.asarray(im.resize((n, n), Image.LANCZOS), dtype=float) / 255.0
        halo = blur_pil(a, 1.3)
        corei = np.clip(a * 1.2, 0, 1)
        rgb = col((70, 150, 255), n) * (1 - corei[..., None]) + col((255, 255, 255), n) * corei[..., None]
        tex("bolt_spark_%d" % i, rgb, np.clip(halo * 1.8 + a * 0.2, 0, 1) * np.clip(1.2 - r, 0, 1))

    m = 64
    x2, y2 = grid(m)
    r2 = np.sqrt(x2 * x2 + y2 * y2)
    th = np.arctan2(y2, x2)
    rmod = 0.78 + 0.05 * np.sin(th * 13 + 1.2) + 0.04 * np.sin(th * 29)
    band = np.exp(-((r2 - rmod) / 0.05) ** 2)
    spikes = np.exp(-((r2 - rmod - 0.07) / 0.05) ** 2) * (np.sin(th * 17) > 0.8) * 0.8
    glowb = np.exp(-((r2 - rmod) / 0.16) ** 2) * 0.4
    c = ramp(np.clip(band + spikes, 0, 1), [(0, (60, 140, 255)), (0.5, (150, 210, 255)), (1, (255, 255, 255))])
    tex("electric_ring", c, np.clip(band + spikes + glowb, 0, 1))

    # ---- fire
    nz = fbm(n, 700)
    wob = r + (nz - 0.5) * 0.25
    v = np.clip(1 - wob / 0.9, 0, 1)
    c = ramp(v, [(0, (190, 35, 10)), (0.35, (255, 120, 30)), (0.75, (255, 205, 100)), (1, (255, 250, 210))])
    tex("fire_orb", c, np.clip(sstep(0.95 - wob, 0, 0.35) * 0.95 + glow * 0.2, 0, 1))

    for i in range(6):
        ph = i * 1.05
        t = (y + 1) / 2.0  # 0 top .. 1 bottom
        up = 1 - t
        width = 0.52 * np.clip(1 - up ** 1.15, 0, 1) ** 0.75 + 0.02
        off = 0.14 * np.sin(up * 7 + ph) * up + 0.05 * np.sin(up * 17 - ph * 1.7) * up
        nzf = fbm(n, 800 + i * 5)
        d = np.abs(x - off) / width
        inten = np.clip(1 - d ** 1.6, 0, 1) * np.clip(1.15 - up ** 1.3 * 0.7, 0, 1)
        inten = np.clip(inten * (0.75 + nzf * 0.5), 0, 1)
        inten *= sstep(1.0 - t, 0.0, 0.06) * sstep(t, 0.0, 0.1)
        c = ramp(inten, [(0, (150, 20, 5)), (0.25, (235, 60, 15)), (0.55, (255, 150, 40)), (0.8, (255, 220, 120)), (1, (255, 250, 220))])
        tex("flame_%d" % i, c, np.clip(inten * 1.3, 0, 1))

    n16 = 16
    x16, y16 = grid(n16)
    r16 = np.sqrt(x16 * x16 + y16 * y16)
    g16 = np.clip(1 - r16, 0, 1) ** 1.6
    c = ramp(np.clip(g16 * 1.4, 0, 1), [(0, (255, 90, 20)), (0.6, (255, 190, 90)), (1, (255, 250, 215))])
    tex("ember", c, np.clip(g16 * 1.4, 0, 1))

    nz = fbm(n, 900)
    tex("fire_smoke", col((235, 215, 200), n), np.clip(sstep(1 - r, 0, 0.8) * (0.25 + nz * 1.0) * 0.8, 0, 1))

    m = 64
    x2, y2 = grid(m)
    r2 = np.sqrt(x2 * x2 + y2 * y2)
    th = np.arctan2(y2, x2)
    for i in range(6):
        p = (i + 1) / 6.0
        tongues = 0.5 + 0.5 * np.cos(th * 10 + i * 0.7)
        edge = (0.35 + 0.55 * p) * (0.7 + 0.45 * tongues)
        inten = np.clip(1 - r2 / edge, 0, 1) ** 0.9
        inten *= (1 - 0.55 * p ** 2)
        nzb = fbm(m, 950 + i)
        inten = np.clip(inten * (0.7 + 0.6 * nzb), 0, 1)
        c = ramp(inten, [(0, (170, 25, 8)), (0.3, (255, 90, 20)), (0.65, (255, 175, 60)), (1, (255, 245, 200))])
        tex("fire_burst_%d" % i, c, np.clip(inten * 1.4, 0, 1))

    # ---- arcane (expulso)
    swirl = 0.5 + 0.5 * np.cos(np.arctan2(y, x) * 3 + r * 7)
    v = np.clip((1 - r / 0.9) * 0.9 + swirl * 0.18, 0, 1)
    c = ramp(v, [(0, (110, 40, 215)), (0.5, (190, 130, 255)), (1, (255, 250, 255))])
    tex("arcane_orb", c, np.clip(sstep(0.95 - r, 0, 0.3) * 0.95 + glow * 0.25, 0, 1))

    th = np.arctan2(y, x)
    for i in range(4):
        ph = i * math.pi / 2
        arms = 0.5 + 0.5 * np.cos(3 * th + 7 * r - ph)
        a = arms * np.clip(1 - r, 0, 1) ** 1.2
        c = ramp(np.clip(a * 1.2, 0, 1), [(0, (95, 35, 190)), (0.5, (170, 105, 255)), (1, (235, 205, 255))])
        tex("arcane_trail_%d" % i, c, np.clip(a * 1.3, 0, 1))

    m = 64
    x2, y2 = grid(m)
    r2 = np.sqrt(x2 * x2 + y2 * y2)
    th = np.arctan2(y2, x2)
    rays = np.exp(-np.abs(np.sin(th * 4)) * 9) * np.clip(1 - r2, 0, 1) ** 1.3
    rays2 = np.exp(-np.abs(np.sin(th * 4 + math.pi / 4)) * 14) * np.clip(1 - r2, 0, 1) ** 1.6 * 0.6
    cg = np.exp(-r2 * r2 * 9)
    a = np.clip(rays + rays2 + cg * 1.1, 0, 1)
    c = ramp(np.clip(a, 0, 1), [(0, (150, 80, 240)), (0.5, (225, 195, 255)), (1, (255, 255, 255))])
    tex("arcane_flash", c, a)

    band = np.exp(-((r2 - 0.8) / 0.09) ** 2)
    dots = np.exp(-((r2 - 0.62) / 0.025) ** 2) * (np.sin(th * 12) > 0.4) * 0.8
    band2 = np.exp(-((r2 - 0.88) / 0.03) ** 2) * 0.6
    a = np.clip(band + dots + band2 + np.exp(-((r2 - 0.8) / 0.2) ** 2) * 0.25, 0, 1)
    c = ramp(np.clip(band + dots, 0, 1), [(0, (125, 65, 225)), (0.6, (195, 140, 255)), (1, (255, 240, 255))])
    tex("arcane_ring", c, a)

    n16 = 16
    x16, y16 = grid(n16)
    dia = np.abs(x16) * 1.0 + np.abs(y16) * 0.55
    shard = sstep(0.8 - dia, 0, 0.18)
    sh = np.clip(1 - dia / 0.8, 0, 1)
    c = ramp(sh, [(0, (150, 80, 235)), (0.6, (215, 175, 255)), (1, (255, 255, 255))])
    tex("arcane_mote", c, np.clip(shard * 0.95 + np.clip(1 - r16 * 1.0, 0, 1) ** 2 * 0.3, 0, 1))

    band = np.exp(-((r2 - 0.86) / 0.14) ** 2)
    inner = sstep(r2, 0.0, 0.86) * 0.18
    c = ramp(np.clip(band, 0, 1), [(0, (140, 90, 230)), (1, (225, 200, 255))])
    tex("arcane_wave", c, np.clip((band * 0.7 + inner) * sstep(1.0 - r2, 0, 0.1), 0, 1))

    tex("arcane_glow", col((190, 140, 255), n), np.clip(glow * 1.1, 0, 1))

PARTICLES = {
    "wand_glow": 1, "wand_spark": 1, "aqua_orb": 1, "aqua_stream": 4, "aqua_droplet": 1, "aqua_spiral": 1,
    "aqua_splash": 6, "aqua_mist": 1, "water_ring": 1, "bolt_core": 1, "bolt_glow": 1, "bolt_spark": 4,
    "electric_ring": 1, "fire_orb": 1, "flame": 6, "ember": 1, "fire_smoke": 1, "fire_burst": 6,
    "arcane_orb": 1, "arcane_trail": 4, "arcane_flash": 1, "arcane_ring": 1, "arcane_mote": 1,
    "arcane_wave": 1, "arcane_glow": 1,
}

def write_particle_defs():
    for name, frames in PARTICLES.items():
        if frames == 1:
            textures = ["%s:%s" % (MOD, name)]
        else:
            textures = ["%s:%s_%d" % (MOD, name, i) for i in range(frames)]
        with open(os.path.join(ASSETS, "particles", name + ".json"), "w") as f:
            json.dump({"textures": textures}, f, indent=2)

# ------------------------------------------------------------------ wands
WANDS = [
    # id, wood, grain(dark), core, ring, handle style
    ("holly_phoenix", (128, 74, 42), (84, 46, 26), (255, 138, 43), (232, 192, 84), "bulb"),
    ("elder_thestral", (203, 199, 214), (150, 146, 168), (220, 228, 255), (190, 196, 214), "spiral"),
    ("yew_dragon", (92, 62, 42), (56, 36, 26), (53, 224, 125), (160, 124, 64), "plain"),
    ("willow_unicorn", (182, 146, 100), (130, 98, 64), (168, 232, 255), (200, 206, 222), "bulb"),
    ("ebony_thunderbird", (34, 28, 34), (14, 10, 16), (111, 168, 255), (170, 190, 220), "spiral"),
    ("vine_veela", (104, 98, 60), (66, 70, 38), (224, 139, 255), (224, 156, 138), "plain"),
]

def make_wand_textures():
    for wid, wood, grain, core, ring, style in WANDS:
        r = np.random.default_rng(abs(hash(wid)) % 100000 + 7)
        img = np.zeros((16, 16, 4))
        def put(x0, y0, w, h, f):
            for yy in range(h):
                for xx in range(w):
                    img[y0 + yy, x0 + xx] = f(xx, yy)
        def woodpx(base, dark):
            def f(xx, yy):
                t = 0.5 + 0.5 * math.sin(yy * 1.7 + xx * 0.9 + r.uniform(0, 0.8))
                c = np.array(base) * (1 - 0.35 * t) + np.array(dark) * 0.35 * t
                c = c * (0.9 + r.uniform(0, 0.2))
                return [*(np.clip(c, 0, 255) / 255.0), 1]
            return f
        put(0, 0, 8, 8, woodpx(wood, grain))
        put(8, 0, 8, 8, woodpx(tuple(int(v * 0.72) for v in wood), tuple(int(v * 0.6) for v in grain)))
        def metal(xx, yy):
            hi = 1.15 if (xx + yy) % 4 == 0 else (0.8 if yy % 3 == 0 else 1.0)
            c = np.array(ring) * hi
            return [*(np.clip(c, 0, 255) / 255.0), 1]
        put(0, 8, 8, 8, metal)
        def gem(xx, yy):
            d = math.hypot(xx - 3.5, yy - 3.5) / 5.0
            t = np.clip(1 - d, 0, 1)
            c = np.array(core) * (0.55 + 0.45 * t) + np.array([255, 255, 255]) * (t ** 3) * 0.7
            return [*(np.clip(c, 0, 255) / 255.0), 1]
        put(8, 8, 8, 8, gem)
        Image.fromarray((img * 255 + 0.5).astype(np.uint8), "RGBA").save(os.path.join(ASSETS, "textures/item", "wand_%s.png" % wid))

UV = {"wood": [0, 0, 4, 4], "grip": [8, 0, 12, 4], "ring": [0, 8, 4, 12], "gem": [8, 8, 12, 12]}

def wand_elements(style):
    els = []
    def cube(i, s, part, off=0.0):
        cx = 2.0 + i * 1.0 + off
        cy = 2.0 + i * 1.0 + off
        z0, z1 = 8 - s / 2, 8 + s / 2
        uv = UV[part]
        face = {"uv": uv, "texture": "#0"}
        els.append({
            "from": [round(cx - s / 2, 3), round(cy - s / 2, 3), round(z0, 3)],
            "to": [round(cx + s / 2, 3), round(cy + s / 2, 3), round(z1, 3)],
            "faces": {k: dict(face) for k in ("north", "south", "east", "west", "up", "down")},
        })
    if style == "bulb":
        cube(0, 2.4, "ring")
        cube(0.8, 1.9, "grip")
    for i in range(0, 5):
        if style == "spiral":
            cube(i, 1.9 if i % 2 == 0 else 1.55, "grip")
        elif style == "bulb":
            if i >= 1:
                cube(i, 1.7, "grip")
        else:
            cube(i, 1.75, "grip")
    cube(5, 2.15, "ring")
    if style == "plain":
        cube(2, 2.0, "ring")
    for i in range(6, 11):
        cube(i, 1.35 - (i - 6) * 0.06, "wood")
    cube(11, 1.1, "wood")
    cube(11.8, 1.3, "gem")
    cube(12.5, 1.0, "gem")
    return els

DISPLAY = {
    "thirdperson_righthand": {"rotation": [0, -90, 55], "translation": [0, 4.0, 0.5], "scale": [0.85, 0.85, 0.85]},
    "thirdperson_lefthand": {"rotation": [0, 90, -55], "translation": [0, 4.0, 0.5], "scale": [0.85, 0.85, 0.85]},
    "firstperson_righthand": {"rotation": [0, -90, 25], "translation": [1.13, 3.2, 1.13], "scale": [0.68, 0.68, 0.68]},
    "firstperson_lefthand": {"rotation": [0, 90, -25], "translation": [1.13, 3.2, 1.13], "scale": [0.68, 0.68, 0.68]},
    "ground": {"rotation": [0, 0, 0], "translation": [0, 2, 0], "scale": [0.5, 0.5, 0.5]},
    "gui": {"rotation": [0, 0, 0], "translation": [0, 0, 0], "scale": [1, 1, 1]},
    "fixed": {"rotation": [0, 0, 0], "translation": [0, 0, 0], "scale": [1, 1, 1]},
    "head": {"rotation": [0, 180, 0], "translation": [0, 13, 7], "scale": [1, 1, 1]},
}

def make_wand_models():
    for wid, wood, grain, core, ring, style in WANDS:
        model = {
            "textures": {"0": "%s:item/wand_%s" % (MOD, wid), "particle": "%s:item/wand_%s" % (MOD, wid)},
            "gui_light": "front",
            "elements": wand_elements(style),
            "display": DISPLAY,
        }
        with open(os.path.join(ASSETS, "models/item", "wand_%s.json" % wid), "w") as f:
            json.dump(model, f, indent=1)
        with open(os.path.join(ASSETS, "items", "wand_%s.json" % wid), "w") as f:
            json.dump({"model": {"type": "minecraft:model", "model": "%s:item/wand_%s" % (MOD, wid)}}, f, indent=2)

# ------------------------------------------------------------------ icons
def make_icons():
    S = 128
    def finish(im, name):
        im = im.resize((16, 16), Image.LANCZOS)
        im.save(os.path.join(ASSETS, "textures/gui", name + ".png"))
    # aqua: droplet
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(64, 8), (104, 70), (24, 70)], fill=(70, 170, 255, 255))
    d.ellipse([22, 44, 106, 120], fill=(70, 170, 255, 255))
    d.ellipse([34, 62, 56, 96], fill=(225, 248, 255, 230))
    finish(im, "spell_aqua")
    # lightning
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(76, 4), (30, 70), (60, 70), (46, 124), (100, 52), (68, 52), (86, 4)], fill=(120, 190, 255, 255))
    d.polygon([(76, 20), (48, 64), (70, 64), (62, 96), (86, 58), (66, 58), (80, 20)], fill=(255, 255, 255, 255))
    finish(im, "spell_lightning")
    # inferno
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(64, 4), (98, 52), (108, 84), (90, 120), (38, 120), (20, 84), (30, 50), (50, 66)], fill=(255, 110, 30, 255))
    d.polygon([(64, 50), (84, 82), (78, 114), (50, 114), (44, 84)], fill=(255, 200, 80, 255))
    d.polygon([(64, 80), (72, 100), (64, 114), (56, 100)], fill=(255, 250, 215, 255))
    finish(im, "spell_inferno")
    # expulso
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for k in range(8):
        a = k * math.pi / 4
        d.line([(64 + math.cos(a) * 20, 64 + math.sin(a) * 20), (64 + math.cos(a) * 58, 64 + math.sin(a) * 58)], fill=(170, 100, 255, 255), width=10)
    d.ellipse([34, 34, 94, 94], fill=(165, 80, 255, 255))
    d.ellipse([48, 48, 80, 80], fill=(235, 205, 255, 255))
    d.ellipse([56, 56, 72, 72], fill=(255, 255, 255, 255))
    finish(im, "spell_expulso")

    # mod icon
    N = 128
    xx, yy = grid(N)
    bg = ramp(np.clip((yy + 1) / 2, 0, 1), [(0, (40, 18, 80)), (1, (12, 6, 30))])
    rr = np.sqrt(xx * xx + yy * yy)
    glowm = np.exp(-rr * rr * 3)[..., None] * np.array([0.5, 0.25, 0.9])
    base = np.clip(bg + glowm * 0.5, 0, 1)
    im = Image.fromarray((base * 255).astype(np.uint8), "RGB").convert("RGBA")
    d = ImageDraw.Draw(im)
    d.line([(30, 100), (96, 34)], fill=(122, 74, 42, 255), width=9)
    d.line([(30, 100), (50, 80)], fill=(70, 40, 24, 255), width=11)
    d.ellipse([92, 26, 108, 42], fill=(255, 200, 120, 255))
    for (sx, sy, s) in [(100, 20, 14), (112, 40, 8), (84, 14, 6), (30, 40, 7), (24, 70, 5)]:
        d.polygon([(sx, sy - s), (sx + s * 0.25, sy - s * 0.25), (sx + s, sy), (sx + s * 0.25, sy + s * 0.25), (sx, sy + s), (sx - s * 0.25, sy + s * 0.25), (sx - s, sy), (sx - s * 0.25, sy - s * 0.25)], fill=(255, 245, 255, 255))
    im.save(os.path.join(ASSETS, "icon.png"))

# ------------------------------------------------------------------ sounds
SR = 44100

def t_axis(dur):
    return np.arange(int(SR * dur)) / SR

def env(n, a, d, power=2.0):
    e = np.ones(n)
    na = max(1, int(a * SR)); e[:na] = np.linspace(0, 1, na)
    nd = n - na
    if nd > 0:
        e[na:] = np.exp(-np.linspace(0, 1, nd) * d * power)
    return e

def white(n):
    return rng.standard_normal(n)

def filt(x, kind, f, order=3):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    wn = f / (SR / 2)
    if wn.size == 1:
        wn = float(wn[0])
    b, a = signal.butter(order, wn, btype=kind)
    return signal.lfilter(b, a, x)

def sweep(f0, f1, dur, shape="exp"):
    t = t_axis(dur)
    if shape == "exp":
        f = f0 * (f1 / f0) ** (t / dur)
    else:
        f = np.linspace(f0, f1, len(t))
    return np.sin(2 * np.pi * np.cumsum(f) / SR)

def saw(f, dur):
    t = t_axis(dur)
    ff = f if np.ndim(f) else np.ones(len(t)) * f
    ph = np.cumsum(ff) / SR
    return 2 * (ph % 1) - 1

def norm(x, peak=0.85):
    return x / (np.max(np.abs(x)) + 1e-9) * peak

def write_sound(name, x):
    x = norm(np.asarray(x))
    # tiny fades to avoid clicks
    k = int(0.004 * SR)
    x[:k] *= np.linspace(0, 1, k); x[-k:] *= np.linspace(1, 0, k)
    wav = os.path.join(ASSETS, "sounds", name + ".wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    out = os.path.join(ASSETS, "sounds", name + ".ogg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-c:a", "libvorbis", "-q:a", "4", "-ac", "1", out], check=True)
    os.remove(wav)

def make_sounds():
    # AQUA
    d = 0.55; t = t_axis(d); n = len(t)
    bub = sweep(260, 1100, d) * (0.5 + 0.5 * np.sin(2 * np.pi * 13 * t)) * 0.5
    flow = filt(white(n), "bandpass", [400, 2500]) * np.linspace(0.1, 1, n)
    write_sound("charge_aqua", (bub + flow * 0.6) * env(n, 0.05, 0.4, 0.8))
    d = 0.7; t = t_axis(d); n = len(t)
    whoosh = filt(white(n), "bandpass", [300, 3500]) * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 0.8
    bloop = sweep(500, 140, 0.25) ; bloop = np.pad(bloop, (0, n - len(bloop))) * env(n, 0.002, 7)
    write_sound("cast_aqua", whoosh * 0.9 + bloop * 0.7)
    d = 1.0; t = t_axis(d); n = len(t)
    splash = filt(white(n), "bandpass", [500, 5000]) * env(n, 0.003, 5)
    drops = np.zeros(n)
    for _ in range(14):
        p = int(rng.uniform(0.04, 0.9) * SR); f = rng.uniform(900, 2600); L = int(0.07 * SR)
        if p + L < n:
            drops[p:p + L] += np.sin(2 * np.pi * f * np.arange(L) / SR * (1 + 0.5 * np.arange(L) / L)) * np.exp(-np.arange(L) / (0.02 * SR)) * rng.uniform(0.2, 0.6)
    thump = np.pad(sweep(180, 60, 0.2), (0, n - int(0.2 * SR))) * env(n, 0.002, 12)
    write_sound("impact_aqua", splash * 1.0 + drops * 0.8 + thump * 0.8)

    # LIGHTNING
    d = 0.55; t = t_axis(d); n = len(t)
    buzz = saw(np.linspace(110, 420, n), d) * 0.35
    buzz = filt(buzz, "lowpass", 3000) * (0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 50 * t)))
    clicks = (rng.random(n) > 0.9975) * rng.uniform(0.5, 1, n)
    clicks = filt(clicks, "highpass", 2000)
    write_sound("charge_lightning", (buzz * np.linspace(0.2, 1, n) + clicks * 1.5))
    d = 0.5; t = t_axis(d); n = len(t)
    zap = filt(white(n), "highpass", 1500) * env(n, 0.001, 9)
    desc = saw(np.linspace(2200, 180, n), d) * env(n, 0.001, 7) * 0.5
    write_sound("cast_lightning", zap + desc)
    d = 0.9; t = t_axis(d); n = len(t)
    crack = white(n) * env(n, 0.0005, 30)
    thump = np.pad(sweep(130, 40, 0.3), (0, n - int(0.3 * SR))) * env(n, 0.001, 9)
    sizzle = filt(white(n), "highpass", 3500) * env(n, 0.02, 3) * (rng.random(n) > 0.55)
    write_sound("impact_lightning", crack * 1.1 + thump * 0.9 + sizzle * 0.5)

    # INFERNO
    d = 0.65; t = t_axis(d); n = len(t)
    roar = filt(white(n), "lowpass", np.linspace(400, 1400, n).mean()) * np.linspace(0.15, 1, n) ** 1.5
    crackle = filt((rng.random(n) > 0.995) * rng.uniform(0.4, 1, n), "highpass", 1200)
    write_sound("charge_inferno", roar + crackle * 0.8)
    d = 0.75; t = t_axis(d); n = len(t)
    wh = filt(white(n), "bandpass", [200, 2200]) * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 0.6
    low = sweep(160, 70, d) * env(n, 0.01, 4) * 0.5
    write_sound("cast_inferno", wh + low)
    d = 1.1; t = t_axis(d); n = len(t)
    boom = filt(white(n), "lowpass", 900) * env(n, 0.002, 4)
    sub = np.pad(sweep(110, 35, 0.6), (0, n - int(0.6 * SR))) * env(n, 0.001, 4)
    crack = filt((rng.random(n) > 0.994) * rng.uniform(0.4, 1, n), "highpass", 1500) * env(n, 0.05, 2)
    write_sound("impact_inferno", boom * 1.0 + sub * 0.9 + crack * 1.2)

    # EXPULSO
    d = 0.65; t = t_axis(d); n = len(t)
    hum = (np.sin(2 * np.pi * np.cumsum(np.linspace(70, 240, n)) / SR) + 0.5 * np.sin(2 * np.pi * np.cumsum(np.linspace(72, 246, n)) / SR))
    pulse = 0.6 + 0.4 * np.sin(2 * np.pi * np.linspace(6, 22, n) * t)
    shimmer = filt(white(n), "bandpass", [3000, 8000]) * np.linspace(0, 0.3, n)
    write_sound("charge_expulso", hum * pulse * np.linspace(0.2, 1, n) + shimmer)
    d = 0.5; t = t_axis(d); n = len(t)
    pew = sweep(1100, 120, d) * env(n, 0.001, 6)
    click = white(n) * env(n, 0.0005, 40)
    write_sound("cast_expulso", pew * 0.9 + click * 0.5 + filt(white(n), "bandpass", [800, 3000]) * env(n, 0.01, 8) * 0.3)
    d = 1.7; t = t_axis(d); n = len(t)
    sub = np.pad(sweep(130, 28, 1.0), (0, n - int(1.0 * SR))) * env(n, 0.002, 2.6)
    body = filt(white(n), "lowpass", 600) * env(n, 0.002, 3.2)
    flash = white(n) * env(n, 0.0005, 35)
    chime = np.zeros(n)
    for f, dl in [(880, 0.05), (1318, 0.09), (1760, 0.14), (2349, 0.2)]:
        p = int(dl * SR)
        L = n - p
        chime[p:] += np.sin(2 * np.pi * f * np.arange(L) / SR) * np.exp(-np.arange(L) / (0.45 * SR)) * 0.18
    write_sound("impact_expulso", sub * 1.0 + body * 0.9 + flash * 0.8 + chime)

    # UI
    d = 0.3; t = t_axis(d); n = len(t)
    write_sound("select", (np.sin(2 * np.pi * 988 * t) + 0.6 * np.sin(2 * np.pi * 1480 * t)) * env(n, 0.002, 7))
    d = 0.25; t = t_axis(d); n = len(t)
    write_sound("no_mana", np.sin(2 * np.pi * 150 * t) * env(n, 0.002, 9) + 0.3 * filt(white(n), "lowpass", 400) * env(n, 0.001, 12))

    sounds = {}
    for spell in ["aqua", "lightning", "inferno", "expulso"]:
        for kind in ["charge", "cast", "impact"]:
            nm = "%s_%s" % (kind, spell)
            sounds[nm] = {"sounds": ["%s:%s" % (MOD, nm)], "subtitle": "subtitles.%s.%s" % (MOD, nm)}
    for nm in ["select", "no_mana"]:
        sounds[nm] = {"sounds": ["%s:%s" % (MOD, nm)], "subtitle": "subtitles.%s.%s" % (MOD, nm)}
    with open(os.path.join(ASSETS, "sounds.json"), "w") as f:
        json.dump(sounds, f, indent=2)
    return list(sounds.keys())

# ------------------------------------------------------------------ lang + recipes
def make_lang(sound_names):
    en = {
        "itemGroup.wizmagic": "Wizarding Magic",
        "key.category.wizmagic.main": "Wizarding Magic",
        "key.wizmagic.spell_aqua": "Select Aqua Magic",
        "key.wizmagic.spell_lightning": "Select Lightning Arc",
        "key.wizmagic.spell_inferno": "Select Inferno",
        "key.wizmagic.spell_expulso": "Select Expulso",
        "key.wizmagic.menu": "Open Spell Selection",
        "screen.wizmagic.spells": "Choose your spell",
        "message.wizmagic.no_mana": "Not enough magical energy!",
        "tooltip.wizmagic.wood": "Wood: %s",
        "tooltip.wizmagic.core": "Core: %s",
        "tooltip.wizmagic.hint": "Right-click to cast the selected spell",
        "spell.wizmagic.aqua": "Aqua Magic",
        "spell.wizmagic.aqua.desc": "A stream of water. Pushes, slows and extinguishes fire.",
        "spell.wizmagic.lightning": "Lightning Arc",
        "spell.wizmagic.lightning.desc": "Branching lightning. Stuns and chains between targets.",
        "spell.wizmagic.inferno": "Inferno",
        "spell.wizmagic.inferno.desc": "A blazing orb. Burns targets and ignites blocks.",
        "spell.wizmagic.expulso": "Expulso",
        "spell.wizmagic.expulso.desc": "An explosive bolt. Area damage, knockback, breaks blocks.",
        "wand.wizmagic.wood.holly": "Holly", "wand.wizmagic.wood.elder": "Elder", "wand.wizmagic.wood.yew": "Yew",
        "wand.wizmagic.wood.willow": "Willow", "wand.wizmagic.wood.ebony": "Ebony", "wand.wizmagic.wood.vine": "Vine",
        "wand.wizmagic.core.phoenix": "Phoenix Feather", "wand.wizmagic.core.thestral": "Thestral Hair",
        "wand.wizmagic.core.dragon": "Dragon Heartstring", "wand.wizmagic.core.unicorn": "Unicorn Hair",
        "wand.wizmagic.core.thunderbird": "Thunderbird Tail Feather", "wand.wizmagic.core.veela": "Veela Hair",
        "item.wizmagic.wand_holly_phoenix": "Holly & Phoenix Feather Wand",
        "item.wizmagic.wand_elder_thestral": "Elder & Thestral Hair Wand",
        "item.wizmagic.wand_yew_dragon": "Yew & Dragon Heartstring Wand",
        "item.wizmagic.wand_willow_unicorn": "Willow & Unicorn Hair Wand",
        "item.wizmagic.wand_ebony_thunderbird": "Ebony & Thunderbird Feather Wand",
        "item.wizmagic.wand_vine_veela": "Vine & Veela Hair Wand",
    }
    pt = dict(en)
    pt.update({
        "itemGroup.wizmagic": "Magia dos Feiticeiros",
        "key.category.wizmagic.main": "Magia dos Feiticeiros",
        "key.wizmagic.spell_aqua": "Selecionar Magia da Água",
        "key.wizmagic.spell_lightning": "Selecionar Arco Elétrico",
        "key.wizmagic.spell_inferno": "Selecionar Inferno",
        "key.wizmagic.spell_expulso": "Selecionar Expulso",
        "key.wizmagic.menu": "Abrir seleção de feitiços",
        "screen.wizmagic.spells": "Escolhe o teu feitiço",
        "message.wizmagic.no_mana": "Energia mágica insuficiente!",
        "tooltip.wizmagic.wood": "Madeira: %s",
        "tooltip.wizmagic.core": "Núcleo: %s",
        "tooltip.wizmagic.hint": "Clique direito para lançar o feitiço selecionado",
        "spell.wizmagic.aqua": "Magia da Água",
        "spell.wizmagic.aqua.desc": "Jato de água. Empurra, abranda e apaga o fogo.",
        "spell.wizmagic.lightning": "Arco Elétrico",
        "spell.wizmagic.lightning.desc": "Relâmpago ramificado. Atordoa e salta entre alvos.",
        "spell.wizmagic.inferno": "Inferno",
        "spell.wizmagic.inferno.desc": "Orbe em chamas. Queima alvos e acende blocos.",
        "spell.wizmagic.expulso": "Expulso",
        "spell.wizmagic.expulso.desc": "Raio explosivo. Dano em área, empurra e parte blocos.",
        "wand.wizmagic.wood.holly": "Azevinho", "wand.wizmagic.wood.elder": "Sabugueiro", "wand.wizmagic.wood.yew": "Teixo",
        "wand.wizmagic.wood.willow": "Salgueiro", "wand.wizmagic.wood.ebony": "Ébano", "wand.wizmagic.wood.vine": "Videira",
        "wand.wizmagic.core.phoenix": "Pena de Fénix", "wand.wizmagic.core.thestral": "Crina de Testrálio",
        "wand.wizmagic.core.dragon": "Fibra de Coração de Dragão", "wand.wizmagic.core.unicorn": "Crina de Unicórnio",
        "wand.wizmagic.core.thunderbird": "Pena de Cauda de Ave-Trovão", "wand.wizmagic.core.veela": "Cabelo de Veela",
        "item.wizmagic.wand_holly_phoenix": "Varinha de Azevinho e Pena de Fénix",
        "item.wizmagic.wand_elder_thestral": "Varinha de Sabugueiro e Crina de Testrálio",
        "item.wizmagic.wand_yew_dragon": "Varinha de Teixo e Coração de Dragão",
        "item.wizmagic.wand_willow_unicorn": "Varinha de Salgueiro e Crina de Unicórnio",
        "item.wizmagic.wand_ebony_thunderbird": "Varinha de Ébano e Pena de Ave-Trovão",
        "item.wizmagic.wand_vine_veela": "Varinha de Videira e Cabelo de Veela",
    })
    sub_en = {"aqua": "Aqua Magic", "lightning": "Lightning Arc", "inferno": "Inferno", "expulso": "Expulso"}
    for nm in sound_names:
        if nm == "select":
            en["subtitles.wizmagic.select"] = "Spell selected"; pt["subtitles.wizmagic.select"] = "Feitiço selecionado"
        elif nm == "no_mana":
            en["subtitles.wizmagic.no_mana"] = "Out of magical energy"; pt["subtitles.wizmagic.no_mana"] = "Sem energia mágica"
        else:
            kind, spell = nm.split("_", 1)
            en["subtitles.wizmagic." + nm] = {"charge": "Spell charges", "cast": "Spell cast", "impact": "Spell impact"}[kind] + " (" + sub_en[spell] + ")"
            pt["subtitles.wizmagic." + nm] = {"charge": "Feitiço a carregar", "cast": "Feitiço lançado", "impact": "Impacto do feitiço"}[kind]
    for code, d in [("en_us", en), ("pt_pt", pt), ("pt_br", pt)]:
        with open(os.path.join(ASSETS, "lang", code + ".json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

def make_recipes():
    cores = {
        "holly_phoenix": "minecraft:blaze_powder", "elder_thestral": "minecraft:phantom_membrane",
        "yew_dragon": "minecraft:fire_charge", "willow_unicorn": "minecraft:amethyst_shard",
        "ebony_thunderbird": "minecraft:lightning_rod", "vine_veela": "minecraft:ender_pearl",
    }
    for wid, core in cores.items():
        recipe = {
            "type": "minecraft:crafting_shaped", "category": "equipment",
            "pattern": ["  C", " S ", "S  "],
            "key": {"C": core, "S": "minecraft:stick"},
            "result": {"id": "%s:wand_%s" % (MOD, wid), "count": 1},
        }
        with open(os.path.join(DATA, "recipe", "wand_%s.json" % wid), "w") as f:
            json.dump(recipe, f, indent=2)

if __name__ == "__main__":
    make_particles(); write_particle_defs(); make_wand_textures(); make_wand_models(); make_icons()
    names = make_sounds(); make_lang(names); make_recipes()
    print("assets generated")
