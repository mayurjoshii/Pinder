"""Synthesizes the promo soundtrack (promo/pinder-promo.wav).

Upbeat 133 BPM track. The beat grid is phased so downbeats land on the key
visual moments in pinder-promo.html (logo reveal 2.65s, payoff 7.6s,
end card ~8.9s), and one-shot SFX sit on each swipe/transition.

    python3 promo/music.py
"""
import wave
import numpy as np

SR = 44100
DUR = 10.0
N = int(SR * DUR)
BEAT = 0.45            # 133.3 BPM
PHASE = 2.65 - 5 * BEAT  # a beat lands exactly on 2.65s
rng = np.random.default_rng(3)
mix = np.zeros(N)


def t_arr(d):
    return np.arange(int(d * SR)) / SR


def add(sig, at, gain=1.0):
    i = int(at * SR)
    if i >= N:
        return
    j = min(N, i + len(sig))
    mix[i:j] += sig[: j - i] * gain


def onepole_lp(x, cutoff):
    """One-pole lowpass; cutoff may be a scalar or per-sample array (Hz)."""
    c = np.broadcast_to(np.asarray(cutoff, float), x.shape)
    a = 1 - np.exp(-2 * np.pi * c / SR)
    y = np.empty_like(x)
    acc = 0.0
    for k in range(len(x)):
        acc += a[k] * (x[k] - acc)
        y[k] = acc
    return y


def hp(x, cutoff):
    return x - onepole_lp(x, cutoff)


def saw(freq, d, detune=0.0):
    tt = t_arr(d)
    f = freq * (1 + detune)
    return 2 * ((tt * f + rng.random()) % 1.0) - 1


def note(n):  # semitones from A4
    return 440.0 * 2 ** (n / 12)


# ---------- instruments ----------
def kick():
    tt = t_arr(0.35)
    f = 45 + 110 * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-tt * 9) + 0.3 * np.sin(ph * 2) * np.exp(-tt * 40)


def clap():
    tt = t_arr(0.2)
    nz = hp(rng.standard_normal(len(tt)), 900)
    env = np.exp(-tt * 22) + 0.6 * np.exp(-((tt - 0.012) % 0.012) * 300) * (tt < 0.036)
    return nz * env * 0.5


def hat(open_=False):
    tt = t_arr(0.18 if open_ else 0.05)
    nz = hp(rng.standard_normal(len(tt)), 7000)
    return nz * np.exp(-tt * (18 if open_ else 80)) * 0.35


def bass(freq, d):
    tt = t_arr(d)
    x = sum(np.sin(2 * np.pi * freq * k * tt) / k for k in range(1, 5))
    return x * np.minimum(1, tt * 200) * np.exp(-tt * 5) * 0.55


def pluck(freq, d=0.18):
    tt = t_arr(d)
    x = np.sign(np.sin(2 * np.pi * freq * tt)) * 0.5 + np.sin(2 * np.pi * freq * 2 * tt) * 0.3
    return onepole_lp(x, 5000) * np.exp(-tt * 18) * 0.35


def supersaw(freqs, d, cutoff):
    x = np.zeros(int(d * SR))
    for f in freqs:
        for dt in (-0.008, 0, 0.007):
            x += saw(f, d, dt)
    x /= 3 * len(freqs)
    return onepole_lp(x, cutoff)


def whoosh(d, f0, f1, gain=0.6):
    tt = t_arr(d)
    nz = rng.standard_normal(len(tt))
    cut = f0 * (f1 / f0) ** (tt / d)
    band = onepole_lp(nz, cut) - onepole_lp(nz, cut * 0.35)
    env = np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 1.5
    return band * env * gain


def impact():
    tt = t_arr(1.6)
    boom = np.sin(2 * np.pi * np.cumsum(38 + 60 * np.exp(-tt * 12)) / SR) * np.exp(-tt * 3.2)
    crash = hp(rng.standard_normal(len(tt)), 5000) * np.exp(-tt * 2.6) * 0.35
    return boom * 0.9 + crash


def riser(d):
    tt = t_arr(d)
    p = tt / d
    nz = hp(rng.standard_normal(len(tt)), 400 + 6000 * p)
    tone = np.sin(2 * np.pi * np.cumsum(200 + 900 * p ** 2) / SR)
    return (nz * 0.3 + tone * 0.12) * p ** 2.2


def click():
    tt = t_arr(0.03)
    return np.sin(2 * np.pi * 2400 * tt) * np.exp(-tt * 250) * 0.5


# ---------- arrangement ----------
beats = [PHASE + i * BEAT for i in range(-1, 40) if PHASE + i * BEAT >= -0.01]
CHORDS = [  # (bass root, pad voicing) per bar: Am F C G
    (note(-36), [note(-12), note(-9), note(-5)]),
    (note(-40), [note(-16), note(-12), note(-9)]),
    (note(-33), [note(-14), note(-9), note(-5)]),
    (note(-38), [note(-14), note(-10), note(-7)]),
]
BAR = 4 * BEAT
bar_start = PHASE - BAR  # so bar boundaries line up with the beat grid


def chord_at(t):
    return CHORDS[int((t - bar_start) // BAR) % 4]


def groove_on(t):
    return (0.3 <= t < 2.2) or (3.55 <= t < 8.85)


# drums + bass
for b in beats:
    for half in (0, 0.5):
        t = b + half * BEAT
        if t < 0 or t >= DUR:
            continue
        if groove_on(t):
            if half == 0:
                add(kick(), t, 0.9)
                bi = round((b - PHASE) / BEAT)
                if bi % 2 == 1:
                    add(clap(), t, 0.7)
            add(hat(open_=half == 0.5), t, 0.5 if half else 0.3)
            add(bass(chord_at(t)[0], BEAT * 0.48), t, 0.8)

# snare roll into the logo reveal and into the payoff
for start, end in ((1.75, 2.65), (7.15, 7.6)):
    t, k = start, 0
    while t < end - 0.01:
        add(clap(), t, 0.25 + 0.45 * (t - start) / (end - start))
        t += BEAT / 4

# pad with sidechain pump, filter opens up across the video
pad = np.zeros(N)
bar_t = bar_start
while bar_t < DUR:
    s = max(0.0, bar_t)
    d = min(bar_t + BAR, DUR) - s
    if d > 0.01:
        cut = 900 if s < 3.55 else 2600
        seg = supersaw(chord_at(s + 0.01)[1], d, cut)
        i = int(s * SR)
        pad[i:i + len(seg)] += seg[: N - i]
    bar_t += BAR
tt = np.arange(N) / SR
pump = 1 - 0.65 * np.exp(-((tt - PHASE) % BEAT) / 0.09)
pad_gain = np.where(tt < 0.3, tt / 0.3, 1.0) * np.where(tt > 8.9, np.exp(-(tt - 8.9) * 1.6), 1.0)
mix += pad * pump * pad_gain * 0.55

# 16th-note arp over the swipe section and payoff
t = 3.55
while t < 8.85:
    chord = chord_at(t)[1]
    step = int(round((t - 3.55) / (BEAT / 4)))
    f = chord[[0, 1, 2, 1][step % 4]] * (2 if step % 8 >= 4 else 1) * 2
    add(pluck(f), t, 0.45 if t > 7.6 else 0.3)
    t += BEAT / 4

# tiles popping in: little blips
for i, t in enumerate(np.linspace(0.12, 1.2, 12)):
    add(pluck(note([3, 7, 10, 15][i % 4])), t, 0.18)

# ---------- SFX synced to picture ----------
add(riser(1.25), 1.4, 0.8)                     # tiles collapse
add(impact(), 2.65, 1.0)                       # logo reveal
add(whoosh(0.5, 300, 4000, 0.5), 3.55)         # logo up, deck in
for t0 in (3.95, 4.75, 5.35, 7.05):            # card swipes
    add(whoosh(0.45, 600, 7000), t0 - 0.12)
for tk in (3.8, 4.6, 5.2, 6.9):                # arrow key presses
    add(click(), tk, 0.6)
add(whoosh(0.35, 1500, 5000, 0.35), 5.95)      # duplicate twin slides out
add(whoosh(0.4, 5000, 400, 0.55), 6.6)         # twin dropped in trash
add(impact(), 7.6, 1.0)                        # "GB freed"
for i in range(14):                            # confetti sparkle
    add(pluck(note(19 + [0, 4, 7, 12][i % 4]), 0.12), 7.75 + i * 0.045, 0.2)
add(supersaw(CHORDS[0][1] + [note(0)], 1.1, 3000), 8.9, 0.6)  # end card chord
add(impact(), 8.9, 0.6)

# ---------- master ----------
mix = np.tanh(mix * 1.2)
fade = np.where(tt > 9.4, np.clip((10 - tt) / 0.6, 0, 1), 1.0)
mix *= fade
mix /= np.max(np.abs(mix)) / 0.89
out = (mix * 32767).astype(np.int16)
stereo = np.repeat(out[:, None], 2, axis=1)
with wave.open('promo/pinder-promo.wav', 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(stereo.tobytes())
print('wrote promo/pinder-promo.wav')
