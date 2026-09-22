"""
song_lights.py — SongLights plugin: music-driven lighting bake.

CLAUDE-NOTE (2026-09-21): Design choice — we BAKE Sequencer keyframes from Audio Synesthesia
NRT analysis instead of running a Tick blueprint. Reasons: the show is prerendered through
Movie Render Queue for disguise, so determinism matters more than latency; every light curve
is visible/editable in Sequencer; and switching songs is a single re-bake, no graph rewiring.

Entry points (called by BP_SongDirector's Details-panel buttons, or from the Python console):
    import song_lights; song_lights.bake_song('/Game/Music/SW_MySong', camera='/Game/Map.Map:PersistentLevel.CineCameraActor_0', fps=30)
    song_lights.setup_rig(origin=(0, 0, 0))   # (re)create the starter fixture rig — idempotent by label
The Level Sequence is created next to the SoundWave as LS_<SongName>, one per song.

Light groups are discovered by ACTOR TAG so fixtures can be added/moved freely in the level:
    Song_Bass   -> low bands   (kick / 808)           slow-ish decay, blue-violet uplights
    Song_Mids   -> mid bands   (keys / melody)        smooth, amber beams
    Song_Highs  -> high bands  (hats / air)           fast attack+decay, cyan
    Song_Kick   -> onsets      (transient hits)       strobe impulse
    Song_Loud   -> loudness    (overall energy)       slow wash
Per-song auto-ranging: each signal is remapped from its 10th..98th percentile over the whole
track to 0..1, so a quiet acoustic song and a loud trap beat both use the full lighting range.
"""
import math
import unreal
import numpy as np

FPS = 30

# group -> (tag, base_intensity_fraction, peak_multiplier_of_actor_intensity, attack_s, decay_s, gamma)
# The actor's authored Intensity is treated as the PEAK; base = peak * base_frac.
GROUPS = {
    # lo = percentile treated as "dark" — raised for bass/highs so the sustained 808 bed and hat bed
    # sit at black and only the hits register (tuned by eye on Longevity, 2026-09-21).
    'Song_Bass':  dict(base=0.00, attack=0.01, decay=0.16, gamma=2.2, lo=35),
    'Song_Mids':  dict(base=0.10, attack=0.08, decay=0.35, gamma=1.6, lo=25),
    'Song_Highs': dict(base=0.00, attack=0.00, decay=0.07, gamma=2.0, lo=45),
    'Song_Kick':  dict(base=0.00, attack=0.00, decay=0.12, gamma=1.0, lo=10),
    'Song_Loud':  dict(base=0.30, attack=0.30, decay=0.60, gamma=1.0, lo=10),
}

# ------------------------------------------------------------- shaft show -------------------------
# CLAUDE-NOTE (2026-09-22): "Who My God Is" wants a CAVE LIGHT SHAFT, not a concert rig — one
# aperture, overlapping concentric cones that widen and brighten across the song's eight sections.
# All layers share ONE origin so they read as a single beam that opens and closes.
#
# Two things must be true in the level or no beam renders at all (both cost a session to find):
#   * volumetric fog must be enabled AND not owned by Ultra Dynamic Sky — UDS re-applies its own
#     ExponentialHeightFog values every tick and silently reverts writes. Delete UDS in interiors.
#   * the shaft light must hang BELOW the ceiling geometry, or the ceiling occludes it completely.
# Beyond ~10k cd the beam core clips to white and bloom smears it into a ball: brightness belongs in
# volumetric_scattering_intensity, not in candelas.

SHAFT_TAGS = ('Song_Shaft', 'Song_ShaftSide')
SHAFT_COLOR = (0.62, 0.78, 1.00)          # cool daylight through rock
SHAFT_LAYERS = [
    # label,          tag,               (pitch,yaw), cone(in,out), cd,    vol,  music signal
    ('L_Shaft_Main',  'Song_Shaft',      (-90,   0),  (3.0,   8.0), 7000,  18.0, 'Song_Loud'),
    ('L_Shaft_L',     'Song_ShaftSide',  (-87,  -6),  (7.0,  16.0), 2600,  11.0, 'Song_Mids'),
    ('L_Shaft_R',     'Song_ShaftSide',  (-87,   6),  (12.0, 26.0), 1400,   7.0, 'Song_Bass'),
]

# The eight treatment looks. Values are MULTIPLIERS on each layer's authored peak, so retuning the
# rig in the level automatically retunes the whole show.
#   core/halo/wide = per-layer intensity scale   cone = cone-angle scale
#   vol            = volumetric scattering scale  accent = level for non-shaft "accent" lights
#   snap           = seconds to reach the look (short = hits on the downbeat, long = eases in)
PHASE_LOOKS = {
    'dark':         dict(core=0.15, halo=0.00, wide=0.00, cone=0.55, vol=0.70, accent=0.03, snap=3.0),
    'discover':     dict(core=0.35, halo=0.10, wide=0.00, cone=0.70, vol=0.90, accent=0.10, snap=2.5),
    'grow':         dict(core=0.55, halo=0.30, wide=0.05, cone=0.85, vol=1.00, accent=0.20, snap=2.5),
    'reveal':       dict(core=1.00, halo=0.75, wide=0.45, cone=1.15, vol=1.20, accent=0.45, snap=0.25),
    'deeper':       dict(core=0.40, halo=0.18, wide=0.05, cone=0.75, vol=1.00, accent=0.15, snap=2.0),
    'tension':      dict(core=0.80, halo=0.55, wide=0.30, cone=0.95, vol=1.30, accent=0.35, snap=0.3),
    'isolated':     dict(core=0.45, halo=0.05, wide=0.00, cone=0.50, vol=1.10, accent=0.05, snap=1.5),
    'breakthrough': dict(core=1.35, halo=1.00, wide=0.85, cone=1.35, vol=1.40, accent=0.65, snap=0.25),
    'out':          dict(core=0.00, halo=0.00, wide=0.00, cone=0.50, vol=0.60, accent=0.00, snap=4.0),
}
# The treatment's storyboard order. auto_phases() lays these onto detected section boundaries.
LOOK_ARC = ['dark', 'discover', 'grow', 'reveal', 'deeper', 'tension', 'isolated', 'breakthrough']
LAYER_KEY = {'L_Shaft_Main': 'core', 'L_Shaft_L': 'halo', 'L_Shaft_R': 'wide'}

# CLAUDE-NOTE (2026-09-22): the first version drove each layer from a SUSTAINED signal (Song_Loud /
# Song_Mids). Those are slow envelopes auto-ranged over the whole track, so inside a loud section
# they pin near 1.0 and hold — `1 + gain*music` collapsed to a constant and the beam looked like it
# only stepped between phases. Layers are now driven mostly by TRANSIENTS (deviation above a rolling
# baseline, see _transient) so hits punch through at every phase level, loud section or quiet.
#   per layer: (sustain signal, sustain weight, transient signal, transient weight)
#
# CLAUDE-NOTE: the halo deliberately does NOT use Song_Highs. The drums stem above 4 kHz in
# "Who My God Is" carries a CLICK TRACK — autocorrelation 0.973 at a fixed 0.910 s lag, where
# musical hi-hats score ~0.6 — so the halo ticked like a metronome. Driving it from Song_Mids
# (the melodic stem) keeps the layer and its pulse musical. If a song's highs are clean, Song_Highs
# is still a fine transient source; check regularity before trusting it.
SHAFT_DRIVE = {
    'core': ('Song_Loud', 0.20, 'Song_Kick', 1.00),
    'halo': ('Song_Mids', 0.30, 'Song_Mids', 0.85),
    'wide': ('Song_Bass', 0.25, 'Song_Bass', 0.95),
}
REACT_GAIN = {'core': 0.55, 'halo': 0.85, 'wide': 1.10}   # intensity swing around the phase level
CONE_REACT = {'core': 0.10, 'halo': 0.16, 'wide': 0.22}   # beam flares wider on hits
VOL_REACT = 0.35                                          # haze pulses with the beam


def _transient(x, dt, window=1.5):
    """Deviation of `x` above its own rolling baseline, renormalised to 0..1.
    This is what makes a kick read identically in a quiet verse and a loud chorus — absolute level
    is handled by the phase curve, so the music term must carry only the *change*."""
    k = max(1, int(round(window / dt)))
    base = np.convolve(x, np.ones(k) / float(k), mode='same')
    d = np.clip(x - base, 0.0, None)
    hi = float(np.percentile(d, 97))
    return np.clip(d / hi, 0.0, 1.0) if hi > 1e-6 else np.zeros_like(d)


def layer_drive(key, sig, dt):
    """0..~1.2 music drive for one shaft layer: a little sustain plus a lot of transient."""
    s_key, s_w, t_key, t_w = SHAFT_DRIVE[key]
    sustain = sig.get(s_key, sig['Song_Loud'])
    raw = sig.get(t_key, sig['Song_Kick'])
    # Song_Kick is already an onset impulse train; everything else needs baseline removal.
    trans = raw if t_key == 'Song_Kick' else _transient(raw, dt)
    return np.clip(s_w * sustain + t_w * trans, 0.0, 1.5)


# Wall washes. The shaft alone leaves the cave walls black; these lift the rock on the bigger
# sections so "full reveal" actually reveals something. They are driven by the phase 'accent' level
# only (never per-frame music) so the walls breathe with the song form instead of flickering.
ACCENT_RIG = [
    # label,            loc,                   (pitch,yaw,roll),  color RGB,            cd,   cone(in,out)
    # CLAUDE-NOTE: these are a WHISPER, not a wash. At ~2500 cd they flooded the cave, flattened the
    # shaft's contrast and exposed the flat ceiling plates. Low hundreds keeps the rock readable
    # while the cave stays a dark room with one beam in it. Aimed low so they miss the ceiling.
    # Aimed for the wide 4:1 framing (camera at ~(2312,679,645) looking yaw -143.6): each wash
    # rakes one flank of rock that the frame's edges land on.
    ('L_Wall_Wash_L',   (  900,  -300, 620),   (-12, 148, 0),  (0.45, 0.58, 0.95),  300, (26, 58)),
    ('L_Wall_Wash_R',   (  600, -1400, 620),   (-12, -53, 0),  (0.50, 0.62, 0.95),  260, (26, 58)),
    ('L_Wall_Wash_B',   ( 1200,   200, 700),   (-14, -135, 0), (0.40, 0.54, 0.92),  220, (30, 62)),
]


def setup_accent_rig(origin=(0.0, 0.0, 0.0)):
    """Create (or re-place) the cave wall washes. Idempotent by actor label."""
    sub = _actor_subsystem()
    o = unreal.Vector(*origin)
    for label, loc, pyr, rgb, cd, cone in ACCENT_RIG:
        a = _find_actor(label)
        pos = unreal.Vector(*loc) + o
        rot = unreal.Rotator(roll=pyr[2], pitch=pyr[0], yaw=pyr[1])
        if a is None:
            a = sub.spawn_actor_from_class(unreal.SpotLight, pos, rot)
            a.set_actor_label(label)
        else:
            a.set_actor_location_and_rotation(pos, rot, False, True)
        a.tags = [unreal.Name('Song_Accent')]
        lc = a.light_component
        lc.set_mobility(unreal.ComponentMobility.MOVABLE)
        lc.set_editor_property('intensity_units', unreal.LightUnits.CANDELAS)
        lc.set_editor_property('intensity', float(cd))
        lc.set_editor_property('light_color', unreal.Color(
            int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255), 255))   # BGRA
        lc.set_editor_property('inner_cone_angle', float(cone[0]))
        lc.set_editor_property('outer_cone_angle', float(cone[1]))
        lc.set_editor_property('attenuation_radius', 5000.0)
        # CLAUDE-NOTE: wall washes deliberately do NOT scatter — a second volumetric source competes
        # with the shaft and muddies it. They light surfaces only.
        lc.set_editor_property('volumetric_scattering_intensity', 0.0)
    unreal.log('song_lights.setup_accent_rig: %d wall washes' % len(ACCENT_RIG))
    return [r[0] for r in ACCENT_RIG]


def setup_shaft_rig(origin=(138.0, -925.0, 1290.0)):
    """Create (or re-place) the cave shaft: concentric spotlights sharing one origin.
    Idempotent by actor label. `origin` should sit just BELOW the ceiling at the aperture."""
    sub = _actor_subsystem()
    o = unreal.Vector(*origin)
    made = []
    for label, tag, (pitch, yaw), cone, cd, vol, _sig in SHAFT_LAYERS:
        a = _find_actor(label)
        rot = unreal.Rotator(roll=0.0, pitch=float(pitch), yaw=float(yaw))
        if a is None:
            a = sub.spawn_actor_from_class(unreal.SpotLight, o, rot)
            a.set_actor_label(label)
        else:
            a.set_actor_location_and_rotation(o, rot, False, True)
        a.tags = [unreal.Name(tag)]
        lc = a.light_component
        lc.set_mobility(unreal.ComponentMobility.MOVABLE)
        lc.set_editor_property('intensity_units', unreal.LightUnits.CANDELAS)
        lc.set_editor_property('intensity', float(cd))
        lc.set_editor_property('light_color', unreal.Color(
            int(SHAFT_COLOR[2] * 255), int(SHAFT_COLOR[1] * 255), int(SHAFT_COLOR[0] * 255), 255))  # BGRA
        lc.set_editor_property('inner_cone_angle', float(cone[0]))
        lc.set_editor_property('outer_cone_angle', float(cone[1]))
        lc.set_editor_property('volumetric_scattering_intensity', float(vol))
        lc.set_editor_property('attenuation_radius', 4000.0)
        lc.set_editor_property('source_radius', 4.0)
        lc.set_editor_property('cast_shadows', True)
        made.append(label)
    unreal.log('song_lights.setup_shaft_rig: %d shaft layers at %s' % (len(made), origin))
    return made


def shaft_lights():
    """{label: actor} for the shaft layers present in the level."""
    out = {}
    for a in _actor_subsystem().get_all_level_actors():
        if isinstance(a, unreal.Light) and a.get_actor_label() in LAYER_KEY:
            out[a.get_actor_label()] = a
    return out


def accent_lights():
    """Every non-shaft light — dimmed and lifted by the phase 'accent' level so the cave walls
    only read once the song opens up."""
    # CLAUDE-NOTE: SkyLight is excluded — animating ambient to zero crushes the whole cave to black
    # and fights the shaft instead of supporting it. It stays a constant low bounce.
    return [a for a in _actor_subsystem().get_all_level_actors()
            if isinstance(a, unreal.Light) and not isinstance(a, unreal.SkyLight)
            and a.get_actor_label() not in LAYER_KEY]


def auto_phases(an, fps=FPS):
    """Detect section boundaries from the mix and lay the treatment's look arc onto them.
    Returns [(start_seconds, look_name), ...] always beginning at 0 and ending with 'out'."""
    mix = an['mix']
    b = mix['bands']
    f = (b - b.mean(axis=0)) / (b.std(axis=0) + 1e-6)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    sim = f @ f.T
    hop_s = HOP / float(an['sr'])
    L = max(8, int(round(6.0 / hop_s)))                    # ~6 s checkerboard half-kernel
    g = np.exp(-((np.arange(2 * L) - L + 0.5) / (L * 0.5)) ** 2)
    kern = np.outer(g, g) * np.where(
        (np.arange(2 * L)[:, None] < L) == (np.arange(2 * L)[None, :] < L), 1.0, -1.0)
    n = sim.shape[0]
    nov = np.zeros(n)
    for i in range(L, n - L):
        nov[i] = (sim[i - L:i + L, i - L:i + L] * kern).sum()
    nov = np.clip(nov, 0, None)
    nov /= (nov.max() + 1e-9)
    min_gap = int(round(12.0 / hop_s))                     # sections are at least 12 s apart
    peaks = sorted(((nov[i], i) for i in range(1, n - 1)
                    if nov[i] > 0.18 and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]), reverse=True)
    chosen = []
    for _v, i in peaks:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
        if len(chosen) >= len(LOOK_ARC) - 1:
            break
    times = [0.0] + sorted(i * hop_s for i in chosen)
    looks = [LOOK_ARC[min(k, len(LOOK_ARC) - 1)] for k in range(len(times))]

    # CLAUDE-NOTE: assigning the arc purely by position put 'isolated' on the loudest chorus and
    # 'breakthrough' on the silent tail. The three structural extremes are pinned by ENERGY instead,
    # and only the connective sections keep their positional look.
    edges = times + [an['duration']]
    lt, loud = mix['times'], mix['loud']
    energy = []
    for a0, a1 in zip(edges[:-1], edges[1:]):
        m = (lt >= a0) & (lt < a1)
        energy.append(float(loud[m].mean()) if m.any() else -120.0)
    peak = max(energy)
    if len(energy) > 2:
        if energy[-1] < peak - 18.0:                       # near-silent trailing section = outro
            looks[-1] = 'out'
        tail_i = len(energy) - (1 if looks[-1] == 'out' else 0)
        loud_i = max(range(len(energy) // 2, tail_i), key=lambda i: energy[i])
        # everything from the loudest late section to the outro stays at full breakthrough —
        # otherwise the sections after it fall back to their positional look (the final chorus
        # was being assigned 'isolated' and dropping the beam out at the song's biggest moment).
        for i in range(loud_i, tail_i):
            looks[i] = 'breakthrough'
        quiet = list(range(max(1, len(energy) // 3), loud_i))
        if quiet:
            looks[min(quiet, key=lambda i: energy[i])] = 'isolated'   # quietest before it = bridge
    phases = list(zip(times, looks))
    if looks[-1] != 'out':
        tail = max(0.0, an['duration'] - 9.0)
        if tail > phases[-1][0] + 4.0:
            phases.append((tail, 'out'))
    unreal.log('song_lights.auto_phases: ' + ', '.join('%.1fs %s' % p for p in phases))
    return phases


def phase_tracks(phases, duration, fps=FPS):
    """Sample the phase look table to per-frame arrays: {field: ndarray(n)} for
    core/halo/wide/cone/vol/accent. Each transition ramps over that look's `snap` seconds."""
    n = int(math.ceil(duration * fps)) + 1
    t = np.arange(n) / float(fps)
    fields = ('core', 'halo', 'wide', 'cone', 'vol', 'accent')
    out = {k: np.zeros(n) for k in fields}
    seq = sorted(phases, key=lambda p: p[0])
    for k in fields:
        y = np.zeros(n)
        prev = PHASE_LOOKS[seq[0][1]][k]
        for idx, (start, look) in enumerate(seq):
            look_d = PHASE_LOOKS[look]
            target = look_d[k]
            snap = max(1.0 / fps, float(look_d['snap']))
            end = seq[idx + 1][0] if idx + 1 < len(seq) else duration + 1.0
            ramp = (t >= start) & (t < start + snap)
            hold = (t >= start + snap) & (t < end)
            y[ramp] = prev + (target - prev) * ((t[ramp] - start) / snap)
            y[hold] = target
            prev = target
        y[t < seq[0][0]] = PHASE_LOOKS[seq[0][1]][k]
        out[k] = y
    return out


# ---------------------------------------------------------------- rig -----------------------------

# CLAUDE-NOTE: candela falls off 1/d^2 — the 13-16 m throws (mid beams, strobe) need thousands of cd
# to register, while the 1-3 m uplights work at ~150 cd. Tuned by eye against the hero camera.
# CLAUDE-NOTE: the cave ceiling slab (actor Boolean3) tops out at z~1425; anything above it is occluded.
RIG = [
    # label,            class,                  tag,          location,              rotation (pitch,yaw,roll), color RGB,           intensity(cd), inner/outer cone
    ('L_Bass_UL_FL',    unreal.SpotLight,  'Song_Bass',  (-1500, -900, 20),    (65, 180, 0),   (0.20, 0.10, 1.00), 150, (35, 60)),
    ('L_Bass_UL_FR',    unreal.SpotLight,  'Song_Bass',  ( 1500, -900, 20),    (65,   0, 0),   (0.20, 0.10, 1.00), 150, (35, 60)),
    ('L_Bass_UL_BL',    unreal.SpotLight,  'Song_Bass',  (-1000, -2800, 20),   (70, 200, 0),   (0.35, 0.10, 1.00), 150, (35, 60)),
    ('L_Bass_UL_BR',    unreal.SpotLight,  'Song_Bass',  ( 1000, -2800, 20),   (70, -20, 0),   (0.35, 0.10, 1.00), 150, (35, 60)),
    ('L_Mid_Beam_L',    unreal.SpotLight,  'Song_Mids',  ( -700, -1200, 1250), (-24, -118, 0), (1.00, 0.55, 0.15), 8000, (12, 30)),
    ('L_Mid_Beam_R',    unreal.SpotLight,  'Song_Mids',  (  700, -1200, 1250), (-24, -62, 0),  (1.00, 0.45, 0.10), 8000, (12, 30)),
    ('L_High_1',        unreal.SpotLight,  'Song_Highs', ( -900, -600, 500),   (10, 160, 0),   (0.20, 0.95, 1.00), 25, (20, 45)),
    ('L_High_2',        unreal.SpotLight,  'Song_Highs', (  900, -600, 500),   (10,  20, 0),   (0.20, 0.95, 1.00), 25, (20, 45)),
    ('L_High_3',        unreal.SpotLight,  'Song_Highs', ( -900, -1800, 600),  (15, 170, 0),   (0.10, 0.80, 1.00), 25, (20, 45)),
    ('L_High_4',        unreal.SpotLight,  'Song_Highs', (  900, -1800, 600),  (15,  10, 0),   (0.10, 0.80, 1.00), 25, (20, 45)),
    ('L_High_5',        unreal.SpotLight,  'Song_Highs', ( -400, -3000, 700),  (20, 200, 0),   (0.30, 1.00, 0.90), 25, (20, 45)),
    ('L_High_6',        unreal.SpotLight,  'Song_Highs', (  400, -3000, 700),  (20, -20, 0),   (0.30, 1.00, 0.90), 25, (20, 45)),
    ('L_Kick_Strobe',   unreal.SpotLight,  'Song_Kick',  (  764, -1693, 1100), (-35, 119, 0),  (1.00, 1.00, 1.00), 12000, (35, 65)),
    ('L_Loud_Wash',     unreal.RectLight,  'Song_Loud',  ( -148, 1200, 350),   (-8, -90, 0),   (1.00, 0.60, 0.30), 6, None),
]


def _actor_subsystem():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def _rot(pyr):
    # CLAUDE-NOTE: unreal.Rotator's positional order is (roll, pitch, yaw) — RIG tuples are (pitch, yaw, roll).
    return unreal.Rotator(roll=pyr[2], pitch=pyr[0], yaw=pyr[1])


def _find_actor(label):
    for a in _actor_subsystem().get_all_level_actors():
        if a.get_actor_label() == label:
            return a
    return None


def setup_rig(origin=(0.0, 0.0, 0.0)):
    """Create (or refresh) the starter fixture rig, offset from `origin`. Idempotent: matches by actor label.
    The RIG table was laid out for a ~40 m cave; treat it as a starting point and move/retag freely."""
    sub = _actor_subsystem()
    o = unreal.Vector(*origin)
    made = []
    for label, cls, tag, loc, rot, rgb, intensity, cone in RIG:
        a = _find_actor(label)
        pos = unreal.Vector(*loc) + o
        if a is None:
            a = sub.spawn_actor_from_class(cls, pos, _rot(rot))
            a.set_actor_label(label)
        else:
            a.set_actor_location_and_rotation(pos, _rot(rot), False, True)
        a.tags = [unreal.Name(tag)]
        lc = a.light_component
        lc.set_mobility(unreal.ComponentMobility.MOVABLE)
        # CLAUDE-NOTE: mobility must be set before the light properties or the stationary/static
        # gate silently rejects the writes (see memory: ue-light-setters-silently-noop).
        lc.set_editor_property('intensity_units', unreal.LightUnits.CANDELAS)
        lc.set_editor_property('intensity', float(intensity))
        lc.set_editor_property('light_color', unreal.Color(int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255), 255))  # BGRA
        lc.set_editor_property('cast_volumetric_shadow', False)
        lc.set_editor_property('volumetric_scattering_intensity', 1.0)
        if cone and isinstance(lc, unreal.SpotLightComponent):
            lc.set_editor_property('inner_cone_angle', float(cone[0]))
            lc.set_editor_property('outer_cone_angle', float(cone[1]))
        if isinstance(lc, unreal.RectLightComponent):
            lc.set_editor_property('source_width', 1200.0)
            lc.set_editor_property('source_height', 300.0)
        made.append(label)
    unreal.log('song_lights.setup_rig: %d fixtures ready' % len(made))
    return made


def tagged_lights():
    out = {g: [] for g in GROUPS}
    for a in _actor_subsystem().get_all_level_actors():
        if not isinstance(a, unreal.Light):
            continue
        for t in a.tags:
            if str(t) in out:
                out[str(t)].append(a)
    return out


# ---------------------------------------------------------------- analysis ------------------------
# CLAUDE-NOTE (2026-09-21): analysis is done HERE in numpy, not with Audio Synesthesia NRT assets.
# Driving those assets from Python crashed the editor twice (LoudnessNRT.cpp:148
# `IsSortedChronologically` assert — the analysis is async and re-triggering/sampling it early
# corrupts the result). Reading the WAV ourselves is deterministic and has no engine state.

import wave

N_FFT = 2048
HOP = 512
BAND_EDGES_HZ = [40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]   # 8 octave bands


def _source_wav(sound):
    """Path of the file the SoundWave was imported from (the bake reads that, not the uasset)."""
    aid = sound.get_editor_property('asset_import_data')
    path = aid.get_first_filename() if aid else ''
    if not path or not unreal.Paths.file_exists(path):
        raise RuntimeError('song_lights: source WAV for %s not found (%r). Re-import it from a WAV on disk.'
                           % (sound.get_name(), path))
    return path


def _read_wav_mono(path):
    """Minimal RIFF/WAVE reader -> (mono float32, sample_rate).
    CLAUDE-NOTE: the stdlib `wave` module rejects float WAVs (format tag 3) and WAVE_FORMAT_EXTENSIBLE
    (0xFFFE) — both are common exports from DAWs and converters — so we parse the chunks ourselves.
    Handles 8/16/24/32-bit PCM and 32/64-bit float, any channel count (downmixed to mono)."""
    import struct
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise RuntimeError('not a RIFF/WAVE file: %s' % path)
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack('<I', data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b'fmt ':
            tag, ch, sr, _, _, bits = struct.unpack('<HHIIHH', body[:16])
            if tag == 0xFFFE and len(body) >= 26:            # extensible: real tag is in the sub-format GUID
                tag = struct.unpack('<H', body[24:26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b'data':
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise RuntimeError('WAV missing fmt/data chunk: %s' % path)
    tag, ch, sr, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(pcm, dtype='<f4').astype(np.float32)
    elif tag == 3 and bits == 64:
        x = np.frombuffer(pcm, dtype='<f8').astype(np.float32)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(pcm, dtype='<i4').astype(np.float32) / 2147483648.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(pcm[:len(pcm) - len(pcm) % 3], dtype=np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8).astype(np.float32) / 8388608.0
    elif tag == 1 and bits == 8:
        x = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError('unsupported WAV encoding (format tag %d, %d-bit): %s' % (tag, bits, path))
    if ch > 1:
        x = x[:len(x) - len(x) % ch].reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(x), sr


def _stft_features(x, sr):
    """Frame-wise features: bands[n,8] (dB), loud[n] (dB), flux / flux_low (<200 Hz) / flux_high (>4 kHz)."""
    win = np.hanning(N_FFT).astype(np.float32)
    n_frames = 1 + max(0, (len(x) - N_FFT) // HOP)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    band_idx = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])]
    low_idx, high_idx = freqs < 200.0, freqs > 4000.0
    bands = np.zeros((n_frames, 8), dtype=np.float32)
    loud = np.zeros(n_frames, dtype=np.float32)
    flux = np.zeros(n_frames, dtype=np.float32)
    flux_low = np.zeros(n_frames, dtype=np.float32)
    flux_high = np.zeros(n_frames, dtype=np.float32)
    prev = None
    for i in range(n_frames):
        seg = x[i * HOP:i * HOP + N_FFT] * win
        mag = np.abs(np.fft.rfft(seg))
        power = mag * mag
        for b, idx in enumerate(band_idx):
            bands[i, b] = 10.0 * np.log10(power[idx].mean() + 1e-10)
        loud[i] = 10.0 * np.log10(power[1:].mean() + 1e-10)
        if prev is not None:
            d = np.clip(mag - prev, 0, None)             # positive spectral flux = onset strength
            flux[i] = np.sqrt((d ** 2).sum())
            flux_low[i] = np.sqrt((d[low_idx] ** 2).sum())
            flux_high[i] = np.sqrt((d[high_idx] ** 2).sum())
        prev = mag
    times = np.arange(n_frames) * HOP / float(sr)
    return dict(times=times, bands=bands, loud=loud, flux=flux, flux_low=flux_low, flux_high=flux_high,
                sr=sr, duration=len(x) / float(sr))


# ---------------------------------------------------------------- stems ---------------------------
# CLAUDE-NOTE (2026-09-21): stem separation runs OUTSIDE the editor in Tools/stems/venv
# (python-audio-separator, the same Demucs family UVR ships) because torch cannot live inside UE's
# Python. Stems are cached per song under SourceAudio/stems/<Song>/ so a re-bake never re-separates.
# If the venv is missing or separation fails we fall back to band analysis of the mix — the bake
# must never fail just because the separator is unavailable.

import os
import glob
import subprocess

STEM_MODEL = 'htdemucs_ft.yaml'          # 4 stems: drums / bass / other / vocals
STEM_NAMES = ('Drums', 'Bass', 'Other')


def _project_dir():
    return unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())


def _plugin_dir():
    # this file lives at <Plugin>/Content/Python/song_lights.py
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def _separator_exe():
    exe = os.path.join(_plugin_dir(), 'Tools', 'stems', 'venv', 'Scripts', 'audio-separator.exe')
    return exe if os.path.exists(exe) else None


def separate(wav_path, force=False):
    """Return {'Drums': path, 'Bass': path, 'Other': path} or None when stems are unavailable."""
    exe = _separator_exe()
    if exe is None:
        unreal.log_warning('song_lights: no stem separator venv — run Plugins/SongLights/Tools/stems/setup_stems.bat once. Using band analysis of the mix.')
        return None
    song = _short_name(os.path.splitext(os.path.basename(wav_path))[0])
    out_dir = os.path.join(_project_dir(), 'Saved', 'SongLights', 'stems', song)
    model_dir = os.path.join(_plugin_dir(), 'Tools', 'stems', 'models')

    def find_stems():
        found = {}
        for name in STEM_NAMES:
            hits = glob.glob(os.path.join(out_dir, '*(%s)*.wav' % name))
            if hits:
                found[name] = hits[0]
        return found if len(found) == len(STEM_NAMES) else None

    if not force:
        cached = find_stems()
        if cached:
            unreal.log('song_lights: using cached stems in %s' % out_dir)
            return cached
    os.makedirs(out_dir, exist_ok=True)
    cmd = [exe, wav_path, '-m', STEM_MODEL, '--output_dir', out_dir, '--output_format', 'WAV',
           '--model_file_dir', model_dir]
    unreal.log('song_lights: separating stems (first time per song, ~1 min on GPU): %s' % ' '.join(cmd))
    with unreal.ScopedSlowTask(1, 'Separating stems with Demucs...') as task:
        task.make_dialog(True)
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if r.returncode != 0:
        unreal.log_warning('song_lights: separator failed (%d) — using band analysis\n%s' % (r.returncode, r.stderr[-2000:]))
        return None
    stems = find_stems()
    if stems is None:
        unreal.log_warning('song_lights: separator produced no stems in %s — using band analysis' % out_dir)
    return stems


def analyze(sound):
    """Analyze the source WAV (and its stems when available).
    Returns dict(mix=<features>, stems={'Drums':..,'Bass':..,'Other':..} or None, duration, sr)."""
    path = _source_wav(sound)
    x, sr = _read_wav_mono(path)
    mix = _stft_features(x, sr)
    unreal.log('song_lights.analyze: %s  %.1fs  %d frames  sr=%d' % (path, mix['duration'], len(mix['times']), sr))
    stems = None
    stem_paths = separate(path)
    if stem_paths:
        stems = {}
        for name, p in stem_paths.items():
            sx, ssr = _read_wav_mono(p)
            stems[name] = _stft_features(sx, ssr)
        unreal.log('song_lights.analyze: stems %s' % ', '.join(sorted(stems)))
    return dict(mix=mix, stems=stems, duration=mix['duration'], sr=sr)


def _autorange(x, lo_pct=10, hi_pct=98):
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    if hi - lo < 1e-4:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _envelope(x, dt, attack, decay):
    y = np.zeros_like(x)
    ka = 1.0 if attack <= 0 else 1.0 - math.exp(-dt / attack)
    kd = 1.0 if decay <= 0 else 1.0 - math.exp(-dt / decay)
    v = 0.0
    for i, s in enumerate(x):
        v = v + (s - v) * (ka if s > v else kd)
        y[i] = v
    return y


def _resample(times, values, t_out):
    return np.interp(t_out, times, values)


def _onsets(times, flux, n, fps, thresh=0.15, min_gap=0.09):
    """Peaks of spectral flux above a local median -> impulse train at `fps`."""
    med = np.array([np.median(flux[max(0, i - 43):i + 43]) for i in range(len(flux))])   # ~1 s window at HOP res
    excess = _autorange(np.clip(flux - med * 1.5, 0, None), 50, 99)
    hits = np.zeros(n)
    last_hit = -1.0
    for i in range(1, len(excess) - 1):
        if excess[i] > thresh and excess[i] >= excess[i - 1] and excess[i] >= excess[i + 1]:
            tt = times[i]
            if tt - last_hit >= min_gap:
                idx = int(round(tt * fps))
                if 0 <= idx < n:
                    hits[idx] = max(hits[idx], excess[i])
                last_hit = tt
    return hits


def build_signals(an, fps=FPS):
    """Turn the analysis into one 0..1 control signal per light group, sampled at `fps`.

    With stems each group follows an INSTRUMENT (drums / bass / other); without them it follows a
    frequency band of the mix. Same output either way, so the bake does not care which ran.
    """
    duration = an['duration']
    n = int(math.ceil(duration * fps)) + 1
    dt = 1.0 / fps
    t = np.arange(n) * dt
    mix, stems = an['mix'], an['stems']
    if stems:
        d, b, o = stems['Drums'], stems['Bass'], stems['Other']
        raw = {
            'Song_Bass':  _resample(b['times'], b['bands'][:, 0:3].mean(axis=1), t),   # bass stem 40-320 Hz
            'Song_Mids':  _resample(o['times'], o['bands'][:, 2:6].mean(axis=1), t),   # keys/melody stem 160-2560 Hz
            'Song_Highs': _resample(d['times'], d['bands'][:, 6:8].mean(axis=1), t),   # drums stem 2.5-10 kHz = hats
            'Song_Loud':  _resample(mix['times'], mix['loud'], t),
        }
        kick_src = (d['times'], d['flux_low'])                                         # drums stem < 200 Hz = kick
    else:
        bands = mix['bands']
        raw = {
            'Song_Bass':  _resample(mix['times'], bands[:, 0:2].mean(axis=1), t),
            'Song_Mids':  _resample(mix['times'], bands[:, 2:5].mean(axis=1), t),
            'Song_Highs': _resample(mix['times'], bands[:, 5:8].mean(axis=1), t),
            'Song_Loud':  _resample(mix['times'], mix['loud'], t),
        }
        kick_src = (mix['times'], mix['flux'])
    sig = {}
    for g, x in raw.items():
        p = GROUPS[g]
        v = _autorange(x, p.get('lo', 10)) ** p['gamma']
        sig[g] = _envelope(v, dt, p['attack'], p['decay'])
    p = GROUPS['Song_Kick']
    sig['Song_Kick'] = _envelope(_onsets(kick_src[0], kick_src[1], n, fps), dt, p['attack'], p['decay'])
    return sig


# ---------------------------------------------------------------- sequencer -----------------------

def _short_name(asset_name):
    """'SW_Praise__feat__Brandon_Lake_...__128kbit_AAC_' -> 'Praise'. Downloaded files carry long
    titles with odd characters; the sequence (and stem cache) get the first word-ish chunk."""
    import re
    n = asset_name[3:] if asset_name.startswith('SW_') else asset_name
    n = n.split('__')[0]                                   # importer turns ' (' into '__' — cut the "(feat. ...)" tail
    n = re.sub(r'[^0-9A-Za-z]+', '_', n).strip('_')
    return n[:40].rstrip('_') or 'Song'


def _get_or_create_sequence(sound):
    """One Level Sequence per song, next to the SoundWave: /Game/Music/SW_Foo -> /Game/Music/LS_Foo."""
    pkg_dir = os.path.dirname(sound.get_path_name().split('.')[0])
    name = 'LS_' + _short_name(sound.get_name())
    path = '%s/%s' % (pkg_dir, name)
    seq = unreal.load_asset(path)
    if seq is None:
        seq = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, pkg_dir, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    return seq


def _resolve_camera(camera):
    """`camera` may be an actor, an actor path name, or an actor label. Returns the actor or None."""
    if camera is None or camera in ('', 'None'):     # BP GetPathName(None) -> 'None'
        return None
    if isinstance(camera, unreal.Actor):
        return camera
    for a in _actor_subsystem().get_all_level_actors():
        if a.get_path_name() == camera or a.get_actor_label() == camera:
            return a
    unreal.log_warning('song_lights: camera %r not found in the level; no camera cut track' % camera)
    return None


def _clear_sequence(seq):
    for b in list(seq.get_bindings()):
        b.remove()
    for t in list(seq.get_tracks()):
        seq.remove_track(t)


def _bind_component(seq, actor, component):
    ab = seq.add_possessable(actor)
    cb = seq.add_possessable(component)
    cb.set_parent(ab)
    return ab, cb


# ---------------------------------------------------------------- director ------------------------
# CLAUDE-NOTE (2026-09-21): the director's "Light Groups" arrays are the user-facing way to choose
# which lights follow which element (Details panel, eyedropper). Tags remain a fallback, and if the
# level has neither, Create spawns the starter rig — so a fresh project works with just Song + Create.

DIRECTOR_CLASS = 'BP_SongDirector_C'
GROUP_PROPS = {'Song_Kick': 'KickLights', 'Song_Bass': 'BassLights', 'Song_Mids': 'MidsLights',
               'Song_Highs': 'HighsLights', 'Song_Loud': 'LoudLights'}


def _find_director(sound=None):
    """The BP_SongDirector in the level (if several, the one whose Song matches)."""
    found = [a for a in _actor_subsystem().get_all_level_actors() if a.get_class().get_name() == DIRECTOR_CLASS]
    if sound is not None:
        for a in found:
            if a.get_editor_property('Song') == sound:
                return a
    return found[0] if found else None


def _director_groups(director):
    return {g: [a for a in director.get_editor_property(p) if a is not None] for g, p in GROUP_PROPS.items()}


def _write_director_groups(director, groups):
    for g, p in GROUP_PROPS.items():
        director.set_editor_property(p, list(groups.get(g, [])))


def resolve_groups(director, auto_rig=True):
    """Light groups for the bake: director arrays -> tags -> (optionally) spawn the starter rig."""
    groups = _director_groups(director) if director else {g: [] for g in GROUPS}
    if any(groups.values()):
        return groups
    groups = tagged_lights()
    if not any(groups.values()) and auto_rig:
        origin = director.get_editor_property('RigOrigin') if director else unreal.Vector(0, 0, 0)
        unreal.log('song_lights: no lights assigned or tagged — spawning the starter rig at %s' % origin)
        setup_rig((origin.x, origin.y, origin.z))
        groups = tagged_lights()
    if director:
        _write_director_groups(director, groups)   # show the assignment in the Details panel
    return groups


def setup_rig_for_director(origin=(0.0, 0.0, 0.0)):
    """Setup Rig button: (re)place the starter rig and assign it to the director's Light Groups."""
    setup_rig(origin)
    d = _find_director()
    if d:
        _write_director_groups(d, tagged_lights())
    return True


def create(sound_path, camera=None, fps=FPS):
    """Create button: everything with defaults. Returns the sequence path."""
    return bake_song(sound_path, camera=camera, fps=fps)


def _bake_float_track(cbind, prop, values, n, fps_unused=None):
    """One float property track on a component binding, one linear key per frame."""
    tr = cbind.add_track(unreal.MovieSceneFloatTrack)
    tr.set_property_name_and_path(prop, prop)
    sec = tr.add_section()
    sec.set_range(0, n)
    ch = sec.get_all_channels()[0]
    for i, v in enumerate(values):
        ch.add_key(unreal.FrameNumber(i), float(v), 0.0,
                   unreal.MovieSceneTimeUnit.DISPLAY_RATE, unreal.MovieSceneKeyInterpolation.LINEAR)
    return n


def bake_song(sound_path, camera=None, fps=FPS, phases=None):
    """`phases` = [(start_seconds, look_name), ...] using PHASE_LOOKS keys, or None to auto-detect.
    When shaft lights are present the bake drives Intensity, cone angles and volumetric scattering
    as phase_level * (1 + gain * music), so the section arc and the music reaction are separable."""
    sound = unreal.load_asset(sound_path)
    if sound is None or not isinstance(sound, unreal.SoundWave):
        raise RuntimeError('bake_song: %s is not a SoundWave' % sound_path)
    # CLAUDE-NOTE: a light's authored Intensity is its peak. If a sequence is open in Sequencer the
    # light currently holds an *evaluated* value, and baking from that drifts the peak every run
    # (seen: 25 cd read back as 30.3). Closing the sequence restores the pre-animated state first.
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    shafts = shaft_lights()
    # CLAUDE-NOTE: auto_rig would respawn the legacy concert rig whenever no lights are tagged. In a
    # shaft show that is exactly wrong — the cave wants one beam, not 14 fixtures.
    groups = resolve_groups(_find_director(sound), auto_rig=not shafts)
    an = analyze(sound)
    duration = an['duration']
    sig = build_signals(an, fps)
    n = len(next(iter(sig.values())))
    ph = None
    if shafts:
        if phases is None:
            phases = auto_phases(an, fps)
        ph = phase_tracks(phases, duration, fps)

    fps = int(fps) if fps else FPS
    seq = _get_or_create_sequence(sound)
    seq.set_display_rate(unreal.FrameRate(fps, 1))
    _clear_sequence(seq)
    seq.set_playback_start(0)
    seq.set_playback_end(n)
    seq.set_work_range_start(0.0)
    seq.set_work_range_end(duration)
    seq.set_view_range_start(0.0)
    seq.set_view_range_end(duration)

    # audio
    at = seq.add_track(unreal.MovieSceneAudioTrack)
    asec = at.add_section()
    asec.set_sound(sound)
    asec.set_range(0, n)

    # camera cut
    cam = _resolve_camera(camera)
    if cam:
        cb = seq.add_possessable(cam)
        ct = seq.add_track(unreal.MovieSceneCameraCutTrack)
        cs = ct.add_section()
        cs.set_range(0, n)
        cs.set_camera_binding_id(seq.get_binding_id(cb))

    # lights
    total_keys = 0
    for g, actors in groups.items():
        s = sig[g]
        base_frac = GROUPS[g]['base']
        for a in actors:
            lc = a.light_component
            if lc.get_editor_property('mobility') != unreal.ComponentMobility.MOVABLE:
                lc.set_mobility(unreal.ComponentMobility.MOVABLE)   # stationary/static ignore Sequencer intensity
            peak = float(lc.get_editor_property('intensity'))
            base = peak * base_frac
            _, cbind = _bind_component(seq, a, lc)
            tr = cbind.add_track(unreal.MovieSceneFloatTrack)
            tr.set_property_name_and_path('Intensity', 'Intensity')
            sec = tr.add_section()
            sec.set_range(0, n)
            ch = sec.get_all_channels()[0]
            keys = [unreal.FrameNumber(i) for i in range(n)]
            vals = (base + (peak - base) * s).astype(float).tolist()
            for k, v in zip(keys, vals):
                ch.add_key(k, v, 0.0, unreal.MovieSceneTimeUnit.DISPLAY_RATE, unreal.MovieSceneKeyInterpolation.LINEAR)
            total_keys += n

    # shaft show: section arc (phase) x music reaction, on four properties per layer
    if shafts:
        for label in sorted(shafts):
            a = shafts[label]
            key = LAYER_KEY[label]
            lc = a.light_component
            if lc.get_editor_property('mobility') != unreal.ComponentMobility.MOVABLE:
                lc.set_mobility(unreal.ComponentMobility.MOVABLE)
            peak_i = float(lc.get_editor_property('intensity'))
            peak_in = float(lc.get_editor_property('inner_cone_angle'))
            peak_out = float(lc.get_editor_property('outer_cone_angle'))
            peak_vol = float(lc.get_editor_property('volumetric_scattering_intensity'))
            drive = layer_drive(key, sig, 1.0 / float(fps))
            # Every property reacts around its phase value: the phase sets where the beam SITS,
            # the music decides how it moves there. Cone and haze react too, or the beam only
            # flashes brighter without ever changing shape.
            react = 1.0 + REACT_GAIN[key] * drive
            cone_react = 1.0 + CONE_REACT[key] * drive
            vol_react = 1.0 + VOL_REACT * drive
            _, cbind = _bind_component(seq, a, lc)
            total_keys += _bake_float_track(cbind, 'Intensity', peak_i * ph[key] * react, n)
            total_keys += _bake_float_track(cbind, 'VolumetricScatteringIntensity',
                                            peak_vol * ph['vol'] * vol_react, n)
            # cone angles are clamped: UE rejects <=0 and >=80 degrees on a spot light
            total_keys += _bake_float_track(cbind, 'InnerConeAngle',
                                            np.clip(peak_in * ph['cone'] * cone_react, 1.0, 78.0), n)
            total_keys += _bake_float_track(cbind, 'OuterConeAngle',
                                            np.clip(peak_out * ph['cone'] * cone_react, 1.5, 79.0), n)

        # Wall washes get a gentler, slower reaction than the shaft — enough that the rock breathes
        # with the song rather than sitting flat, but not so much that the walls strobe.
        acc_peak = float(max(ph['accent'].max(), 1e-6))
        acc_drive = 1.0 + 0.35 * _transient(sig['Song_Loud'], 1.0 / float(fps), window=2.5) \
                        + 0.20 * sig['Song_Bass']
        for a in accent_lights():
            lc = a.light_component
            peak = float(lc.get_editor_property('intensity'))
            if peak <= 0.0:
                continue
            if lc.get_editor_property('mobility') != unreal.ComponentMobility.MOVABLE:
                lc.set_mobility(unreal.ComponentMobility.MOVABLE)
            _, cbind = _bind_component(seq, a, lc)
            total_keys += _bake_float_track(cbind, 'Intensity',
                                            peak * (ph['accent'] / acc_peak) * acc_drive, n)
        unreal.log('song_lights: shaft show baked over %d phases' % len(phases))

    unreal.EditorAssetLibrary.save_loaded_asset(seq)
    unreal.log('song_lights: baked %d lights, %d keys, %d frames @ %dfps -> %s' % (
        sum(len(v) for v in groups.values()), total_keys, n, fps, seq.get_path_name()))
    unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(seq)   # show the result
    return seq.get_path_name()
