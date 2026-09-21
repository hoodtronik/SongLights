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


def bake_song(sound_path, camera=None, fps=FPS):
    sound = unreal.load_asset(sound_path)
    if sound is None or not isinstance(sound, unreal.SoundWave):
        raise RuntimeError('bake_song: %s is not a SoundWave' % sound_path)
    # CLAUDE-NOTE: a light's authored Intensity is its peak. If a sequence is open in Sequencer the
    # light currently holds an *evaluated* value, and baking from that drifts the peak every run
    # (seen: 25 cd read back as 30.3). Closing the sequence restores the pre-animated state first.
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    groups = resolve_groups(_find_director(sound))
    an = analyze(sound)
    duration = an['duration']
    sig = build_signals(an, fps)
    n = len(next(iter(sig.values())))

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
    unreal.EditorAssetLibrary.save_loaded_asset(seq)
    unreal.log('song_lights: baked %d lights, %d keys, %d frames @ %dfps -> %s' % (
        sum(len(v) for v in groups.values()), total_keys, n, fps, seq.get_path_name()))
    unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(seq)   # show the result
    return seq.get_path_name()
