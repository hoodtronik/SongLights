# SongLights — music-driven lighting for Unreal Engine 5.6

SongLights analyzes a song and **bakes light-intensity curves into a Level Sequence**, so your
lights react to the music identically in the editor, in Play, and in a Movie Render Queue render.
Pick a song, press one button, get a sequence. Switching songs is the same button.

- **Instrument-aware.** With the optional stem separator (Demucs, the same model family
  Ultimate Vocal Remover uses), each light group follows an *instrument*: drum hits, bassline,
  keys/melody, hi-hats, overall loudness. Without it, groups follow frequency bands of the mix.
- **Deterministic.** Everything is keyframes in Sequencer — no runtime audio analysis, no drift,
  frame-locked for prerendered playback (e.g. disguise / media servers).
- **Editable.** Every curve is a normal Sequencer track you can hand-tweak, and every light's
  assignment is a pick-list in the Details panel.
- **Content-only plugin.** No C++, nothing to compile. Blueprint + Python.

Built and tested on **UE 5.6.1, Windows 11**.

---

## Contents

> **Changelog** — 0.1.1: float/extensible WAV support, short sequence names. 0.1.0: first release.

1. [Requirements](#requirements)
2. [Install](#install)
3. [Quick start](#quick-start)
4. [What Create does](#what-create-does)
5. [Customizing](#customizing)
6. [Rendering for a show](#rendering-for-a-show)
7. [Switching songs](#switching-songs)
8. [How it works](#how-it-works)
9. [Tuning the feel](#tuning-the-feel)
10. [Troubleshooting](#troubleshooting)
11. [Project layout](#project-layout)

---

## Requirements

| | Required | Notes |
|---|---|---|
| Unreal Engine | **5.6** | Content-only plugin; other 5.x may work but are untested. |
| Engine plugins | Python Editor Script Plugin, Sequencer Scripting, Movie Render Pipeline | All ship with the engine. SongLights declares them as dependencies, so the editor enables them for you. |
| Song file | **WAV** (16/24/32-bit PCM or 32-bit float) | Import a WAV. MP3/M4A/etc. must be converted first (e.g. `ffmpeg -i song.m4a -ar 48000 song.wav`). |
| Stems (optional) | NVIDIA GPU + internet for a one-time setup | `uv` or Python 3.10–3.12 on PATH. ~3 GB download (CUDA PyTorch + model weights). |

---

## Install

### Option A — download the zip (easiest for teammates)

**[⬇ Download SongLights (latest release zip)](https://github.com/hoodtronik/SongLights/releases/latest/download/SongLights.zip)**

Extract it into your project's `Plugins` folder so that this file exists:

```
YourProject\Plugins\SongLights\SongLights.uplugin
```

(Create the `Plugins` folder next to your `.uproject` if it doesn't exist yet.)

### Option B — clone into your project

```bat
cd C:\Path\To\YourProject
git clone https://github.com/hoodtronik/SongLights.git Plugins\SongLights
```

### Then

1. Open the project. If the editor asks to enable **SongLights** (or its dependencies), say yes and
   let it restart. Otherwise enable it under *Edit → Plugins → Virtual Production → SongLights*.
2. In the Content Browser, click **Settings** (top-right) → tick **Show Plugin Content**. A
   `SongLights Content` folder appears.
3. **Optional — stems.** Double-click `Plugins\SongLights\Tools\stems\setup_stems.bat`. It creates a
   private Python environment next to itself (`Tools\stems\venv`) and installs CUDA PyTorch and
   `audio-separator`. Takes a few minutes. The Demucs model itself downloads on your first bake.
   You can do this later; the plugin works without it.

---

## Quick start

1. **Import the song.** Drag a `.wav` into the Content Browser. You get a SoundWave asset
   (e.g. `SW_MySong`). Keep the original `.wav` where it is — the analysis reads it from disk.
2. **Place the director.** Drag `SongLights Content/BP_SongDirector` into your level.
3. **Set Song.** Select the director, and in the Details panel set **Song** to your SoundWave.
   Optionally set **Camera** to your CineCameraActor.
4. **Click Create** (a button in the Details panel, under the *Default* category).
5. Sequencer opens on `LS_MySong`. Press play.

That's it. On a fresh level you'll get a 14-light starter rig too (see below).

---

## What Create does

In order:

1. **Analyzes the song.** If the stems environment exists, the song is separated into
   drums / bass / other (first time per song, ~1 minute on a modern GPU; cached afterwards in
   `Saved/SongLights/stems/<Song>/`). Then a short-time FFT produces band energies, loudness and
   onset (hit) strength for each stem and for the mix.
2. **Resolves the light groups**, in this priority:
   1. the **Light Groups** lists on the director (if any are filled),
   2. otherwise any lights in the level carrying a `Song_*` actor tag (see [Tags](#tags)),
   3. otherwise it **spawns the starter rig** — 14 tagged, movable lights at **Rig Origin** —
      so a blank level works with nothing but Song + Create.
   Whichever source it used, it writes the result back into the **Light Groups** lists so you can
   see (and change) which light follows what.
3. **Creates or refreshes `LS_<SongName>`** next to the SoundWave (the name is shortened — a
   downloaded "Praise (feat. …) ¦ Elevation Worship (128kbit_AAC)" becomes `LS_Praise`): audio track, camera cut (if
   **Camera** is set), and one **Intensity** track per light with a key every frame at **FPS**.
   Re-running Create rebuilds the sequence from scratch.
4. **Opens the sequence** in Sequencer.

Lights that aren't Movable are switched to Movable automatically (Sequencer can't animate
stationary/static light intensity).

---

## Customizing

Everything is on the director's Details panel.

### Song
| Property | Meaning |
|---|---|
| **Song** | The SoundWave to analyze. |
| **Camera** | Optional. Actor used for the Camera Cut track (usually a CineCameraActor). |
| **FPS** | Frame rate of the baked sequence and its keys. Match your render. Default 30. |

### Light Groups (optional)
Five lists of Light actors. Add with **+** and pick from the dropdown, or use the **eyedropper**
to click a light in the viewport. Click **Create** again after changing them.

| List | Follows | Feel | Starter-rig look |
|---|---|---|---|
| **Kick Lights** | drum onsets (kick, <200 Hz transients) | instant on, 0.12 s decay, 0 base | white strobe |
| **Bass Lights** | bass stem energy | 0.16 s decay, 0 base | 4 blue-violet uplights |
| **Mids Lights** | keys / melody ("other" stem) | slow, 10 % base | 2 amber beams |
| **Highs Lights** | drums stem 2.5–10 kHz (hi-hats) | very fast, 0 base | 6 cyan spots |
| **Loud Lights** | overall loudness of the mix | 0.3 s attack / 0.6 s decay, 30 % base | 1 warm wash |

A light's **authored Intensity is its peak**. The bake writes `base + (peak − base) × signal`, so
to make a light hit harder, raise its Intensity; to make it react more subtly, lower it.
Leaving all five lists empty means "use defaults" (tags, then the starter rig).

### Starter Rig
| Property | Meaning |
|---|---|
| **Rig Origin** | World-space offset applied to the starter rig. |
| **Setup Rig** (button) | (Re)places the starter rig at Rig Origin and assigns it to the groups. Idempotent — it moves existing rig lights rather than duplicating them. |

The rig was laid out for a roughly 40 m cave. Treat it as a starting point: move the lights,
recolor them, delete the ones you don't want. Its long throws (beams, strobe) use thousands of
candela on purpose — spot intensity falls off with distance squared.

### Tags
If you'd rather not use the lists, tag any light actor (Details → Actor → Tags) with one of
`Song_Kick`, `Song_Bass`, `Song_Mids`, `Song_Highs`, `Song_Loud`. Tags are only consulted when
all five lists are empty.

---

## Shaft mode — one beam, eight sections

For a cave / shaft-of-light look (one aperture, overlapping cones that open and close with the
song) rather than a stage rig:

```python
import song_lights
song_lights.setup_shaft_rig(origin=(138, -925, 1290))   # just BELOW the ceiling, at the aperture
song_lights.setup_accent_rig()                          # optional low wall washes
song_lights.bake_song('/Game/Music/SW_MySong', camera='MyCineCamera', fps=30)
```

Three spotlights (`L_Shaft_Main` / `_L` / `_R`) share **one origin** and differ only in cone width,
so they read as a single beam with a bright core and a soft halo. The bake drives four properties
per layer — `Intensity`, `InnerConeAngle`, `OuterConeAngle`, `VolumetricScatteringIntensity` — as:

    value = phase_level * (1 + react_gain * music_signal)

`phase_level` is the section arc, `music_signal` is the per-frame audio reaction, so you can retune
the storyboard without touching the music response and vice versa.

### Phases

`PHASE_LOOKS` holds nine named looks (`dark`, `discover`, `grow`, `reveal`, `deeper`, `tension`,
`isolated`, `breakthrough`, `out`). Each is a set of multipliers on the rig's authored values plus
a `snap` time — short snaps hit on the downbeat, long ones ease in.

Pass your own map when you know the arrangement:

```python
song_lights.bake_song(song, phases=[(0,'dark'), (15,'discover'), (54.5,'reveal'),
                                    (138.7,'isolated'), (178,'breakthrough'), (270,'out')])
```

Omit `phases` and `auto_phases()` detects section boundaries from the mix (spectral self-similarity
+ a checkerboard novelty kernel) and pins the three structural extremes by energy: the loudest late
section becomes `breakthrough`, the quietest section before it becomes `isolated`, and a near-silent
tail becomes `out`. Treat it as a first pass and hand-correct the times.

### Two things that stop the beam rendering

* **Volumetric fog must be enabled and not owned by another system.** Ultra Dynamic Sky re-applies
  its own `ExponentialHeightFog` values every tick and silently reverts writes — fog edits appear to
  do nothing. In an interior, delete UDS and add your own fog actor.
* **The shaft light must hang below the ceiling geometry**, or the ceiling occludes it entirely and
  you get no beam at all.

Also: past roughly 10k candelas the beam core clips to white and bloom smears it into a ball.
Brightness belongs in `volumetric_scattering_intensity`, not candelas. And lock exposure with an
unbound Post Process Volume (`min == max` auto-exposure brightness) — otherwise auto-exposure
re-normalises every section to mid-grey and flattens the whole dark-to-breakthrough arc.

---

## Rendering for a show

1. *Window → Cinematics → Movie Render Queue*, add `LS_<SongName>`.
2. Set **Output** resolution and frame rate (match **FPS**), choose an encoder (Apple ProRes is a
   good choice for media servers; enable the *Apple ProRes Media* plugin if it isn't).
3. MRQ captures the sequence's **audio track** into the output, so the file is already in sync.
4. For a quick check, use *Anti-aliasing → Temporal Sample Count = 1*; raise it for the final.

The bake is frame-locked to the sequence, so what you see in Sequencer is what renders.

---

## Switching songs

1. Import the new WAV.
2. On the director, change **Song**.
3. Click **Create**.

A new `LS_<NewSong>` appears next to that SoundWave; the previous song's sequence is untouched.
Every control signal is auto-ranged per song (its 10th–98th percentile is mapped to 0–1), so a
quiet acoustic track and a loud beat both drive the lights through their full range without
re-tuning.

---

## How it works

```
WAV on disk ──► (optional) Demucs stems: drums / bass / other  (cached per song)
      │
      ▼
 STFT features per source: 8 octave-band energies (40 Hz–10 kHz), loudness, spectral flux
      │
      ▼
 5 control signals ──► per-song auto-range ──► gamma ──► attack/decay envelope   (0..1 at FPS)
      │                (Kick = onset impulses from drums <200 Hz flux)
      ▼
 Level Sequence LS_<Song>: audio + camera cut + Intensity keys for every light in every group
```

- Analysis is plain **numpy** inside the editor's Python (numpy ships with UE 5.6). It deliberately
  does *not* use the engine's Audio Synesthesia NRT assets — driving those from Python asserts the
  editor (`LoudnessNRT.cpp:148 IsSortedChronologically`).
- Stem separation runs in a separate Python environment (PyTorch can't live inside UE's Python) via
  [`python-audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) with the
  `htdemucs_ft` model. If the environment is missing or separation fails, the bake logs a warning
  and falls back to band analysis of the mix.
- The Blueprint is thin: the two buttons build a one-line Python command
  (`song_lights.create(...)` / `song_lights.setup_rig_for_director(...)`) and execute it.

---

## Tuning the feel

Open `Content/Python/song_lights.py` and edit the `GROUPS` table near the top:

```python
GROUPS = {
    'Song_Bass':  dict(base=0.00, attack=0.01, decay=0.16, gamma=2.2, lo=35),
    'Song_Mids':  dict(base=0.10, attack=0.08, decay=0.35, gamma=1.6, lo=25),
    'Song_Highs': dict(base=0.00, attack=0.00, decay=0.07, gamma=2.0, lo=45),
    'Song_Kick':  dict(base=0.00, attack=0.00, decay=0.12, gamma=1.0, lo=10),
    'Song_Loud':  dict(base=0.30, attack=0.30, decay=0.60, gamma=1.0, lo=10),
}
```

| Key | Effect |
|---|---|
| `base` | Fraction of the light's peak that stays on at silence (0 = fully dark between hits). |
| `attack` / `decay` | Envelope times in seconds. Smaller = snappier. |
| `gamma` | Curve shaping; >1 keeps quiet passages dark and makes hits pop. |
| `lo` | Percentile treated as "silence" for auto-ranging. Raise it if a sustained bed (an 808, a hat loop) keeps the group lit. |

The starter rig's positions, colors, cone angles and peak intensities are in the `RIG` table in the
same file. Click **Create** after editing; the module is reloaded on every run.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| **Create does nothing** | Check *Window → Output Log* for lines starting `song_lights:`. Most often **Song** is unset or isn't a SoundWave. |
| `unknown format` / `unsupported WAV encoding` | 0.1.0 only read 16/24/32-bit integer WAVs; 0.1.1 also reads 32-bit float and extensible WAVs (what most DAWs and converters export). Update the plugin. |
| `source WAV for SW_… not found` | The bake reads the file the SoundWave was imported from. Re-import from a WAV that stays on disk (or on a shared drive teammates can reach). |
| `no stem separator venv` warning | Stems are optional. Run `Tools\stems\setup_stems.bat` to enable them; the bake continues with band analysis meanwhile. |
| First Create takes a minute | Stem separation runs once per song and is then cached in `Saved/SongLights/stems/`. |
| A light doesn't react | Is it in a group list (or tagged)? Is its Intensity > 0? Long throws need far more candela than nearby lights (inverse-square). |
| Lights look "always on" | Lower the light's Intensity, or raise that group's `lo` / `gamma` in `GROUPS`. |
| Editor viewport doesn't update while scrubbing | Enable *Realtime* on the viewport (viewport menu ▾ → Realtime). |
| Peaks change between bakes | The bake closes any open sequence before reading light intensities, so it never reads Sequencer-evaluated values. If you edit a light while its sequence is open, close Sequencer first. |
| Lights are Stationary/Static | The bake switches group lights to Movable; if you change them back, Sequencer intensity is ignored. |

---

## Project layout

```
SongLights/
├── SongLights.uplugin           content-only plugin descriptor (UE 5.6)
├── Content/
│   ├── BP_SongDirector.uasset   the actor with the Create / Setup Rig buttons
│   └── Python/song_lights.py    analysis, stems, rig, Sequencer bake
├── Tools/stems/
│   ├── setup_stems.bat          one-time venv setup for stem separation
│   └── venv/, models/           created by the setup script; git-ignored, machine-specific
└── README.md
```

Per-project data lives in the project, not the plugin: sequences next to your SoundWaves, stem
caches under `<Project>/Saved/SongLights/`.

---

## License

MIT — see `LICENSE`.
