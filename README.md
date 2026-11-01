# Bloom

An audio-reactive ambient display inspired by Apple's **HomePod**. Bloom listens to whatever
your computer is playing and renders a soft, glowing blob that moves with the
music, coloured by the album art of the current track.

<video src="./demo/demo_usage_1.mp4" autoplay muted loop controls width="20%"></video>

https://github.com/user-attachments/assets/5420308f-1b89-4a2e-86b2-ee3136e6396f





*Developed with Claude (Anthropic) as a coding assistant. I set the design, direction and feature decisions and tested and tuned it against real audio.*

## Features

- Reacts to live system audio (kicks, snares, hats, bass and tonal parts are picked out separately with plain DSP, no machine learning)
- Takes its colour from the now-playing album art and fades between tracks
- Circle, fullscreen, or any filled SVG as the display shape
- Control panel for reactivity, blob size, intensity, pulses and snare waves
- Idle glow and edge fade options
- Settings are saved between runs

## Noise Model

- **Noise blob.** A soft round noise grows out from the centre of the shape. Its radius comes from the music's volume and bass, and 4D simplex noise (position + time) gives the outline a slow wobble.

- **Noise pulses.** Randomized noise areas are generated at the center and drift outwards while fading in intensity. Fields are added together so the main noise body swells towards the pulses and melts back into one smooth shape instead of visually detaching from the main noise body.

- **Brightness and colour.** Intensity is highest at the centre and tapers off towards the edge, plus extra from kicks and snare waves which appear as intensity waves which move outwards as they lose intensity. Low intensities are represented by the album colour, higher intensities fade towards white.

- **Reactivity.** A velocity gives higher intensity audio moments a longer lasting corresponding visual intensity so that noise does not rapidly increase and decrease.

## Requirements

- Windows 10/11 (live audio capture and now-playing info use Windows APIs)
- Python 3.11+

`--wav` and `--demo` work on other platforms too.

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[all,dev]"
```

## Running

```powershell
python -m bloom # live system audio
python -m bloom --demo # built-in drum loop
python -m bloom --wav song.wav # play a WAV file through the visualizer
```

Two windows open: the display and a control panel. Press Esc in the display
to quit.

Some useful options:

| Option | Description |
|-|-|
| `--shape` | `circle`, `fullscreen`, or a path to an SVG |
| `--no-panel` | Display only |
| `--size W H` | Window size in pixels |
| `--reset-settings` | Ignore saved settings and start from defaults |

Run `python -m bloom --help` for the full list.

Settings are stored in `%APPDATA%\reactance\settings.json`.

## Short Demos

### Calm
<video src="./demo/demo_usage_3.mp4" autoplay muted loop controls width="20%"></video>

https://github.com/user-attachments/assets/77cd2497-bf7c-47e7-a509-538a5d1d73cb



### More Reactive
<video src="./demo/demo_usage_2.mp4" autoplay muted loop controls width="20%"></video>

https://github.com/user-attachments/assets/b70e305b-728f-4814-b8fa-dcdf0cd0b9b9




### Pausing
<video src="./demo/demo_pause_1.mp4" autoplay muted loop controls width="20%"></video>

https://github.com/user-attachments/assets/918f2810-649d-4cd6-afc8-db880da92df5
