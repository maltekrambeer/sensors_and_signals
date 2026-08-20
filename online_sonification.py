"""

===============================================================================
Hackathon Starter Script: Sensors, Signals, Surveillance, AND Sonification ???
===============================================================================

OVERVIEW & ARCHITECTURE
-----------------------
This script provides an asynchronous, multi-threaded pipeline that continuously 
polls smartphone motion sensors over HTTP via the Phyphox REST API and synthesizes 
sound in real time.

The architecture decouples network I/O from audio rendering across two threads 
to provide audio stability with as little stutter or buffer underflows as possible:
	
1. Data Fetcher Thread ("Producer"):
   - Polls all configured IP addresses concurrently using a thread pool.
   - Extracts raw sensor vectors, computes motion intensities (magnitudes), 
     and maps these metrics to synthetic control signals (pitch, volume, beat rate).
   - Applies temporal smoothing (glides) and attack/release envelope following.
   - Updates shared state variables protected by a thread lock (`state_lock`).

2. Audio Synthesis Thread ("Consumer" / Real-Time Callback):
   - Triggered periodically by the sounddevice / PortAudio backend every `BLOCK_SIZE` frames.
   - Takes a thread-safe snapshot of target control parameters.
   - Generates a phase-continuous sine carrier modulated by a shaped Low-Frequency 
     Oscillator (LFO) for rhythmic pulsing.
   - Updates phase accumulators modulo 2*pi across block boundaries to prevent 
     auditory clicks, pops, and discontinuities.


EXECUTION LIFECYCLE & CALL ORDER
--------------------------------
1. Program Start (`main()`):
   ├── 1.1 Configures network and audio parameters.
   ├── 1.2 Spawns `data_fetcher_thread()` as a daemon worker.
   ├── 1.3 Initializes `sounddevice.OutputStream` bound to `audio_callback()`.
   └── 1.4 Enters idle wait loop while audio/fetch threads run.

2. Network & Control Loop (`data_fetcher_thread()`, running every `FETCH_INTERVAL_S`):
   └── while running_event.is_set():
       ├── 2.1 ThreadPoolExecutor dispatches `_fetch_single_phone(ip)` for each IP.
       │   └── `_fetch_accel_from_ip(ip)`: Sends HTTP GET to `/get?accX&accY&accZ&clear=1`.
       │   └── `_extract_latest_axis()`: Parses newest float sample per axis.
       │   └── Computes Euclidean norm (magnitude): sqrt(x^2 + y^2 + z^2).
       ├── 2.2 Routes sensor data to target sound roles via `_get_feature_phone_ip()`.
       ├── 2.3 Feature Transformation & Conditioning:
       │   ├── `linear_map()`: Normalizes sensor range to target sound parameters.
       │   ├── `smooth_toward()`: Exponential smoothing for continuous pitch/tempo glide.
       │   └── `envelope_follow()`: Dynamic attack/release follower for natural volume decay.
       └── 2.4 Acquires `state_lock` -> updates global synth state -> sleeps `FETCH_INTERVAL_S`.

3. Audio Rendering Engine (`audio_callback()`, triggered by OS sound driver):
   ├── 3.1 Acquires `state_lock` -> reads frequency, volume, beat rate, and start phases.
   ├── 3.2 Computes per-sample phase arrays: `phase_0 + phase_step * arange(frames)`.
   ├── 3.3 Synthesizes carrier wave: `np.sin(phase_values)`.
   ├── 3.4 Synthesizes beat envelope: `(1 - depth) + depth * (sin(beat_phase))^exponent`.
   ├── 3.5 Computes output buffer: `outdata[:] = volume * beat_envelope * carrier`.
   └── 3.6 Acquires `state_lock` -> saves `(end_phase % 2*pi)` for next chunk continuity.

4. Teardown / Exit (on `KeyboardInterrupt` / SIGINT):
   ├── 4.1 `running_event.clear()` signals background threads to terminate.
   ├── 4.2 Closes `sounddevice.OutputStream` safely.
   └── 4.3 Joins `fetch_thread` with timeout -> exits cleanly.


MODIFICATION & EXTENSION POINTS
-------------------------------
[1] Changing Sensor Modality (Gyroscope, Light, Barometer, etc.):
    - Modify the query string in `_fetch_accel_from_ip()`:
      * Gyroscope:     `f"http://{ip}:{PORT}/get?gyrX&gyrY&gyrZ&clear=1"`
      * Illuminance:   `f"http://{ip}:{PORT}/get?lux&clear=1"`
      * Pressure:      `f"http://{ip}:{PORT}/get?pressure&clear=1"`
    - Adjust parsing logic in `_extract_latest_axis()` to match returned JSON keys.
    - Adapt feature derivation inside `_fetch_single_phone()` (e.g., raw value vs. 3D norm).

[2] Multi-Phone Routing & Roles:
    - Add IP addresses to `PHONE_IPS`.
    - Adjust `FEATURE_PHONE_INDEX` to map specific phone indices to sound roles:
      * Example: `{"pitch": 0, "volume": 1, "beat": 2}` lets Phone 0 control pitch,
        Phone 1 control volume, and Phone 2 control rhythmic pulse rate.

[3] Parameter Mapping & Calibration:
    - Tweak mapping limits (`FREQ_MIN/MAX_HZ`, `VOL_MIN/MAX`, `PITCH_MOTION_MIN/MAX`).
    - Replace `linear_map()` with non-linear functions (e.g., logarithmic pitch scaling 
      or MIDI note mapping: f = 440 * 2^((m - 69) / 12)).
    - Tune `VOLUME_ATTACK` and `VOLUME_RELEASE` for sharper percussive hits or ambient swells.

[4] Sound Design & Audio Synthesis:
    - Inside `audio_callback()`:
      * Additive Synthesis: Sum multiple harmonic sines (`np.sin(phase) + 0.5 * np.sin(2 * phase)`).
      * FM Synthesis: Modulate the carrier phase with a secondary modulator oscillator.
      * Stereo Panning: Output 2 channels (`channels=2`) and map phone tilt (X-axis) to L/R balance.
===============================================================================

Suggested exploration:
change one thing at a time, run, listen, and document how behavior changes.
then go wild.
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd
from concurrent.futures import ThreadPoolExecutor


# Globals and Constants (play around wiht those)

# phones to poll each cycle; comment out IPs that are surely not connected to avoid stutter
PHONE_IPS = [  # ordered phone list used for feature routing; changing order changes which device controls which sound feature
	# "192.168.71.244", # eduroam
	# "192.168.0.154", # home
	# "10.25.1.226", # gemma, eduroam
	# "192.168.0.241", # lan_w
	"192.168.0.100", # lan_b
]

# assign each sonification feature to a PHONE_IPS index
# to extend, add new features and process them in data_fetcher_thread/audio_callback
FEATURE_PHONE_INDEX = {  # maps each sound feature to the index in PHONE_IPS that controls it
	# "pitch": 0,  # uncomment to let this phone's motion control oscillator pitch
	# "volume": 0,  # selects which phone controls loudness from motion magnitude
	"beat": 0,  # selects which phone controls beat/pulse rate from motion magnitude
}

PORT = 8080  # phyphox network port; wrong value means no sensor updates and therefore static/fading sound
REQUEST_TIMEOUT_S = 0.15  # max wait per phone request; larger values can introduce rhythmic control dropouts when a phone is unreachable
FETCH_INTERVAL_S = 0.02  # control update interval; smaller values track motion faster, larger values make control changes more stepped

SAMPLE_RATE = 48_000  # audio sample rate; higher rates improve high-frequency fidelity and reduce aliasing artifacts
BLOCK_SIZE = 64  # audio callback block size; smaller blocks reduce latency, larger blocks can feel less responsive but more stable

# frequency mapping: motion intensity from pitch phone -> pitch (Hz)
FREQ_MIN_HZ = 110.0  # lowest oscillator pitch reached at low mapped pitch-motion values
FREQ_MAX_HZ = 440.0  # highest oscillator pitch reached at high mapped pitch-motion values
PITCH_MOTION_MIN = 0.0  # motion level treated as minimum pitch control input
PITCH_MOTION_MAX = 5.5  # motion level treated as maximum pitch control input
FREQ_SMOOTHING = 0.25  # pitch glide amount; higher values follow motion faster, lower values produce smoother/slower pitch movement

# volume mapping: motion intensity from volume phone -> loudness
VOL_MIN = 0.01  # minimum output loudness floor
VOL_MAX = 1.00  # maximum output loudness ceiling
VOL_MOTION_MIN = 0.0  # motion level mapped to minimum loudness
VOL_MOTION_MAX = 5.5  # motion level mapped to maximum loudness

# beat mapping: motion intensity from beat phone -> beat rate (Hz)
BEAT_MIN_HZ = 0.5  # slowest beat rate (pulses per second) at low mapped beat-motion values
BEAT_MAX_HZ = 8.0  # fastest beat rate (pulses per second) at high mapped beat-motion values
BEAT_MOTION_MIN = 0.0  # motion level treated as minimum beat-rate control input
BEAT_MOTION_MAX = 5.5  # motion level treated as maximum beat-rate control input
BEAT_DEPTH = 0.9  # beat modulation strength; 0 gives no pulsing, 1 gives full pulse contrast
BEAT_SHAPE_EXPONENT = 8.0  # pulse sharpness; higher values create shorter/tighter pulses, lower values sound rounder

# envelope constants for expressive volume behavior
VOLUME_ATTACK = 0.85  # loudness rise speed when motion increases; higher values create snappier accents
VOLUME_RELEASE = 0.35  # loudness fall speed when motion decreases; lower values create longer tails/sustain

# on fetch failure, reduce volume gradually to avoid abrupt silence
FAIL_DECAY = 0.70  # loudness multiplier on missing data; lower values mute faster, higher values decay more gently



# shared state (thread-protected)

state_lock = threading.Lock()
running_event = threading.Event()
running_event.set()

current_frequency_hz = 220.0
current_volume = 0.0
current_phase = 0.0
current_beat_hz = 2.0
current_beat_phase = 0.0
had_any_connection = False


def clamp(value: float, lo: float, hi: float) -> float:
	"""Clamp a numeric value to an inclusive range.

	Args:
		value: Number to clamp.
		lo: Lower allowed bound.
		hi: Upper allowed bound.

	Returns:
		The clamped value in [lo, hi].
	"""
	return max(lo, min(hi, value))


# mapping can be linear or nonlinear; you can experiment with different kinds
def linear_map(
	value: float,
	in_min: float,
	in_max: float,
	out_min: float,
	out_max: float,
) -> float:
	"""Map a value linearly from one range to another with endpoint clamping.

	Args:
		value: Input value to convert.
		in_min: Lower bound of source range.
		in_max: Upper bound of source range.
		out_min: Lower bound of output range.
		out_max: Upper bound of output range.

	Returns:
		Mapped value in [out_min, out_max].
	"""
	if math.isclose(in_min, in_max):
		return out_min
	t = (value - in_min) / (in_max - in_min)  # normalize to 0-1
	t = clamp(t, 0.0, 1.0)
	return out_min + t * (out_max - out_min)


def smooth_toward(previous: float, target: float, alpha: float) -> float:
	"""Move a value toward a target by a smoothing factor.

	Args:
		previous: Current value.
		target: Desired value.
		alpha: Smoothing coefficient in [0, 1].

	Returns:
		A value between previous and target.
	"""
	alpha = clamp(alpha, 0.0, 1.0)
	return previous + alpha * (target - previous)


def envelope_follow(previous: float, target: float, attack: float, release: float) -> float:
	"""Apply an attack/release envelope follower.

	When the target rises, attack is used so volume reacts quickly.
	When the target falls, release is used so volume decays slowly, creating
	longer sustained sound from brief motion events.

	Args:
		previous: Current envelope level.
		target: New target level.
		attack: Smoothing factor when target >= previous.
		release: Smoothing factor when target < previous.

	Returns:
		Smoothed envelope level.
	"""
	alpha = attack if target >= previous else release
	return smooth_toward(previous, target, alpha)


def _extract_latest_axis(payload: dict, axis_name: str) -> float:
	"""Read the latest sample for one axis from the Phyphox JSON payload.

	Args:
		payload: Decoded object returned by the Phyphox /get endpoint.
		axis_name: Axis key to read (for example 'accY').

	Returns:
		Latest axis sample as float.

	Raises:
		ValueError: If axis buffer is empty.
		KeyError/TypeError: If the payload shape is unexpected.
	"""
	axis_block = payload["buffer"][axis_name]["buffer"]
	if not axis_block:
		raise ValueError(f"No samples in axis '{axis_name}'")
	return float(axis_block[-1])  # take newest sample to minimize control latency


def _get_feature_phone_ip(feature_name: str) -> Optional[str]:
	"""Resolve which phone IP is assigned to a given sound feature.

	Args:
		feature_name: Feature key, for example 'pitch' or 'volume'.

	Returns:
		Assigned phone IP if mapping is valid, otherwise None.
	"""
	phone_index = FEATURE_PHONE_INDEX.get(feature_name)
	if phone_index is None:
		return None
	if phone_index < 0 or phone_index >= len(PHONE_IPS):
		return None
	return PHONE_IPS[phone_index]


def _fetch_accel_from_ip(ip: str) -> Tuple[float, float, float]:
	"""Fetch latest accelerometer axis values from a single phone.

	Args:
		ip: Phone IP hosting Phyphox remote endpoint.

	Returns:
		Tuple of (accX, accY, accZ) in m/s^2.

	Raises:
		requests.RequestException: For HTTP/network errors.
		json.JSONDecodeError: If response is not valid JSON.
		KeyError/TypeError/ValueError: For malformed or empty buffers.
	"""
	url = f"http://{ip}:{PORT}/get?accX&accY&accZ&clear=1"
	response = requests.get(url, timeout=REQUEST_TIMEOUT_S)  # blocking HTTP call inside fetch thread by design
	response.raise_for_status()
	payload = response.json()

	ax = _extract_latest_axis(payload, "accX")
	ay = _extract_latest_axis(payload, "accY")
	az = _extract_latest_axis(payload, "accZ")
	return ax, ay, az


def _fetch_single_phone(ip: str) -> Tuple[str, Optional[float]]:
	"""Fetch one phone sample and convert it into a scalar control feature.

	Why this helper exists:
	- It isolates per-phone network/parsing work so we can run it in a thread pool.
	- It returns a stable tuple shape for easy aggregation in data_fetcher_thread.

	Args:
		ip: Phone address to query via the Phyphox remote endpoint.

	Returns:
		(ip, magnitude) on success, where magnitude = sqrt(ax^2 + ay^2 + az^2).
		(ip, None) on any network or payload parsing failure.
	"""
	try:
		ax, ay, az = _fetch_accel_from_ip(ip)  # pull the newest accelerometer sample triplet
		magnitude = math.sqrt(ax * ax + ay * ay + az * az)  # collapse 3D acceleration into one motion value
		return ip, magnitude  # return both source identity and value for routing
	except (
		requests.RequestException,
		ValueError,
		KeyError,
		TypeError,
		json.JSONDecodeError,
	):
		return ip, None  # keep loop resilient: one failing phone should not stop synthesis


def data_fetcher_thread() -> None:
	"""Continuously fetch phone motion features and update shared synth controls.

	This thread is the control-rate engine of the sonifier:
	1) Acquire per-phone motion values concurrently.
	2) Route values to sound features (pitch, volume, beat) by phone assignment.
	3) Smooth and clamp parameters before exposing them to the audio callback.

	Design intent:
	- Keep all network I/O outside the audio callback.
	- Degrade gracefully when one device drops out.
	"""
	global current_frequency_hz, current_volume, current_beat_hz, had_any_connection

	if not PHONE_IPS:
		print("No PHONE_IPS configured. Audio will stay silent until IPs are added.")
	else:
		print(f"Fetcher started. Polling phones concurrently: {PHONE_IPS}")

	pitch_ip = _get_feature_phone_ip("pitch")  # phone controlling oscillator frequency
	volume_ip = _get_feature_phone_ip("volume")  # phone controlling loudness envelope
	beat_ip = _get_feature_phone_ip("beat")  # phone controlling beat/pulse rate

	last_wait_log = 0.0
	phone_connected: Dict[str, bool] = {ip: False for ip in PHONE_IPS}

	# persistent worker pool sized to active phones; avoids re-creating threads each cycle
	max_workers = max(1, len(PHONE_IPS))
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		while running_event.is_set():
			motion_by_ip: Dict[str, float] = {}  # stores successful per-phone motion values for this cycle

			# dispatch all phone requests concurrently and gather one result tuple per phone
			results = list(executor.map(_fetch_single_phone, PHONE_IPS))

			for ip, motion in results:
				if motion is not None:
					motion_by_ip[ip] = motion  # keep latest successful value for this phone
					if not phone_connected.get(ip, False):
						print(f"Connected to Phyphox at {ip}:{PORT}")  # edge-triggered status print on reconnect
					phone_connected[ip] = True
					had_any_connection = True
				else:
					if phone_connected.get(ip, False):
						print(f"Lost connection to Phyphox at {ip}:{PORT}")  # edge-triggered status print on dropout
					phone_connected[ip] = False

			# resolve the routed control values for this cycle (None means unavailable this frame)
			pitch_motion = motion_by_ip.get(pitch_ip) if pitch_ip else None
			volume_motion = motion_by_ip.get(volume_ip) if volume_ip else None
			beat_motion = motion_by_ip.get(beat_ip) if beat_ip else None

			# update shared synth parameters atomically so the callback sees consistent snapshots
			with state_lock:
				if pitch_motion is not None:
					target_freq = linear_map(
						pitch_motion,
						PITCH_MOTION_MIN,
						PITCH_MOTION_MAX,
						FREQ_MIN_HZ,
						FREQ_MAX_HZ,
					)
					current_frequency_hz = smooth_toward(
						current_frequency_hz, target_freq, FREQ_SMOOTHING
					)  # smoothing avoids zipper noise from control-rate stepping
					current_frequency_hz = clamp(
						current_frequency_hz, FREQ_MIN_HZ, FREQ_MAX_HZ
					)  # keep frequency in audible and intended range

				if volume_motion is not None:
					target_vol = linear_map(
						volume_motion,
						VOL_MOTION_MIN,
						VOL_MOTION_MAX,
						VOL_MIN,
						VOL_MAX,
					)
					current_volume = envelope_follow(
						current_volume, target_vol, VOLUME_ATTACK, VOLUME_RELEASE
					)  # attack/release follower shapes musical dynamics
					current_volume = clamp(current_volume, VOL_MIN, VOL_MAX)  # keep gain bounded for safety
				else:
					current_volume *= FAIL_DECAY  # graceful fade when mapped volume source is missing
					current_volume = clamp(current_volume, VOL_MIN, VOL_MAX)

				if beat_motion is not None:
					target_beat_hz = linear_map(
						beat_motion,
						BEAT_MOTION_MIN,
						BEAT_MOTION_MAX,
						BEAT_MIN_HZ,
						BEAT_MAX_HZ,
					)
					current_beat_hz = smooth_toward(
						current_beat_hz, target_beat_hz, 0.2
					)  # smooth beat tempo so pulse rate glides instead of jumping
					current_beat_hz = clamp(
						current_beat_hz, BEAT_MIN_HZ, BEAT_MAX_HZ
					)  # keep modulation rate inside configured rhythmic range

			if not motion_by_ip:
				now = time.time()
				if not had_any_connection and (now - last_wait_log) >= 3.0:
					print("Waiting for reachable Phyphox servers...")  # periodic user feedback when no data arrives
					last_wait_log = now

			time.sleep(FETCH_INTERVAL_S)  # control-rate pacing for network and mapping loop


def audio_callback(outdata, frames, time_info, status) -> None:
	"""Audio callback that renders phase-continuous sine output.

	Args:
		outdata: Output buffer (frames x channels) to be filled in-place.
		frames: Number of frames requested for this callback.
		time_info: Timing metadata from the audio backend.
		status: Callback status flags (underflow, overflow, etc.).

	Real-time safety:
		This callback should only do lightweight math and memory operations.
		Never perform network requests or sleeping here.
	"""
	_ = time_info  # required by callback signature; not needed in this implementation

	global current_phase, current_beat_phase

	if status:
		print(f"Audio stream status: {status}")

	if frames <= 0:
		outdata.fill(0.0)
		return

	with state_lock:
		freq_hz = float(current_frequency_hz)
		volume = float(current_volume)
		phase_0 = float(current_phase)  # snapshot shared phase at callback start
		beat_hz = float(current_beat_hz)  # snapshot beat rate for this audio block
		beat_phase_0 = float(current_beat_phase)  # snapshot beat LFO phase for continuity

	phase_step = (2.0 * math.pi * freq_hz) / SAMPLE_RATE  # radians advanced per sample
	phase_values = phase_0 + phase_step * np.arange(frames, dtype=np.float64)
	beat_phase_step = (2.0 * math.pi * beat_hz) / SAMPLE_RATE  # beat LFO radians advanced per sample
	beat_phase_values = beat_phase_0 + beat_phase_step * np.arange(frames, dtype=np.float64)

	carrier = np.sin(phase_values) + 0.5 * np.sin(2 * phase_values) + 0.25 * np.sin(3 * phase_values)  # main tonal oscillator
	beat_lfo = np.power(0.5 * (1.0 + np.sin(beat_phase_values)), BEAT_SHAPE_EXPONENT)  # 0..1 pulse-shaped LFO
	beat_envelope = (1.0 - BEAT_DEPTH) + BEAT_DEPTH * beat_lfo  # blend dry level with pulsed modulation depth
	samples = (volume * beat_envelope * carrier).astype(np.float32)  # apply simple pulsing envelope for a steady beat
	outdata[:, 0] = samples

	with state_lock:
		current_phase = float((phase_values[-1] + phase_step) % (2.0 * math.pi))  # persist phase to avoid clicks
		current_beat_phase = float((beat_phase_values[-1] + beat_phase_step) % (2.0 * math.pi))  # persist beat phase for stable rhythm


def main() -> None:
	"""Start system components and keep the script alive until interruption.

	Lifecycle:
		1) Start fetch thread.
		2) Open sounddevice OutputStream.
		3) Keep main thread alive.
		4) On Ctrl+C or errors, shutdown cleanly.
	"""
	print("Starting online sonification...")  # startup banner for quick terminal diagnostics
	print(f"Audio config: sample_rate={SAMPLE_RATE}, block_size={BLOCK_SIZE}, channels=1")
	print(f"Network config: port={PORT}, fetch_interval={FETCH_INTERVAL_S}s")

	fetch_thread = threading.Thread(target=data_fetcher_thread, daemon=True)  # daemon thread exits automatically with main process
	fetch_thread.start()

	try:
		default_output = sd.query_devices(kind="output")  # confirm which playback device PortAudio selected
		print(f"Using output device: {default_output['name']}")

		stream = sd.OutputStream(
			samplerate=SAMPLE_RATE,
			blocksize=BLOCK_SIZE,
			channels=1,
			dtype="float32",
			callback=audio_callback,  # sounddevice invokes this in the real-time audio thread
		)

		with stream:
			print("Live sonification running. Press Ctrl+C to stop.")
			while True:
				time.sleep(0.2)  # keep main thread alive while callback thread renders audio
	except KeyboardInterrupt:
		print("Keyboard interrupt received. Shutting down...")
	except sd.PortAudioError as exc:
		print(f"Audio device error: {exc}")
		print("Available audio output devices:")
		for idx, dev in enumerate(sd.query_devices()):
			if int(dev.get("max_output_channels", 0)) > 0:
				print(f"  [{idx}] {dev['name']}")
	except Exception as exc:
		print(f"Unhandled error: {exc}")
	finally:
		running_event.clear()
		fetch_thread.join(timeout=2.0)  # allow control thread to finish cleanly
		print("Shutdown complete.")


if __name__ == "__main__":
	main()
