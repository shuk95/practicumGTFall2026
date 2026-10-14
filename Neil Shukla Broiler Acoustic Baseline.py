"""
broiler_acoustic_baseline.py

Unsupervised acoustic baseline for broiler-house recordings.

The control-group recordings (e.g. Ctrl_W4_D1_PoS_1.mp3) arrive without
Audacity labels, so rather than training a classifier this script learns what
"normal" control-flock sound looks like and measures every recording against
that baseline:

    1. Load audio, mix to mono, resample to 32 kHz and band-pass 2-12 kHz.
       The supplied MP3s carry almost no energy below ~2 kHz (they were
       high-passed upstream) or above ~12-15 kHz (MP3 encoder), so mel bins
       outside that band would only describe silence.
    2. Compute a band-limited log-mel spectrogram (64 mels, 10 ms hop).
    3. Detect acoustic events with a hysteresis threshold on band energy
       relative to a rolling noise floor.
    4. Describe every event (duration, SNR, peak frequency, centroid,
       bandwidth, spectral flatness, band coverage, frequency modulation),
       flag broadband transients, and cut a fixed 0.48 s log-mel patch
       centred on its energy peak.
    5. Train a denoising convolutional autoencoder on the patches.  The
       bottleneck is a compact embedding of each sound's shape and the
       reconstruction error is its anomaly score.
    6. Cluster the embeddings into call types (k chosen by silhouette) and set
       an anomaly threshold from held-out control events.
    7. Report per-recording welfare indicators (vocal activity, call-type mix,
       call pitch, broadband-transient rate, anomaly rate, shift in call-type
       mix) as CSV files and figures, and export Audacity label files so the
       detected events can be reviewed and relabelled by ear.

Step 7's label files use the same start / end / label layout the team's
supervised loaders read, so relabelled events (e.g. renaming some "type3" tags
to "cough") can feed the ResNet18 / random forest code directly once the audio
is converted to FLAC:  ffmpeg -i Ctrl_W4_D1_PoS_1.mp3 Ctrl_W4_D1_PoS_1.flac

Usage:
    # Fit the baseline on control recordings, then score them
    python broiler_acoustic_baseline.py ./data/ctrl --out-dir ./runs/ctrl_baseline

    # Score new recordings (other weeks, treatment or challenge groups)
    python broiler_acoustic_baseline.py ./data/new \
        --checkpoint ./runs/ctrl_baseline/ctrl_baseline.pt --out-dir ./runs/new_vs_ctrl
"""

import argparse
import re
import sys
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from copy                      import deepcopy

from matplotlib.ticker         import MaxNLocator

from pathlib                   import Path

from scipy.ndimage             import percentile_filter
from scipy.signal              import butter
from scipy.signal              import find_peaks
from scipy.signal              import sosfiltfilt
from scipy.spatial.distance    import jensenshannon

from sklearn.cluster           import KMeans
from sklearn.metrics           import silhouette_score

from torch.utils.data          import DataLoader
from torch.utils.data          import TensorDataset

from typing                    import Dict
from typing                    import List
from typing                    import Optional
from typing                    import Tuple


DATA_DIR               = './data/raw'
OUTPUT_DIR             = './runs/ctrl_baseline'
CHECKPOINT_NAME        = 'ctrl_baseline.pt'
AUDIO_PATTERNS         = ('*.mp3', '*.flac', '*.wav')
NAME_REGEX             = re.compile(r'(?P<group>[A-Za-z]+)_W(?P<week>\d+)_D(?P<day>\d+)_PoS_(?P<pos>\d+)',
                                    re.IGNORECASE)

# Signal conditioning
TARGET_SR              = 32000
BAND_LOW_HZ            = 2000.0
BAND_HIGH_HZ           = 12000.0
FILTER_ORDER           = 4          # applied forward and backward (zero phase)

# Spectrogram
N_FFT                  = 1024
WIN_LENGTH             = 800        # 25 ms
HOP_LENGTH             = 320        # 10 ms
DESCRIPTOR_HOP         = 160        # 5 ms, finer track for frequency modulation
N_MELS                 = 64

# Event detection: hysteresis on band energy above a rolling noise floor
FLOOR_WINDOW_S         = 5.0
FLOOR_PERCENTILE       = 10.0
ABSOLUTE_FLOOR_DB      = -100.0
ONSET_DB               = 10.0       # an event must rise this far above the floor ...
OFFSET_DB              = 5.0        # ... and lasts while it stays this far above it
MIN_EVENT_S            = 0.03
MERGE_GAP_S            = 0.03
MAX_EVENT_S            = 1.0
VALLEY_DB              = 6.0        # split a run where energy dips this far between two calls
TRIM_DB                = 15.0       # keep only frames within this of the event's own peak
SMOOTH_FRAMES          = 3

# Broadband transients (candidate coughs, sneezes, pecks, wing flaps): energy
# spread evenly across frequency AND covering nearly the whole band at once.
# Flatness alone fails here because the recordings' steep spectral tilt keeps
# it low even for clicks; coverage alone fails because loud calls with
# harmonics light up most bands too.
BROADBAND_FLATNESS_MIN = 0.15
BROADBAND_COVERAGE_MIN = 0.90
COVERAGE_DB            = 12.0       # a band is "covered" this far above its quiet level

# Autoencoder
PATCH_FRAMES           = 48         # 0.48 s, must be divisible by 8
EMBED_DIM              = 32
BASE_CHANNELS          = 16
MAX_FREQ_MASK          = 8          # SpecAugment-style masking during training
MAX_TIME_MASK          = 6

# Training
DEFAULT_NUM_EPOCHS     = 60
DEFAULT_BATCH_SIZE     = 128
DEFAULT_LEARNING_RATE  = 1.0e-3
DEFAULT_PATIENCE       = 8
DEFAULT_RANDOM_STATE   = 77
VAL_BLOCK_S            = 60.0       # every Nth minute of each recording is held out
VAL_EVERY_NTH_BLOCK    = 5

# Baseline
ANOMALY_PERCENTILE     = 99.0
CLUSTER_K_RANGE        = range(3, 9)
SILHOUETTE_SAMPLES     = 4000
EPS                    = 1.0e-10

# Figures
SERIES_COLORS          = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                          '#e87ba4', '#008300', '#6250d6', '#e34948']
INK_PRIMARY            = '#0b0b0b'
INK_SECONDARY          = '#52514e'
GRID_COLOR             = '#e4e3df'


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 3,
                        padding = 1, bias = False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size = 3,
                        padding = 1, bias = False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size = 2)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.out_channels,
                        kernel_size = 2, stride = 2, bias = False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size = 3,
                        padding = 1, bias = False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class MelPatchAutoencoder(nn.Module):
    """
    Convolutional autoencoder over (mel x time) patches.  Three 2x poolings
    take a 64 x 48 patch to an 8 x 6 map, which is flattened to a small
    embedding and decoded back to the full patch.
    """
    def __init__(self, num_mels = N_MELS, num_frames = PATCH_FRAMES,
                        embed_dim = EMBED_DIM, base_channels = BASE_CHANNELS):
        super().__init__()
        if (((num_mels % 8) != 0) or ((num_frames % 8) != 0)):
            raise ValueError('num_mels and num_frames must be divisible by 8')
        self.num_mels = num_mels
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.base_channels = base_channels

        channels = [base_channels, (2 * base_channels), (4 * base_channels)]
        self.bottleneck_shape = (channels[2], (num_mels // 8), (num_frames // 8))
        flat_size = int(np.prod(self.bottleneck_shape))

        self.encoder = nn.Sequential(
            DownBlock(1, channels[0]),
            DownBlock(channels[0], channels[1]),
            DownBlock(channels[1], channels[2])
        )
        self.to_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, self.embed_dim)
        )
        self.from_embedding = nn.Sequential(
            nn.Linear(self.embed_dim, flat_size),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            UpBlock(channels[2], channels[1]),
            UpBlock(channels[1], channels[0]),
            UpBlock(channels[0], channels[0]),
            nn.Conv2d(channels[0], 1, kernel_size = 1)
        )

    def encode(self, x):
        return self.to_embedding(self.encoder(x.unsqueeze(1)))

    def decode(self, z):
        h = self.from_embedding(z).view(-1, *self.bottleneck_shape)
        return self.decoder(h).squeeze(1)

    def forward(self, x):
        z = self.encode(x)
        return (self.decode(z), z)


class EarlyStopper(object):
    """Keeps the weights from the best validation epoch."""
    def __init__(self, patience = DEFAULT_PATIENCE, min_delta = 1.0e-4):
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.best_state = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.wait = 0

    def update(self, model: nn.Module, val_loss: float, epoch: int) -> bool:
        """Record one epoch; returns True when training should stop."""
        if (val_loss < (self.best_loss - self.min_delta)):
            self.best_state = deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
        return (self.wait >= self.patience)

    def restore(self, model: nn.Module) -> nn.Module:
        if (self.best_state is not None):
            model.load_state_dict(self.best_state)
        return model


# --------------------------------------------------------------------------
# Signal processing and event detection
# --------------------------------------------------------------------------

def parse_recording_name(path: Path) -> Dict[str, object]:
    """Ctrl_W4_D1_PoS_3.mp3 -> group 'Ctrl', week 4, day 1, position 3."""
    match = NAME_REGEX.search(path.stem)
    if (match is None):
        return {'recording': path.stem, 'group': 'unknown', 'week': -1,
                        'day': -1, 'position': -1}
    return {'recording': match.group(0), 'group': match.group('group'),
                        'week': int(match.group('week')),
                        'day': int(match.group('day')),
                        'position': int(match.group('pos'))}


def list_recordings(directory: str) -> List[Path]:
    root = Path(directory)
    if (root.is_file()):
        return [root]
    return sorted({f for pattern in AUDIO_PATTERNS for f in root.rglob(pattern)})


def load_audio(path: Path, sample_rate: int = TARGET_SR) -> np.ndarray:
    # Stereo channels in these recordings are near-identical (r > 0.99),
    # so a mono mix loses nothing
    (samples, _) = librosa.load(str(path), sr = sample_rate, mono = True)
    return samples.astype(np.float32)


def bandpass(samples: np.ndarray, sample_rate: int = TARGET_SR,
                        low_hz: float = BAND_LOW_HZ,
                        high_hz: float = BAND_HIGH_HZ,
                        order: int = FILTER_ORDER) -> np.ndarray:
    sos = butter(order, [low_hz, high_hz], btype = 'bandpass',
                        fs = sample_rate, output = 'sos')
    return sosfiltfilt(sos, samples).astype(np.float32)


def compute_logmel(samples: np.ndarray,
                        sample_rate: int = TARGET_SR) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (log-mel in dB, shape [mels, frames]; band energy in dB per frame)."""
    mel = librosa.feature.melspectrogram(y = samples, sr = sample_rate,
                        n_fft = N_FFT, hop_length = HOP_LENGTH,
                        win_length = WIN_LENGTH, n_mels = N_MELS,
                        fmin = BAND_LOW_HZ, fmax = BAND_HIGH_HZ, power = 2.0)
    logmel_db = librosa.power_to_db(mel, ref = 1.0, amin = EPS, top_db = None)
    band_energy_db = 10.0 * np.log10(mel.sum(axis = 0) + EPS)
    return (logmel_db.astype(np.float32), band_energy_db.astype(np.float32))


def rolling_noise_floor(band_energy_db: np.ndarray, frame_rate: float) -> np.ndarray:
    size = max(int(round(FLOOR_WINDOW_S * frame_rate)), 1)
    floor_db = percentile_filter(band_energy_db, FLOOR_PERCENTILE, size = size,
                        mode = 'nearest')
    return np.maximum(floor_db, ABSOLUTE_FLOOR_DB)


def detect_events(band_energy_db: np.ndarray, floor_db: np.ndarray,
                        frame_rate: float, onset_db: float = ONSET_DB,
                        offset_db: float = OFFSET_DB) -> List[Tuple[int, int]]:
    """
    Returns (start_frame, end_frame) pairs, end exclusive.

    A rolling floor alone cannot separate calls in a dense chorus: the
    flock's overlapping calls keep the energy above the floor for seconds at a
    time.  Each above-floor run is therefore split again at every valley that
    dips VALLEY_DB below the calls on both sides, and each piece is trimmed to
    the frames within TRIM_DB of its own peak.
    """
    min_frames = max(int(round(MIN_EVENT_S * frame_rate)), 1)
    gap_frames = int(round(MERGE_GAP_S * frame_rate))
    max_frames = int(round(MAX_EVENT_S * frame_rate))
    smooth_db = np.convolve(band_energy_db, (np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES),
                        mode = 'same')
    excess_db = band_energy_db - floor_db
    strong = excess_db > onset_db
    weak = excess_db > offset_db

    # Runs of "weak" frames are candidates; keep runs holding a "strong" frame
    edges = np.flatnonzero(np.diff(np.concatenate(([0], weak.astype(np.int8), [0]))))
    candidates = [(s, e) for (s, e) in zip(edges[0::2], edges[1::2])
                        if (strong[s : e].any())]

    # Bridge short dips inside a single call
    merged = []
    for (start, end) in candidates:
        if (merged and ((start - merged[-1][1]) <= gap_frames)):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    events = []
    for (start, end) in merged:
        for (s, e) in split_at_valleys(smooth_db, start, end):
            if (not strong[s : e].any()):
                continue
            (s, e) = trim_to_peak(smooth_db, s, e)
            # Anything still too long is cut at its quietest frame; anything
            # too short is a click below the resolution of the descriptors
            pending = [(s, e)]
            while (pending):
                (a, b) = pending.pop()
                if ((b - a) > max_frames):
                    lo = a + min_frames
                    cut = lo + int(np.argmin(smooth_db[lo : (b - min_frames)]))
                    pending.extend([(cut, b), (a, cut)])
                elif ((b - a) >= min_frames):
                    events.append((int(a), int(b)))
    return sorted(events)


def split_at_valleys(energy_db: np.ndarray, start: int, end: int,
                        depth_db: float = VALLEY_DB) -> List[Tuple[int, int]]:
    # A valley's prominence in the negated signal is how far it sits below the
    # lower of the two peaks around it
    (valleys, _) = find_peaks(-energy_db[start : end], prominence = depth_db)
    bounds = [start] + [(start + int(v)) for v in valleys] + [end]
    return [(bounds[ii], bounds[ii + 1]) for ii in range(len(bounds) - 1)]


def trim_to_peak(energy_db: np.ndarray, start: int, end: int,
                        trim_db: float = TRIM_DB) -> Tuple[int, int]:
    segment = energy_db[start : end]
    peak = int(np.argmax(segment))
    quiet = np.flatnonzero(segment < (segment[peak] - trim_db))
    left = quiet[quiet < peak]
    right = quiet[quiet > peak]
    s = start + ((int(left[-1]) + 1) if (left.size) else 0)
    e = start + (int(right[0]) if (right.size) else segment.size)
    return (s, e)


def describe_event(segment: np.ndarray, sample_rate: int = TARGET_SR) -> Dict[str, float]:
    """Spectral descriptors of one event, restricted to the analysis band."""
    if (segment.size < N_FFT):
        segment = np.pad(segment, (0, N_FFT - segment.size))
    power = np.abs(librosa.stft(segment, n_fft = N_FFT, hop_length = DESCRIPTOR_HOP,
                        win_length = WIN_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr = sample_rate, n_fft = N_FFT)
    in_band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    (power, freqs) = (power[in_band], freqs[in_band])

    spectrum = power.mean(axis = 1) + EPS
    total = spectrum.sum()
    centroid = float((freqs * spectrum).sum() / total)
    bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * spectrum).sum() / total))
    flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))

    # Dominant-frequency track over frames within 10 dB of the loudest frame
    frame_energy = power.sum(axis = 0)
    loud = np.flatnonzero(frame_energy >= (0.1 * frame_energy.max()))
    track = freqs[np.argmax(power[:, loud], axis = 0)]
    if (track.size >= 3):
        times = loud * DESCRIPTOR_HOP / sample_rate
        fm_range = float(np.percentile(track, 90) - np.percentile(track, 10))
        fm_slope = float(np.polyfit(times, track, 1)[0])
    else:
        (fm_range, fm_slope) = (np.nan, np.nan)

    return {'peak_freq_hz': float(freqs[np.argmax(spectrum)]),
                        'centroid_hz': centroid, 'bandwidth_hz': bandwidth,
                        'flatness': flatness, 'fm_range_hz': fm_range,
                        'fm_slope_hz_per_s': fm_slope}


def band_coverage(whitened_db: np.ndarray, band_energy_db: np.ndarray,
                        start: int, end: int) -> float:
    """Mean fraction of mel bands lit up at once, over the event's loud frames."""
    energy = band_energy_db[start : end]
    loud = energy >= (energy.max() - 10.0)
    lit = whitened_db[:, start : end][:, loud] > COVERAGE_DB
    return float(lit.mean(axis = 0).mean())


def process_recording(path: Path) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    meta = parse_recording_name(path)
    samples = bandpass(load_audio(path))
    (logmel_db, band_energy_db) = compute_logmel(samples)
    frame_rate = TARGET_SR / HOP_LENGTH
    floor_db = rolling_noise_floor(band_energy_db, frame_rate)
    events = detect_events(band_energy_db, floor_db, frame_rate)

    # Express every band relative to its own quiet level in this recording.
    # Each microphone position has its own frequency response (the upstream
    # high-pass corner differs between files), and without this the model
    # learns to recognise microphones instead of calls
    whitened_db = logmel_db - np.percentile(logmel_db, 20, axis = 1, keepdims = True)

    # Pad with each band's quiet level so patches near the start or end of the
    # file keep their full width
    half = PATCH_FRAMES // 2
    pad = np.repeat(np.percentile(whitened_db, 5, axis = 1, keepdims = True), half, axis = 1)
    padded = np.concatenate((pad, whitened_db, pad), axis = 1)

    rows = []
    patches = np.zeros((len(events), N_MELS, PATCH_FRAMES), dtype = np.float32)
    for (ii, (start, end)) in enumerate(events):
        peak = start + int(np.argmax(band_energy_db[start : end]))
        row = dict(meta)
        row.update({'start_s': start / frame_rate, 'end_s': end / frame_rate,
                        'duration_s': (end - start) / frame_rate,
                        'peak_db': float(band_energy_db[peak]),
                        'snr_db': float(band_energy_db[peak] - floor_db[peak])})
        row.update(describe_event(samples[(start * HOP_LENGTH) : (end * HOP_LENGTH)]))
        row['band_coverage'] = band_coverage(whitened_db, band_energy_db, start, end)
        row['is_broadband'] = ((row['flatness'] >= BROADBAND_FLATNESS_MIN) and
                        (row['band_coverage'] >= BROADBAND_COVERAGE_MIN))
        rows.append(row)
        patches[ii] = padded[:, peak : (peak + PATCH_FRAMES)]

    info = dict(meta)
    info.update({'path': str(path), 'duration_s': samples.size / TARGET_SR,
                        'noise_floor_db': float(np.median(floor_db))})
    return (pd.DataFrame(rows), patches, info)


def load_dataset(files: List[Path]) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    (frames, patch_list, infos) = ([], [], [])
    for f in files:
        (events, patches, info) = process_recording(f)
        if (info['recording'] in [i['recording'] for i in infos]):
            info['recording'] = f.stem
            if (len(events) > 0):
                events['recording'] = f.stem
        minutes = info['duration_s'] / 60.0
        print(f'    {info["recording"]}:  {minutes:0.1f} min,  {len(events)} events  '
                        f'({len(events) / max(minutes, EPS):0.1f}/min),  '
                        f'floor {info["noise_floor_db"]:0.1f} dB')
        if (len(events) > 0):
            frames.append(events)
            patch_list.append(patches)
        infos.append(info)
    if (not frames):
        raise RuntimeError('No acoustic events were detected in any recording.')
    events = pd.concat(frames, ignore_index = True)
    return (events, np.concatenate(patch_list, axis = 0), pd.DataFrame(infos))


# --------------------------------------------------------------------------
# Training and scoring
# --------------------------------------------------------------------------

def blocked_validation_mask(events: pd.DataFrame) -> np.ndarray:
    """
    Hold out every Nth minute of every recording.  Neighbouring calls are
    strongly correlated, so a random event-level split would leak; whole
    blocks keep validation honest while every position stays represented.
    """
    block = np.floor(events['start_s'].to_numpy() / VAL_BLOCK_S).astype(int)
    return (block % VAL_EVERY_NTH_BLOCK) == (VAL_EVERY_NTH_BLOCK - 1)


def estimate_db_scale(raw_patches: np.ndarray) -> float:
    centred = raw_patches - np.median(raw_patches, axis = (1, 2), keepdims = True)
    return float(max(centred.std(), EPS))


def normalize_patches(raw_patches: np.ndarray, db_scale: float) -> np.ndarray:
    # Removing each patch's median cancels microphone gain differences between
    # positions, so the model compares the shape of a sound, not its level
    centred = raw_patches - np.median(raw_patches, axis = (1, 2), keepdims = True)
    return (centred / db_scale).astype(np.float32)


def mask_spectrogram(batch: torch.Tensor, max_freq: int = MAX_FREQ_MASK,
                        max_time: int = MAX_TIME_MASK) -> torch.Tensor:
    """One random frequency band and one random time span per patch set to 0."""
    (b, f, t) = batch.shape
    device = batch.device

    def band_mask(size, max_width):
        width = torch.randint(0, (max_width + 1), (b, 1), device = device)
        start = (torch.rand((b, 1), device = device) * (size - width + 1)).long()
        index = torch.arange(size, device = device).unsqueeze(0)
        return (index >= start) & (index < (start + width))

    mask = band_mask(f, max_freq).unsqueeze(2) | band_mask(t, max_time).unsqueeze(1)
    return batch.masked_fill(mask, 0.0)


@torch.no_grad()
def embed_and_score(model: MelPatchAutoencoder, patches: np.ndarray,
                        batch_size: int = DEFAULT_BATCH_SIZE,
                        device: torch.device = torch.device('cpu')) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (per-patch reconstruction MSE, embeddings)."""
    model.to(device)
    model.eval()
    (errors, embeddings) = ([], [])
    for start in range(0, len(patches), batch_size):
        batch = torch.from_numpy(patches[start : (start + batch_size)]).to(device)
        (recon, z) = model(batch)
        errors.append(((recon - batch) ** 2).mean(dim = (1, 2)).cpu().numpy())
        embeddings.append(z.cpu().numpy())
    if (not errors):
        return (np.zeros(0, dtype = np.float32), np.zeros((0, model.embed_dim), dtype = np.float32))
    return (np.concatenate(errors), np.concatenate(embeddings))


def train_autoencoder(model: MelPatchAutoencoder, train_patches: np.ndarray,
                        val_patches: np.ndarray,
                        num_epochs: int = DEFAULT_NUM_EPOCHS,
                        batch_size: int = DEFAULT_BATCH_SIZE,
                        learning_rate: float = DEFAULT_LEARNING_RATE,
                        device: torch.device = torch.device('cpu')) -> Tuple[MelPatchAutoencoder, Dict[str, list]]:
    generator = torch.Generator().manual_seed(DEFAULT_RANDOM_STATE)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_patches)),
                        batch_size = batch_size, shuffle = True, generator = generator,
                        drop_last = (len(train_patches) > batch_size))
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = 1.0e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3)
    stopper = EarlyStopper()
    history = {'train_loss': [], 'val_loss': []}

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        (running, count) = (0.0, 0)
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # Denoising objective: reconstruct the clean patch from a masked one
            (recon, _) = model(mask_spectrogram(batch))
            loss = criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5.0)
            optimizer.step()
            running += loss.item() * batch.size(0)
            count += batch.size(0)

        train_loss = running / max(count, 1)
        val_loss = float(embed_and_score(model, val_patches, batch_size, device)[0].mean())
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1:3d} of {num_epochs}   train MSE {train_loss:0.5f}   '
                        f'val MSE {val_loss:0.5f}   lr {lr:0.1e}')
        scheduler.step(val_loss)
        if (stopper.update(model, val_loss, (epoch + 1))):
            print(f'Stopping early: best epoch was {stopper.best_epoch}')
            break

    stopper.restore(model)
    history['best_epoch'] = stopper.best_epoch
    return (model, history)


def fit_call_types(embeddings: np.ndarray,
                        k_range: range = CLUSTER_K_RANGE) -> Dict[str, np.ndarray]:
    """K-means on standardised embeddings; k picked by silhouette score."""
    mean = embeddings.mean(axis = 0)
    std = embeddings.std(axis = 0) + EPS
    z = (embeddings - mean) / std
    (best, best_score, scores) = (None, -np.inf, {})
    for k in k_range:
        if (k >= len(z)):
            break
        km = KMeans(n_clusters = k, n_init = 10, random_state = DEFAULT_RANDOM_STATE).fit(z)
        score = float(silhouette_score(z, km.labels_,
                        sample_size = min(SILHOUETTE_SAMPLES, len(z)),
                        random_state = DEFAULT_RANDOM_STATE))
        scores[k] = score
        print(f'    k = {k}:  silhouette {score:0.4f}')
        if (score > best_score):
            (best, best_score) = (km, score)
    if (best is None):
        raise RuntimeError('Too few events to cluster.')

    # Re-index so that type 0 is the most common call type
    counts = np.bincount(best.labels_, minlength = best.n_clusters)
    centers = best.cluster_centers_[np.argsort(-counts, kind = 'stable')]
    return {'centers': centers, 'mean': mean, 'std': std, 'silhouette': scores}


def assign_call_types(embeddings: np.ndarray, centers: np.ndarray, mean: np.ndarray,
                        std: np.ndarray) -> np.ndarray:
    z = (embeddings - mean) / std
    distances = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis = -1)
    return np.argmin(distances, axis = 1)


def save_baseline(filename: Path, model: MelPatchAutoencoder, baseline: Dict[str, object]) -> None:
    checkpoint = {
        'model_state': model.state_dict(),
        'model_config': {'num_mels': model.num_mels, 'num_frames': model.num_frames,
                        'embed_dim': model.embed_dim, 'base_channels': model.base_channels},
        'signal_config': signal_config(),
        'db_scale': float(baseline['db_scale']),
        'anomaly_threshold': float(baseline['anomaly_threshold']),
        'centers': torch.tensor(baseline['centers'], dtype = torch.float32),
        'embed_mean': torch.tensor(baseline['mean'], dtype = torch.float32),
        'embed_std': torch.tensor(baseline['std'], dtype = torch.float32),
        'baseline_mix': torch.tensor(baseline['baseline_mix'], dtype = torch.float32),
        'best_epoch': int(baseline['best_epoch']),
        'trained_on': list(baseline['trained_on'])
    }
    torch.save(checkpoint, filename)


def load_baseline(filename: str) -> Tuple[MelPatchAutoencoder, Dict[str, object]]:
    checkpoint = torch.load(filename, map_location = 'cpu', weights_only = True)
    if (checkpoint['signal_config'] != signal_config()):
        raise ValueError('Checkpoint was built with different signal settings: '
                        f'{checkpoint["signal_config"]}')
    model = MelPatchAutoencoder(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state'])
    baseline = {'db_scale': checkpoint['db_scale'],
                        'anomaly_threshold': checkpoint['anomaly_threshold'],
                        'centers': checkpoint['centers'].numpy(),
                        'mean': checkpoint['embed_mean'].numpy(),
                        'std': checkpoint['embed_std'].numpy(),
                        'baseline_mix': checkpoint['baseline_mix'].numpy(),
                        'best_epoch': checkpoint['best_epoch'],
                        'trained_on': checkpoint['trained_on']}
    return (model, baseline)


def signal_config() -> Dict[str, float]:
    return {'target_sr': TARGET_SR, 'band_low_hz': BAND_LOW_HZ,
                        'band_high_hz': BAND_HIGH_HZ, 'n_fft': N_FFT,
                        'win_length': WIN_LENGTH, 'hop_length': HOP_LENGTH,
                        'n_mels': N_MELS, 'patch_frames': PATCH_FRAMES}


# --------------------------------------------------------------------------
# Welfare indicators
# --------------------------------------------------------------------------

def summarize_recordings(events: pd.DataFrame, recordings: pd.DataFrame,
                        num_types: int, baseline_mix: np.ndarray) -> pd.DataFrame:
    rows = []
    for (_, rec) in recordings.iterrows():
        ev = events[events['recording'] == rec['recording']]
        minutes = rec['duration_s'] / 60.0
        calls = ev[~ev['is_broadband']]
        held_out = ev[ev['split'] == 'val']

        # Calls per complete minute, to describe how bursty the activity is
        full_minutes = int(np.floor(minutes))
        minute_index = (ev['start_s'] // 60).astype(int).to_numpy()
        per_minute = np.bincount(minute_index[minute_index < full_minutes],
                        minlength = full_minutes)
        activity_cv = (per_minute.std() / per_minute.mean()) if (per_minute.size and per_minute.mean() > 0) else np.nan

        mix = np.bincount(ev['call_type'], minlength = num_types) / max(len(ev), 1)
        row = {'recording': rec['recording'], 'group': rec['group'], 'week': rec['week'],
                        'day': rec['day'], 'position': rec['position'],
                        'duration_min': minutes,
                        'noise_floor_db': rec['noise_floor_db'],
                        'n_events': len(ev),
                        'events_per_min': len(ev) / minutes,
                        'activity_fraction': ev['duration_s'].sum() / rec['duration_s'],
                        'activity_cv': activity_cv,
                        'calls_per_min': len(calls) / minutes,
                        'broadband_per_min': (len(ev) - len(calls)) / minutes,
                        'broadband_share': ev['is_broadband'].mean() if len(ev) else np.nan,
                        'call_peak_freq_median_hz': calls['peak_freq_hz'].median(),
                        'call_peak_freq_iqr_hz': (calls['peak_freq_hz'].quantile(0.75)
                                        - calls['peak_freq_hz'].quantile(0.25)),
                        'call_duration_median_s': calls['duration_s'].median(),
                        'call_fm_range_median_hz': calls['fm_range_hz'].median(),
                        'anomaly_score_median': ev['anomaly_score'].median(),
                        'anomaly_rate': ev['is_anomaly'].mean() if len(ev) else np.nan,
                        'anomaly_rate_heldout': held_out['is_anomaly'].mean() if len(held_out) else np.nan,
                        'call_mix_js_distance': (float(jensenshannon(mix, baseline_mix, base = 2))
                                        if len(ev) else np.nan)}
        for k in range(num_types):
            row[f'type{k}_share'] = mix[k]
        rows.append(row)
    return pd.DataFrame(rows)


def describe_call_type(broadband_share: float, duration_s: float, peak_freq_hz: float) -> str:
    if (broadband_share >= 0.5):
        texture = 'broadband'
    elif (broadband_share >= 0.15):
        texture = 'mixed'
    else:
        texture = 'tonal'
    if (duration_s < 0.08):
        length = 'short'
    elif (duration_s < 0.25):
        length = 'medium'
    else:
        length = 'long'
    return f'{texture}, {length}, ~{peak_freq_hz / 1000.0:0.1f} kHz'


def summarize_call_types(events: pd.DataFrame, num_types: int) -> pd.DataFrame:
    rows = []
    for k in range(num_types):
        ev = events[events['call_type'] == k]
        if (len(ev) == 0):
            rows.append({'call_type': k, 'n_events': 0, 'share': 0.0, 'description': 'not observed'})
            continue
        rows.append({'call_type': k, 'n_events': len(ev), 'share': len(ev) / len(events),
                        'median_duration_s': ev['duration_s'].median(),
                        'median_peak_freq_hz': ev['peak_freq_hz'].median(),
                        'median_flatness': ev['flatness'].median(),
                        'median_band_coverage': ev['band_coverage'].median(),
                        'broadband_share': ev['is_broadband'].mean(),
                        'median_snr_db': ev['snr_db'].median(),
                        'median_fm_range_hz': ev['fm_range_hz'].median(),
                        'anomaly_rate': ev['is_anomaly'].mean(),
                        'description': describe_call_type(ev['is_broadband'].mean(),
                                        ev['duration_s'].median(),
                                        ev['peak_freq_hz'].median())})
    return pd.DataFrame(rows)


def export_audacity_labels(events: pd.DataFrame, out_dir: Path) -> Path:
    """One <recording>_label.txt per recording: start, end, tag (tab separated)."""
    label_dir = out_dir / 'labels'
    label_dir.mkdir(parents = True, exist_ok = True)
    for (recording, ev) in events.groupby('recording'):
        with open((label_dir / f'{recording}_label.txt'), mode = 'w', encoding = 'utf-8') as fp:
            for row in ev.sort_values('start_s').itertuples():
                tag = f'type{row.call_type}' + ('_anomaly' if row.is_anomaly else '')
                fp.write(f'{row.start_s:0.6f}\t{row.end_s:0.6f}\t{tag}\n')
    return label_dir


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _style_axes(ax) -> None:
    ax.grid(True, axis = 'y', color = GRID_COLOR, linewidth = 0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors = INK_SECONDARY, labelsize = 8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)


def plot_learning_curves(history: Dict[str, list], path: Path) -> None:
    epochs = np.arange(1, (len(history['train_loss']) + 1))
    (fig, ax) = plt.subplots(figsize = (7, 4))
    ax.plot(epochs, history['train_loss'], color = SERIES_COLORS[0], linewidth = 2,
                        label = 'Train (masked input)')
    ax.plot(epochs, history['val_loss'], color = SERIES_COLORS[1], linewidth = 2,
                        label = 'Validation (held-out minutes)')
    ax.axvline(history['best_epoch'], color = INK_SECONDARY, linestyle = ':', linewidth = 1)
    ax.set_title('Autoencoder reconstruction loss', loc = 'left')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (normalised dB)')
    ax.legend(frameon = False, fontsize = 8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi = 120)


def plot_activity_timeline(events: pd.DataFrame, recordings: pd.DataFrame, path: Path) -> None:
    num = len(recordings)
    max_minutes = int(np.ceil(recordings['duration_s'].max() / 60.0))
    height = (1.6 * num) + 1.2
    (fig, axes) = plt.subplots(num, 1, figsize = (9, height), sharex = True,
                        sharey = True, squeeze = False)
    for (ax, (_, rec)) in zip(axes[:, 0], recordings.iterrows()):
        ev = events[events['recording'] == rec['recording']]
        minute = (ev['start_s'] // 60).astype(int)
        calls = np.bincount(minute[~ev['is_broadband']], minlength = max_minutes)[: max_minutes]
        broad = np.bincount(minute[ev['is_broadband']], minlength = max_minutes)[: max_minutes]
        x = np.arange(max_minutes) + 0.5
        ax.bar(x, calls, width = 0.9, color = SERIES_COLORS[0], edgecolor = 'white',
                        linewidth = 1, label = 'Calls')
        ax.bar(x, broad, width = 0.9, bottom = calls, color = SERIES_COLORS[1],
                        edgecolor = 'white', linewidth = 1, label = 'Broadband transients')
        ax.set_title(rec['recording'], loc = 'left', fontsize = 9)
        ax.set_ylabel('per min', fontsize = 8)
        _style_axes(ax)
    axes[-1, 0].set_xlabel('Minute of recording (last bar may be a partial minute)')
    axes[-1, 0].xaxis.set_major_locator(MaxNLocator(integer = True))
    # Title on the left and legend on the right share a reserved top band
    (handles, labels) = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon = False, fontsize = 8, ncol = 2,
                        loc = 'upper right')
    fig.suptitle('Detected events per minute', x = 0.01, ha = 'left', color = INK_PRIMARY)
    fig.tight_layout(rect = [0, 0, 1, (1.0 - (0.45 / height))])
    fig.savefig(path, dpi = 120)


def plot_peak_frequency(events: pd.DataFrame, recordings: pd.DataFrame, path: Path) -> None:
    names = list(recordings['recording'])
    data = [events.loc[(events['recording'] == name) & ~events['is_broadband'], 'peak_freq_hz'] / 1000.0
                        for name in names]
    (fig, ax) = plt.subplots(figsize = (7, (0.5 * len(names)) + 1.5))
    # matplotlib 3.10 replaced vert = False with orientation = 'horizontal'
    version = tuple(int(v) for v in matplotlib.__version__.split('.')[: 2])
    direction = {'orientation': 'horizontal'} if (version >= (3, 10)) else {'vert': False}
    ax.boxplot(data, **direction, widths = 0.5, showfliers = False, patch_artist = True,
                        boxprops = {'facecolor': SERIES_COLORS[0], 'edgecolor': SERIES_COLORS[0], 'alpha': 0.35},
                        medianprops = {'color': SERIES_COLORS[0], 'linewidth': 2},
                        whiskerprops = {'color': INK_SECONDARY}, capprops = {'color': INK_SECONDARY})
    ax.set_yticks(np.arange(1, (len(names) + 1)), names)
    ax.invert_yaxis()
    ax.set_xlabel('Peak frequency of calls (kHz)')
    ax.set_title('Call pitch by recording', loc = 'left')
    _style_axes(ax)
    ax.grid(True, axis = 'x', color = GRID_COLOR, linewidth = 0.8)
    ax.grid(False, axis = 'y')
    fig.tight_layout()
    fig.savefig(path, dpi = 120)


def plot_call_type_mix(summary: pd.DataFrame, type_summary: pd.DataFrame, path: Path) -> None:
    num_types = len(type_summary)
    legend_height = (0.22 * int(np.ceil(num_types / 2))) + 0.2
    height = (0.5 * len(summary)) + 1.5 + legend_height
    (fig, ax) = plt.subplots(figsize = (9, height))
    left = np.zeros(len(summary))
    y = np.arange(len(summary))
    for k in range(num_types):
        share = summary[f'type{k}_share'].to_numpy()
        label = f'type{k}: {type_summary.loc[k, "description"]}'
        ax.barh(y, share, left = left, height = 0.6, color = SERIES_COLORS[k],
                        edgecolor = 'white', linewidth = 2, label = label)
        left += share
    ax.set_yticks(y, summary['recording'])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Share of detected events')
    ax.set_title('Call-type mix by recording', loc = 'left')
    _style_axes(ax)
    ax.grid(False)
    # Legend gets its own band under the axes so it never covers the x label
    (handles, labels) = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon = False, fontsize = 7, ncol = 2,
                        loc = 'lower left', bbox_to_anchor = (0.01, 0.0))
    fig.tight_layout(rect = [0, (legend_height / height), 1, 1])
    fig.savefig(path, dpi = 120)


def plot_call_type_gallery(patches: np.ndarray, events: pd.DataFrame,
                        type_summary: pd.DataFrame, path: Path) -> None:
    num_types = len(type_summary)
    mel_freqs = librosa.mel_frequencies(n_mels = N_MELS, fmin = BAND_LOW_HZ, fmax = BAND_HIGH_HZ)
    extent = [-(PATCH_FRAMES // 2) * HOP_LENGTH / TARGET_SR * 1000.0,
                        (PATCH_FRAMES // 2) * HOP_LENGTH / TARGET_SR * 1000.0, 0, N_MELS]
    (fig, axes) = plt.subplots(1, num_types, figsize = ((2.2 * num_types) + 0.5, 3.2),
                        sharey = True, squeeze = False)
    tick_rows = [int(np.argmin(np.abs(mel_freqs - hz))) for hz in (2000, 4000, 6000, 8000, 12000)]
    for k in range(num_types):
        ax = axes[0, k]
        members = patches[events['call_type'].to_numpy() == k]
        if (len(members) > 0):
            ax.imshow(np.median(members, axis = 0), origin = 'lower', aspect = 'auto',
                        cmap = 'magma', extent = extent)
        row = type_summary.loc[k]
        ax.set_title(f'type{k}  (n = {row["n_events"]})\n{row["description"]}', fontsize = 7,
                        loc = 'left', color = INK_PRIMARY)
        ax.set_xlabel('ms from peak', fontsize = 7, color = INK_SECONDARY)
        ax.tick_params(labelsize = 7, colors = INK_SECONDARY)
    axes[0, 0].set_yticks([r + 0.5 for r in tick_rows],
                        [f'{mel_freqs[r] / 1000.0:0.0f}k' for r in tick_rows])
    axes[0, 0].set_ylabel('Frequency (Hz)', fontsize = 7, color = INK_SECONDARY)
    fig.suptitle('Median spectrogram of each call type', x = 0.01, ha = 'left',
                        color = INK_PRIMARY, fontsize = 10)
    fig.tight_layout()
    fig.savefig(path, dpi = 120)


def plot_anomaly_scores(events: pd.DataFrame, threshold: float, path: Path) -> None:
    scores = np.log10(events['anomaly_score'].to_numpy() + EPS)
    (fig, ax) = plt.subplots(figsize = (7, 4))
    ax.hist(scores, bins = 60, color = SERIES_COLORS[0], edgecolor = 'white', linewidth = 0.5)
    ax.axvline(np.log10(threshold), color = INK_PRIMARY, linestyle = '--', linewidth = 1.2)
    ax.text(np.log10(threshold), ax.get_ylim()[1] * 0.95,
                        f'  threshold ({ANOMALY_PERCENTILE:0.0f}th pct of held-out Ctrl)',
                        fontsize = 8, color = INK_PRIMARY, va = 'top')
    ax.set_xlabel('log10 reconstruction error')
    ax.set_ylabel('Events')
    ax.set_title('Anomaly scores', loc = 'left')
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi = 120)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'Unsupervised acoustic baseline for broiler recordings.')
    parser.add_argument('data_dir', nargs = '?', default = DATA_DIR,
                        help = 'Folder (searched recursively) or single audio file')
    parser.add_argument('--checkpoint', default = '',
                        help = 'Score against an existing baseline instead of fitting one')
    parser.add_argument('--out-dir', default = OUTPUT_DIR)
    parser.add_argument('--epochs', type = int, default = DEFAULT_NUM_EPOCHS)
    parser.add_argument('--batch-size', type = int, default = DEFAULT_BATCH_SIZE)
    parser.add_argument('--device', default = '', help = 'cpu or cuda (default: auto)')
    parser.add_argument('--show', action = 'store_true', help = 'Display figures as well as saving them')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if (not args.show):
        plt.switch_backend('Agg')
    torch.manual_seed(DEFAULT_RANDOM_STATE)
    np.random.seed(DEFAULT_RANDOM_STATE)
    device = torch.device(args.device if (args.device) else
                        ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents = True, exist_ok = True)

    files = list_recordings(args.data_dir)
    if (not files):
        print(f'No audio files found in {args.data_dir}')
        return 1
    print(f'Detecting events in {len(files)} recording(s)...')
    (events, raw_patches, recordings) = load_dataset(files)
    print(f'Events:  {len(events)};  Patch Size:  {raw_patches.shape[1:]}')

    history = None
    if (args.checkpoint):
        (model, baseline) = load_baseline(args.checkpoint)
        print(f'Loaded baseline trained on {len(baseline["trained_on"])} recording(s)')
        events['split'] = 'score'
        patches = normalize_patches(raw_patches, baseline['db_scale'])
    else:
        is_val = blocked_validation_mask(events)
        events['split'] = np.where(is_val, 'val', 'train')
        db_scale = estimate_db_scale(raw_patches[~is_val])
        patches = normalize_patches(raw_patches, db_scale)
        print(f'Train Events:  {(~is_val).sum()};  Validation Events:  {is_val.sum()}')

        model = MelPatchAutoencoder()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable Parameters:  {trainable}')
        (model, history) = train_autoencoder(model, patches[~is_val], patches[is_val],
                        num_epochs = args.epochs, batch_size = args.batch_size,
                        device = device)

        (val_errors, _) = embed_and_score(model, patches[is_val], args.batch_size, device)
        (_, embeddings) = embed_and_score(model, patches, args.batch_size, device)
        print('Choosing the number of call types...')
        call_types = fit_call_types(embeddings)
        baseline = dict(call_types)
        baseline.update({'db_scale': db_scale,
                        'anomaly_threshold': float(np.percentile(val_errors, ANOMALY_PERCENTILE)),
                        'best_epoch': history['best_epoch'],
                        'trained_on': list(recordings['recording'])})

    (errors, embeddings) = embed_and_score(model, patches, args.batch_size, device)
    num_types = len(baseline['centers'])
    events['anomaly_score'] = errors
    events['is_anomaly'] = errors > baseline['anomaly_threshold']
    events['call_type'] = assign_call_types(embeddings, baseline['centers'],
                        baseline['mean'], baseline['std'])

    if (not args.checkpoint):
        baseline['baseline_mix'] = np.bincount(events['call_type'], minlength = num_types) / len(events)
        save_baseline((out_dir / CHECKPOINT_NAME), model, baseline)
        print(f'Saved baseline to {out_dir / CHECKPOINT_NAME}')

    summary = summarize_recordings(events, recordings, num_types, baseline['baseline_mix'])
    type_summary = summarize_call_types(events, num_types)
    events.to_csv((out_dir / 'events.csv'), index = False)
    summary.to_csv((out_dir / 'recording_summary.csv'), index = False)
    type_summary.to_csv((out_dir / 'call_types.csv'), index = False)
    label_dir = export_audacity_labels(events, out_dir)

    if (history is not None):
        plot_learning_curves(history, (fig_dir / 'learning_curves.png'))
    plot_activity_timeline(events, recordings, (fig_dir / 'activity_timeline.png'))
    plot_peak_frequency(events, recordings, (fig_dir / 'call_pitch.png'))
    plot_call_type_mix(summary, type_summary, (fig_dir / 'call_type_mix.png'))
    plot_call_type_gallery(patches, events, type_summary, (fig_dir / 'call_type_gallery.png'))
    plot_anomaly_scores(events, baseline['anomaly_threshold'], (fig_dir / 'anomaly_scores.png'))

    with pd.option_context('display.width', 160, 'display.max_columns', 20):
        print('\nCall types:')
        print(type_summary[['call_type', 'n_events', 'share', 'description', 'anomaly_rate']]
                        .to_string(index = False, float_format = lambda v: f'{v:0.3f}'))
        print('\nPer-recording indicators:')
        columns = ['recording', 'events_per_min', 'activity_fraction', 'calls_per_min',
                        'broadband_per_min', 'call_peak_freq_median_hz', 'anomaly_rate',
                        'call_mix_js_distance']
        print(summary[columns].to_string(index = False, float_format = lambda v: f'{v:0.3f}'))
        if (summary['group'].nunique() * summary['day'].nunique() > 1):
            print('\nBy group / week / day:')
            print(summary.groupby(['group', 'week', 'day'])[columns[1 :]].mean()
                        .to_string(float_format = lambda v: f'{v:0.3f}'))
    print(f'\nOutputs:  {out_dir}  (labels in {label_dir.name}/, figures in {fig_dir.name}/)')
    if (args.show):
        plt.show()
    return 0


if (__name__ == '__main__'):
    sys.exit(main())
