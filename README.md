# Spoken Digit Recognition Using Time-Domain, DFT, and STFT Analysis

## Project Overview

This project focuses on spoken digit recognition using audio recordings from the Free Spoken Digit Dataset (FSDD). The goal is to classify spoken digits from 0 to 9 by extracting useful audio features using Digital Signal Processing (DSP) techniques and training machine learning classifiers.

Three main analysis approaches are implemented and compared:

1. Time-Domain Analysis
2. Discrete Fourier Transform (DFT) Analysis
3. Short-Time Fourier Transform (STFT) Analysis

Each approach represents the audio signal differently. The extracted features are converted into numerical feature vectors and used to train a Random Forest classifier.

---

## Dataset

The project uses the Free Spoken Digit Dataset.

Each audio file follows this naming format:

```text
digit_speaker_index.wav
```

Example:

```text
0_george_0.wav
```

This means:

```text
digit   = 0
speaker = george
index   = 0
```

The first value in the filename is used as the class label.

---

## Project Objective

The main objective of this project is to compare different DSP-based audio feature extraction methods for spoken digit classification.

The project aims to:

- Extract meaningful features from speech signals.
- Compare time-domain, DFT-based, and STFT-based representations.
- Train Random Forest classifiers using the extracted features.
- Evaluate classification performance using accuracy, classification report, confusion matrix, and feature importance.
- Understand which signal representation is more suitable for spoken digit recognition.

---

## General Workflow

The general workflow of the project is:

```text
Audio files
    ↓
Audio preprocessing
    ↓
Feature extraction
    ↓
Feature matrix construction
    ↓
Train-test split
    ↓
Random Forest training
    ↓
Model evaluation
```

Although the three approaches use different feature extraction methods, they follow the same general machine learning pipeline.

---

# Approach 1: Time-Domain Analysis

## Description

The time-domain approach analyzes the waveform directly without converting it into the frequency domain.

This approach focuses on the amplitude behavior, energy, signal shape, silence ratio, and periodicity of the audio signal.

Time-domain features are simple, interpretable, and computationally efficient.

---

## Audio Preprocessing

Each audio file is loaded using `librosa`:

```python
y, sr = librosa.load(file_path, sr=None, mono=True)
```

Silence is removed from the beginning and end of the recording:

```python
y, _ = librosa.effects.trim(y, top_db=20)
```

The signal amplitude is then normalized:

```python
max_abs = np.max(np.abs(y))
if max_abs > 0:
    y = y / max_abs
```

Normalization makes the recordings more comparable by scaling the maximum absolute amplitude to 1.

---

## Extracted Time-Domain Features

The time-domain model extracts the following groups of features.

---

## 1. Zero-Crossing Rate

Zero-crossing rate measures how often the waveform crosses the zero-amplitude axis.

```python
zcr = librosa.feature.zero_crossing_rate(y=y)[0]
```

This feature helps describe how rapidly the signal changes sign.

For the zero-crossing rate sequence, the following statistics are extracted:

```text
minimum
maximum
mean
standard deviation
skewness
kurtosis
```

---

## 2. RMS Energy

RMS energy measures the short-time energy or loudness of the signal.

```python
rms = librosa.feature.rms(y=y)[0]
```

This helps describe how the energy of the spoken digit changes over time.

The same six statistical values are extracted:

```text
minimum
maximum
mean
standard deviation
skewness
kurtosis
```

---

## 3. Signal Envelope

The signal envelope is extracted using the Hilbert transform:

```python
envelope = np.abs(hilbert(y))
```

The envelope describes the smooth amplitude shape of the signal. It shows how the loudness rises and falls during pronunciation.

The following statistics are extracted from the envelope:

```text
minimum
maximum
mean
standard deviation
skewness
kurtosis
```

---

## 4. Global Time-Domain Features

The code also extracts global waveform features:

```text
duration
energy
absolute mean amplitude
variance
peak amplitude
crest factor
silence ratio
```

Duration is calculated as:

```python
duration = len(y) / sr
```

Signal energy is calculated as:

```python
energy = np.sum(y ** 2)
```

Crest factor is calculated as:

```python
signal_rms = np.sqrt(np.mean(y ** 2)) + 1e-8
crest_factor = peak / signal_rms
```

Silence ratio is calculated as:

```python
silence_ratio = np.mean(np.abs(y) < 0.02)
```

---

## 5. Autocorrelation Features

Autocorrelation measures how similar the signal is to a delayed version of itself.

```python
autocorr = np.correlate(y, y, mode="full")
```

The code focuses on a pitch-related delay region between 5 ms and 30 ms:

```python
start = int(0.005 * sr)
end = int(0.030 * sr)
```

The extracted autocorrelation features are:

```text
autocorrelation peak
autocorrelation mean
autocorrelation standard deviation
peak lag in milliseconds
```

These features help capture periodic patterns in the speech signal.

---

## Time-Domain Feature Vector Size

The final time-domain feature vector contains:

```text
6 zero-crossing rate features
6 RMS energy features
6 envelope features
7 global waveform features
4 autocorrelation features
```

Therefore:

```text
6 + 6 + 6 + 7 + 4 = 29 features
```

Each audio file is represented by 29 time-domain features.

---

## Time-Domain Model

The time-domain features are scaled using `StandardScaler`:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Then a Random Forest classifier is trained:

```python
model = RandomForestClassifier(
    n_estimators=600,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42
)
```

---

# Approach 2: Discrete Fourier Transform Analysis

## Description

The Discrete Fourier Transform (DFT) converts the audio signal from the time domain into the frequency domain.

In this project, the DFT is implemented using the Fast Fourier Transform (FFT).

Unlike STFT, which analyzes the signal using short time segments, the DFT analyzes the entire signal at once. Therefore, it captures the overall frequency content of the audio signal but does not show how frequency content changes over time.

This makes DFT useful for general frequency analysis, but less suitable for non-stationary signals such as speech.

---

## Mathematical Definition

For a discrete-time signal \( x[n] \), the DFT is defined as:

```text
X[k] = Σ x[n] e^(-j2πkn/N)
```

where:

```text
x[n] = time-domain signal
X[k] = frequency-domain representation
N    = number of samples
k    = frequency-bin index
```

---

## DFT Audio Preprocessing

Each audio file is processed using the following steps:

1. Load the audio signal.
2. Remove silence from the beginning and end.
3. Normalize the amplitude.
4. Apply a Hanning window.
5. Compute the FFT.

The audio is loaded using:

```python
y, sr = librosa.load(file_path, sr=None, mono=True)
```

Silence is removed using:

```python
y, _ = librosa.effects.trim(y, top_db=20)
```

The audio is normalized:

```python
y = y / np.max(np.abs(y))
```

A Hanning window is applied:

```python
x = y * np.hanning(N)
```

The Hanning window helps reduce spectral leakage.

---

## FFT Computation

The FFT is computed using:

```python
X = np.fft.rfft(x)
```

The magnitude spectrum is obtained using:

```python
mag = np.abs(X)
```

The frequency axis is generated using:

```python
freqs = np.fft.rfftfreq(N, d=1/sr)
```

The magnitude spectrum represents the strength of each frequency component in the audio signal.

---

## Extracted DFT Features

The DFT approach extracts the following features:

```text
spectral centroid
spectral bandwidth
spectral rolloff at 85%
dominant frequency
spectral entropy
band energy ratio from 0–500 Hz
band energy ratio from 500–1000 Hz
band energy ratio from 1000–2000 Hz
band energy ratio from 2000–4000 Hz
spectral flatness
spectral skewness
spectral kurtosis
zero-crossing rate
```

Therefore, each audio file is represented by 13 DFT-based features.

Although zero-crossing rate is a time-domain feature, it is included in the DFT feature vector as an additional supporting feature.

---

## Important DFT Features

### Spectral Centroid

Spectral centroid represents the center of mass of the spectrum.

```python
centroid = np.sum(freqs * power) / power_sum
```

A higher centroid means the signal has more high-frequency content.

---

### Spectral Bandwidth

Spectral bandwidth measures how spread out the spectrum is around the centroid.

```python
bandwidth = np.sqrt(
    np.sum(((freqs - centroid) ** 2) * power) / power_sum
)
```

A larger bandwidth means the signal energy is distributed over a wider frequency range.

---

### Spectral Rolloff

Spectral rolloff is the frequency below which 85% of the total spectral energy is contained.

```python
rolloff_threshold = 0.85 * cumulative_energy[-1]
```

This feature indicates whether most of the signal energy is located in lower or higher frequencies.

---

### Dominant Frequency

Dominant frequency is the frequency with the highest magnitude in the spectrum.

```python
mag_no_dc = mag.copy()
mag_no_dc[0] = 0
dominant_idx = np.argmax(mag_no_dc)
dominant_freq = freqs[dominant_idx]
```

The DC component is removed before finding the dominant frequency.

---

### Spectral Entropy

Spectral entropy measures the spread or randomness of the spectral energy distribution.

```python
entropy = -np.sum(p * np.log2(p + 1e-12))
```

Low entropy means the energy is concentrated in fewer frequency bins. High entropy means the energy is more widely spread across the spectrum.

---

### Band Energy Ratios

The spectrum is divided into four frequency bands:

```text
0–500 Hz
500–1000 Hz
1000–2000 Hz
2000–4000 Hz
```

For each band, the ratio of band energy to total energy is calculated.

```text
band energy ratio = band energy / total energy
```

These features describe how the signal energy is distributed across low, mid, and high frequency regions.

---

### Spectral Flatness

Spectral flatness measures whether the signal is more tone-like or noise-like.

```python
flatness = np.exp(np.mean(np.log(mag + 1e-12))) / (np.mean(mag) + 1e-12)
```

A lower value usually indicates a more tonal signal, while a higher value indicates a more noise-like signal.

---

### Spectral Skewness and Kurtosis

Spectral skewness describes the asymmetry of the frequency distribution.

Spectral kurtosis describes the sharpness or peakiness of the spectrum.

These features help describe the overall shape of the frequency spectrum.

---

## DFT Experiments

The DFT section also includes experiments to study how signal-processing choices affect the spectrum and classification.

### FFT Size Experiment

Different FFT sizes are tested:

```text
256
512
1024
2048
4096
```

The frequency resolution is calculated as:

```text
Δf = fs / N
```

where:

```text
Δf = frequency resolution
fs = sampling frequency
N  = FFT size
```

A larger FFT size gives smaller spacing between frequency bins.

However, if the larger FFT size is created only by zero-padding, it does not add new information to the signal. It only makes the plotted spectrum appear smoother.

---

### Windowing Effect

The DFT analysis compares a rectangular window with a Hanning window.

The rectangular window is equivalent to applying no window, while the Hanning window reduces spectral leakage.

The project plots both spectra to show how the selected window affects the magnitude spectrum.

---

### Window Length Effect

The DFT analysis also compares different window lengths:

```text
256
512
1024
full signal length
```

This shows how using shorter or longer portions of the signal affects the resulting frequency spectrum.

---

## DFT Model

The DFT features are scaled using `RobustScaler`:

```python
scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

`RobustScaler` is used to reduce the effect of outliers.

The classifier used is a Random Forest model:

```python
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=1,
    random_state=42
)
```

The DFT model is evaluated using accuracy, classification report, confusion matrix, and feature importance.

---

# Approach 3: Short-Time Fourier Transform Analysis

## Description

The Short-Time Fourier Transform (STFT) analyzes the audio signal over short overlapping time frames.

This is useful for speech because speech is non-stationary. Its frequency content changes over time.

Instead of analyzing the full audio signal at once, STFT divides the signal into small windows and applies FFT to each window. This produces a time-frequency representation called a spectrogram.

---

## STFT Visualization Tool

The project includes an interactive visualization tool that allows the user to select:

```text
speaker
digit
recording index
window type
N_FFT
window length
hop length
```

The tool plots:

```text
1. Full-signal FFT spectrum
2. Linear STFT spectrogram
```

The FFT spectrum shows the overall frequency content of the audio signal.

The STFT spectrogram shows how the frequency content changes over time.

The visualization helps compare DFT and STFT behavior and shows the effect of changing STFT parameters.

---

## STFT Parameters

The STFT classification code uses parameters such as:

```python
n_fft = 512
win_length = 512
hop_length = 64
window = "hann"
```

The meaning of each parameter is:

```text
n_fft      = number of FFT points used in each frame
win_length = number of samples in each analysis window
hop_length = number of samples between consecutive frames
window     = window function applied to each frame
```

---

## STFT Computation

The STFT is computed using:

```python
S = librosa.stft(
    y,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    window=window
)
```

The magnitude spectrogram is calculated as:

```python
magnitude = np.abs(S)
```

The power spectrogram is calculated as:

```python
power = magnitude ** 2
```

---

## STFT Feature Extraction: Statistical Spectral Features

The first STFT-based classification approach extracts frame-wise spectral features from the STFT magnitude spectrum.

The extracted frame-wise features are:

```text
spectral centroid
spectral bandwidth
spectral rolloff
spectral flatness
RMS energy
dominant frequency
band energy ratio from 0–500 Hz
band energy ratio from 500–1000 Hz
band energy ratio from 1000–2000 Hz
band energy ratio from 2000–4000 Hz
```

Each feature is calculated across multiple STFT frames.

---

## Statistical Summary

Since each audio file may produce a different number of frames, statistical summaries are used to convert each frame-wise feature sequence into a fixed-length vector.

For each feature sequence, the following statistics are calculated:

```text
mean
standard deviation
```

The `safe_stats` function is used to calculate these values while avoiding invalid skewness or kurtosis values when the standard deviation is zero.

---

## STFT Statistical Feature Vector Size

The STFT statistical approach extracts:

```text
10 frame-wise feature groups
6 statistical values for each group
```

Therefore:

```text
10 × 6 = 60 features
```

Each audio file is represented by 60 STFT-based statistical features.

---

## STFT Feature Extraction: Spectrogram Block Features

The second STFT-based classification approach treats the spectrogram as a time-frequency image.

The main steps are:

1. Compute the STFT.
2. Take the magnitude spectrogram.
3. Normalize the spectrogram.
4. Force all spectrograms to have the same number of time frames.
5. Divide the spectrogram into frequency-time blocks.
6. Extract the mean and standard deviation from each block.

---

## Spectrogram Block Parameters

The spectrogram block method uses:

```python
N_FFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 64
WINDOW = "hann"

N_FREQ_BLOCKS = 10
N_TIME_BLOCKS = 8
FIXED_TIME_FRAMES = 40
```

The spectrogram is divided into:

```text
10 frequency blocks
8 time blocks
```

Therefore:

```text
10 × 8 = 80 blocks
```

For each block, two features are extracted:

```text
mean
standard deviation
```

Therefore, the final feature vector size is:

```text
80 × 2 = 160 features
```

Each audio file is represented by 160 spectrogram-block features.

---

## STFT Train-Test Split

The STFT classification approach uses an index-based split:

```text
index 0–4   → test set
index 5–49  → training set
```

Since each digit-speaker combination has 50 recordings, this gives approximately:

```text
10% testing
90% training
```

---

## STFT Models

The STFT statistical feature model uses a Random Forest classifier with parameters such as:

```python
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 15,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42
}
```

The spectrogram block feature model uses:

```python
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42
)
```

Both models are evaluated using accuracy, classification report, confusion matrix, and feature importance.

---

# Train-Test Splitting

Two train-test splitting methods are used in the project.

## Stratified Random Split

The time-domain and DFT approaches use stratified random splitting:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    stratify=y,
    random_state=42
)
```

This gives:

```text
90% training
10% testing
```

The `stratify=y` option ensures that all digit classes are fairly represented in both training and testing sets.

---

## Index-Based Split

The STFT approaches use the recording index from the filename:

```text
index 0–4   → test set
index 5–49  → training set
```

This also gives approximately:

```text
90% training
10% testing
```

---

# Model Evaluation

The models are evaluated using:

```text
accuracy
classification report
confusion matrix
feature importance
```

## Accuracy

Accuracy measures the percentage of correctly classified audio files.

## Classification Report

The classification report includes:

```text
precision
recall
F1-score
support
```

## Confusion Matrix

The confusion matrix shows the relationship between true labels and predicted labels.

Rows represent true classes, and columns represent predicted classes.

A strong model should have most values concentrated along the diagonal.

## Feature Importance

Random Forest provides feature importance values.

These values show which extracted features contributed most to the classification decision.

Feature importance is used in this project to identify which time-domain, DFT, or STFT features were most useful for classification.

---

# Required Libraries

The project requires the following Python libraries:

```text
numpy
librosa
scipy
matplotlib
pandas
scikit-learn
pathlib
os
ipywidgets
```

Install the required libraries using:

```bash
pip install numpy librosa scipy matplotlib pandas scikit-learn ipywidgets
```

---

# How to Run

1. Download or clone the Free Spoken Digit Dataset.
2. Place the `.wav` files inside the dataset folder.
3. Set the dataset path correctly:

```python
DATA_DIR = Path("/content/free-spoken-digit-dataset/recordings")
```

4. Run the desired analysis section:

```text
time-domain analysis
DFT analysis
STFT statistical feature analysis
STFT spectrogram-block feature analysis
```

5. Train the Random Forest classifier.
6. View the accuracy, classification report, confusion matrix, and feature importance.

---

# Project Structure

```text
project-folder/
│
├── recordings/
│   ├── 0_george_0.wav
│   ├── 0_george_1.wav
│   ├── 1_jackson_0.wav
│   └── ...
│
├── time_domain_analysis.py
├── dft_analysis.py
├── stft_analysis.py
└── README.md
```

---

# Comparison of the Three Main Approaches

## Time-Domain Analysis

The time-domain approach extracts features directly from the waveform.

It captures amplitude, energy, silence behavior, waveform shape, and periodicity.

This approach is simple and computationally efficient.

## DFT Analysis

The DFT approach analyzes the full signal in the frequency domain.

It captures the overall frequency content of the audio signal.

However, it does not capture how the frequency content changes over time.

## STFT Analysis

The STFT approach analyzes the signal over short overlapping frames.

It captures both time and frequency information.

This makes it more suitable for speech signals because speech changes over time.

---

# Future Improvements

Possible future improvements include:

```text
1. Extract MFCC-based features and compare them with the current features.
2. Use deeper AI models such as CNN or CRNN on spectrogram inputs.
3. Expand the dataset to include more speakers, accents, noise conditions, and languages such as Arabic.
```

---

# Summary

This project compares three DSP-based feature extraction approaches for spoken digit recognition:

```text
1. Time-Domain Analysis
2. DFT Analysis
3. STFT Analysis
```

Each approach converts the audio recordings into fixed-length feature vectors.

The extracted features are used to train Random Forest classifiers to classify spoken digits from 0 to 9.

The comparison helps show how different audio representations affect classification performance.


# AI Assistance
creating the STFT interactive widget
creating dropdown menus for speaker, digit, index, window type, N_FFT, WIN_LENGTH, and HOP_LENGTH
updating the FFT spectrum and STFT spectrogram plots based on the selected options
plotting the confusion matrix
plotting feature importance
plotting the normalized spectrogram used by the STFT block-feature model
