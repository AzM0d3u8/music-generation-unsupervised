# Music Generation - Unsupervised Learning

An end-to-end unsupervised pipeline for multi-genre MIDI music generation using deep learning models. This project implements three complementary neural architectures for symbolic music synthesis trained on the MAESTRO dataset.

## Overview

This project demonstrates a comprehensive approach to unsupervised music generation, progressing from deterministic reconstruction to probabilistic generation and long-horizon autoregressive modeling.

### Three-Model Architecture

1. **Task 1: LSTM Autoencoder** - Baseline reconstruction model
   - Deterministic encoder-decoder architecture
   - Focuses on compression and faithfully preserving local musical structure
   - Serves as a performance baseline for comparison

2. **Task 2: Conditional VAE** - Style-controlled generative model
   - Variational autoencoder with style conditioning
   - Enables multi-genre music generation with explicit style control
   - Supports interpolation between stylistic regions
   - Uses one-hot style encoding (7 musical periods: Baroque, Classical, Romantic, Impressionist, Modern, Contemporary, Other Classical)

3. **Task 3: Transformer Decoder** - Long-horizon sequence generation
   - Autoregressive Transformer with causal self-attention
   - Generates extended musical sequences with coherent long-term structure
   - Evaluated using perplexity metric
   - Produces novel multi-step MIDI continuations

## Dataset

- **MAESTRO v3.0.0**: Large-scale dataset of classical piano performances
- 200+ hours of carefully matched audio-MIDI pairs
- Spans multiple composers and musical periods
- Preprocessing: 88-key piano-roll representation at 16 Hz sampling rate

## Technical Details

### Data Representation
- **Piano-roll format**: Binary (128 × T) matrices indicating note activity
- **Pitch range**: 88 piano keys (MIDI 21–108)
- **Temporal resolution**: 16 Hz (62.5 ms per frame)
- **Context window**: 128 timesteps (~8 seconds)

### Model Specifications

| Model | Architecture | Loss | Key Parameters |
|-------|------|------|---|
| LSTM-AE | 2×LSTM encoder-decoder | MSE | latent_dim=64, lstm_units=256 |
| VAE | Conditional LSTM encoder + decoder | MSE + β-KL | latent_dim=64, beta=0.002 |
| Transformer | Multi-head attention decoder | Sparse categorical cross-entropy | embed_dim=128, num_heads=4 |

### Evaluation Metrics

- **Reconstruction models** (LSTM-AE, VAE): Mean Squared Error (MSE)
- **Generative model** (Transformer): Perplexity on held-out test set
- **Qualitative**: Generated MIDI files for listening evaluation


## Output Artifacts

- **Task 1**: Reconstructed MIDI files from test set samples
- **Task 2**: 8 MIDI files, one per musical style, sampled from the learned latent space
- **Task 3**: 10 long-sequence MIDI files (512+ tokens each) generated autoregressively

## Requirements

```
tensorflow >= 2.0
pretty_midi
numpy
pandas
matplotlib
```

## Usage

1. Update `DATASET_PATH` in the notebook to point to your MAESTRO dataset
2. Run all cells sequentially (Task 1 → Task 2 → Task 3)
3. Generated MIDI files will be saved to `generated_midi_*` directories
4. Play outputs with any standard MIDI player

## Key Findings

- **LSTM Autoencoder**: Effective for deterministic compression; reconstructions are faithful but conservative
- **Conditional VAE**: Successfully learns style representations; quality improves with careful β scheduling
- **Transformer Decoder**: Superior for extended sequences; self-attention captures long-range musical dependencies
- **Trade-offs**: Reconstruction vs. generation, determinism vs. diversity, local accuracy vs. global coherence

## Future Work

- Preserve velocity and duration information more faithfully
- Implement overlapping windows to increase training coverage
- Adopt richer polyphonic tokenization schemes (beyond monophonic simplification)
- Incorporate additional music-theoretic metrics (note onset, pitch distributions)
- Conduct structured human listening evaluations

## Contributions

**EDA**
- Nafiur Rahman Afnan (ID: 24141074)

**Task 1: LSTM Autoencoder**
- Nafiur Rahman Afnan (ID: 24141074)

**Baseline Models**
- Nafiur Rahman Afnan (ID: 24141074)

**Task 2: Conditional VAE**
- Adnan Safin (ID: 2299478)

**Task 3: Transformer Decoder**
- Adnan Safin (ID: 2299478)

**Model Evaluation**
- Nafiur Rahman Afnan (ID: 24141074)

**Writing Latex Report**
- Nafiur Rahman Afnan (ID: 24141074)

## License

This project is provided for educational and research purposes.

## Acknowledgments

- MAESTRO dataset: https://magenta.tensorflow.org/maestro
- TensorFlow and Keras communities
- Research on unsupervised music generation
