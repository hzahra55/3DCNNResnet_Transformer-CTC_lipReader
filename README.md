# Lipreading with Transformer-CTC and Character-Level Language Model
This project implements a Transformer-based CTC (TM-CTC) model for lipreading using pre-extracted spatio-temporal visual features from video frames. It leverages beam search decoding with a character-level language model (LM) to improve transcription accuracy on the LRS2 dataset.

# Project Overview
- Input: 512-dimensional visual feature vectors extracted from the lip region of video frames.

- Architecture: Transformer encoder with CTC (Connectionist Temporal Classification) loss.

- Decoding: Beam search enhanced with a character-level language model.

- Dataset: LRS2 (processed with custom scripts to generate feature files).

- Goal: Predict accurate text transcriptions from silent video input.
