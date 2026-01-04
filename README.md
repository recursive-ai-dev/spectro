# SSGS — Spectral-State Guided Synthesis
### © 2025 Damien Davison & Michael Maillet & Sacha Davison 
### Recursive AI Devs  
Licensed under the Apache License 2.0
psychcoherence@gmail.com OR therealmichaelmaillet@gmail.com

SSGS (Spectral-State Guided Synthesis) is a hybrid generative audio model that
combines **linear predictive coding (LPC)**, **hidden Markov models (HMMs)**,
**spectral clustering**, **global graph search**, and a **physically-inspired
Karplus–Strong excitation engine** into a single synthesis framework.

Unlike classical systems that commit to a single paradigm (neural, statistical,
or physical), SSGS deliberately **smashes algorithms together** and extracts the
useful computational primitives from each.  
The result is a model that:

- learns spectral structure from real signals,  
- organizes that structure into Markov states,  
- decodes the optimal state trajectory over time via A* search,  
- generates excitation signals using string-model dynamics, and  
- reconstructs audio by filtering excitation through LPC envelopes.

SSGS is not meant to imitate any existing model family.  
It is a new hybrid class: detailed like DSP, global like HMMs, and expressive like
physical models.

---

## Key Idea: “Algorithm Deconstruction”
SSGS is built using the same philosophy behind the Symbo project:

> **Break apart many algorithms, strip them to their mathematical essence,
> recombine only the primitives that are actually useful.**

Instead of inheriting algorithms whole (e.g., a standard HMM, standard LPC vocoder,
standard Karplus–Strong), SSGS takes:

- LPC → spectral envelope representation  
- HMM → temporal clustering + statistical state transitions  
- EM → unsupervised parameter refinement  
- A* search → globally optimal state sequence  
- Karplus–Strong → natural resonance and excitation noise  
- Heuristics → spectral smoothness constraints  
- Graph theory → prune invalid or degenerate state structures  

SSGS then recomposes these pieces into a **single generative pipeline that didn’t
exist before**.

This approach is extremely flexible:
swap the envelope model, swap the excitation, modify the search heuristic — the
system keeps working.

---

## Pipeline Overview

### **1. LPC Analysis**
The training signal is segmented and analyzed with LPC to extract:

- LPC coefficients  
- excitation residual  
- power envelope

These become the feature vectors for clustering.

---

### **2. Spectral Clustering via HMM Initialization**
Frames are embedded into a feature space (typically derived from LPC spectra).
States are initialized using **k-means**, then upgraded into a full **HMM** with:

- initial state distribution  
- transition matrix  
- mean vectors and covariance matrices for each state

---

### **3. Expectation-Maximization (EM) Refinement**
A custom EM implementation updates:

- state responsibilities (γ)  
- pairwise transitions (ξ)  
- transition probabilities  
- Gaussian parameters (means, covariances)

Result: the HMM becomes a structured map of repeating spectral “modes.”

---

### **4. Graph Constraint Pruning**
To prevent degenerate solutions, SSGS analyzes the transition graph:

- identifies strongly connected components (SCCs)  
- removes invalid or isolated states  
- ensures state sequences remain musically plausible

---

### **5. Global State Decoding (A* Search)**
Instead of using Viterbi (which is greedy and purely local), SSGS uses **A***:

- cost function = negative log-likelihood + spectral smoothness heuristic  
- ensures global consistency in the decoded state path  
- supports long-range structure better than classical decoding

---

### **6. Physically Inspired Excitation**
The decoded state sequence modulates a **Karplus–Strong** string model,
producing a dynamic excitation that is:

- rich in overtones  
- noisy where appropriate  
- resonant and evolving

---

### **7. LPC Synthesis**
Finally, excitation is passed through the LPC filters of each decoded state:

- reconstructs spectral envelopes  
- restores formants and resonances  
- yields new but structurally coherent audio

This is how SSGS generates *novel signals* even from short or simple training data.

---

## Model Export, Compression, and Checkpointing

After training you can persist the learned parameters in a compact archive:

```python
ssgs.export_model("models/ssgs_baseline", use_compression=True, pack_covariances=True)
```

- `.npz` output uses ZIP compression by default when `use_compression=True`.
- Covariance matrices are serialized as lower-triangular slices to halve their
   stored size.
- Set `include_training_artifacts=True` if you also need cached LPC and residual
   buffers for later analysis.
- Reload the model with `SpectralStateGuidedSynthesis.load_model(path)`.

For long-running training, SSGS also supports `.safetensors` checkpoints:

```python
ssgs.save_checkpoint("models/checkpoint_epoch_5.safetensors")
ssgs = SpectralStateGuidedSynthesis.load_checkpoint("models/checkpoint_epoch_5.safetensors")
```

Checkpoints can optionally persist adaptive statistics so continuous learning can resume
without reprocessing previous audio.

---

## Generative Capability Levels

SSGS now offers enhanced generative capabilities through configurable state counts:

- **Standard (16 states)**: Original baseline configuration
- **Enhanced (34 states)**: 112% increase - recommended for most applications
- **Maximum (42 states)**: 163% increase - highest generative capacity

```python
# Enhanced generative capabilities (default)
ssgs = SpectralStateGuidedSynthesis(n_states=34)

# Maximum generative capabilities
ssgs = SpectralStateGuidedSynthesis(n_states=42)

# Original baseline
ssgs = SpectralStateGuidedSynthesis(n_states=16)
```

More states = more spectral patterns the model can learn and generate = richer, more diverse output.

---

## Example Output

SSGS produces time-domain and spectral visualizations like this:

![SSGS demo output](ssgs_demo.png)

For a deeper comparison diagnostic, see `ssgs_analysis.png`.

The generated signal is not a copy —  
it is a **new trajectory** through learned spectral states.

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the demo training + generation script:

```bash
python test_ssgs.py
```

Run the enhanced capabilities demo (defaults to 34 states):

```bash
python demo_enhanced_capabilities.py
```

Train on folders of audio with checkpointing:

```bash
python train_on_folder.py --folders training_001 training_002
```

## Adaptive Learning

SSGS can adapt to new audio using persistent statistics:

```python
ssgs.adapt_to_audio(new_audio, sample_rate=16000, adaptation_rate=0.5, memory_blend=0.3)
```

Adaptation can be exported through `.npz` or `.safetensors` artifacts and reloaded later.

## Feature Indexing & Retrieval

Use built-in nearest-neighbor search over frame features for exploration and analysis:

```python
index = ssgs.build_feature_index(feature_space="lpc", normalize=True)
result = ssgs.query_similar_frames(frame_idx=42, k=8, feature_space="lpc")
vector_result = ssgs.search_by_feature_vector(
    vector=index.tree.data[0],
    k=5,
    feature_space="lpc",
)
```

The returned `FrameQueryResult` includes indices, optional distances, and frame metadata.

## Testing

Run individual checks as standalone scripts:

```bash
python test_ssgs.py
python test_checkpoints.py
python test_adaptive_persistence.py
python test_data_indexing.py
python test_gain_control.py
python test_generative_capabilities.py
python test_utils.py
```


### Requirements

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
soundfile>=0.10.0
PyWavelets>=1.1.0
safetensors>=0.4.0
```

### Project Structure

ssgs.py                        # Full model implementation \
train_on_folder.py             # Folder-based training with checkpointing \
demo_enhanced_capabilities.py  # State-count showcase \
example_checkpoints.py         # Checkpoint usage example \
example_fidelity.py            # Fidelity-focused generation example \
test_ssgs.py                   # Training + generation demo \
test_checkpoints.py            # Checkpoint save/load tests \
test_adaptive_persistence.py   # Adaptive learning tests \
test_data_indexing.py          # Feature index tests \
test_gain_control.py           # Gain control tests \
test_generative_capabilities.py # State-count tests \
requirements.txt               # Dependencies \
ssgs_demo.png                  # Example output \
ssgs_analysis.png              # Diagnostic output \
README.md                      # This document \


### Authors

Damien Davison & Michael Maillet & Sacha Davison
Recursive AI Devs

We build hybrid-symbolic neural, statistical, and physical AI systems by
algorithm decomposition — extracting the useful primitives and recombining them
into new model classes.

If you use SSGS in research or production, please cite the authors.


### License — Apache 2.0

This project is licensed under the Apache License, Version 2.0.

Copyright 2025
Damien Davison & Michael Maillet
Recursive AI Devs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy at:

   http://www.apache.org/licenses/LICENSE-2.0

## NOTICE

This product includes original work by:  
**Damien Davison & Michael Maillet & Sacha Davison (Recursive AI Devs)**  
Additional details can be found in the project's LICENSE and source headers.
