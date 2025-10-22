# HDC Baselines

This repository provides three end-to-end notebooks implementing Hyperdimensional Computing (HDC) baselines for Image (MNIST), Voice (ISOLET), and Text (European Languages).

## Notebooks

1. **HDC_Image_BinarizedMNIST.ipynb**  
   - Task: Digit classification (MNIST) → 10 Classes
   - Encoding: **1D positional item memory**, **foreground (1) pixels only** 
   - Classifier: Single prototype per class (majority sign), cosine similarity  

2. **HDC_Voice_ISOLET_VoiceHD.ipynb**  
   - Task: ISOLET (letters A–Z spoken) → 26 Classes
   - Encoding: **VoiceHD-style** — feature item memory (iM) ⊙ continuous item memory (CiM) for quantized value; bundle over features; majority sign for class prototypes  
   - Classifier: Single prototype per class, cosine similarity  
   - Notes: Min–max feature quantization (train-only stats), LEVELS=21 (classic).

3. **HDC_Text_EU_Languages.ipynb**  
   - Task: European Languages (21 languages) → 21 Classes
   - Encoding: **TorchHD baseline** — `embeddings.Random` + `functional.ngrams(n=3)` (fixed cyclic permutations), `functional.hard_quantize` per sentence  
   - Classifier: Prototype accumulation in a linear layer (weights), row-wise L2 normalization, dot product (cosine-like)  
   - Notes: Fixed-length ASCII (a–z + space) per original TorchHD example.

## How to run 

- Open each notebook.
- Run all cells top-to-bottom.  
- GPU is optional but recommended for Text (ngrams) and large HDC dimensions.


## References

- **Image / General HDC encoding**  
  Manabat, Alec Xavier, et al. "Performance analysis of hyperdimensional computing for character recognition." 2019 International Symposium on Multimedia and Communication Technology (ISMAC). IEEE, 2019.

- **Voice (ISOLET / VoiceHD)**  
  Imani, Mohsen, et al. "Voicehd: Hyperdimensional computing for efficient speech recognition." 2017 IEEE international conference on rebooting computing (ICRC). IEEE, 2017.

- **Text (European Languages)**  
  Joshi, Aditya, Johan T. Halseth, and Pentti Kanerva. "Language geometry using random indexing." International Symposium on Quantum Interaction. Cham: Springer International Publishing, 2016.

