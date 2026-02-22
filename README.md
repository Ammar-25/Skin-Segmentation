# Skin Segmentation
## 📝Overview
This project implements a Naive Bayes Classifier from scratch in C# to perform image segmentation, specifically identifying and isolating human skin pixels in color images. Instead of relying on external Machine Learning libraries, the algorithm calculates probabilistic models directly from training images and their corresponding ground-truth binary masks.
## ⚙️How It Works
The algorithm operates by analyzing the color distribution of pixels in the HSV (Hue, Saturation, Value) color space. It isolates the Hue channel to make the model robust against lighting and shadow variations.
1. **Training Phase:** The model processes a dataset of images and binary masks. It calculates the **Prior** probabilities (the overall likelihood of any pixel being skin vs. non-skin) and the **Likelihoods** $`(P(Hue \mid Skin),  P(Hue \mid NonSkin))`$.
2. **Prediction Phase:** For a new image, the model calculates the unnormalized Posterior probability for each pixel using Bayes' Theorem. If the probability of being skin outweighs the probability of being non-skin (and surpasses a defined threshold), the pixel is classified as skin.
3. **Evaluation:** The system evaluates its own accuracy by comparing predictions against ground-truth test masks, generating a Confusion Matrix to calculate **Precision** and **Recall**.

## ⚡High-Performance Multi-Threading
Image processing is computationally expensive. To ensure rapid execution times, this project heavily utilizes **Nested Parallelism**. 
* Using C#'s `Parallel.For`, the system processes multiple images simultaneously.
* Within each image, threads are further sub-divided row-by-row.
* Custom thread-safe `lock` hierarchies and local caching arrays are used to prevent race conditions and bottlenecking, allowing the algorithm to fully utilize 100% of the available CPU cores.
  
## 🛠️Tech Stack
* **Language:** C# (.NET)
* **UI:** Windows Forms (WinForms) / Charting
* **Core Concepts:** Probabilistic Machine Learning (Naive Bayes), Computer Vision, Advanced Multithreading / Concurrency.
