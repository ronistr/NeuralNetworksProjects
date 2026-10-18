# Neural Networks Projects
 
A collection of four neural-network projects built as coursework at Metropolia. They progress from basic image classifiers to audio classification and transformer models. Each project is a set of Jupyter notebooks (TensorFlow/Keras, plus PyTorch and Hugging Face for the speech models) with the training code, plots, evaluation and written conclusions.
 
## Projects at a glance
 
| # | Project | Task | Data | Models | Best result |
|---|---------|------|------|--------|-------------|
| 1 | [CIFAR-10](projects/CIFAR-10) | Image classification | CIFAR-10 (60k 32×32 images, 10 classes) | Fully connected network vs. CNN | CNN: ~80% test accuracy |
| 2 | [CNN_part2](projects/CNN_part2) | Image classification, transfer learning | Self-collected photos (book, cat, dog, mug, rabbit) | Custom CNN, VGG16 feature extraction, VGG16 fine-tuning | VGG16 feature extraction: 96.1% |
| 3 | [Final_project](projects/Final_project) | Birdsong classification | Bird song recordings (5 species) as mel spectrograms and raw audio | Custom CNN, EfficientNetB0, YAMNet | Custom CNN: 91.9% |
| 4 | [Transformers](projects/Transformers) | Speech recognition and text generation | Finnish audio clips; *Frankenstein* text | Whisper, MarianMT, decoder-only transformer | Medium transformer (11.3M params) |
 
---
 
## 1. CIFAR-10: FCN vs. CNN
 
**Folder:** `projects/CIFAR-10`
 
Two classifiers on the CIFAR-10 benchmark, compared against each other to show why convolutions suit images.
 
| Notebook | Approach | Test accuracy |
|----------|----------|---------------|
| `CIFAR10_FCN.ipynb` | Flattened 3072-value input, Dense(1024) → Dense(256) → softmax, dropout, Adam, early stopping | ~52% |
| `CIFAR10_CNN.ipynb` | Stacked Conv2D blocks (32 → 64 → 128 filters) with batch normalization, max pooling, dropout, and flip/rotation data augmentation | ~80% |
 
**Takeaways**
- The CNN beats the FCN by about 28 percentage points. It keeps the spatial structure of the image, while the FCN needs many more parameters and cannot capture local patterns.
- Vehicles (automobile, ship, truck) are classified best. Animals, especially cats, are the hardest classes for both models.
- Both notebooks include training/validation curves, a confusion matrix, a classification report, and examples of correct and wrong predictions.
---
 
## 2. CNN Part 2: Custom CNN vs. Transfer Learning
 
**Folder:** `projects/CNN_part2`
 
An image classifier trained on a small, self-collected dataset of five classes: **book, cat, dog, mug, rabbit**. Photos are in `data/train` and `data/test`. Three approaches are compared on the same data pipeline.
 
| Notebook | Approach | Test accuracy |
|----------|----------|---------------|
| `preprocessing.ipynb` | Shared data pipeline: loading, 15% validation split, 224×224 resizing, class-distribution checks, augmentation (flip, rotation, zoom), caching and shuffling | n/a |
| Model 1, custom CNN | Three Conv2D + MaxPooling blocks, Dense(128), dropout, trained from scratch | 54.9% |
| `model2_feature_extraction_pretrained.ipynb` | Frozen VGG16 (ImageNet) as feature extractor with a small dense classifier | 96.1% |
| `model3_fine_tuned_pretrained.ipynb.ipynb` | VGG16 with a frozen-base warm-up, then the top layers unfrozen and fine-tuned at a low learning rate (1e-5) | 88.2% |
 
**Takeaways**
- A small dataset is too limited to train a CNN from scratch, and pretrained features help a lot.
- On this dataset, plain feature extraction (Model 2) beat fine-tuning (Model 3).
- The few mistakes of Model 2 were cats and dogs photographed lying down.
---
 
## 3. Final Project: Birdsong Classification
 
**Folder:** `projects/Final_project`
 
Identifies bird species from their songs, using the [bird song data set](https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set) (Xeno-canto recordings). Audio is converted to 128×128 RGB mel spectrograms with `librosa`, so image CNNs can be applied. The five classes used are *melodia*, *polyglottos*, *cardinalis*, *migratorius* and *bewickii*.
 
| Notebook | Approach | Test result |
|----------|----------|-------------|
| `bird_CNN.ipynb` | Preprocessing: audio → mel spectrograms → train/val/test splits (imported by the other notebooks) | n/a |
| `bird_custom_CNN.ipynb` | Four-block custom CNN, dense head, augmentation, early stopping and model checkpointing | **91.9%** accuracy (loss 0.257) |
| `bird_CNN_pretrained_model.ipynb` | EfficientNetB0 (ImageNet), first frozen, then the last ~30 layers unfrozen | Weak: frozen model collapsed to one class (~21–23%); fine-tuning reached only ~30% validation accuracy |
| `bird_yamnet.ipynb` | Google's YAMNet on 16 kHz mono audio, with its 1024-dim embeddings fed to a small dense classifier | ~80% accuracy |
| `bird_CNN_woodpecker_great_tit.ipynb` | Custom CNN on a tiny self-recorded dataset (great spotted woodpecker vs. great tit, 20–30 samples per class) | 50% accuracy; heavy overfitting |
 
**Takeaways**
- A custom CNN trained on spectrograms works best here. ImageNet-pretrained backbones transfer poorly to spectrograms.
- YAMNet is a strong audio-native alternative that needs no spectrogram step.
- The self-recorded set is too small to generalize, which shows how much data quantity and quality matter.
**Data note:** the metadata CSV and the woodpecker/great tit recordings are included. The full bird song `.wav` files are not committed (`projects/Final_project/data/wavfiles` is git-ignored). Download them from Kaggle and place them in `projects/Final_project/data/wavfiles/`. `bird_yamnet.ipynb` downloads them itself with `kagglehub`.
 
---
 
## 4. Transformers: Speech Recognition and Text Generation
 
**Folder:** `projects/Transformers`
 
### 4a. Speech recognition with Whisper (`speech_recognition_whisper_model/`)
 
Transcribes and translates seven Finnish audio clips, grouped by difficulty:
 
- **Easy:** clear, slow learner-oriented speech and a commercial
- **Medium:** children's TV and normal dialogue with some background sound
- **Hard:** fast speech over loud music
It compares **Whisper base** and **Whisper large** (`openai/whisper-*`) for transcription. For Finnish → English translation, it compares both Whisper models with the dedicated **MarianMT** model (`Helsinki-NLP/opus-mt-fi-en`). The notebook also visualizes waveforms and mel spectrograms to explain why noisy clips are harder.
 
**Findings:** Whisper large transcribes much better than base, whose translations were mostly nonsense. MarianMT gave the best translations. Music under fast speech was hard for every model.
 
### 4b. Text generator (`text_generator/`)
 
A small GPT-style, decoder-only transformer built from scratch in Keras. It has token and positional embeddings, causal masking, multi-head attention and feed-forward blocks. It is trained on Mary Shelley's *Frankenstein* (`frankenstein.txt`) with a SentencePiece tokenizer and 64-token sequences.
 
It compares three model sizes: **Small** (5.3M parameters), **Medium** (11.3M) and **Large** (28.9M). Text is generated at temperatures 0.5, 1.0 and 1.5.
 
**Findings:** the medium model was the best balance of coherence, style and stability. The small model produced incoherent text, and the large model took about three hours to train with little benefit.
 
---
 
## Repository structure
 
```
.
├── docker-compose.yml
├── requirements.txt
└── projects/
    ├── CIFAR-10/        # FCN vs. CNN on CIFAR-10
    ├── CNN_part2/       # custom CNN vs. VGG16 transfer learning (+ image data)
    ├── Final_project/   # birdsong classification (CNN, EfficientNetB0, YAMNet)
    └── Transformers/
        ├── speech_recognition_whisper_model/   # Whisper + MarianMT on Finnish audio
        └── text_generator/                     # transformer trained on Frankenstein
```
 
## Getting started
 
### Option 1: Docker (Jupyter)
 
```bash
docker compose up
```
 
Open <http://localhost:8888>. The `projects/` folder is mounted at `/home/jovyan/tasks`. Jupyter starts without a token or password, so use it only on a trusted machine. The compose file mounts `requirements.txt` but does not install it, so run this once in a notebook cell or terminal inside the container:
 
```bash
pip install -r /tmp/requirements.txt
```
 
### Option 2: Local Python
 
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
 
`requirements.txt` covers the core stack: TensorFlow, NumPy, Pillow, Matplotlib, pandas, scikit-learn and seaborn. Some notebooks need extra packages:
 
| Used by | Extra packages |
|---------|----------------|
| CNN_part2, Final_project (notebooks that import other notebooks) | `import_ipynb` |
| Final_project | `librosa`, `soundfile`, `scipy`, `tensorflow_hub`, `kagglehub` |
| Transformers: Whisper | `torch`, `transformers`, `librosa` |
| Transformers: text generator | `sentencepiece` |
 
```bash
pip install import_ipynb librosa soundfile scipy tensorflow-hub kagglehub torch transformers sentencepiece
```
 
Some notebooks import others (for example the models in `CNN_part2` use `preprocessing.ipynb`, and the bird notebooks use `bird_CNN.ipynb`), so run them from inside their own project folder.
 
## Notes
 
- Several notebooks note where AI assistance was used (code checks, plots, language corrections). These notes are in each notebook's "Use of AI" section.
- Accuracy figures above are taken from the notebooks' saved outputs and can vary slightly between runs.
