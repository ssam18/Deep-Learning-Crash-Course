"""Generate a short README.md for every chapter folder.

Chapter folders and notebook filenames use **zero-padded** numbers so that
GitHub (which sorts directory names as strings) lists them in natural order:
Chapter_01, Chapter_02, ..., Chapter_09, Chapter_10, Chapter_11, Chapter_12.
"""
from pathlib import Path

ROOT = Path(__file__).parent
CHAPTERS = [
    (1, 'Neural Network Foundations Using Python and NumPy',
     'Neural_Network_Foundations',
     'Perceptron, activations, forward/backprop, optimisers, from-scratch network on the moons dataset and a Keras equivalent.'),
    (2, 'Setting Up a Modern Deep Learning Development Environment',
     'Environment_Setup',
     'Verification scripts for Python, TensorFlow, PyTorch and CUDA; mixed-precision smoke test; reproducibility helpers; TensorBoard demo.'),
    (3, 'Your First Neural Network on the MNIST Dataset',
     'First_Neural_Network_MNIST',
     'MNIST preprocessing, Keras + PyTorch training loops, confusion matrix, misclassified-sample inspection, model save/load.'),
    (4, 'Convolutional Neural Networks',
     'Convolutional_Neural_Networks',
     'Conv from scratch, baseline + residual + BatchNorm CNNs on CIFAR-10, augmentation, Grad-CAM and MobileNetV2 transfer learning preview.'),
    (5, 'Recurrent Neural Networks and Long Short-Term Memory',
     'RNNs_and_LSTMs',
     'Vanilla RNN, vanishing-gradient demo, text preprocessing, Bi-LSTM on IMDb, GRU comparison in PyTorch, temperature sampling.'),
    (6, 'Transformers and Attention Models',
     'Transformers_and_Attention',
     'Scaled dot-product and multi-head attention in NumPy, positional encoding, PyTorch encoder block, tiny end-to-end Transformer, BERT/GPT tokenizers, top-p sampling.'),
    (7, 'Generative Models',
     'Generative_Models',
     'Toy 1-D GAN, DCGAN on MNIST, VAE, 2-D denoising diffusion, FID-style metric.'),
    (8, 'Transfer Learning',
     'Transfer_Learning',
     'MobileNetV2 feature extraction + two-phase fine-tuning on CIFAR-10, discriminative LRs, MixUp, from-scratch LoRA, MMD for domain adaptation.'),
    (9, 'Model Optimization and Compression',
     'Model_Optimization',
     'Magnitude pruning, dynamic INT8 quantization, quantization-aware training, knowledge distillation; side-by-side accuracy / size / latency comparison.'),
    (10, 'Reinforcement Learning',
     'Reinforcement_Learning',
     'Tabular Q-learning on FrozenLake, DQN on CartPole, REINFORCE with baseline, PPO with GAE from scratch.'),
    (11, 'MLOps and Deployment',
     'MLOps_and_Deployment',
     'Model packaging (PyTorch / TorchScript / ONNX), model card, FastAPI inference server, multi-stage Dockerfile, MLflow tracking, GitHub Actions CI, deployment-strategy visualisations.'),
    (12, 'Monitoring and Scaling',
     'Monitoring_and_Scaling',
     'KS / PSI / CUSUM drift detection, fairness audit with TPR/FPR gap, tamper-evident audit record, retraining-trigger logic, scaling cost calculator, dashboard sketch.'),
]

REPO = 'ssam18/Deep-Learning-Crash-Course'

for num, title, topic, summary in CHAPTERS:
    folder_name = f'Chapter_{num:02d}'
    nb = f'Chapter_{num:02d}_{topic}.ipynb'
    rel = f'{folder_name}/{nb}'
    colab = f'https://colab.research.google.com/github/{REPO}/blob/main/{rel}'
    binder = f'https://mybinder.org/v2/gh/{REPO}/main?filepath={rel}'
    folder = ROOT / folder_name
    folder.mkdir(exist_ok=True)
    (folder / 'images').mkdir(exist_ok=True)
    readme = folder / 'README.md'
    readme.write_text(f"""# Chapter {num} — {title}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab}) [![Open in Binder](https://mybinder.org/badge_logo.svg)]({binder})

**Notebook**: [`{nb}`]({nb})

{summary}

## Running

Click a badge above to launch this notebook in your browser (Colab or Binder), or run it locally:

```bash
pip install -r ../requirements.txt
jupyter lab {nb}
```

Run every cell top-to-bottom. Figures are written into [`images/`](images/) as PNG files at 150 dpi.

When running in **Colab**, uncomment the `!pip install` line in the second cell of the notebook to install the chapter's dependencies. **Binder** picks them up automatically from `requirements.txt`.

## Exercises

The final section of the notebook contains worked solutions for every multiple-choice question and short-answer / coding exercise from the chapter.
""")
    print('Wrote', readme.relative_to(ROOT))
