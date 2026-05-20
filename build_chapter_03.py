"""Build Chapter 3 notebook: First Neural Network on the MNIST Dataset."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _nbutils import build_notebook, md, code

CHAPTER = 3
OUT = pathlib.Path(__file__).parent / f"Chapter_{CHAPTER}" / f"Chapter_{CHAPTER}_First_Neural_Network_MNIST.ipynb"

cells = []
cells.append(md(
    "# Chapter 3: Your First Neural Network on the MNIST Dataset\n"
    "\n"
    "*Deep Learning Crash Course - BPB Publications*\n"
    "\n"
    "We use the MNIST dataset of handwritten digits to walk through the full lifecycle of a "
    "supervised learning project: load, explore, preprocess, split, build, train, evaluate and "
    "save the model. We do this twice — first in TensorFlow / Keras and then in PyTorch — so "
    "you can see how the same pipeline maps onto each framework.\n"
))

cells.append(md("## 1. Setup"))
cells.append(code(
    "import os\n"
    "# CPU-mode guard: hide the GPU from TensorFlow before it is imported.\n"
    "# Many machines have an NVIDIA driver but a missing CUDA toolkit\n"
    "# (no `ptxas`), in which case TF's XLA JIT crashes the kernel. Remove\n"
    "# these two lines once you have a complete CUDA install.\n"
    "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')\n"
    "os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')\n"
    "import random\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "IMG_DIR = Path('images'); IMG_DIR.mkdir(exist_ok=True)\n"
    "\n"
    "def set_seed(seed=42):\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    random.seed(seed); np.random.seed(seed)\n"
    "    try:\n"
    "        import tensorflow as tf; tf.random.set_seed(seed)\n"
    "    except ModuleNotFoundError: pass\n"
    "    try:\n"
    "        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)\n"
    "    except ModuleNotFoundError: pass\n"
    "set_seed(42)\n"
))

cells.append(md("## 2. Load and explore MNIST\n"
                "\n"
                "Keras downloads MNIST in NumPy form. If you only have PyTorch, the `torchvision.datasets` "
                "version is shown later."))
cells.append(code(
    "import tensorflow as tf\n"
    "(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()\n"
    "print(f'Training images: {X_train_full.shape}, labels: {y_train_full.shape}')\n"
    "print(f'Test images:     {X_test.shape}, labels: {y_test.shape}')\n"
    "print(f'Pixel range: [{X_train_full.min()}, {X_train_full.max()}], dtype {X_train_full.dtype}')\n"
    "\n"
    "# Class balance\n"
    "labels, counts = np.unique(y_train_full, return_counts=True)\n"
    "print('Per-class counts:', dict(zip(labels.tolist(), counts.tolist())))\n"
))
cells.append(code(
    "fig, axes = plt.subplots(2, 5, figsize=(9, 4))\n"
    "for ax, img, lab in zip(axes.flat, X_train_full[:10], y_train_full[:10]):\n"
    "    ax.imshow(img, cmap='gray'); ax.set_title(str(lab)); ax.axis('off')\n"
    "fig.suptitle('MNIST sample digits')\n"
    "fig.tight_layout(); fig.savefig(IMG_DIR / '01_mnist_samples.png', dpi=150); plt.show()\n"
))

cells.append(md("## 3. Preprocess: normalise, flatten, encode\n"
                "\n"
                "Scaling pixels to `[0, 1]`, flattening 28x28 to a 784-vector, and one-hot encoding the "
                "labels for use with categorical cross-entropy."))
cells.append(code(
    "X_train_full = X_train_full.astype('float32') / 255.0\n"
    "X_test = X_test.astype('float32') / 255.0\n"
    "X_train_full = X_train_full.reshape(-1, 28 * 28)\n"
    "X_test = X_test.reshape(-1, 28 * 28)\n"
    "y_train_full_oh = tf.keras.utils.to_categorical(y_train_full, 10)\n"
    "y_test_oh = tf.keras.utils.to_categorical(y_test, 10)\n"
    "print('Flattened train shape:', X_train_full.shape, 'one-hot label shape:', y_train_full_oh.shape)\n"
))

cells.append(md("## 4. Train / validation / test split\n"
                "\n"
                "Keras gives us 60k training + 10k test. We carve a further 6k validation set out of "
                "the training set."))
cells.append(code(
    "VAL_SIZE = 6000\n"
    "idx = np.random.permutation(len(X_train_full))\n"
    "val_idx, train_idx = idx[:VAL_SIZE], idx[VAL_SIZE:]\n"
    "X_train, y_train_oh = X_train_full[train_idx], y_train_full_oh[train_idx]\n"
    "X_val, y_val_oh = X_train_full[val_idx], y_train_full_oh[val_idx]\n"
    "y_train = y_train_full[train_idx]; y_val = y_train_full[val_idx]\n"
    "print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')\n"
))

cells.append(md("## 5. Build and train a Keras model"))
cells.append(code(
    "model = tf.keras.Sequential([\n"
    "    tf.keras.layers.Input(shape=(784,)),\n"
    "    tf.keras.layers.Dense(128, activation='relu'),\n"
    "    tf.keras.layers.Dropout(0.2),\n"
    "    tf.keras.layers.Dense(64, activation='relu'),\n"
    "    tf.keras.layers.Dense(10, activation='softmax'),\n"
    "])\n"
    "model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),\n"
    "              loss='categorical_crossentropy', metrics=['accuracy'])\n"
    "model.summary()\n"
))
cells.append(code(
    "callbacks = [\n"
    "    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),\n"
    "    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-5),\n"
    "]\n"
    "history = model.fit(X_train, y_train_oh,\n"
    "                    validation_data=(X_val, y_val_oh),\n"
    "                    epochs=15, batch_size=128, callbacks=callbacks, verbose=2)\n"
))
cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
    "axes[0].plot(history.history['loss'], label='train')\n"
    "axes[0].plot(history.history['val_loss'], label='val')\n"
    "axes[0].set_title('Loss'); axes[0].set_xlabel('epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)\n"
    "axes[1].plot(history.history['accuracy'], label='train')\n"
    "axes[1].plot(history.history['val_accuracy'], label='val')\n"
    "axes[1].set_title('Accuracy'); axes[1].set_xlabel('epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)\n"
    "fig.tight_layout(); fig.savefig(IMG_DIR / '02_keras_curves.png', dpi=150); plt.show()\n"
))

cells.append(md("## 6. Evaluate: accuracy, confusion matrix, misclassified examples"))
cells.append(code(
    "test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)\n"
    "print(f'Test accuracy: {test_acc:.4f}')\n"
    "y_pred = model.predict(X_test, verbose=0).argmax(axis=1)\n"
    "\n"
    "from sklearn.metrics import classification_report, confusion_matrix\n"
    "print(classification_report(y_test, y_pred, digits=4))\n"
    "cm = confusion_matrix(y_test, y_pred)\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "im = ax.imshow(cm, cmap='Blues')\n"
    "for i in range(10):\n"
    "    for j in range(10):\n"
    "        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='black' if cm[i, j] < cm.max()/2 else 'white')\n"
    "ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title('Confusion matrix')\n"
    "fig.colorbar(im, ax=ax)\n"
    "fig.tight_layout(); fig.savefig(IMG_DIR / '03_confusion_matrix.png', dpi=150); plt.show()\n"
))
cells.append(code(
    "wrong = np.where(y_pred != y_test)[0]\n"
    "print('Total misclassified:', len(wrong))\n"
    "sample = wrong[:10]\n"
    "fig, axes = plt.subplots(2, 5, figsize=(10, 4))\n"
    "for ax, i in zip(axes.flat, sample):\n"
    "    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')\n"
    "    ax.set_title(f'pred {y_pred[i]} / true {y_test[i]}'); ax.axis('off')\n"
    "fig.suptitle('Selected misclassifications')\n"
    "fig.tight_layout(); fig.savefig(IMG_DIR / '04_misclassified.png', dpi=150); plt.show()\n"
))

cells.append(md("## 7. PyTorch equivalent\n"
                "\n"
                "Same architecture, written from scratch in PyTorch — note we output raw logits and "
                "let `CrossEntropyLoss` apply softmax internally for numerical stability."))
cells.append(code(
    "try:\n"
    "    import torch\n"
    "    import torch.nn as nn\n"
    "    from torch.utils.data import DataLoader, TensorDataset\n"
    "    device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "    set_seed(42)\n"
    "\n"
    "    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).long())\n"
    "    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val).long())\n"
    "    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test).long())\n"
    "    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)\n"
    "    val_loader   = DataLoader(val_ds, batch_size=256)\n"
    "    test_loader  = DataLoader(test_ds, batch_size=256)\n"
    "\n"
    "    class TorchMLP(nn.Module):\n"
    "        def __init__(self):\n"
    "            super().__init__()\n"
    "            self.fc = nn.Sequential(\n"
    "                nn.Linear(784, 128), nn.ReLU(), nn.Dropout(0.2),\n"
    "                nn.Linear(128, 64), nn.ReLU(),\n"
    "                nn.Linear(64, 10))\n"
    "        def forward(self, x): return self.fc(x)\n"
    "\n"
    "    net = TorchMLP().to(device)\n"
    "    opt = torch.optim.Adam(net.parameters(), lr=1e-3)\n"
    "    loss_fn = nn.CrossEntropyLoss()\n"
    "\n"
    "    history_t = {'train': [], 'val': []}\n"
    "    for epoch in range(8):\n"
    "        net.train()\n"
    "        running = 0.0\n"
    "        for xb, yb in train_loader:\n"
    "            xb, yb = xb.to(device), yb.to(device)\n"
    "            opt.zero_grad()\n"
    "            loss = loss_fn(net(xb), yb)\n"
    "            loss.backward(); opt.step()\n"
    "            running += loss.item() * xb.size(0)\n"
    "        running /= len(train_loader.dataset)\n"
    "        net.eval()\n"
    "        v_correct = 0\n"
    "        with torch.no_grad():\n"
    "            for xb, yb in val_loader:\n"
    "                xb, yb = xb.to(device), yb.to(device)\n"
    "                v_correct += (net(xb).argmax(1) == yb).sum().item()\n"
    "        v_acc = v_correct / len(val_loader.dataset)\n"
    "        history_t['train'].append(running); history_t['val'].append(v_acc)\n"
    "        print(f'Epoch {epoch+1:2d} | train loss {running:.4f} | val acc {v_acc:.4f}')\n"
    "    # final test\n"
    "    net.eval(); correct = 0\n"
    "    with torch.no_grad():\n"
    "        for xb, yb in test_loader:\n"
    "            xb, yb = xb.to(device), yb.to(device)\n"
    "            correct += (net(xb).argmax(1) == yb).sum().item()\n"
    "    print(f'Test accuracy: {correct / len(test_loader.dataset):.4f}')\n"
    "except ModuleNotFoundError:\n"
    "    print('PyTorch not installed - skipping PyTorch section.')\n"
))

cells.append(md("## 8. Save and reload the model"))
cells.append(code(
    "model.save('mnist_keras.keras')\n"
    "loaded = tf.keras.models.load_model('mnist_keras.keras')\n"
    "print('Reloaded model test accuracy:', loaded.evaluate(X_test, y_test_oh, verbose=0)[1])\n"
    "\n"
    "try:\n"
    "    torch.save(net.state_dict(), 'mnist_torch.pt')\n"
    "    loaded_net = TorchMLP().to(device)\n"
    "    loaded_net.load_state_dict(torch.load('mnist_torch.pt'))\n"
    "    loaded_net.eval()\n"
    "    print('Reloaded PyTorch state_dict OK.')\n"
    "except NameError:\n"
    "    pass\n"
))

cells.append(md("## 9. Exercise solutions\n"
                "\n"
                "### 9.1 MCQ answer key\n"
                "\n"
                "| Q | Answer | Why |\n"
                "|---|--------|-----|\n"
                "| 1 | (c) 70,000 | 60k train + 10k test. |\n"
                "| 2 | (c) Stable gradients and faster convergence | Scaling to [0,1] keeps activations and gradients well-behaved. |\n"
                "| 3 | (b) Validation set | Test set is reserved for final reporting. |\n"
                "| 4 | (b) 100,480 | $784 \\times 128 + 128$ bias terms. |\n"
                "| 5 | (c) Softmax | Produces a probability distribution summing to 1. |\n"
                "| 6 | (b) Overfitting | Train loss falling while val loss rises is the textbook signature. |\n"
                "| 7 | (b) `CrossEntropyLoss` applies softmax internally | More numerically stable than computing softmax then log. |\n"
                "| 8 | (b) Correctly classified samples per class | Off-diagonal entries are errors. |\n"
                "| 9 | (b) `net.eval()` | Disables dropout and switches BatchNorm to running stats. |\n"
                "| 10 | (c) Loss explodes or becomes NaN | Adam's effective LR is much larger than 0.1 implies. |\n"
                "| 11 | (c) `torch.save(model.state_dict())` | Recommended portable serialisation. |\n"
                "| 12 | (b) Number of epochs to wait before reducing | `patience` is the patience window. |\n"))

cells.append(md("### 9.2 Why normalisation accelerates training\n"
                "Without normalisation, the input range `[0, 255]` produces large pre-activations: "
                "$Wx$ with random $W$ and large $x$ saturates sigmoid/tanh and pushes ReLU into "
                "very large regions. The first-layer gradient $\\partial L / \\partial W \\propto x$ "
                "scales with the input, so for un-normalised data the optimal learning rate is "
                "smaller by a factor of $\\sim 255$. Scaling to `[0, 1]` keeps activations and "
                "gradients close to unit scale and lets a standard learning rate (e.g. $10^{-3}$ "
                "with Adam) converge in a handful of epochs."))
cells.append(code(
    "# Confirm with a small experiment: train the same network on raw vs normalised inputs\n"
    "def train_quick(X_tr, X_va, y_tr_oh, y_va_oh, epochs=3, lr=1e-3, label=''):\n"
    "    set_seed(0)\n"
    "    m = tf.keras.Sequential([\n"
    "        tf.keras.layers.Input(shape=(784,)),\n"
    "        tf.keras.layers.Dense(128, activation='relu'),\n"
    "        tf.keras.layers.Dense(10, activation='softmax'),\n"
    "    ])\n"
    "    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])\n"
    "    h = m.fit(X_tr, y_tr_oh, validation_data=(X_va, y_va_oh), epochs=epochs, batch_size=128, verbose=0)\n"
    "    print(f'{label:20s} val acc after {epochs} epochs: {h.history[\"val_accuracy\"][-1]:.4f}')\n"
    "    return h\n"
    "\n"
    "h_raw  = train_quick((X_train * 255).astype('float32'), (X_val * 255).astype('float32'),\n"
    "                     y_train_oh, y_val_oh, label='raw [0,255]')\n"
    "h_norm = train_quick(X_train, X_val, y_train_oh, y_val_oh, label='normalised [0,1]')\n"
))

cells.append(md("### 9.3 99.2% train vs 95.8% test - over-fitting playbook\n"
                "The 3.4 pp gap is classic over-fitting. Four interventions in order of impact:\n"
                "1. **More aggressive regularisation** - increase dropout to 0.4-0.5 in the wide layer and add weight decay (`tf.keras.regularizers.l2(1e-4)`).\n"
                "2. **Data augmentation** - random shifts of +/-2 pixels, +/-10 degree rotations and small zooms turn 60k MNIST examples into many millions.\n"
                "3. **Early stopping** - restore the weights from the epoch with lowest val loss.\n"
                "4. **Smaller model** - if the train accuracy keeps climbing toward 100 % while val stalls, the model has more capacity than the data supports.\n"))

cells.append(md("### 9.4 One iteration of the PyTorch training loop\n"
                "- `opt.zero_grad()` - PyTorch accumulates gradients into `param.grad` by default, "
                "so each step must reset them; otherwise the next backward pass adds to the previous gradient.\n"
                "- `loss.backward()` - walks the computation graph backwards from `loss` and "
                "computes $\\partial L / \\partial \\theta$ for every leaf tensor that has "
                "`requires_grad=True`. Writes the result into `param.grad`.\n"
                "- `opt.step()` - applies the optimiser's update rule (e.g. Adam's bias-corrected "
                "moments) using the gradients stored on each parameter, and writes the new "
                "values back in-place."))

cells.append(md("### 9.5 Controlled batch-size experiment\n"
                "Hold these constant: optimiser, LR schedule, weights initialisation, data split, number of epochs, augmentation, hardware.\n"
                "Vary: batch size only (32 vs 256).\n"
                "Measure: best validation accuracy across 15 epochs, repeated for at least 3 seeds, "
                "report mean and standard error. To keep the comparison fair when batch size changes, "
                "you should either keep the LR fixed (smaller batch dominates by noise) or use the "
                "linear-scaling rule (`lr = base_lr * batch_size / 256`)."))
cells.append(code(
    "def experiment(batch_size, epochs=5, lr=1e-3, seeds=(0, 1, 2)):\n"
    "    accs = []\n"
    "    for s in seeds:\n"
    "        set_seed(s)\n"
    "        m = tf.keras.Sequential([\n"
    "            tf.keras.layers.Input(shape=(784,)),\n"
    "            tf.keras.layers.Dense(128, activation='relu'),\n"
    "            tf.keras.layers.Dropout(0.2),\n"
    "            tf.keras.layers.Dense(10, activation='softmax'),\n"
    "        ])\n"
    "        m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])\n"
    "        h = m.fit(X_train, y_train_oh, validation_data=(X_val, y_val_oh),\n"
    "                  epochs=epochs, batch_size=batch_size, verbose=0)\n"
    "        accs.append(max(h.history['val_accuracy']))\n"
    "    return float(np.mean(accs)), float(np.std(accs))\n"
    "\n"
    "for bs in (32, 256):\n"
    "    mu, sd = experiment(bs, epochs=3)  # 3 epochs to keep it quick; bump to 15 for real comparison\n"
    "    print(f'batch={bs:4d} val acc {mu:.4f} +/- {sd:.4f}')\n"
))

cells.append(md("### 9.6 Investigating a weak class (digit 8)\n"
                "1. **Inspect the confusion matrix** to see what other digit class 8 is most often confused with.\n"
                "2. **Look at misclassified 8s** to spot systematic mistakes (e.g. ambiguous loops mistaken for 0 or 9).\n"
                "3. **Per-class metrics**: precision, recall, F1 for class 8 - low recall + high precision = many 8s misclassified as something else.\n"
                "4. **Targeted augmentation**: rotational + thickness jitter only on the 8 class.\n"
                "5. **Class-weighted loss**: `class_weight={8: 1.5, ...}` to up-weight the rare class.\n"))

cells.append(md("### 9.7 One-hot vs sparse integer labels\n"
                "- **One-hot encoding** (`y_oh.shape = (N, K)`) pairs with `categorical_crossentropy`. "
                "Wastes memory at very large `K` but plays nicely with label smoothing and "
                "multi-label problems.\n"
                "- **Sparse integer labels** (`y.shape = (N,)`) pair with `sparse_categorical_crossentropy` "
                "in Keras and `nn.CrossEntropyLoss` in PyTorch. Cheaper, the default choice for "
                "single-label classification with hundreds of classes.\n"
                "Output layer is the same in both cases — `K` logits — only the loss changes."))

cells.append(md("### 9.8 Why `net.eval()` matters\n"
                "Two layer families behave differently in train vs eval mode:\n"
                "- **Dropout** is identity in eval mode but randomly zeroes activations in train mode.\n"
                "- **BatchNorm** uses *batch statistics* during training and the accumulated *running statistics* during eval. Forgetting `net.eval()` means each validation batch's accuracy depends on the other items in that batch.\n"
                "If you forget the call, dropout fires during validation and the network produces a noisy, downward-biased accuracy. Reported numbers can easily look 1-3 pp lower than the true value, with most of the noise concentrated in small batches."))

cells.append(md("---\n*End of Chapter 3.*\n"))

build_notebook(cells, OUT, kernel_name='python3', display_name='Python 3 (Deep Learning)')
print('Built', OUT)
