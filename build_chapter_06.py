"""Build Chapter 6 notebook: Transformers and Attention Models."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _nbutils import build_notebook, md, code

CHAPTER = 6
OUT = pathlib.Path(__file__).parent / f"Chapter_{CHAPTER}" / f"Chapter_{CHAPTER}_Transformers_and_Attention.ipynb"

cells = []
cells.append(md(
    "# Chapter 6: Transformers and Attention Models\n"
    "\n"
    "*Deep Learning Crash Course - BPB Publications*\n"
    "\n"
    "This notebook implements scaled dot-product attention and multi-head attention from scratch in "
    "NumPy, walks through sinusoidal positional encoding, builds a complete Transformer encoder "
    "block in PyTorch, demonstrates BERT fine-tuning and GPT-2 text generation with temperature, "
    "top-k and top-p sampling, and finishes with the chapter exercises.\n"
))

cells.append(md("## 1. Setup"))
cells.append(code(
    "import os, random, math\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "IMG_DIR = Path('images'); IMG_DIR.mkdir(exist_ok=True)\n"
    "def set_seed(s=42):\n"
    "    os.environ['PYTHONHASHSEED'] = str(s); random.seed(s); np.random.seed(s)\n"
    "    try:\n"
    "        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)\n"
    "    except ModuleNotFoundError: pass\n"
    "set_seed(42)\n"
))

cells.append(md("## 2. Scaled dot-product attention in NumPy\n"
                "\n"
                "$\\text{Attention}(Q, K, V) = \\text{softmax}(Q K^\\top / \\sqrt{d_k}) V$.\n"))
cells.append(code(
    "def softmax(x, axis=-1):\n"
    "    z = np.exp(x - x.max(axis=axis, keepdims=True))\n"
    "    return z / z.sum(axis=axis, keepdims=True)\n"
    "\n"
    "def scaled_dot_product_attention(Q, K, V, mask=None):\n"
    "    d_k = Q.shape[-1]\n"
    "    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)\n"
    "    if mask is not None:\n"
    "        scores = scores + mask * -1e9\n"
    "    attn = softmax(scores, axis=-1)\n"
    "    return attn @ V, attn\n"
    "\n"
    "rng = np.random.default_rng(0)\n"
    "Q = rng.standard_normal((1, 4, 8))\n"
    "K = rng.standard_normal((1, 4, 8))\n"
    "V = rng.standard_normal((1, 4, 8))\n"
    "out, attn = scaled_dot_product_attention(Q, K, V)\n"
    "print('Output shape:', out.shape, 'attention shape:', attn.shape)\n"
    "print('Row sums (should be 1):', attn.sum(axis=-1))\n"
))
cells.append(code(
    "# Visualise the attention pattern with and without a causal mask\n"
    "L = 8\n"
    "Q = rng.standard_normal((1, L, 16))\n"
    "K = rng.standard_normal((1, L, 16))\n"
    "V = rng.standard_normal((1, L, 16))\n"
    "causal = np.triu(np.ones((L, L)), k=1)[None]  # 1s above diagonal\n"
    "_, attn_full = scaled_dot_product_attention(Q, K, V)\n"
    "_, attn_causal = scaled_dot_product_attention(Q, K, V, mask=causal)\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))\n"
    "axes[0].imshow(attn_full[0], cmap='viridis'); axes[0].set_title('Full attention')\n"
    "axes[1].imshow(attn_causal[0], cmap='viridis'); axes[1].set_title('Causal (GPT-style)')\n"
    "for a in axes: a.set_xlabel('key index'); a.set_ylabel('query index')\n"
    "fig.tight_layout(); fig.savefig(IMG_DIR / '01_attention_patterns.png', dpi=150); plt.show()\n"
))

cells.append(md("## 3. Multi-head attention from scratch"))
cells.append(code(
    "class MultiHeadAttention:\n"
    "    def __init__(self, d_model, num_heads, seed=0):\n"
    "        assert d_model % num_heads == 0\n"
    "        self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads\n"
    "        rng = np.random.default_rng(seed)\n"
    "        scale = 1.0 / np.sqrt(d_model)\n"
    "        self.W_q = rng.standard_normal((d_model, d_model)) * scale\n"
    "        self.W_k = rng.standard_normal((d_model, d_model)) * scale\n"
    "        self.W_v = rng.standard_normal((d_model, d_model)) * scale\n"
    "        self.W_o = rng.standard_normal((d_model, d_model)) * scale\n"
    "\n"
    "    def _split(self, x):\n"
    "        B, T, _ = x.shape\n"
    "        return x.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)\n"
    "\n"
    "    def forward(self, X, mask=None):\n"
    "        Q = self._split(X @ self.W_q)\n"
    "        K = self._split(X @ self.W_k)\n"
    "        V = self._split(X @ self.W_v)\n"
    "        scores = Q @ K.swapaxes(-2, -1) / math.sqrt(self.d_k)\n"
    "        if mask is not None:\n"
    "            scores = scores + mask * -1e9\n"
    "        attn = softmax(scores, axis=-1)\n"
    "        ctx = attn @ V\n"
    "        B, H, T, dk = ctx.shape\n"
    "        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, H * dk)\n"
    "        return ctx @ self.W_o, attn\n"
    "\n"
    "mha = MultiHeadAttention(d_model=64, num_heads=8)\n"
    "X = rng.standard_normal((2, 6, 64))\n"
    "out, attn = mha.forward(X)\n"
    "print('Output shape:', out.shape, 'attention shape:', attn.shape)\n"
))

cells.append(md("## 4. Sinusoidal positional encoding"))
cells.append(code(
    "def positional_encoding(seq_len, d_model):\n"
    "    pos = np.arange(seq_len)[:, None]\n"
    "    i = np.arange(d_model)[None, :]\n"
    "    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)\n"
    "    pe = np.zeros((seq_len, d_model))\n"
    "    pe[:, 0::2] = np.sin(angle[:, 0::2])\n"
    "    pe[:, 1::2] = np.cos(angle[:, 1::2])\n"
    "    return pe\n"
    "\n"
    "pe = positional_encoding(64, 128)\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "im = ax.imshow(pe, cmap='RdBu', aspect='auto')\n"
    "ax.set_xlabel('embedding dim'); ax.set_ylabel('position')\n"
    "ax.set_title('Sinusoidal positional encoding (64 positions x 128 dims)')\n"
    "fig.colorbar(im, ax=ax)\n"
    "fig.tight_layout(); fig.savefig(IMG_DIR / '02_pos_encoding.png', dpi=150); plt.show()\n"
))

cells.append(md("## 5. Complete Transformer encoder block in PyTorch\n"
                "\n"
                "Pre-LayerNorm variant - the modern default - because it trains stably without "
                "warmup."))
cells.append(code(
    "try:\n"
    "    import torch\n"
    "    import torch.nn as nn\n"
    "\n"
    "    class TransformerEncoderBlock(nn.Module):\n"
    "        def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1):\n"
    "            super().__init__()\n"
    "            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)\n"
    "            self.ff = nn.Sequential(\n"
    "                nn.Linear(d_model, d_ff), nn.GELU(),\n"
    "                nn.Dropout(dropout), nn.Linear(d_ff, d_model))\n"
    "            self.ln1 = nn.LayerNorm(d_model)\n"
    "            self.ln2 = nn.LayerNorm(d_model)\n"
    "            self.drop = nn.Dropout(dropout)\n"
    "        def forward(self, x, src_mask=None, src_key_padding_mask=None):\n"
    "            h = self.ln1(x)\n"
    "            attn, _ = self.attn(h, h, h, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)\n"
    "            x = x + self.drop(attn)\n"
    "            x = x + self.drop(self.ff(self.ln2(x)))\n"
    "            return x\n"
    "\n"
    "    block = TransformerEncoderBlock()\n"
    "    x = torch.randn(2, 10, 128)\n"
    "    print('Block output:', block(x).shape)\n"
    "    print('Param count:', sum(p.numel() for p in block.parameters()))\n"
    "except ModuleNotFoundError:\n"
    "    print('PyTorch not installed - skipping.')\n"
))

cells.append(md("## 6. A tiny end-to-end Transformer classifier\n"
                "\n"
                "Trains a 2-layer Transformer encoder on a synthetic 'sum-greater-than-threshold' "
                "task so it runs in a few seconds without internet access."))
cells.append(code(
    "try:\n"
    "    import torch\n"
    "    import torch.nn as nn\n"
    "    set_seed(0)\n"
    "    VOCAB, SEQ = 30, 16\n"
    "    def make_batch(batch=256):\n"
    "        x = torch.randint(0, VOCAB, (batch, SEQ))\n"
    "        y = (x.sum(dim=1) > SEQ * VOCAB / 2).long()\n"
    "        return x, y\n"
    "\n"
    "    class TinyTransformer(nn.Module):\n"
    "        def __init__(self):\n"
    "            super().__init__()\n"
    "            self.emb = nn.Embedding(VOCAB, 64)\n"
    "            self.pos = nn.Embedding(SEQ, 64)\n"
    "            self.blocks = nn.ModuleList([TransformerEncoderBlock(64, 4, 128) for _ in range(2)])\n"
    "            self.head = nn.Linear(64, 2)\n"
    "        def forward(self, x):\n"
    "            pos = torch.arange(x.size(1), device=x.device)\n"
    "            h = self.emb(x) + self.pos(pos)\n"
    "            for b in self.blocks: h = b(h)\n"
    "            return self.head(h.mean(dim=1))\n"
    "\n"
    "    net = TinyTransformer()\n"
    "    opt = torch.optim.Adam(net.parameters(), lr=1e-3)\n"
    "    loss_fn = nn.CrossEntropyLoss()\n"
    "    accs = []\n"
    "    for step in range(400):\n"
    "        x, y = make_batch(); opt.zero_grad()\n"
    "        out = net(x); loss = loss_fn(out, y); loss.backward(); opt.step()\n"
    "        if step % 40 == 0:\n"
    "            with torch.no_grad():\n"
    "                xv, yv = make_batch(512)\n"
    "                acc = (net(xv).argmax(1) == yv).float().mean().item()\n"
    "                accs.append(acc); print(f'step {step:3d} loss {loss.item():.4f} acc {acc:.3f}')\n"
    "    fig, ax = plt.subplots(figsize=(6, 3.5))\n"
    "    ax.plot(np.arange(len(accs)) * 40, accs)\n"
    "    ax.set_xlabel('step'); ax.set_ylabel('val acc'); ax.set_title('Tiny Transformer on a toy task')\n"
    "    ax.grid(alpha=0.3); fig.tight_layout()\n"
    "    fig.savefig(IMG_DIR / '03_tiny_transformer.png', dpi=150); plt.show()\n"
    "except ModuleNotFoundError:\n"
    "    print('PyTorch not installed - skipping.')\n"
))

cells.append(md("## 7. Tokenisation strategies side-by-side\n"
                "\n"
                "BERT uses WordPiece, GPT uses byte-pair encoding (BPE). The cell below shows what "
                "each tokenizer does to the same sentence (requires `transformers` to be installed)."))
cells.append(code(
    "sentence = 'The Transformer architecture revolutionised natural language processing in 2017.'\n"
    "try:\n"
    "    from transformers import AutoTokenizer\n"
    "    for name in ['bert-base-uncased', 'gpt2']:\n"
    "        tok = AutoTokenizer.from_pretrained(name)\n"
    "        ids = tok.encode(sentence, add_special_tokens=False)\n"
    "        tokens = tok.convert_ids_to_tokens(ids)\n"
    "        print(f'{name:25s} {len(tokens):2d} tokens -> {tokens}')\n"
    "except (ImportError, OSError) as e:\n"
    "    print('Tokeniser download not available -', type(e).__name__, e)\n"
))

cells.append(md("## 8. Text generation - temperature, top-k, top-p"))
cells.append(code(
    "def temperature_sample(logits, T=1.0):\n"
    "    if T <= 0: return int(np.argmax(logits))\n"
    "    z = np.exp((logits - logits.max()) / T)\n"
    "    return int(np.random.choice(len(z), p=z / z.sum()))\n"
    "\n"
    "def top_k_sample(logits, k=10, T=1.0):\n"
    "    idx = np.argpartition(-logits, k)[:k]\n"
    "    sub = logits[idx]\n"
    "    z = np.exp((sub - sub.max()) / T); z /= z.sum()\n"
    "    return int(idx[np.random.choice(k, p=z)])\n"
    "\n"
    "def top_p_sample(logits, p=0.9, T=1.0):\n"
    "    z = np.exp((logits - logits.max()) / T); z /= z.sum()\n"
    "    order = np.argsort(-z)\n"
    "    cum = np.cumsum(z[order])\n"
    "    cut = np.searchsorted(cum, p) + 1\n"
    "    keep = order[:cut]\n"
    "    sub = z[keep] / z[keep].sum()\n"
    "    return int(keep[np.random.choice(len(keep), p=sub)])\n"
    "\n"
    "rng = np.random.default_rng(0)\n"
    "logits = rng.standard_normal(50) * 1.5\n"
    "for name, fn in [('greedy',    lambda l: int(np.argmax(l))),\n"
    "                 ('temp 0.7',  lambda l: temperature_sample(l, T=0.7)),\n"
    "                 ('top-k=5',   lambda l: top_k_sample(l, k=5)),\n"
    "                 ('top-p=0.9', lambda l: top_p_sample(l, p=0.9))]:\n"
    "    np.random.seed(0)\n"
    "    samples = [fn(logits) for _ in range(20)]\n"
    "    print(f'{name:10s} -> {samples}')\n"
))

cells.append(md("## 9. Exercise solutions\n"
                "\n"
                "### 9.1 MCQ answer key\n"
                "\n"
                "| Q | Answer | Why |\n"
                "|---|--------|-----|\n"
                "| 1 | (b) Parallelisable, gradient-direct, no recurrence | RNNs are sequential, slow on GPUs, and lose long-range context. |\n"
                "| 2 | (b) Stabilise softmax gradients | Dot products grow with $d_k$; scaling keeps the softmax inputs bounded. |\n"
                "| 3 | (b) Multiple attention subspaces in parallel | Each head captures a different relation. |\n"
                "| 4 | (c) Attention itself is permutation-invariant | Positional encoding injects order. |\n"
                "| 5 | (b) Masked Language Modelling | Random masking forces bidirectional context. |\n"
                "| 6 | (b) A summary representation for the whole sequence | Used as classifier input. |\n"
                "| 7 | (b) Allows attention only to previous tokens | Decoder-only auto-regressive. |\n"
                "| 8 | (b) Aligning sub-word tokens to original word labels | Sub-word tokenisation splits entities. |\n"
                "| 9 | (d) Top-p sampling | Sets dynamically based on cumulative probability. |\n"
                "| 10 | (c) Sharper distribution / less diversity | T<1 makes the softmax peakier. |\n"
                "| 11 | (b) WordPiece | BERT's tokenisation algorithm. |\n"
                "| 12 | (c) Free-form text generation | Decoder-only is auto-regressive. |\n"))

cells.append(md("### 9.2 Why RNNs were replaced\n"
                "Three concrete reasons exposed in the section 2 visualisations:\n"
                "1. **Sequential compute**: each step depends on the previous step's hidden state, so the GPU sits idle for most of training. The Transformer's attention matrix is one big GEMM that fully utilises modern accelerators.\n"
                "2. **Gradient decay**: BPTT vanishing gradients limit effective memory to a few dozen tokens. Attention has $O(1)$ gradient path length between any two positions.\n"
                "3. **Context compression**: every RNN hidden state is a fixed-size summary of the past. Attention can re-look at any past token directly."))

cells.append(md("### 9.3 Why the $\\sqrt{d_k}$ scaling\n"
                "If $q, k \\in \\mathbb{R}^{d_k}$ have iid components with mean 0 and variance 1, "
                "$q \\cdot k$ has variance $d_k$. Large pre-softmax scores push softmax into "
                "saturation regions where almost all probability mass collapses onto one position, "
                "and the gradient of softmax vanishes everywhere else. Dividing by $\\sqrt{d_k}$ keeps "
                "the variance of $q \\cdot k$ at $1$ regardless of $d_k$, which keeps the softmax in "
                "its useful range."))

cells.append(md("### 9.4 What multi-head attention buys you\n"
                "Each head operates on a subspace of $d_k = d_{\\text{model}} / h$ dimensions. "
                "Empirically the heads specialise: one head ends up tracking syntactic dependencies "
                "(verb-subject), another tracks coreference, another tracks position-relative patterns. "
                "A single head with the same number of parameters would have to learn one weighted "
                "average that compromises between all of these patterns."))

cells.append(md("### 9.5 Why positional encoding is necessary\n"
                "Self-attention is set-of-vectors $\\rightarrow$ set-of-vectors: permuting the input "
                "permutes the output identically. To recover order information the input embeddings "
                "are augmented with positional encodings (either fixed sinusoids or learned vectors) "
                "before the first attention layer."))

cells.append(md("### 9.6 BERT [CLS] token\n"
                "BERT prepends a special `[CLS]` token to every input. Its final hidden state is "
                "trained, during pre-training, to summarise the whole sequence (next-sentence "
                "prediction). At fine-tuning time it is the natural input to a classification "
                "head: `Linear(d_model, num_classes)`."))

cells.append(md("### 9.7 GPT causal mask vs BERT bidirectional\n"
                "BERT lets every position attend to every other position (bidirectional). GPT applies "
                "an upper-triangular mask that prevents a position from attending to future positions "
                "- this is what makes GPT a left-to-right language model and lets it be used "
                "autoregressively at inference."))

cells.append(md("### 9.8 NER fine-tuning challenge\n"
                "The tokenizer may split a single word like *Anthropic* into *An*, *##throp*, *##ic*. "
                "Named-entity labels are at the word level; the standard fix is to label only the first "
                "subword token and use `-100` (ignored by the loss) for the rest, then aggregate at "
                "post-processing time."))

cells.append(md("### 9.9 Sampling strategies\n"
                "Already implemented in section 8. **Top-p** (nucleus sampling) is the modern default: "
                "it adapts the candidate-set size based on how confident the distribution is. If only "
                "two tokens cover 90 % of probability mass, you sample from a tiny set; if many tokens "
                "share probability, you sample from a wider set. Top-k cannot adapt this way."))

cells.append(md("### 9.10 Temperature $T<1$\n"
                "Dividing logits by $T<1$ multiplies their magnitude, which after softmax pushes more "
                "mass onto the most likely token. The distribution becomes sharper and generation more "
                "deterministic. $T \\to 0$ recovers greedy decoding."))

cells.append(md("### 9.11 Tokenisation algorithms\n"
                "BERT: WordPiece. GPT: BPE. Modern open-source models often use SentencePiece (a "
                "wrapper that can produce BPE or unigram outputs from raw bytes). The choice "
                "affects vocabulary size, handling of unknown characters, and how multilingual "
                "tokens are represented."))

cells.append(md("### 9.12 Encoder-only vs decoder-only\n"
                "- **Encoder-only (BERT)**: best for classification, NER, extractive QA - tasks "
                "where the model consumes the whole input and produces a single label or set of "
                "spans.\n"
                "- **Decoder-only (GPT)**: best for free-form text generation, summarisation, "
                "instruction following.\n"
                "- **Encoder-decoder (T5, BART)**: best for sequence-to-sequence tasks like "
                "translation."))

cells.append(md("---\n*End of Chapter 6.*\n"))

build_notebook(cells, OUT, kernel_name='python3', display_name='Python 3 (Deep Learning)')
print('Built', OUT)
