# mojo-gpu-tutorials
Hands-on tutorials for GPU programming in Mojo 🔥 | Learn kernels, threads, blocks, memory layouts, and LayoutTensor with practical examples | Write once, run on NVIDIA, AMD, or Apple Silicon

---

## 📚 Available Tutorials

| # | Topic | MDX Notebook | Colab Notebook |
|---|-------|-------------|----------------|
| 001 | GPU Introduction & Basics | [📓 MDX](notebooks/mojo/001_mojo_gpu_intro.mdx) | [📙 Colab](notebooks/colab/001_mojo_gpu_intro.ipynb) |
| 002 | Memory Layout & LayoutTensor | [📓 MDX](notebooks/mojo/002_memory_layout.mdx) | [📙 Colab](notebooks/colab/002_memory_layout.ipynb) |
| 003 | Vector Addition | [📓 MDX](notebooks/mojo/003_vector_addition.mdx) | [📙 Colab](notebooks/colab/003_vector_addition.ipynb) |
| 004 | Thread Management | [📓 MDX](notebooks/mojo/004_thread_mgmt.mdx) | [📙 Colab](notebooks/colab/004_thread_mgmt.ipynb) |
| 005 | Shared Memory | [📓 MDX](notebooks/mojo/005_shared_memory.md) | [📙 Colab](notebooks/colab/005_shared_memory.ipynb) |
| 006 | Pooling Operations | [📓 MDX](notebooks/mojo/006_pooling.mdx) | [📙 Colab](notebooks/colab/006_pooling.ipynb) |
| 007 | Convolution | [📓 MDX](notebooks/mojo/007_convolution.mdx) | [📙 Colab](notebooks/colab/007_convolution.ipynb) |
| 008 | MAX Graph Introduction | - | [📙 Colab](notebooks/max-graph/008_max_graph_intro.ipynb) |
| 009 | Matrix Multiplication | [📓 MDX](notebooks/mojo/009_matmul.mdx) | [📙 Colab](notebooks/colab/009_matmul.ipynb) |
| 010 | Functional Programming in Mojo | [📓 MDX](notebooks/mojo/010_mojo_functional_pgm.mdx) | [📙 Colab](notebooks/colab/010_mojo_functional_pgm.ipynb) |
| 011 | Warp-Level Programming | [📓 MDX](notebooks/mojo/011_warp.mdx) | [📙 Colab](notebooks/colab/011_warp.ipynb) |

---

## 🎨 Interactive Visualizations

Enhance your learning with **interactive HTML visualizers** that help you understand GPU concepts visually!

### 📊 Available Visualizations:
- 🔄 **[GPU Algorithm Visualizer](visualize/gpu_algo_visualizer.html)** - Step through GPU algorithms
- ⚡ **[GPU Execution Visualizer](visualize/gpu_exec_visualizer.html)** - See how threads execute
- 📐 **[GPU Layout Visualizer](visualize/gpu_layout_visualizer.html)** - Understand memory layouts
- 🌊 **[GPU Warp Visualizer](visualize/gpu_warp_visualizer.html)** - Visualize warp-level operations

### 🖥️ How to View (Local - VS Code):
1. Install **Live Preview** extension:
   - Open Extensions (Ctrl+Shift+X / Cmd+Shift+X)
   - Search for **"Live Preview"** by Microsoft
   - Click **Install**
2. Right-click any HTML file in the `visualize/` folder
3. Select **"Show Preview"** or **"Open with Live Server"**
4. Interact with the visualization in your browser!

### ☁️ How to View (Colab Users):
1. Download the HTML files from the `visualize/` folder to your local machine
2. Follow the VS Code steps above to view them locally
3. (HTML visualizations cannot run directly in Colab, but work great locally!)

---

## 🎯 Choose Your Learning Path

### 📓 **Method 1: MDX Notebooks (Local - Recommended for Best Experience)**

✅ **Pros:**
- 🎨 Pure notebook experience with interactive cells
- 🔄 Split code across multiple cells for step-by-step execution
- 🏃 Run cells independently and experiment freely
- 💻 Full local development environment

❌ **Cons:**
- 🖥️ Requires local setup (VS Code + Pixi + GPU)
- 🐧 Linux/macOS only (Windows users need WSL2)

**Perfect for:** Deep learning and experimentation with full flexibility

---

### 📙 **Method 2: Colab Notebooks (Cloud - Quick Start)**

✅ **Pros:**
- ☁️ Run on Google Colab with free T4 GPU
- 🚀 No local installation required
- 🌐 Access from anywhere
- 💰 Free GPU compute

❌ **Cons:**
- ⚠️ **Important Limitation:** Entire Mojo program must be in a single code block
- 🚫 Cannot split code across multiple cells for true notebook experience
- 📦 Less flexibility for experimentation

**Perfect for:** Quick start, testing, or if you don't have a local GPU

> **📝 Note:** Due to how IPython notebooks execute Mojo code in Colab, you must write your complete program in one cell. This means you cannot define a function in one cell and call it in another - everything must be together.

---

# 🚀 Getting Started

## 📋 Prerequisites

### For MDX Notebooks (Local):
- 🖥️ **VS Code** installed ([Download here](https://code.visualstudio.com/))
- 🐧 **Linux** or **macOS** (Windows users: use WSL2)
- 🎮 **NVIDIA/AMD GPU** (optional, but recommended for GPU tutorials)

### For Colab Notebooks (Cloud):
- 🌐 Google account for Colab access
- ☁️ Free T4 GPU provided by Colab

---

## ⚡ Setup Instructions

### 🏠 Local Setup (MDX Notebooks)

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/abhisheksreesaila/mojo-gpu-tutorials.git
cd mojo-gpu-tutorials
```

### 2️⃣ Install Pixi (Package Manager)

If you don't have `pixi` installed:

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

> 💡 Pixi manages all dependencies automatically - no manual setup needed!

### 3️⃣ Install Mojo

```sh
pixi add mojo
```

Choose your version:
- 🌙 **Nightly** - Latest features (for adventurers)
- ✅ **Stable** - Recommended for learning

### 4️⃣ Activate the Environment

```sh
pixi shell
```

You should see `(mojo_monday)` in your terminal prompt 🎉

### 5️⃣ Open in VS Code

```sh
code .
```

### 6️⃣ Install Markdown Lab Extension

In VS Code:
1. Open Extensions (Ctrl+Shift+X / Cmd+Shift+X)
2. Search for **"Markdown Lab"**
3. Click **Install**

---

## 🎯 Run Your First Notebook

### For MDX Notebooks:
1. Open any `.mdx` file in the `notebooks/mojo/` folder
2. Click the **▶️ Run** button in any code cell
3. Watch Mojo magic happen! 🔥

### For Colab Notebooks:
1. Open any `.ipynb` file in the `notebooks/colab/` folder in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run cells sequentially
4. **Remember:** Keep your complete Mojo program in a single code block

---

## 💬 Need Help?

- 📖 [Mojo Notebook Setup](https://docs.modular.com/mojo/manual/gpu/basics/#setup)
- 🐛 [Mojo Installation Instructions](https://docs.modular.com/mojo/manual/install)
- 💡 [Google Colab Guide](https://colab.research.google.com/)

---

**Happy GPU Programming! 🔥🚀**