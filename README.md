# mojo-gpu-tutorials
Hands-on tutorials for GPU programming in Mojo 🔥 | Learn kernels, threads, blocks, memory layouts, and LayoutTensor with practical examples | Write once, run on NVIDIA, AMD, or Apple Silicon

# 🚀 Getting Started

## 📋 Prerequisites

Before you begin, make sure you have:
- 🖥️ **VS Code** installed ([Download here](https://code.visualstudio.com/))
- 🐧 **Linux** or **macOS** (Windows users: use WSL2)
- 🎮 **NVIDIA/AMD GPU** (optional, but recommended for GPU tutorials)

---

## ⚡ Quick Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/abhisheksreesaila/mojo-gpu-tutorials.git
cd mojo-gpu-tutorials
```

### 2️⃣ Install Pixi (Package Manager)

If you don't have `pixi` installed:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

> 💡 Pixi manages all dependencies automatically - no manual setup needed!

### 3️⃣ Install Mojo

```bash
pixi add mojo
```

Choose your version:
- 🌙 **Nightly** - Latest features (for adventurers)
- ✅ **Stable** - Recommended for learning

### 4️⃣ Activate the Environment

```bash
pixi shell
```

You should see `(mojo_monday)` in your terminal prompt 🎉

### 5️⃣ Open in VS Code

```bash
code .
```

### 6️⃣ Install Markdown Lab Extension

In VS Code:
1. Open Extensions (Ctrl+Shift+X / Cmd+Shift+X)
2. Search for **"Markdown Lab"**
3. Click **Install**

---

## 🎯 Run Your First Notebook

1. Open any `.mdx` file in the `notebooks/` folder
2. Click the **▶️ Run** button in any code cell
3. Watch Mojo magic happen! 🔥

---

## 💬 Need Help?

- 📖 [Mojo Notebook Setup](https://docs.modular.com/mojo/manual/gpu/basics/#setup)
- 🐛 [Mojo Installation Instructions](https://docs.modular.com/mojo/manual/install)
---

**Happy GPU Programming! 🔥🚀**