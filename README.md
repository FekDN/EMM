# Elastic Memory Model (EMM) v26
### A Concept for Unified Merging of Heterogeneous Neural Networks

**Disclaimer:** This project is an **early-stage proof-of-concept (PoC)** and a research prototype. It is not a production-ready library. Its primary goal is to demonstrate and explore novel approaches to intelligently merging multiple pre-trained models into a single, consolidated architecture.

## Core Concept

Modern neural networks are often highly specialized "experts." The Elastic Memory Model (EMM) is an experimental framework designed to create a single, unified model that assimilates knowledge from a multitude of such experts, even if they possess **different architectures** (e.g., BERT, RoBERTa, ESM, ViT), operate on **different data types** (text, protein sequences, images), and have **different dimensionalities** (`d_model`).

The core philosophy of EMM is **knowledge consolidation, not destructive averaging**. Instead of naively averaging the weights of incompatible models, EMM constructs a complex, heterogeneous computational graph where knowledge from diverse experts is preserved, reused, and enriched.

## Merging Principles & Architectural Components

The EMM sequentially "absorbs" expert models by decomposing them into fundamental components and integrating them into its evolving structure.

### 1. Parsers (Tokenizers & Embeddings)
- **Identification:** Parsers are grouped by a unique architectural key (e.g., `bert-base-uncased-v30522`).
- **Merging:** When a new expert with an existing parser type is introduced, EMM performs:
    - **Vocabulary Expansion:** New tokens from the expert's vocabulary are added to EMM's master vocabulary.
    - **Intelligent Vector Merging:** Embedding vectors for common tokens are not replaced but are merged using an adaptive coefficient based on their cosine similarity. This allows the semantic representation of existing tokens to be enriched.

### 2. The Hybrid Encoder Graph
This is the heart of the EMM. The encoders from expert models are not stored monolithically. Instead:
- **Decomposition:** An encoder is broken down into its individual layers (e.g., `BertLayer`).
- **Unification:** Each layer becomes a node in EMM's shared knowledge graph.
- **Layer Fusion:** If a new layer is structurally and functionally similar to an existing one in the graph (assessed using CKA, RSA, and SVCCA metrics), the `UnifiedAssimilationEngine` attempts to merge them. This process involves aligning their neurons using the Hungarian algorithm before fusing weights.
- **Quality Control:** A merge is reverted if it leads to a significant performance degradation on previously learned tasks.

### 3. Cognitive Links — The "4D" Connections
This is one of EMM's key innovations for handling architectural incompatibility.

- **The Problem:** What happens when a layer from a new model (e.g., a `GPT2Block`) is functionally similar to a layer within EMM (e.g., a `BertLayer`), but their architectures are incompatible, making a direct weight merge impossible or lossy?
- **The Solution:** Instead of a structural merge, EMM creates a **Cognitive Link**. This is an indirect, functional connection between the two layer-nodes in the graph.
- **How It Works:** During a forward pass, a layer with a Cognitive Link receives not only its "native" input but also the activation tensor from its linked peer in a different architecture. These two information streams are fused on-the-fly by a dedicated `CognitiveFusionUnit`, which dynamically weighs and combines them.
- **Why "4D"?** The standard data flow in a transformer can be seen as a 3D tensor (`Batch, Sequence, Dimension`). A Cognitive Link represents a conceptual **"fourth dimension"** that connects two parallel, architecturally distinct 3D spaces. It acts as a "hyperlink" between different model universes, enabling them to exchange information without needing an identical internal structure.

### 4. Skill Branches
- **The Idea:** Unique, task-specific knowledge is encapsulated not in the core graph, but in lightweight "branches" (`AdaptiveBranch`), implemented using a low-rank decomposition (LoRA-like) approach.
- **Assimilation:** When a new skill is added, EMM analyzes its similarity to existing ones:
    - **New, Unique Skill:** It's added as a new, independent branch.
    - **High Similarity:** A "task arithmetic" operation is performed, merging the skills into a single, more generalized branch.
    - **Moderate Similarity:** This triggers a **Hierarchical Refactoring**. Two similar skills are transformed into a `HierarchicalBranch`—a structure with a shared "trunk" (common knowledge) and two specific "sub-branches" (unique knowledge from each original skill).

### 5. Classification Heads
- **Grouping:** Heads for similar tasks (e.g., text classification with N classes and an input dimension of D) are grouped together.
- **Merging:** When a new head is added to an existing group, their weights are merged after performing neuron alignment to prevent sign conflicts.

## Web Interface and Visualizer

The project includes a simple web-based visualizer built with Flask and Socket.IO.

- **Purpose:** Its primary function is to help developers **visualize** the complex graph structure that EMM builds during the assimilation process. It provides a real-time view of how layers, parsers, heads, and—most importantly—**Cognitive Links** are added and interconnected.
- **Status:** **Please Note:** The web interface is a debugging and demonstration tool. It has not been optimized for UI/UX or production stability. Its main value is in providing insight into the model's internal state.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision transformers datasets scikit-learn scipy numpy pandas flask flask-socketio timm
    ```

2.  **Download Models:**
    The script will automatically attempt to download models from the Hugging Face Hub. It creates a `scientific_models_cache` subdirectory to store the model files. Ensure you have an internet connection for the first run to populate this cache.

3.  **Execute the Script:**
    ```bash
    python EMMv26.py
    ```

4.  **Interact:**
    - After running the script, the console will indicate that the web server has started.
    - Open your web browser and navigate to `http://127.0.0.1:5001`.
    - You will see the model's graph being constructed in real-time as each expert is assimilated.

---

*   feklindn@gmail.com 

---

## License

The source code of this project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.

The accompanying documentation, including this README and the project's White Paper, is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.
