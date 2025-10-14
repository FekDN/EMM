# Elastic Memory Model (EMM) v26
### A Concept for Unified Merging of Heterogeneous Neural Networks

**Disclaimer:** This project is an **early-stage proof-of-concept (PoC)** and a research prototype. It is not a production-ready library. Its primary goal is to demonstrate and explore novel approaches to intelligently merging multiple pre-trained models into a single, consolidated architecture.

## Abstract

The Elastic Memory Model (EMM) is an experimental framework designed to create a single, unified model that assimilates knowledge from a multitude of specialized "expert" models. It is uniquely capable of merging experts with **heterogeneous architectures** (e.g., BERT, RoBERTa, ESM, ViT), **different data modalities** (text, protein sequences, images), and varying **internal dimensionalities** (`d_model`). The core philosophy of EMM is **knowledge consolidation, not destructive averaging**, achieved by constructing a dynamic, heterogeneous computational graph where unique knowledge is preserved, reused, and functionally interconnected.

## Key Features

- **Heterogeneous by Design:** Natively merges models with different architectures and `d_model` sizes into a single graph.
- **Intelligent Unification Engine:** Employs a sophisticated engine (`UnifiedAssimilationEngine`) that uses a hybrid of structural and functional similarity metrics (CKA, RSA, SVCCA, weight correlation) to decide whether to merge, link, or separate layers.
- **Cognitive Links ("4D" Coupling):** A novel mechanism to create functional connections between architecturally incompatible but functionally similar layers, allowing for information exchange without requiring identical structures.
- **Hierarchical Skill Refactoring:** Automatically identifies related "skills" (task-specific knowledge) and refactors them into hierarchical structures with a shared "trunk" of common knowledge and specific "branches" for unique capabilities.
- **Real-time Visualization:** Includes a web-based interface to visualize the complex graph structure, showing how models are decomposed and interconnected during the assimilation process.

## Core Concept

Modern neural networks are often highly specialized "experts." The Elastic Memory Model (EMM) aims to overcome this specialization by creating a holistic model that learns from them all.

> The core philosophy of EMM is **knowledge consolidation, not destructive averaging**.

Instead of naively averaging the weights of incompatible models, EMM constructs a complex computational graph. Here, knowledge from diverse experts is preserved in its native form, intelligently fused where possible, and functionally linked where a direct merge is impossible.

## The Assimilation Process: How It Works

The EMM sequentially "absorbs" expert models by decomposing them into fundamental components and integrating them into its evolving structure.

### 1. Parsers (Tokenizers & Embeddings)
- **Grouping:** Parsers are grouped by a shared architectural key (e.g., `bert-d768-l12`).
- **Merging & Expansion:** When a new expert with an existing parser type is introduced, EMM performs:
    - **Vocabulary Expansion:** New tokens are added to the master vocabulary.
    - **Intelligent Vector Merging:** Embedding vectors for common tokens are merged using an adaptive coefficient based on their cosine similarity, enriching the semantic representation of existing tokens.

### 2. The Hybrid Encoder Graph
This is the heart of the EMM. Encoders are not stored monolithically:
- **Decomposition:** An encoder is broken down into its individual layers (e.g., `BertLayer`).
- **Unification:** Each layer becomes a node in EMM's shared knowledge graph.
- **Layer Fusion:** If a new layer is structurally and functionally similar to an existing one, the `UnifiedAssimilationEngine` attempts to merge them. This process involves aligning their neurons using the Hungarian algorithm before fusing weights.
- **Quality Control:** A merge is automatically reverted (`ROLLBACK`) if it leads to a significant performance degradation on any previously learned task.

### 3. Spotlight on Cognitive Links — The "4D" Connections
This is EMM's key innovation for handling architectural incompatibility.

- **The Problem:** What happens when a layer from a new model (e.g., a `GPT2Block`) is functionally similar to a layer within EMM (e.g., a `BertLayer`), but their architectures make a direct weight merge impossible?
- **The Solution:** Instead of a structural merge, EMM creates a **Cognitive Link**. This is an indirect, functional, and *learnable* connection between the two layer-nodes.
- **How It Works:** During a forward pass, a layer with a Cognitive Link receives activations from its linked peer in a different architecture. These two information streams are fused on-the-fly by a dedicated `CognitiveFusionUnit`, which dynamically combines them.

> **Why "4D"?** The standard data flow in a transformer is a 3D tensor (`Batch, Sequence, Dimension`). A Cognitive Link represents a conceptual **"fourth dimension"** that connects two parallel, architecturally distinct 3D spaces. It acts as a "hyperlink" between different model universes, enabling them to exchange information without an identical structure.

### 4. Skill Branches
- **Concept:** Unique, task-specific knowledge is encapsulated in lightweight "branches" (`AdaptiveBranch`), implemented using a low-rank decomposition (LoRA-like) approach.
- **Assimilation Logic:**
    - **New Skill:** Added as an independent branch.
    - **High Similarity:** A "task arithmetic" operation merges the skills into a single, more generalized branch.
    - **Moderate Similarity:** Triggers a **Hierarchical Refactoring**, creating a structure with a shared "trunk" (common knowledge) and specific "sub-branches" (unique knowledge).

### 5. Classification Heads
- **Grouping:** Heads for similar tasks (e.g., text classification with N classes) are grouped.
- **Merging:** When a new head is added to an existing group, their weights are merged after performing neuron alignment to prevent sign conflicts.

---

## Under the Hood: The Merging Logic

The decision-making process for merging and linking is based on a formal algorithm.

**1. Data Preparation:**
For all encoders `E_k`, activation representations `A_{k,i}` are formed for each layer `i` on a test dataset. These are then normalized and centered.

**2. Functional Similarity Matrix:**
For each pair of layers `(A_{a,i}, A_{b,j})`, a metric vector is computed:
```
v = [CKA, SVCCA, RSA, Corr(W), TypeSim]
```
An integral similarity score `S` is then calculated as a weighted sum of these metrics:
```
S = λ₁·CKA + λ₂·SVCCA + λ₃·RSA + λ₄·Corr(W) + λ₅·TypeSim
```

**3. Decision Thresholds:**
The algorithm classifies pairs based on their similarity score `S`:
- **Merge Pairs (`S ≥ τ_merge` ≈ 0.85):** These layers are considered functionally equivalent. The system performs neuron alignment and weight averaging.
- **Bridge Pairs (`τ_link ≤ S < τ_merge` ≈ 0.75-0.85):** These layers are related but not identical. A `Cognitive Link` is created between them.
- **Independent (`S < τ_link`):** The layers are treated as functionally distinct and exist independently in the graph.

**4. Pair Selection Algorithm:**
To ensure global consistency, the optimal set of pairs is selected by solving a **linear assignment problem** (using the Hungarian algorithm), which maximizes the total similarity across all chosen pairs while ensuring each layer is part of at most one pair.

---

## Web Interface and Visualizer

The project includes a web-based visualizer built with Flask and Socket.IO.

- **Purpose:** Its primary function is to help developers **visualize** the complex graph structure that EMM builds. It provides a real-time, hierarchically-organized view of how layers, parsers, heads, and Cognitive Links are added and interconnected.
- **Layout:** The layout is programmatically controlled to represent each model as a vertical "column." Models with shared components are automatically placed in adjacent columns to highlight architectural similarities.
- **Status:** **Please Note:** The web interface is a debugging and demonstration tool. Its main value is in providing insight into the model's internal state.

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

## Future Work & Limitations

As a PoC, there are many areas for future research and improvement:
- **Dynamic Threshold Adaptation:** Implement dynamic adjustment of merge/link thresholds based on the statistical properties of the similarity matrix, as described in the formal logic.
- **Advanced `alpha` Calculation:** Compute the weight merging coefficient `alpha` based on each layer's contribution to task-specific loss, rather than using a fixed value.
- **Continual Learning & Forgetting:** Explore strategies to fine-tune the consolidated EMM on new data and mitigate catastrophic forgetting.
- **Performance Optimization:** The current implementation is not optimized for speed or memory.

---

*   feklindn@gmail.com 

---

## License

The source code of this project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.

The accompanying documentation, including this README and the project's White Paper, is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.
