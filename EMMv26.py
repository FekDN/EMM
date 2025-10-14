# Copyright (c) 2025  Feklin Dmitry (FeklinDN@gmail.com)

import os
import ssl
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_ALLOW_LOADING_LEGACY_TORCH_LOAD'] = '1'
os.environ["TRANSFORMERS_NO_SAFE_LOAD"] = "1"
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import deque, defaultdict
import math
import random
import copy
from typing import Dict, List, Optional, Tuple, Any, Union
from torchvision import datasets, transforms
from transformers import (
    BertTokenizer, GPT2Tokenizer, BertModel, GPT2Model, AutoTokenizer, AutoModel,
    PreTrainedTokenizer, PreTrainedTokenizerFast
)
from transformers.models.bert.modeling_bert import BertEmbeddings
from datasets import load_dataset
import timm
import torchvision
import torchvision.transforms as transforms
import operator
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.stats import spearmanr
import weakref
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

import logging

def suppress_transformers_logging():
    """
    Forces the logging level for the transformers library to ERROR to suppress informational messages.
    """
    # Gaining access to the Transformers library logger
    transformers_logger = logging.getLogger("transformers")
    
    # Set a level that will only allow ERROR and CRITICAL messages through.
    transformers_logger.setLevel(logging.ERROR)
    
    # Can also do this for datasets if it is also "noisy"
    datasets_logger = logging.getLogger("datasets")
    datasets_logger.setLevel(logging.ERROR)

# Call the function to apply the settings
suppress_transformers_logging()

EMM_CONFIG = {
    # General settings
    "SEED": 42,
    "CACHE_DIR_NAME": "scientific_models_cache",
    # Key architectural parameters of EMM
    "INITIAL_BACKBONE_BLOCKS": 1,
    "DEFAULT_BRANCH_RANK": 8,
    "ENRICHMENT_HORIZON": 3,

    # Genetic Algorithm Parameters (for weight merging)
    "GA_PARAMS": {
        "pop_size": 15,
        "generations": 15,
        "mutation_rate": 0.2
    },
    
    # Encoder unification parameters
    "UNIFICATION_PARAMS": {
        "direct_merge_threshold": 0.85, # Threshold for direct layer merging
        "cognitive_link_threshold": 0.75, # Threshold for creating a cognitive link during a failed merger
        "post_merge_quality_loss_tolerance": 0.02, # Acceptable drop in quality (2%)
        "svcca_explained_variance": 0.98,
        "randomized_svd_threshold": 512,
        "structural_similarity_weight": 0.3, # Weight of structural similarity in matching
        "functional_similarity_weight": 0.7, # Functional similarity weight
    },

    # Component Merge Parameters
    "MERGE_PARAMS": {
        "TOKENIZER": {
            "merge_similarity_threshold": 0.9,
            "alpha_base": 0.5
        },
        "HEADS": {
            "threshold": 0.7,
            "alpha": 0.5
        }
    },

    "BRANCH_ASSIMILATION_PARAMS": {
        "energy_threshold": 0.99,
        "threshold_relevance": 0.4,
        "threshold_task_arithmetic": 0.95,
        "threshold_hierarchical_refactor": 0.75
    }
}

def mse_metric(model: nn.Module, dataloader: DataLoader, device: str, task_name: str) -> float:
    """Calculates Mean Squared Error (MSE) for regression tasks."""
    if not HAS_SKLEARN_METRICS:
        print("  -> WARNING: MSE metric skipped, scikit-learn not installed.")
        return float('inf')
        
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            y = batch[1] if isinstance(batch, (list, tuple)) else batch['labels']
            
            if isinstance(model, ElasticMemoryModel):
                # Pass the task name explicitly
                out = model(x, task_name)
            else:
                out = model(**x) if isinstance(x, dict) else model(x)

            if hasattr(out, 'logits'): out = out.logits
            elif isinstance(out, dict): out = out.get('out', next(iter(out.values())))
            
            all_preds.append(out.squeeze().cpu())
            all_labels.append(y.squeeze().cpu())
            
    if not all_preds: return float('inf')
    
    return mean_squared_error(torch.cat(all_labels), torch.cat(all_preds))

def f1_metric(model: nn.Module, dataloader: DataLoader, device: str, task_name: str) -> float:
    """Calculates F1-score for token classification or QA tasks, ignoring padding."""
    if not HAS_SKLEARN_METRICS:
        print("  -> WARNING: F1 metric skipped, scikit-learn not installed.")
        return 0.0

    model.eval()
    all_preds, all_labels = [], []
    ignore_index = -100 
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            if isinstance(model, ElasticMemoryModel):
                # Pass the task name explicitly
                out = model(inputs, task_name)
            else:
                out = model(**inputs)

            # Use robust logic to extract logits
            logits = out
            if hasattr(out, 'logits'):
                logits = out.logits
            elif isinstance(out, dict):
                logits = out.get('out', next(iter(out.values())))

            predictions = logits.argmax(dim=-1)
            
            for i in range(labels.shape[0]):
                mask = labels[i] != ignore_index
                actual_labels = labels[i][mask]
                actual_preds = predictions[i][mask]
                all_labels.extend(actual_labels.cpu().numpy())
                all_preds.extend(actual_preds.cpu().numpy())

    if not all_labels: return 0.0
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)

def accuracy_metric(model: nn.Module, dataloader: DataLoader, device: str, task_name: str) -> float:
    """A general-purpose accuracy metric that can evaluate both EMMs and simple teacher models."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['pixel_values'].to(device) if 'pixel_values' in batch else {k: v.to(device) for k, v in batch.items() if k not in ['labels', 'task_name']}
            y = batch[1].to(device) if isinstance(batch, (list, tuple)) else batch['labels'].to(device)
            
            if isinstance(model, ElasticMemoryModel):
                # Pass the task name explicitly
                out = model(x, task_name)
            else:
                out = model(**x) if isinstance(x, dict) else model(x)

            if hasattr(out, 'logits'): out = out.logits
            elif isinstance(out, dict): out = out.get('out', next(iter(out.values())))

            # Handle cases where the model returns unpooled sequence logits for a sequence-level task.
            if out.dim() == 3 and y.dim() == 1:
                # The output is (batch, seq_len, classes) but labels are (batch,).
                # We pool the logits over the sequence dimension before comparison.
                out = out.mean(dim=1)
            
            if out.dim() > 1 and out.shape[-1] > 1:
                pred = out.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
    return correct / total if total > 0 else 0.0

def mIoU_metric(model: nn.Module, dataloader: DataLoader, device: str, task_name: str) -> float:
    # Calculates the Mean Intersection over Union (mIoU) metric for semantic segmentation tasks.
    model.eval()
    num_classes = None

    try:
        head_module = None
        if task_name in model.hybrid_heads: # Check in EMM
            head_module = model.hybrid_heads[task_name]
        elif hasattr(model, 'classifier'): # For standalone models
            head_module = model.classifier

        if head_module:
            final_layer = next(reversed(list(head_module.modules())), None)
            if hasattr(final_layer, 'out_channels'):
                num_classes = final_layer.out_channels
            elif hasattr(final_layer, 'out_features'):
                num_classes = final_layer.out_features
            else:
                raise ValueError(f"Could not determine num_classes from the head for task '{task_name}'")
        else:
            raise ValueError(f"No head found for task '{task_name}'")

    except Exception as e:
        print(f"Error during mIoU initialization: {e}")
        return 0.0

    conf_matrix = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            targets = batch[1].to(device)
            
            if isinstance(model, ElasticMemoryModel):
                 outputs = model(images, task_name)
            else:
                 outputs = model(images)
            
            logits = outputs
            if isinstance(outputs, dict):
                 logits = outputs.get('out', next(iter(outputs.values())))

            if logits is None: continue

            preds = logits.argmax(1)
            
            if preds.shape[-2:] != targets.shape[-2:]: continue

            mask = (targets >= 0) & (targets < num_classes)
            flat_targets = targets[mask]
            flat_preds = preds[mask]
            
            indices = flat_targets * num_classes + flat_preds
            counts = torch.bincount(indices, minlength=num_classes**2)
            conf_matrix += counts.reshape(num_classes, num_classes)

    intersection = torch.diag(conf_matrix)
    ground_truth_set = conf_matrix.sum(dim=1)
    predicted_set = conf_matrix.sum(dim=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / (union + 1e-6)
    mean_iou = torch.nanmean(iou).item()

    return mean_iou

def center(X):
    return X - X.mean(axis=0, keepdims=True)

def flatten_activations(A):
    if A.ndim > 2:
        return A.reshape(A.shape[0], -1)
    return A

def linear_cka(X, Y):
    X_flat, Y_flat = flatten_activations(X), flatten_activations(Y)
    Xc, Yc = center(X_flat), center(Y_flat)
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
    hsic = np.sum(Kx * Ky)
    # Add a small epsilon to the denominator to prevent division by zero with constant activations.
    denom = math.sqrt(max(1e-20, np.sum(Kx * Kx) * np.sum(Ky * Ky)))
    return float(hsic / denom)

def svcca_score(A, B):
    try:
        # Add a local context manager to suppress known numerical warnings from scikit-learn/numpy.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            if np.std(A) < 1e-6 or np.std(B) < 1e-6:
                return 0.0

            A_flat, B_flat = flatten_activations(A), flatten_activations(B)
            solver = 'randomized' if A_flat.shape[1] > EMM_CONFIG["UNIFICATION_PARAMS"]["randomized_svd_threshold"] else 'full'
            k_A = min(A_flat.shape[1], A_flat.shape[0] - 1)
            k_B = min(B_flat.shape[1], B_flat.shape[0] - 1)

            if k_A <=0 or k_B <=0: return 0.0

            pcaA = PCA(n_components=k_A, svd_solver=solver)
            A_r = pcaA.fit_transform(center(A_flat))
            # The following line can cause a warning if explained_variance_ratio_ contains NaN.
            k_A_var = np.searchsorted(np.cumsum(pcaA.explained_variance_ratio_), EMM_CONFIG["UNIFICATION_PARAMS"]["svcca_explained_variance"]) + 1
            A_r = A_r[:, :max(1, min(k_A_var, A_r.shape[1]))]

            pcaB = PCA(n_components=k_B, svd_solver=solver)
            B_r = pcaB.fit_transform(center(B_flat))
            k_B_var = np.searchsorted(np.cumsum(pcaB.explained_variance_ratio_), EMM_CONFIG["UNIFICATION_PARAMS"]["svcca_explained_variance"]) + 1
            B_r = B_r[:, :max(1, min(k_B_var, B_r.shape[1]))]

            cca_k = min(10, A_r.shape[1], B_r.shape[1])
            if cca_k == 0: return 0.0
            
            cca = CCA(n_components=cca_k)
            cca.fit(A_r, B_r)
            U, V = cca.transform(A_r, B_r)
            
            corrs = [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(U.shape[1])]
            valid_corrs = [c for c in corrs if not np.isnan(c)]
            return float(np.mean(valid_corrs)) if valid_corrs else 0.0
    except Exception:
        return 0.0

def rsa_score(A, B):
    try:
        # Add a local context manager to suppress known numerical warnings from scikit-learn/numpy.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            if np.std(A) < 1e-6 or np.std(B) < 1e-6:
                return 0.0
            
            A_flat, B_flat = flatten_activations(A), flatten_activations(B)
            rdmA = 1.0 - np.corrcoef(center(A_flat))
            rdmB = 1.0 - np.corrcoef(center(B_flat))
            rdmA_vec = rdmA[np.triu_indices(len(rdmA), k=1)]
            rdmB_vec = rdmB[np.triu_indices(len(rdmB), k=1)]
            rho, _ = spearmanr(rdmA_vec, rdmB_vec)
            return 0.0 if np.isnan(rho) else float(rho)
    except Exception:
        return 0.0

def rsa(acts1: torch.Tensor, acts2: torch.Tensor) -> float:
    """Calculates RSA, now robust to 3D tensor inputs."""
    if not HAS_SCIPY_SKLEARN: return 0.0
    
    if torch.std(acts1) < 1e-6 or torch.std(acts2) < 1e-6:
        return 0.0
    
    if acts1.dim() > 2:
        acts1 = acts1.reshape(-1, acts1.shape[-1])
    if acts2.dim() > 2:
        acts2 = acts2.reshape(-1, acts2.shape[-1])
    
    try:
        vec1 = torch.pdist(acts1)
        vec2 = torch.pdist(acts2)
        
        if torch.std(vec1) < 1e-6 or torch.std(vec2) < 1e-6:
            return 0.0
            
        with warnings.catch_warnings():
            # Suppressing the constant input warning, which is a subclass of RuntimeWarning
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            corr, _ = spearmanr(vec1.detach().cpu().numpy(), vec2.detach().cpu().numpy())
            return corr if not np.isnan(corr) else 0.0
            
    except Exception as e:
        print(f"  -> WARNING: RSA calculation failed: {e}. Defaulting to 0.0")
        return 0.0

# Also adding the same check to the older svcca function for consistency
def svcca(acts1: torch.Tensor, acts2: torch.Tensor, k: float = 0.99) -> float:
    """Computes SVCCA, now robust to 3D tensor inputs and handles ConvergenceWarnings."""
    if not HAS_SCIPY_SKLEARN:
        print("  -> WARNING: SVCCA skipped, scikit-learn/scipy not installed.")
        return 0.0
    
    if torch.std(acts1) < 1e-6 or torch.std(acts2) < 1e-6:
        return 0.0

    if acts1.dim() > 2:
        acts1 = acts1.reshape(-1, acts1.shape[-1])
    if acts2.dim() > 2:
        acts2 = acts2.reshape(-1, acts2.shape[-1])

    acts1_np = acts1.detach().cpu().numpy()
    acts2_np = acts2.detach().cpu().numpy()

    try:
        # Add RuntimeWarning to the list of ignored warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning) 
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            
            u1, s1, _ = np.linalg.svd(acts1_np.T, full_matrices=False)
            u2, s2, _ = np.linalg.svd(acts2_np.T, full_matrices=False)

            s1_sum = np.sum(s1); cum_s1 = np.cumsum(s1) / s1_sum if s1_sum > 0 else np.zeros_like(s1)
            rank1 = np.searchsorted(cum_s1, k) + 1
            
            s2_sum = np.sum(s2); cum_s2 = np.cumsum(s2) / s2_sum if s2_sum > 0 else np.zeros_like(s2)
            rank2 = np.searchsorted(cum_s2, k) + 1
            
            svd_acts1 = acts1_np @ u1[:, :rank1]
            svd_acts2 = acts2_np @ u2[:, :rank2]

            n_components = min(svd_acts1.shape[1], svd_acts2.shape[1])
            if n_components == 0: return 0.0

            cca = CCA(n_components=n_components)
            cca.fit(svd_acts1, svd_acts2)

            X_c, Y_c = cca.transform(svd_acts1, svd_acts2)
            corrs = np.array([np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)])
            
            abs_corrs = np.abs(corrs)
            if np.all(np.isnan(abs_corrs)): return 0.0
            return np.nanmean(abs_corrs)

    except Exception as e:
        print(f"  -> WARNING: SVCCA calculation failed: {e}. Defaulting to 0.0")
        return 0.0

def get_layer_type(module):
    if isinstance(module, nn.Conv2d): return 'conv'
    if isinstance(module, (nn.LSTM, nn.GRU)): return 'rnn'
    if isinstance(module, nn.Linear): return 'linear'
    if isinstance(module, (nn.MultiheadAttention, ScaledDotProductAttention)): return 'attention'
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)): return 'norm'
    return 'other'

def get_dynamic_weights(layer_type):
    if layer_type == 'conv': return (0.3, 0.3, 0.4)
    if layer_type == 'attention': return (0.5, 0.3, 0.2)
    if layer_type == 'norm': return (0.6, 0.2, 0.2)
    return (0.5, 0.4, 0.1)

def ensemble_layer_similarity(A_act, B_act, layer_type='linear'):
    if A_act is None or B_act is None or A_act.size == 0 or B_act.size == 0:
        return {'score': 0.0, 'cka': 0.0, 'svcca': 0.0, 'rsa': 0.0}
    weights = get_dynamic_weights(layer_type)
    cka = linear_cka(A_act, B_act)
    sv = svcca_score(A_act, B_act)
    rs = rsa_score(A_act, B_act)
    score = float(weights[0]*cka + weights[1]*sv + weights[2]*rs)
    return {'score': score, 'cka': cka, 'svcca': sv, 'rsa': rs}

def neuron_cost_matrix(A_act, B_act):
    def _act_to_neuron_matrix(X):
        if X is None: return None
        X = np.asarray(X)
        if X.ndim == 4: return X.mean(axis=(2, 3))
        if X.ndim == 3: return X.mean(axis=1)
        if X.ndim == 2: return X
        if X.ndim == 1: return X.reshape(1, -1)
        if X.ndim > 1: return X.reshape(X.shape[0], -1)
        return X.reshape(1, -1)

    A_mat = _act_to_neuron_matrix(A_act)
    B_mat = _act_to_neuron_matrix(B_act)
    if A_mat is None or B_mat is None: return np.ones((1, 1))

    nA, nB = A_mat.shape[1], B_mat.shape[1]
    if nA == 0 or nB == 0: return np.ones((nA, nB))
    
    def _normalize_cols(M):
        M_c = M - M.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(M_c, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        return M_c / norms
    try:
        An, Bn = _normalize_cols(A_mat), _normalize_cols(B_mat)
        sim = np.abs(An.T @ Bn)
        cost = 1.0 - sim
        return np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    except Exception:
        return np.ones((nA, nB)) * 1.0

def find_permutation_matrix_from_cost(cost_np: np.ndarray):
    row_ind, col_ind = linear_sum_assignment(cost_np)
    return np.array(row_ind, dtype=int), np.array(col_ind, dtype=int)

class CognitiveFusionUnit(nn.Module):
    def __init__(self, lambda_align: float = 0.2, eps: float = 1e-9, threshold_center: float = 0.95, threshold_isolation: float = 0.6):
        super().__init__()
        self.lambda_align = float(lambda_align)
        self.eps = float(eps)
        self.threshold_center = float(threshold_center)
        self.threshold_isolation = float(threshold_isolation)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if not inputs or len(inputs) == 0: raise ValueError("CognitiveFusionUnit: empty input list.")
        if len(inputs) == 1: return inputs[0]
        V = [v for v in inputs if v is not None]
        if not V: raise ValueError("CognitiveFusionUnit: no valid inputs.")
        Vn = []
        for v in V:
            nrm = torch.norm(v.flatten(1), dim=-1, keepdim=True).view(v.shape[0], *([1] * (v.dim() - 1)))
            Vn.append(v / (torch.where(nrm == 0, torch.ones_like(nrm), nrm) + self.eps))
        sims = [torch.mean(torch.sum(Vn[i].flatten(1) * Vn[j].flatten(1), dim=-1)) for i in range(len(Vn)) for j in range(i + 1, len(Vn))]
        S = torch.mean(torch.stack(sims)) if sims else torch.tensor(1.0, device=V[0].device)
        if S > self.threshold_center:
            fused = torch.mean(torch.stack(Vn, dim=0), dim=0)
            nrm = torch.norm(fused.flatten(1), dim=-1, keepdim=True).view(fused.shape[0], *([1] * (fused.dim() - 1)))
            return fused / (torch.where(nrm == 0, torch.ones_like(nrm), nrm) + self.eps)
        c = torch.mean(torch.stack(Vn, dim=0), dim=0)
        nrm_c = torch.norm(c.flatten(1), dim=-1, keepdim=True).view(c.shape[0], *([1] * (c.dim() - 1)))
        c = c / (torch.where(nrm_c == 0, torch.ones_like(nrm_c), nrm_c) + self.eps)
        d_stack = torch.stack([1.0 - torch.sum(v.flatten(1) * c.flatten(1), dim=-1) for v in Vn], dim=0)
        lead_idx = torch.argmin(d_stack.mean(dim=1)).item()
        v_lead = Vn[lead_idx]
        if S <= self.threshold_isolation: return v_lead
        V_corr = []
        for i, v in enumerate(Vn):
            alpha = (self.lambda_align * (1.0 - d_stack[i])).view(-1, *([1] * (v.dim()-1)))
            v_corr = v + alpha * c
            nrm = torch.norm(v_corr.flatten(1), dim=-1, keepdim=True).view(v_corr.shape[0], *([1] * (v_corr.dim() - 1)))
            V_corr.append(v_corr / (torch.where(nrm == 0, torch.ones_like(nrm), nrm) + self.eps))
        w = torch.clamp(1.0 / (d_stack + self.eps), 0.0, 10.0)
        w = w / (torch.sum(w, dim=0, keepdim=True) + self.eps)
        w_expanded = w.view(w.shape[0], w.shape[1], *([1] * (Vn[0].dim() - 1)))
        weighted_sum = torch.sum(torch.stack([w_expanded[i] * V_corr[i] for i in range(len(V_corr))]), dim=0)
        w_sum_expanded = torch.sum(w, dim=0, keepdim=True).view(1, -1, *([1]*(Vn[0].dim()-1)))
        fused = (v_lead + weighted_sum) / (1.0 + w_sum_expanded)
        nrm_f = torch.norm(fused.flatten(1), dim=-1, keepdim=True).view(fused.shape[0], *([1] * (fused.dim() - 1)))
        return fused / (torch.where(nrm_f == 0, torch.ones_like(nrm_f), nrm_f) + self.eps)

def calculate_layer_importance(activations, out_grads):
    layer_imp, neuron_imp = {}, {}
    for layer_id, acts in activations.items():
        if layer_id in out_grads and acts is not None and out_grads[layer_id] is not None and acts.ndim > 1:
            saliency = np.abs(out_grads[layer_id] * acts)
            neuron_axis = 1
            neuron_imp[layer_id] = np.mean(saliency, axis=tuple(i for i in range(saliency.ndim) if i != neuron_axis))
            layer_imp[layer_id] = np.linalg.norm(neuron_imp[layer_id])
    return layer_imp, neuron_imp

def extract_layer_signature(module: nn.Module) -> dict:
    sig = {"type": type(module).__name__}
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        sig["out_dim"] = module.weight.shape[0]
        sig["in_dim"] = module.weight.shape[1]
        return sig
    elif isinstance(module, HybridEncoderLayerWrapper):
        return extract_layer_signature(module.layer) # Recursive call for wrapped layer

    # Added logic for analyzing complex blocks (e.g. BertLayer)
    # Trying to find the first and last linear layers to determine the dimensions.
    # This is a more reliable way than looking for the .weight attribute at the top level.
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if linear_layers:
        # Heuristics: in_dim - from the first linear layer, out_dim - from the last.
        # For most transformer blocks they will be the same (hidden_size).
        sig["in_dim"] = linear_layers[0].in_features
        sig["out_dim"] = linear_layers[-1].out_features

    return sig

def structural_similarity(sigA: dict, sigB: dict) -> float:
    if sigA["type"] != sigB["type"]: return 0.1
    in_dim_A, in_dim_B = sigA.get("in_dim", 0), sigB.get("in_dim", 0)
    out_dim_A, out_dim_B = sigA.get("out_dim", 0), sigB.get("out_dim", 0)
    if in_dim_A != in_dim_B or out_dim_A != out_dim_B:
        # Heavily penalize dimension mismatches, as this is key to merging in EMMv24.
        return 0.2 
    return 1.0

import json
from collections import defaultdict
try:
    from flask import Flask, render_template_string
    from flask_socketio import SocketIO
    import threading
    import time
    import logging

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    HAS_FLASK = True
    
    class EMM_Visualizer:
        def __init__(self, emm_instance):
            self.emm = emm_instance
            self.app = Flask(__name__)
            self.socketio = SocketIO(self.app, async_mode='threading')
            
            # Internal state for graph construction
            self.nodes = {} # {node_id: node_dict}
            self.edges = {} # {edge_id: edge_dict}
            
            self.COLUMN_WIDTH = 250  # Horizontal distance between model columns
            self.LEVEL_HEIGHT = 150  # Vertical distance between levels
            self.arch_keys = []      # Ordered list of architectures for column definitions

            self._setup_routes()

        def _calculate_layout_and_get_graph_data(self):
            """
            The central function for recalculating the entire graph layout.
            Determines the column order and the X, Y, and level coordinates for each node.
            """
            print("  -> VISUALIZER: Recalculating full graph layout...")
            
            # 1. Identify all unique architectures (columns)
            all_archs = set()
            for task_info in self.emm.task_to_arch_key.values():
                all_archs.add(task_info.split('-d')[0])
            
            # Add architectures from "ghost" nodes, if any
            for node in self.nodes.values():
                if 'Arch:' in node.get('title', ''):
                    try:
                        arch = node['title'].split("Arch: ")[1].strip()
                        all_archs.add(arch)
                    except IndexError:
                        pass

            # 2. Grouping architectures by common components for optimal order
            shared_parsers = defaultdict(list)
            for task, parser_key in self.emm.task_to_parser_key.items():
                arch_base = self.emm.task_to_arch_key[task].split('-d')[0]
                shared_parsers[parser_key].append(arch_base)

            shared_heads = defaultdict(list)
            for task, head_key in self.emm.task_to_head_key.items():
                arch_base = self.emm.task_to_arch_key[task].split('-d')[0]
                shared_heads[head_key].append(arch_base)

            groups = []
            processed_archs = set()

            # Ordering all_archs for a stable result
            for arch in sorted(list(all_archs)):
                if arch in processed_archs: continue
                
                current_group = {arch}
                # Search for related people through parsers
                for p_key, arch_list in shared_parsers.items():
                    if arch in arch_list:
                        current_group.update(arch_list)
                # Search for those connected through heads
                for h_key, arch_list in shared_heads.items():
                    if arch in arch_list:
                        current_group.update(arch_list)
                
                groups.append(sorted(list(current_group)))
                processed_archs.update(current_group)

            self.arch_keys = [item for sublist in groups for item in sublist]
            arch_to_x = {arch: i * self.COLUMN_WIDTH for i, arch in enumerate(self.arch_keys)}
            
            # 3. Calculate coordinates and levels for nodes
            layout_nodes = []
            max_level = 0
            
            # First, will process the nodes associated with the architectures (layers)
            for node_id, node in self.nodes.items():
                if node_id.startswith('layer_'):
                    try:
                        arch_base = node['title'].split("Arch: ")[1].strip()
                        layer_index = int(node['label'].replace('L', ''))
                        node['level'] = layer_index + 1
                        node['x'] = arch_to_x.get(arch_base, 0)
                        max_level = max(max_level, node['level'])
                    except (IndexError, ValueError, AttributeError):
                        node['level'] = 1 # Fallback
                        node['x'] = 0

            # Now let's process common nodes (parsers, heads, branches)
            for node_id, node in self.nodes.items():
                connected_archs = []
                if node_id.startswith('parser_'):
                    parser_key = node_id.replace('parser_', '')
                    connected_archs = list(set(shared_parsers.get(parser_key, [])))
                    node['level'] = 0
                elif node_id.startswith('head_'):
                    head_key = node_id.replace('head_', '')
                    connected_archs = list(set(shared_heads.get(head_key, [])))
                    node['level'] = max_level + 2 # Place the heads and branches at the bottom
                elif node_id.startswith('branch_'):
                    # Bind branches to their native encoder
                    branch_name = node_id.replace('branch_', '')
                    task_name = self.emm.branch_info.get(branch_name, {}).get('task_name')
                    if task_name and task_name in self.emm.task_to_arch_key:
                        arch_base = self.emm.task_to_arch_key[task_name].split('-d')[0]
                        node['x'] = arch_to_x.get(arch_base, 0)
                    node['level'] = max_level + 2

                if connected_archs:
                    # Center the common node between its columns
                    avg_x = sum(arch_to_x.get(arch, 0) for arch in connected_archs) / len(connected_archs)
                    node['x'] = avg_x
                
                # Set Y and fix the position
                if 'level' in node:
                    node['y'] = node['level'] * self.LEVEL_HEIGHT
                node['fixed'] = True # Blocking the position

                layout_nodes.append(node)

            return {'nodes': layout_nodes, 'edges': list(self.edges.values())}

        def log_full_graph_rebuild(self):
            """
            Recalculates the entire layout and sends it to the client as 'initial_graph'.
            """
            graph_data = self._calculate_layout_and_get_graph_data()
            
            pathway_data = {}
            for task_name, path in self.emm.task_native_paths.items():
                parser_key = self.emm.task_to_parser_key.get(task_name)
                head_key = self.emm.task_to_head_key.get(task_name)
                arch_key = self.emm.task_to_arch_key.get(task_name)
                if path and parser_key and head_key and arch_key:
                    pathway_data[task_name] = {
                        "path": path, "parser_key": parser_key,
                        "head_key": head_key, "arch_key": arch_key.split('-d')[0]
                    }
            
            graph_data['task_pathways'] = pathway_data
            self.socketio.emit('initial_graph', graph_data)
            self.socketio.sleep(0.1) # Give time for rendering

        def animate_reset(self):
            """Sends a command to reset the animation in the web interface."""
            self.socketio.emit('animate_reset')
            self.socketio.sleep(0.05) 

        def animate_step_by_step(self, updates: List[Dict]):
            """Sends a small update package for animation."""
            if not updates: return
            self.socketio.emit('animate_updates', {'updates': updates})
            # The pause depends on the number so that the animation is smooth
            pause_duration = 0.02 + len(updates) * 0.01
            self.socketio.sleep(min(pause_duration, 0.15))

        def log_unify_layer(self, emm_layer_id: str, donor_layer_id: str):
            """Marks an existing EMM node as unified."""
            node_id = f"layer_{emm_layer_id}"
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node['title'] = node.get('title', '') + f"\nUnified with: {donor_layer_id}"
                node['color'] = {'background': '#FFD700', 'border': '#FFA500'}
                node['borderWidth'] = 3
                # SEND ONLY THE UNIT UPDATE, WITHOUT REBUILDING THE LAYOUT
                self._send_update({'update_nodes': [node]})

        def log_cognitive_link(self, from_layer_id: str, to_layer_id: str, donor_arch_key_full: str):
            """Adds an edge representing a cognitive connection, with rigid node positioning."""
            from_node = f"layer_{from_layer_id}"
            to_node = f"layer_{to_layer_id}"
            
            update_pkg = {'add_nodes': [], 'add_edges': []}
            needs_rebuild = False
            
            arch_base_donor = donor_arch_key_full.split('-d')[0]
            if arch_base_donor not in self.arch_keys:
                 # If the donor architecture is completely new, we add it and plan a complete rebuild
                 self.arch_keys.append(arch_base_donor)
                 needs_rebuild = True

            if to_node not in self.nodes:
                # Create a "ghost" node, but IMMEDIATELY with the calculated coordinates
                layer_index = int(to_layer_id.split('_')[-1])
                level = layer_index + 1
                
                # Use a temporary calculation of X if there is no complete restructuring
                temp_arch_to_x = {arch: i * self.COLUMN_WIDTH for i, arch in enumerate(self.arch_keys)}
                x_pos = temp_arch_to_x.get(arch_base_donor, (len(self.arch_keys)-1) * self.COLUMN_WIDTH)
                y_pos = level * self.LEVEL_HEIGHT
                
                node = {
                    'id': to_node, 'label': f"L{layer_index}",
                    'title': f"Ghost Layer: {to_layer_id}\nArch: {arch_base_donor}",
                    'group': 'encoder', 'color': '#A9A9A9',
                    'x': x_pos, 'y': y_pos, 'level': level, 'fixed': True # Key addition
                }
                self.nodes[to_node] = node
                update_pkg['add_nodes'].append(node)

            edge_id = f"{from_node}<->{to_node}_coglink"
            if edge_id not in self.edges:
                edge = {
                    'id': edge_id, 'from': from_node, 'to': to_node, 'dashes': [5, 5], 
                    'color': '#BA55D3', 'title': 'Cognitive Link', 'arrows': 'to, from',
                    'smooth': {'type': 'curvedCW', 'roundness': 0.2}
                }
                self.edges[edge_id] = edge
                update_pkg['add_edges'].append(edge)

            if needs_rebuild:
                # If a new architecture is added, the old X calculations are incorrect and a complete rebuild is required.
                self.log_full_graph_rebuild()
            elif update_pkg['add_nodes'] or update_pkg['add_edges']:
                # If simply added a node and an edge within the framework of known architectures
                self._send_update(update_pkg)

        def _get_html_template(self):
            return """
            <!doctype html>
            <html>
            <head>
                <title>EMM Live Structure Visualizer</title>
                <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
                <style>
                html, body { font-family: Arial, sans-serif; margin: 0; padding: 0; height: 100%; width: 100%; overflow: hidden; background-color: #2D2D2D; color: #EAEAEA;}
                #graph { width: 100%; height: 100%; }
                .panel { position: absolute; background: rgba(0,0,0,0.75); padding: 15px; border-radius: 8px; border: 1px solid #555; z-index: 10; }
                .info-panel { top: 10px; left: 10px; max-width: 350px; max-height: 95%; overflow-y: auto; }
                h1, h2 { margin: 0 0 10px 0; padding: 0; border-bottom: 1px solid #555; padding-bottom: 5px; }
                .legend-item { display: flex; align-items: center; margin-bottom: 5px; text-align: left; }
                .legend-color, .legend-line { margin-right: 10px; flex-shrink: 0; }
                .legend-color { width: 15px; height: 15px; border-radius: 50%; border: 1px solid #fff; }
                .legend-line { width: 25px; height: 2px; border: none; }
                .task-item { padding: 5px; cursor: pointer; border-radius: 3px; }
                .task-item:hover { background-color: #555; }
                </style>
            </head>
            <body>
                <div class="panel info-panel">
                    <h1>EMM Visualizer</h1>
                    <h2>Nodes Legend</h2>
                    <div class="legend-item"><div class="legend-color" style="background-color: #97C2FC;"></div> Parser</div>
                    <div class="legend-item"><div class="legend-color" style="background-color: #F5A623;"></div> Merged Layers</div>
                    <div class="legend-item"><div class="legend-color" style="background-color: #4CAF50;"></div> Layer</div>
                    <div class="legend-item"><div class="legend-color" style="background-color: #F44336;"></div> Head</div>
                    <div class="legend-item"><div class="legend-color" style="background-color: #E040FB;"></div> Hierarchical Skill</div>
                    <h2 id="task-pathways-title">Task Pathways</h2>
                    <div id="task-list"></div>
                </div>

                <div id="graph"></div>

                <script>
                const socket = io();
                const container = document.getElementById('graph');
                const nodes = new vis.DataSet([]);
                const edges = new vis.DataSet([]);
                const data = { nodes: nodes, edges: edges };
                let taskPathways = {};

                const options = {
                    nodes: {
                        shape: 'dot', size: 20, font: { color: '#ffffff', size: 14 }, borderWidth: 2,
                        color: { border: '#FFFFFF', highlight: { border: '#FF1744', background: '#D50000' } }
                    },
                    edges: {
                        width: 2, arrows: 'to',
                        color: { inherit: 'from', highlight: '#FF1744' },
                        smooth: {
                            enabled: true,
                            type: "cubicBezier",
                            forceDirection: "vertical",
                            roundness: 0.4
                        }
                    },
                    physics: {
                        enabled: false // Turn off physics, since we set the coordinates manually
                    },
                    layout: {
                        hierarchical: {
                            enabled: true,
                            direction: 'UD', // Top-down
                            sortMethod: 'directed', // Use the levels that we set
                            nodeSpacing: 150,
                            levelSeparation: 170,
                            treeSpacing: 250
                        }
                    },
                    groups: {
                        parser: { color: '#97C2FC', shape: 'database', size: 25 }, branch: { color: '#F5A623', shape: 'star', size: 22 },
                        hierarchical_branch: { color: '#E040FB', shape: 'hexagon', size: 28 }, encoder: { color: '#4CAF50', shape: 'dot', size: 18 },
                        head: { color: '#F44336', shape: 'triangle', size: 22 }
                    },
                    interaction: { hover: true, tooltipDelay: 200 }
                };
                const network = new vis.Network(container, data, options);

                network.on("click", (params) => { if (params.nodes.length === 0 && params.edges.length === 0) network.unselectAll(); });

                function updateTaskLists() {
                    const taskListDiv = document.getElementById('task-list');
                    taskListDiv.innerHTML = '';
                    
                    const sortedTasks = Object.keys(taskPathways).sort();
                    if (sortedTasks.length === 0) return;

                    sortedTasks.forEach(taskName => {
                        const item = document.createElement('div');
                        item.className = 'task-item';
                        item.innerText = taskName;
                        item.onclick = () => highlightPath(taskName);
                        taskListDiv.appendChild(item);
                    });
                }

                function highlightPath(taskName) {
                    network.unselectAll();
                    const pathInfo = taskPathways[taskName];
                    if (!pathInfo) return;

                    const nodeIds = [`parser_${pathInfo.parser_key}`];
                    const edgeIds = [];
                    
                    for (let i = 0; i < pathInfo.path.length; i++) {
                        const layerId = pathInfo.path[i];
                        nodeIds.push(`layer_${layerId}`);
                        const fromNode = (i === 0) ? `parser_${pathInfo.parser_key}` : `layer_${pathInfo.path[i-1]}`;
                        edgeIds.push(`${fromNode}->layer_${layerId}`);
                    }
                    if (pathInfo.path.length > 0) {
                        nodeIds.push(`head_${pathInfo.head_key}`);
                        const lastLayerId = pathInfo.path[pathInfo.path.length - 1];
                        edgeIds.push(`layer_${lastLayerId}->head_${pathInfo.head_key}`);
                    }
                    network.setSelection({ nodes: nodeIds, edges: edgeIds }, { highlightEdges: true });
                }

                socket.on('animate_reset', () => {
                    network.unselectAll();
                    const allNodeIds = nodes.getIds();
                    if (allNodeIds.length > 0) {
                        nodes.update(allNodeIds.map(id => ({id: id, color: null})));
                    }
                    const allEdgeIds = edges.getIds();
                    if (allEdgeIds.length > 0) {
                        edges.update(allEdgeIds.map(id => ({id: id, color: null, width: 2})));
                    }
                });

                socket.on('animate_updates', (data) => {
                    const updates = data.updates || [];
                    updates.forEach(item => {
                        if (item.type === 'node' && nodes.get(item.id)) {
                            nodes.update({id: item.id, color: { background: '#00BFFF', border: '#1E90FF' }});
                        } else if (item.type === 'edge' && edges.get(item.id)) {
                            edges.update({id: item.id, color: { color: '#00BFFF' }, width: 4});
                        }
                    });
                });
                
                socket.on('connect', () => socket.emit('request_initial_graph'));

                socket.on('initial_graph', (graphData) => {
                    nodes.clear(); edges.clear();
                    nodes.add(graphData.nodes);
                    edges.add(graphData.edges);
                    taskPathways = graphData.task_pathways || {};
                    updateTaskLists();
                    // After loading new data with coordinates, center the view
                    setTimeout(() => network.fit(), 200);
                });

                socket.on('update', (updateData) => {
                    if (updateData.add_nodes) nodes.add(updateData.add_nodes);
                    if (updateData.add_edges) edges.add(updateData.add_edges);
                    if (updateData.update_nodes) nodes.update(updateData.update_nodes);
                    if (updateData.remove_nodes) nodes.remove(updateData.remove_nodes);
                    if (updateData.remove_edges) edges.remove(updateData.remove_edges);
                    if (updateData.task_pathways) {
                        Object.assign(taskPathways, updateData.task_pathways);
                        updateTaskLists();
                    }
                });
                </script>
            </body>
            </html>
            """
        
        def _setup_routes(self):
            @self.app.route('/')
            def index():
                return render_template_string(self._get_html_template())

            @self.socketio.on('request_initial_graph')
            def handle_initial_graph_request():
                # On the first request simply send the current state
                self.log_full_graph_rebuild()

        def run(self):
            print("\n" + "="*80)
            print(" Starting EMM Visualizer... Open http://127.0.0.1:5001 in your browser. ".center(80))
            print("="*80)
            server_thread = threading.Thread(target=lambda: self.socketio.run(self.app, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True))
            server_thread.daemon = True
            server_thread.start()

        def _send_update(self, update_data):
            self.socketio.emit('update', update_data)
            self.socketio.sleep(0.05) 
        
        def log_assimilate_parser(self, parser_key, vocab_size, embed_dim):
            node_id = f"parser_{parser_key}"
            if node_id not in self.nodes:
                node = {
                    'id': node_id, 'label': f"Parser\n{parser_key.split('-v')[0]}",
                    'title': f"ID: {parser_key}\nVocab: {vocab_size}\nDim: {embed_dim}",
                    'group': 'parser'
                }
                self.nodes[node_id] = node
            # The graph rebuild will be called later in `log_assimilate_encoder`

        def log_assimilate_encoder(self, arch_key, layer_ids, parser_key):
            """
            V2.0: Ensures that all native links are created,
            including parser links, even if nodes already exist.
            Assumes that `layer_ids` is a complete, sorted list of layers for `arch_key`.
            """
            if not layer_ids:
                print("  -> VISUALIZER WARNING: log_assimilate_encoder called with empty layer_ids. Skipping.")
                return

            arch_base = arch_key.split('-d')[0]

            # Step 1: Make sure all nodes for layers exist.
            for i, layer_id in enumerate(layer_ids):
                node_id = f"layer_{layer_id}"
                if node_id not in self.nodes:
                    # Extracting the real index from the layer ID
                    try:
                        layer_index = int(layer_id.split('_')[-1])
                        label = f"L{layer_index}"
                    except (ValueError, IndexError):
                        label = f"L{i}" # Fallback
                    
                    node = {'id': node_id, 'label': label, 'title': f"Layer: {layer_id}\nArch: {arch_base}", 'group': 'encoder'}
                    self.nodes[node_id] = node
            
            # Step 2: Ensure all native edges are created.
            # Connection from the parser to the first layer (layer_0).
            parser_node_id = f"parser_{parser_key}"
            first_layer_node_id = f"layer_{layer_ids[0]}"
            edge_id_parser = f"{parser_node_id}->{first_layer_node_id}"

            if parser_node_id in self.nodes and first_layer_node_id in self.nodes:
                if edge_id_parser not in self.edges:
                    edge = {'id': edge_id_parser, 'from': parser_node_id, 'to': first_layer_node_id, 'title': 'Native Data Flow'}
                    self.edges[edge_id_parser] = edge
            else:
                print(f"  -> VISUALIZER WARNING: Cannot create edge from parser '{parser_node_id}' to layer '{first_layer_node_id}'. One or both nodes do not exist.")


            # Connections between layers.
            for i in range(len(layer_ids) - 1):
                from_node_id = f"layer_{layer_ids[i]}"
                to_node_id = f"layer_{layer_ids[i+1]}"
                edge_id_layer = f"{from_node_id}->{to_node_id}"
                if from_node_id in self.nodes and to_node_id in self.nodes and edge_id_layer not in self.edges:
                    edge = {'id': edge_id_layer, 'from': from_node_id, 'to': to_node_id, 'title': 'Native Data Flow'}
                    self.edges[edge_id_layer] = edge
            
            # After updating the structure, we initiate a complete restructuring of the layout.
            self.log_full_graph_rebuild()

        def log_add_task(self, task_name, head_key, last_layer_id):
            node_id = f"head_{head_key}"
            serviced_tasks = self.nodes.get(node_id, {}).get('serviced_tasks', [])
            if task_name not in serviced_tasks: serviced_tasks.append(task_name)
            
            if node_id not in self.nodes:
                node = {'id': node_id, 'label': f"Head\n{head_key.split('-')[0]}", 'group': 'head', 'serviced_tasks': serviced_tasks}
                self.nodes[node_id] = node
            
            self.nodes[node_id]['title'] = f"ID: {head_key}\nTasks: {', '.join(serviced_tasks)}"
            
            from_node = f"layer_{last_layer_id}"
            edge_id = f"{from_node}->{node_id}"
            if edge_id not in self.edges:
                edge = {'id': edge_id, 'from': from_node, 'to': node_id}
                self.edges[edge_id] = edge
            
            # Starting a rebuild, as adding a head may change the grouping
            self.log_full_graph_rebuild()

        def log_absorb_branch(self, action_item):
            action = action_item['action']
            name_m = action_item.get('name') or action_item.get('name_m')
            
            task_name_m = self.emm.branch_info.get(name_m, {}).get('task_name')
            if not task_name_m or not self.emm.task_native_paths.get(task_name_m): 
                return

            last_layer_id_m = self.emm.task_native_paths[task_name_m][-1]
            from_node_id_m = f"layer_{last_layer_id_m}"

            if action == 'absorb_new':
                node_id = f"branch_{name_m}"
                self.nodes[node_id] = {'id': node_id, 'label': f"Skill\n{name_m}", 'title': f"ID: {name_m}\nType: Standalone", 'group': 'branch'}
                edge = {'id': f"{from_node_id_m}->{node_id}", 'from': from_node_id_m, 'to': node_id, 'arrows': 'to', 'color': '#F5A623'}
                self.edges[edge['id']] = edge

            elif action == 'merge_ta':
                name_s = action_item['name_s']
                node_id_s = f"branch_{name_s}"
                if node_id_s in self.nodes:
                    self.nodes[node_id_s]['title'] += f"\nMerged: {name_m}"
                    self.nodes[node_id_s]['size'] = self.nodes[node_id_s].get('size', 22) + 2
                    edge = {'id': f"{from_node_id_m}->{node_id_s}_contrib", 'from': from_node_id_m, 'to': node_id_s, 'dashes': True, 'color': '#FF9800', 'arrows': 'to'}
                    self.edges[edge['id']] = edge

            elif action == 'refactor_into_hierarchical':
                name_s = action_item['name_s']
                parent_name = action_item.get('parent_name') or name_s
                h_node_id = f"branch_{parent_name}"
                
                # Remove old branches that were merged
                old_node_id_s = f"branch_{name_s}"
                if old_node_id_s in self.nodes:
                    del self.nodes[old_node_id_s]
                    edges_to_remove = [eid for eid, edge in self.edges.items() if edge['to'] == old_node_id_s or edge['from'] == old_node_id_s]
                    for eid in edges_to_remove:
                        if eid in self.edges: del self.edges[eid]

                # Adding a new hierarchical node
                self.nodes[h_node_id] = {'id': h_node_id, 'label': f"Trunk\n{parent_name}", 'title': f"ID: {parent_name}\nType: Hierarchical", 'group': 'hierarchical_branch'}
                
                # Adding a connection from the encoder to the node
                task_s = self.emm.branch_info[parent_name]['task_name']
                last_layer_s = self.emm.task_native_paths[task_s][-1]
                edge_trunk = {'id': f"layer_{last_layer_s}->{h_node_id}", 'from': f"layer_{last_layer_s}", 'to': h_node_id, 'arrows': 'to', 'color': '#E040FB'}
                self.edges[edge_trunk['id']] = edge_trunk

                # Adding a connection from the second model encoder
                edge_contrib = {'id': f"{from_node_id_m}->{h_node_id}_contrib", 'from': from_node_id_m, 'to': h_node_id, 'dashes': True, 'color': '#FF9800', 'arrows': 'to'}
                self.edges[edge_contrib['id']] = edge_contrib

                # Adding specific branches as child nodes
                hier_branch_module = next((b for b_dict in self.emm.branched_block.branches_by_arch.values() for name, b in b_dict.items() if name == parent_name), None)
                if isinstance(hier_branch_module, HierarchicalBranch):
                    for spec_name in hier_branch_module.specific_branches.keys():
                        spec_node_id = f"branch_{spec_name}"
                        self.nodes[spec_node_id] = {'id': spec_node_id, 'label': f"Spec\n{spec_name.replace('_specific', '')}", 'title': f"ID: {spec_name}", 'group': 'branch', 'size': 18}
                        spec_edge_id = f"{h_node_id}->{spec_node_id}"
                        self.edges[spec_edge_id] = {'id': spec_edge_id, 'from': h_node_id, 'to': spec_node_id, 'arrows': 'to'}
            
            # After all the manipulations with the branches, we rebuild the layout
            self.log_full_graph_rebuild()

except ImportError:
    HAS_FLASK = False
    class EMM_Visualizer: 
        def __init__(self, emm_instance): pass
        def run(self): pass
        def log_assimilate_parser(self, *args, **kwargs): pass
        def log_assimilate_encoder(self, *args, **kwargs): pass
        def log_add_task(self, *args, **kwargs): pass
        def log_absorb_branch(self, *args, **kwargs): pass
        def log_full_graph_rebuild(self, *args, **kwargs): pass
        def log_unify_layer(self, *args, **kwargs): pass
        def log_cognitive_link(self, *args, **kwargs): pass

def set_seed(seed: int):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multiple GPUs
        # These settings may slow down training, but ensure determinism.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

try:
    from sklearn.cross_decomposition import CCA
    from sklearn.linear_model import LinearRegression
    from scipy.stats import spearmanr
    HAS_SCIPY_SKLEARN = True
except ImportError:
    HAS_SCIPY_SKLEARN = False
    print("Warning: scikit-learn or scipy not installed. SVCCA/RSA metrics will be disabled.")

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not installed. Graph data support will be disabled.")

try:
    from sklearn.metrics import f1_score, r2_score, mean_squared_error
    HAS_SKLEARN_METRICS = True
except ImportError:
    HAS_SKLEARN_METRICS = False
    print("Warning: scikit-learn not installed. F1/MSE/R2 metrics will be disabled.")

def get_d_model_from_config(config: Any) -> int:
    """Generic extracts the d_model dimension from the model configuration."""
    # List of possible names for d_model in order of precedence
    possible_keys = ['hidden_size', 'd_model', 'n_embd', 'hidden_dim']
    for key in possible_keys:
        if hasattr(config, key):
            return getattr(config, key)
    # If nothing is found, return the default value or raise an error.
    print("  -> WARNING: Could not automatically determine d_model. Defaulting to 768.")
    return 768

def get_architecture_key(model_or_config: Any, for_parser: bool = False) -> str:
    config_hf = getattr(model_or_config, 'config', model_or_config)
    arch_type = getattr(config_hf, 'model_type', 'unknown')
    d_model = get_d_model_from_config(config_hf)
    num_layers = getattr(config_hf, 'num_hidden_layers', 'N/A')
    if for_parser:
        vocab_size = getattr(config_hf, 'vocab_size', 'N/A')
        return f"{arch_type}-d{d_model}-l{num_layers}-v{vocab_size}"
    else:
        return f"{arch_type}-d{d_model}-l{num_layers}"

def optimal_transport_map(C: torch.Tensor, epsilon: float = 0.1, n_iters: int = 10):
    """
    Computes the transport plan using the Sinkhorn algorithm.
    C: Cost matrix (e.g. cosine distance between embeddings).
    Added regularization for numerical stability.
    """
    u = torch.ones(C.shape[0], device=C.device) / C.shape[0]
    v = torch.ones(C.shape[1], device=C.device) / C.shape[1]
    
    K = torch.exp(-C / epsilon)
    K = torch.clamp(K, min=1e-10) # Prevent underflow
    
    for _ in range(n_iters):
        u = 1. / (K @ v + 1e-10) # Prevent division by zero
        v = 1. / (K.T @ u + 1e-10)
        
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return P

def find_permutation_matrix(W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    """
    Finds a permutation matrix P for aligning neurons in the second layer W2 with the first layer W1.
    Uses the Sinkhorn algorithm to solve the assignment problem.
    """
    # The cost matrix is ​​the pairwise cosine distance between rows W1 and W2.
    cost_matrix = 1 - F.cosine_similarity(W1.unsqueeze(1), W2.unsqueeze(0), dim=2)
    
    P_soft = optimal_transport_map(cost_matrix, epsilon=0.01, n_iters=20)
    
    # Transform the soft matrix into a hard (0/1) permutation matrix
    P = torch.zeros_like(P_soft)
    indices = torch.argmax(P_soft, dim=1)
    P[torch.arange(P.shape[0]), indices] = 1
    
    return P

class SimpleGA:
    """A simple genetic algorithm for finding optimal coefficients."""
    def __init__(self, num_params: int, pop_size: int = 20, generations: int = 20, mutation_rate: float = 0.2):
        self.num_params = num_params
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = torch.rand(self.pop_size, self.num_params) * 0.5 + 0.25 

    def evaluate(self, population, W1, W2, W_target):
        fitness = torch.zeros(self.pop_size, device=W1.device)
        for i, individual in enumerate(population):
            alpha = individual[0]
            W_merged = (1 - alpha) * W1 + alpha * W2
            fitness[i] = -F.mse_loss(W_merged, W_target)
        return fitness

    def select(self, fitness):
        parents = torch.zeros_like(self.population)
        for i in range(self.pop_size):
            idx1, idx2 = random.sample(range(self.pop_size), 2)
            winner_idx = idx1 if fitness[idx1] > fitness[idx2] else idx2
            parents[i] = self.population[winner_idx]
        return parents

    def crossover(self, parents):
        """Average crossover."""
        offspring = torch.zeros_like(self.population)
        for i in range(self.pop_size):
            # Select two random parents
            p1, p2 = parents[torch.randperm(self.pop_size)[:2]]
            # Their descendant is their average
            offspring[i] = (p1 + p2) / 2
        return offspring

    def mutate(self, offspring):
        """Gaussian mutation."""
        for i in range(self.pop_size):
            if random.random() < self.mutation_rate:
                # Add some random noise
                offspring[i] += torch.randn_like(offspring[i]) * 0.05
        return torch.clamp(offspring, 0.0, 1.0) # The coefficients must be in [0, 1]

    def run(self, W1, W2, W_target):
        for _ in range(self.generations):
            fitness = self.evaluate(self.population, W1, W2, W_target)
            parents = self.select(fitness)
            offspring = self.crossover(parents)
            self.population = self.mutate(offspring)
        
        final_fitness = self.evaluate(self.population, W1, W2, W_target)
        best_idx = torch.argmax(final_fitness)
        return self.population[best_idx]

# Universal dataset analysis function
def analyze_dataset(dataset: Dataset) -> Dict[str, Any]:
    """
    Analyzes a dataset to extract metadata such as class counts.
    Implements a three-tier logic: metadata -> sample analysis -> fallback.
    """
    metadata = {}
    print("Analyzing dataset to determine task properties...")
    try:
        # Level 1: Checking Standard Metadata Attributes
        if hasattr(dataset, 'classes'):
            metadata['num_classes'] = len(dataset.classes)
        elif hasattr(dataset, 'features') and 'label' in dataset.features and hasattr(dataset.features['label'], 'num_classes'):
            metadata['num_classes'] = dataset.features['label'].num_classes
        else:
            # Level 2: Analyze multiple samples to determine the maximum label index
            print("  -> Metadata not found. Analyzing samples to infer num_classes...")
            labels = set()
            max_label_val = -1
            # Check up to 1000 samples
            for i in range(min(1000, len(dataset))):
                try:
                    # A universal way to access data even if __getitem__ is complex
                    item = dataset[i]
                    # Look for the label (usually the second element in the tuple)
                    label = item[1]
                    if isinstance(label, torch.Tensor):
                        # For tensors (can also be for regression)
                        if label.numel() == 1:
                            val = label.item()
                            if val == int(val): # Check that this is an integer label
                                labels.add(int(val))
                                max_label_val = max(max_label_val, int(val))
                    elif isinstance(label, int):
                        labels.add(label)
                        max_label_val = max(max_label_val, label)
                except (IndexError, TypeError, KeyError):
                    # Skipping samples that do not match the format (data, label)
                    continue
            
            if labels:
                 # +1, since classes are usually numbered from 0
                metadata['num_classes'] = max_label_val + 1
            else:
                # Level 3: Fallback for regression or unknown formats
                 print("  -> Could not infer discrete labels. Assuming regression task.")
                 metadata['num_classes'] = 1 
                 
        print(f"Dataset analysis complete. Found num_classes: {metadata.get('num_classes')}")
    except Exception as e:
        print(f"Could not analyze dataset due to an error: {e}. Defaulting num_classes to 10.")
        metadata['num_classes'] = 10
        
    return metadata

# Multimodal encoder with automatic registration and conversion
from torchvision.transforms.functional import resize, rgb_to_grayscale
class MultiModalEncoder(nn.Module):
    """
    A "pure" executor for EMM. It does not own layers, but only performs
    layer-by-layer graph signal propagation, passing to each layer node
    the full set of activations from the previous layer for hybrid fusion.
    """
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.visualizer = None

    def forward(self, 
                x: Any, 
                task_name: str,
                task_to_arch_key: Dict[str, str],
                task_native_paths: Dict[str, List[str]],
                embedding_layers: nn.ModuleDict,
                hybrid_layers: nn.ModuleDict,
                legacy_encoders: nn.ModuleDict,
                external_activations: Optional[Dict[str, torch.Tensor]] = None,
                trace: bool = False
               ) -> Tuple[Dict[str, torch.Tensor], List[str], Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        
        activation_trace = []
        path_log = []
        enrichment_horizon = EMM_CONFIG.get("ENRICHMENT_HORIZON", 3)

        # Parallel Cascade mode
        if external_activations is None:
            raise ValueError("MultiModalEncoder in parallel mode requires 'external_activations'.")

        level_activations = external_activations.copy()
        
        if not hybrid_layers: return {}, [], level_activations, activation_trace

        max_depth = max((len(p) for p in task_native_paths.values() if p), default=0)

        for depth in range(max_depth):
            next_level_activations = {}
            edges_this_level, nodes_this_level = [], []
            
            target_layers_at_this_depth = {
                layer_id: layer for layer_id, layer in hybrid_layers.items() if f"_layer_{depth}" in layer_id
            }

            for layer_id, wrapped_layer in target_layers_at_this_depth.items():
                arch = layer_id.split('_layer_')[0]
                native_source_id = f"{arch}_layer_{depth-1}" if depth > 0 else next((ak for ak in level_activations if ak.startswith(arch) and ak.endswith('_embedding')), None)
                
                # Collect inputs based on the "knowledge horizon"
                # Collect all activations calculated up to the current level
                available_activations = {
                    act_id: tensor for act_id, tensor in level_activations.items()
                    if not act_id.endswith(f"_layer_{depth}") # Exclude "colleagues" from the same level
                }
                
                possible_inputs = {}
                # 1. Always add a native input if it exists.
                if native_source_id and native_source_id in available_activations:
                    possible_inputs[native_source_id] = available_activations[native_source_id]

                # 2. Adding enrichment inputs from the final layers of other models
                for t_name, path in task_native_paths.items():
                    if not path: continue
                    path_len = len(path)
                    # Condition: the path has ended (depth is greater than or equal to the path length)
                    # And yet it ended not too "long ago" (within the horizon)
                    if depth >= path_len and (depth - path_len) < enrichment_horizon:
                        final_layer_id = path[-1]
                        if final_layer_id in available_activations:
                            possible_inputs[final_layer_id] = available_activations[final_layer_id]
                
                # 3. Adding parser activations to the very first layers
                if depth == 0:
                     for act_id, tensor in level_activations.items():
                         if act_id.endswith("_embedding"):
                             possible_inputs[act_id] = tensor

                if not possible_inputs: continue

                output_tensor = wrapped_layer(possible_inputs, native_source_id)

                if output_tensor is not None:
                    output_tensor_main = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                    next_level_activations[layer_id] = output_tensor_main
                    
                    if trace:
                        for parent_id in possible_inputs:
                             from_node_base = parent_id.replace('_embedding', '')
                             from_node = f"parser_{from_node_base}" if parent_id.endswith("_embedding") else f"layer_{parent_id}"
                             is_native = (native_source_id is not None and parent_id == native_source_id)
                             edge_id = f"{from_node}->layer_{layer_id}" if is_native else f"{from_node}->layer_{layer_id}_cross"
                             edges_this_level.append({'type': 'edge', 'id': edge_id})
                        nodes_this_level.append({'type': 'node', 'id': f"layer_{layer_id}"})
            
            if trace:
                activation_trace.extend(edges_this_level)
                activation_trace.extend(nodes_this_level)

            if not next_level_activations: break
            
            level_activations.update(next_level_activations)
        
        return {}, path_log, level_activations, activation_trace

# Adaptive branch with low-rank decomposition
class AdaptiveBranch(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, device: str = "cpu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_rank = rank
        
        # Low-ranking matrices
        self.slices = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, rank).to(device)),
            nn.Parameter(torch.randn(rank, out_dim).to(device))
        ])
        
        # "SMART" INITIALIZATION
        with torch.no_grad():
            # Initialize the first matrix using the standard Kaiming method, and the second to near-zero values.
            # This ensures that the branch output will be close to zero at the start.
            nn.init.kaiming_uniform_(self.slices[0], a=math.sqrt(5))
            self.slices[1].data.normal_(mean=0.0, std=0.01)

        self.alpha = nn.Parameter(torch.tensor(1.0).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.slices[0].device)
        A, B = self.slices
        branch_out = x @ A @ B * self.alpha
        return branch_out

    def resize(self, in_dim: int, out_dim: int, device: str):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
    def grow(self, k: int):
        if self.in_dim == 0 or self.out_dim == 0: return
        max_possible_rank = min(self.in_dim, self.out_dim)
        new_rank = min(self.total_rank + k, max_possible_rank)
        if new_rank <= self.total_rank: return
        A, B = self.slices
        new_A = nn.Parameter(torch.randn(self.in_dim, new_rank - self.total_rank).to(A.device))
        new_B = nn.Parameter(torch.randn(new_rank - self.total_rank, self.out_dim).to(B.device))
        self.slices = nn.ParameterList([
            nn.Parameter(torch.cat([A, new_A], dim=1)),
            nn.Parameter(torch.cat([B, new_B], dim=0))
        ])
        self.total_rank = new_rank

# Block structure with branches
class BranchedBlock(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.branches_by_arch = nn.ModuleDict()
        self.branch_d_models = {}

    def add_branch(self, name: str, arch_key: str, rank: int, d_model_branch: int):
        if arch_key not in self.branches_by_arch:
            self.branches_by_arch[arch_key] = nn.ModuleDict()
        
        new_branch = None
        if name not in self.branches_by_arch[arch_key]:
            print(f"  -> Adding branch '{name}' (d_model={d_model_branch}) for arch '{arch_key}'")
            new_branch = AdaptiveBranch(d_model_branch, d_model_branch, rank, device=self.device)
            self.branches_by_arch[arch_key][name] = new_branch
            self.branch_d_models[name] = d_model_branch
        return new_branch if new_branch is not None else self.branches_by_arch[arch_key][name]

    def remove_branch(self, name: str) -> bool:
        found_and_removed = False
        if name in self.branch_d_models:
            del self.branch_d_models[name]
        for arch_key in list(self.branches_by_arch.keys()):
            if name in self.branches_by_arch[arch_key]:
                del self.branches_by_arch[arch_key][name]
                if not self.branches_by_arch[arch_key]:
                    del self.branches_by_arch[arch_key]
                found_and_removed = True
                break
        return found_and_removed

    def forward(self, x_native: torch.Tensor, current_arch_key: str) -> torch.Tensor:
        """Works directly with native vectors without projectors."""
        total_output = torch.zeros_like(x_native)
        native_dim = x_native.shape[-1]
        
        if current_arch_key not in self.branches_by_arch:
            return total_output

        activated_branches_count = 0
        for name, branch in self.branches_by_arch[current_arch_key].items():
            branch_dim = self.branch_d_models.get(name)
            
            if branch_dim == native_dim:
                branch_out = branch(x_native)
                total_output += branch_out
                activated_branches_count += 1
        
        if activated_branches_count > 1:
            total_output /= activated_branches_count
            
        return total_output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_p = dropout
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        
        # This layer will prepare Q, K, V for attention
        self.in_proj = nn.Linear(d_model, d_model * 3)
        # Output layer
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_bias: Optional[torch.Tensor] = None):
        # query, key, value are here for compatibility, but we will use only query because it is self-attention.
        B, T, C = query.shape # Batch, Sequence Length, Channels (d_model)
        
        # 1. Project Q, K, V and divide into heads
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        
        q = q.view(B, T, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        
        # 2. Use the built-in, fast implementation of attention
        # is_causal=False for encoder, True for decoder
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_bias, 
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False 
        )
        
        # 3. Put the heads back together and project them onto the exit.
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, None # Return None for compatibility with MultiheadAttention

# Prioritized Rehearsal Buffer with Subsampling
class RehearsalBuffer:
    def __init__(self, capacity: int = 1000, subsample_rate: float = 0.1, device: str = "cpu"):
        self.capacity = capacity
        self.subsample_rate = subsample_rate
        self.task_complexity = defaultdict(lambda: 1.0)
        self.buffer = []
        self.priorities = []
        self.device = device

    def add(self, x: torch.Tensor, y: torch.Tensor, task: str, loss: float, model: Optional[nn.Module] = None):
        if random.random() > self.subsample_rate: return
        priority = loss
        if model:
            model.eval()
            with torch.no_grad():
                device = next(model.parameters()).device
                if isinstance(x, dict):
                    x_dev = {k: v.to(device) for k, v in x.items()}
                else:
                    x_dev = x.to(device)
                out = model(x_dev, task)

                if isinstance(out, dict):
                    out = out.get('out', next(iter(out.values())))

                l = F.cross_entropy(out, y.to(out.device))
                priority = l.item()

        self.task_complexity[task] = 0.9 * self.task_complexity[task] + 0.1 * priority
        total_complexity = sum(self.task_complexity.values())
        if total_complexity > 0:
            adjusted_capacity = int(self.capacity * self.task_complexity[task] / total_complexity)
        else:
            # Fallback if complexities are zero
            num_tasks = len(self.task_complexity) if len(self.task_complexity) > 0 else 1
            adjusted_capacity = self.capacity // num_tasks
    
        adjusted_capacity = max(adjusted_capacity, 50)

        if isinstance(x, dict):
            item_to_store = ({k: v.cpu() for k, v in x.items()}, y.cpu(), task)
        else:
            item_to_store = (x.cpu(), y.cpu(), task)

        if len(self.buffer) < self.capacity:
            self.buffer.append(item_to_store)
            self.priorities.append(priority)
        else:
            min_idx = min(range(len(self.priorities)), key=self.priorities.__getitem__)
            if priority > self.priorities[min_idx]:
                self.buffer[min_idx] = item_to_store
                self.priorities[min_idx] = priority
    
        # Pruning if total capacity is exceeded
        while len(self.buffer) > self.capacity:
            min_idx = min(range(len(self.priorities)), key=self.priorities.__getitem__)
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)

    def sample(self, batch_size: int, epsilon: float = 0.1): # Adding epsilon
        if not self.buffer:
            return []
        
        k = min(batch_size, len(self.buffer))
        indices_to_sample = []

    # SAMPLING LOGIC
        for _ in range(k):
            if random.random() < epsilon:
                # With probability epsilon we select a random sample (research)
                indices_to_sample.append(random.randint(0, len(self.buffer) - 1))
            else:
                # With probability 1-epsilon we choose by priority (operation)
                if not hasattr(self, '_cached_probs') or self._cached_probs is None:
                    probs = torch.tensor(self.priorities, dtype=torch.float32) + 1e-8
                    self._cached_probs = probs / probs.sum()
            
                # multinomial expects tensor on CPU or same device
                idx = torch.multinomial(self._cached_probs.to('cpu'), 1).item()
                indices_to_sample.append(idx)
    
        # Resetting the cache because priorities may have changed.
        self._cached_probs = None
    
        return [self.buffer[i] for i in indices_to_sample]

# SVD fusion with adaptive rank
def svd_merge(W1: torch.Tensor, W2: torch.Tensor, target_rank: int, w1: float = 0.5, w2: float = 0.5):
    # TODO: For very large matrices, consider switching to batched SVD,
    # which processes the matrix in parts to avoid OOM on the GPU.
    # For example, using torch.svd_lowrank or a custom implementation.
    W1, W2 = W1.to(W1.device), W2.to(W1.device)
    W = w1 * W1 + w2 * W2
    
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except torch.cuda.OutOfMemoryError:
        print("  -> WARNING: SVD failed due to OOM. Moving to CPU for this operation.")
        W_cpu = W.cpu()
        U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        U, S, Vh = U.to(W.device), S.to(W.device), Vh.to(W.device)

    energy = S.cumsum(0) / S.sum()
    adaptive_rank = (energy < 0.99).sum().item()
    target_rank = min(target_rank, adaptive_rank, min(W.shape))
    return U[:, :target_rank] @ torch.diag(S[:target_rank]) @ Vh[:target_rank, :]

class CrossDimensionalAttentionBridge(nn.Module):
    """A bridge for extracting information from multidimensional spaces using an attention mechanism. Version 2.1: Fixed interpolation logic to ensure sequence lengths match."""
    def __init__(self, native_dim: int, device: str, n_heads: int = 8):
        super().__init__()
        self.native_dim = native_dim
        self.device = device
        self.attention = nn.MultiheadAttention(embed_dim=native_dim, num_heads=n_heads, batch_first=True).to(device)
        self.norm = nn.LayerNorm(native_dim).to(device)
        self.key_projectors = nn.ModuleDict()
        self.value_projectors = nn.ModuleDict()

    def add_source_dimension(self, source_dim: int):
        dim_key = str(source_dim)
        if dim_key not in self.key_projectors:
            self.key_projectors[dim_key] = nn.Linear(source_dim, self.native_dim, device=self.device)
            self.value_projectors[dim_key] = nn.Linear(source_dim, self.native_dim, device=self.device)

    def forward(self, native_query: torch.Tensor, external_activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not external_activations:
            return torch.zeros_like(native_query)
            
        target_seq_len = native_query.shape[1]
        target_batch_size = native_query.shape[0]
        
        all_keys, all_values = [], []
        # Add native_query as K and V for self-attention
        # This stabilizes attention when there is little or no external activation.
        all_keys.append(native_query)
        all_values.append(native_query)

        for _, source_tensor in external_activations.items():
            if source_tensor.shape[0] != target_batch_size:
                continue

            source_dim = source_tensor.shape[-1]
            dim_key = str(source_dim)
            if dim_key in self.key_projectors:
                # Project into the required D-dimensional space
                key = self.key_projectors[dim_key](source_tensor)
                value = self.value_projectors[dim_key](source_tensor)
                
                # If the length of the sequence does not match, we change it BEFORE concatenation
                if key.shape[1] != target_seq_len:
                    # F.interpolate requires a format (Batch, Channels, Length)
                    key_transposed = key.permute(0, 2, 1) 
                    key_interpolated = F.interpolate(key_transposed, size=target_seq_len, mode='linear', align_corners=False)
                    key = key_interpolated.permute(0, 2, 1) # Return to (Batch, Length, Channels)

                    value_transposed = value.permute(0, 2, 1)
                    value_interpolated = F.interpolate(value_transposed, size=target_seq_len, mode='linear', align_corners=False)
                    value = value_interpolated.permute(0, 2, 1)

                all_keys.append(key)
                all_values.append(value)

        if not all_keys:
            return torch.zeros_like(native_query)
            
        # Now all tensors in all_keys and all_values ​​are guaranteed to have seq_len = target_seq_len
        keys = torch.cat(all_keys, dim=1) # Concatenation by seq_len
        values = torch.cat(all_values, dim=1)
        
        # Query remains original, but Keys and Values ​​are expanded
        attn_output, _ = self.attention(query=native_query, key=keys, value=values)
        return self.norm(attn_output)

class HybridEncoderLayerWrapper(nn.Module):
    """VERSION 2.2: Strict batch_size checking has been introduced to prevent RuntimeError errors when merging activations."""
    def __init__(self, layer: nn.Module, layer_id: str, in_shape: Tuple, device: str = "cpu"):
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.in_shape = in_shape
        self.native_dim = in_shape[-1]
        self.device = device

        self.bridge = CrossDimensionalAttentionBridge(native_dim=self.native_dim, device=self.device)
        self.gate = nn.Parameter(torch.tensor(0.0, device=device))
        
        self.cognitive_links = nn.ModuleDict()

    def remove_cognitive_link(self, target_layer_id: str):
        """Removes the cognitive link to the specified layer, if one exists.."""
        mangled_key = target_layer_id.replace('.', '_')
        if mangled_key in self.cognitive_links:
            del self.cognitive_links[mangled_key]
            # print(f"      -> Functional link from '{self.layer_id}' to '{target_layer_id}' pruned.")

    def add_cognitive_link(self, target_layer_id: str):
        # Preventing the creation of a "self-referencing" connection
        if target_layer_id == self.layer_id:
            print(f"      -> SKIPPED self-referencing cognitive link on '{self.layer_id}'.")
            return
        
        if target_layer_id not in self.cognitive_links:
            print(f"      -> VISUALIZER HINT: Creating cognitive link from '{target_layer_id}' to '{self.layer_id}'")
            # Replace the dots so the name is valid for ModuleDict
            self.cognitive_links[target_layer_id.replace('.', '_')] = CognitiveFusionUnit()

    def update_bridges(self, all_dims: set):
        for dim in all_dims:
            if dim != self.native_dim:
                self.bridge.add_source_dimension(dim)
    
    def _horizontal_merge_with_priority(self, 
                                        inputs_map: Dict[str, torch.Tensor], 
                                        coherence_threshold: float,
                                        target_batch_size: int) -> Optional[torch.Tensor]:
        """Performs a merge with native input priority and strict batch_size checking."""
        if not inputs_map:
            return None

        # Filtering by batch_size
        filtered_inputs = {}
        for key, tensor in inputs_map.items():
            if tensor.shape[0] == target_batch_size:
                filtered_inputs[key] = tensor
            else:
                # Logging can be added for debugging.
                # print(f"    -> Layer '{self.layer_id}' dropped input '{key}' due to batch size mismatch ({tensor.shape[0]} vs {target_batch_size})")
                pass
        
        if not filtered_inputs:
            return None
        
        # Further logic works only with filtered tensors.
        input_keys = list(filtered_inputs.keys())
        inputs_list = list(filtered_inputs.values())

        max_len = max(seq.shape[1] for seq in inputs_list)
        padded = [F.pad(s, (0, 0, 0, max_len - s.shape[1]), "constant", 0) for s in inputs_list]
        rep_vectors = torch.stack([seq.mean(dim=(0, 1)) for seq in padded])
        
        if len(inputs_list) == 1:
            return padded[0]

        if len(inputs_list) == 2:
            native_input_exists = '__native__' in filtered_inputs
            sim = F.cosine_similarity(rep_vectors[0], rep_vectors[1], dim=0).item()
            
            if sim >= coherence_threshold:
                return (padded[0] + padded[1]) / 2
            elif native_input_exists:
                native_idx = input_keys.index('__native__')
                return padded[native_idx]
            else:
                # If there is no native and the coherence is low, it is better not to return anything
                return None

        sim_matrix = F.cosine_similarity(rep_vectors.unsqueeze(1), rep_vectors.unsqueeze(0), dim=-1)
        coherence_scores = (sim_matrix.sum(dim=1) - 1) / (len(padded) - 1)
        coherent_mask = coherence_scores > coherence_threshold
        
        native_idx = -1
        if '__native__' in input_keys:
            native_idx = input_keys.index('__native__')
            # If the native input is not coherent with the others, return only it
            if not coherent_mask[native_idx]:
                return padded[native_idx]

        coherent_indices = torch.where(coherent_mask)[0]
        if coherent_indices.numel() == 0:
            return padded[native_idx] if native_idx != -1 else None
        if coherent_indices.numel() == 1:
            return padded[coherent_indices[0]]

        coherent_sequences = [padded[i] for i in coherent_indices]
        coherent_reps = rep_vectors[coherent_indices]

        group_mean_rep = coherent_reps.mean(dim=0)
        similarities_to_mean = F.cosine_similarity(coherent_reps, group_mean_rep.unsqueeze(0), dim=-1)
        base_vector_idx_in_coherent = torch.argmax(similarities_to_mean)
        base_tensor = coherent_sequences[base_vector_idx_in_coherent]
        
        total_delta = torch.zeros_like(base_tensor)
        base_rep = coherent_reps[base_vector_idx_in_coherent].unsqueeze(0)
        similarities_to_base = F.cosine_similarity(base_rep, coherent_reps, dim=-1)
        delta_weights = F.softmax(similarities_to_base, dim=-1)
        
        num_deltas_added = 0
        for i in range(len(coherent_sequences)):
            if i == base_vector_idx_in_coherent: continue
            delta = coherent_sequences[i] - base_tensor
            weighted_delta = delta * delta_weights[i]
            total_delta += weighted_delta
            num_deltas_added += 1

        if num_deltas_added > 0:
            total_delta /= num_deltas_added

        return base_tensor + total_delta

    def forward(self, 
                all_prev_activations: Dict[str, torch.Tensor],
                native_source_id: Optional[str]) -> Optional[torch.Tensor]:
        
        # Determine the target batch_size
        # Give priority to native input
        if native_source_id and native_source_id in all_prev_activations:
            target_batch_size = all_prev_activations[native_source_id].shape[0]
        elif all_prev_activations:
            # If there is no native one, take the batch_size from the first one we come across
            target_batch_size = next(iter(all_prev_activations.values())).shape[0]
        else:
            return None # No entrance - no exit

        horizontal_inputs_map = {}
        vertical_inputs_map = {}
        
        for act_id, tensor in all_prev_activations.items():
            if tensor.shape[-1] == self.native_dim:
                key = '__native__' if act_id == native_source_id else act_id
                horizontal_inputs_map[key] = tensor
            else:
                vertical_inputs_map[act_id] = tensor

        fused_horizontal_input = self._horizontal_merge_with_priority(
            horizontal_inputs_map, coherence_threshold=0.5, target_batch_size=target_batch_size
        )

        if fused_horizontal_input is None:
            return None
        
        context_vector = self.bridge(fused_horizontal_input, vertical_inputs_map)
        
        mix_gate = torch.sigmoid(self.gate)
        final_fused_input = fused_horizontal_input + mix_gate * context_vector
        
        layer_output_raw = self.layer(final_fused_input)
        
        final_output = layer_output_raw[0] if isinstance(layer_output_raw, tuple) else layer_output_raw
        
        if self.cognitive_links:
            linked_activations = [final_output]
            for target_id_mangled, fusion_unit in self.cognitive_links.items():
                target_id = target_id_mangled.replace('_', '.')
                if target_id in all_prev_activations and all_prev_activations[target_id].shape[0] == target_batch_size:
                    act1, act2 = final_output, all_prev_activations[target_id]
                    max_len = max(act1.shape[1], act2.shape[1])
                    padded1 = F.pad(act1, (0, 0, 0, max_len - act1.shape[1]))
                    padded2 = F.pad(act2, (0, 0, 0, max_len - act2.shape[1]))
                    fused_link = fusion_unit([padded1, padded2])
                    linked_activations.append(fused_link)
            
            if len(linked_activations) > 1:
                final_output = torch.mean(torch.stack(linked_activations, dim=0), dim=0)

        return final_output

import pandas as pd
from datasets import Dataset as HFDataset
from transformers import DataCollatorWithPadding

class UnifiedAssimilationEngine:
    """
    Manages the complex process of intelligently merging a new donor model
    with the existing EMM architecture. It encapsulates gradient collection,
    layer matching, weight merging, quality control, and the creation of cognitive links.
    """
    
    def __init__(self, emm_model: 'ElasticMemoryModel', config: Dict, device: str):
        """Initializes the assimilation engine."""
        self.emm = emm_model
        # Safely get the visualizer attribute, default is None
        self.visualizer = getattr(self.emm, 'visualizer', None)
        self.config = config["UNIFICATION_PARAMS"]
        self.device = device

    def _create_agnostic_dataloader(self, tokenizer: Any, model_for_task_type: nn.Module, batch_size=4, num_samples=16, max_length=32):
        """
        Creates a DataLoader that generates
        labels of the correct form depending on the model type (sequence-level
        or token-level classification).
        """
        from transformers import AutoModelForTokenClassification
        
        is_token_class_task = isinstance(model_for_task_type, AutoModelForTokenClassification)

        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Exploring the universe reveals countless wonders.",
            "Artificial intelligence is transforming industries.",
            "Scientific discoveries push the boundaries of knowledge.",
            "The cat sat on the mat, watching the rain.",
            "Complex systems exhibit emergent properties.",
            "Data analysis provides valuable insights.",
            "Machine learning models require extensive training."
        ]
        
        # Generating labels of different shapes
        if is_token_class_task:
            print("    -> Agnostic dataloader: Generating token-level labels.")
            num_classes = getattr(model_for_task_type.config, 'num_labels', 2)
            # Create a list of lists of labels
            labels = [[random.randint(0, num_classes - 1) for _ in range(max_length)] for _ in range(num_samples)]
            data = {'text': (sentences * (num_samples // len(sentences) + 1))[:num_samples], 'labels': labels}
        else:
            # Old logic for sequence-level tasks
            labels = [0] * num_samples
            data = {'text': (sentences * (num_samples // len(sentences) + 1))[:num_samples], 'labels': labels}

        hf_dataset = HFDataset.from_pandas(pd.DataFrame(data))
        
        def tokenize_fn(examples):
            # We tokenize the text; labels will be processed separately.
            return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=max_length)
            
        tokenized_dataset = hf_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        return DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

    def _get_full_grad_info(self, model: nn.Module, dataloader: DataLoader, task_name_for_emm: Optional[str] = None):
        """
        Collects activations and gradients. Version 2.8: Restored the
        model.to(device) command as a critical failsafe to ensure
        the correct placement of the model on the device.
        """
        model.to(self.device)
        activations, out_grads = {}, {}

        def setup_hooks(target_layers, temp_activations_dict):
            hooks = []
            for layer_id, module in target_layers.items():
                def f_hook(name):
                    def hook(_, __, o):
                        output = o[0] if isinstance(o, (list, tuple)) else o
                        if isinstance(output, torch.Tensor) and output.requires_grad:
                            temp_activations_dict[name] = output.detach()
                            output.register_hook(
                                lambda g: out_grads.update({name: g.detach().cpu().numpy() if g is not None else np.array([])})
                            )
                    return hook
                hooks.append(module.register_forward_hook(f_hook(layer_id)))
            return hooks

        def remove_hooks(hooks):
            for h in hooks:
                h.remove()

        def forward_and_loss():
            model.zero_grad()
            try:
                batch = next(iter(dataloader))
                inputs = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                labels = inputs.get('labels')
            except StopIteration:
                return None, None

            local_task_name = task_name_for_emm
            if isinstance(model, ElasticMemoryModel) and local_task_name is None:
                if model.task_to_arch_key:
                    local_task_name = next(iter(model.task_to_arch_key.keys()))
                else:
                    print("  -> CRITICAL WARNING: EMM has no tasks, cannot determine a task name for forward pass.")
                    return None, None

            outputs, loss = None, None
            try:
                if isinstance(model, ElasticMemoryModel):
                    logits = model(inputs, local_task_name)
                    if labels is not None and logits is not None:
                        if logits.shape[0] == labels.shape[0]:
                             loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
                else:
                    raw_outputs = model(**inputs)
                    loss = getattr(raw_outputs, 'loss', None)
                    outputs = getattr(raw_outputs, 'logits', raw_outputs)
            except Exception as e:
                print(f"    -> Info: Model forward pass with labels failed: {e}. Trying without labels.")
                inputs.pop('labels', None)
                if isinstance(model, ElasticMemoryModel):
                    outputs = model(inputs, local_task_name)
                else:
                    raw_outputs = model(**inputs)
                    outputs = getattr(raw_outputs, 'logits', raw_outputs)
                loss = None
            
            return outputs, loss

        target_layers = {}
        if isinstance(model, ElasticMemoryModel):
            target_layers = {name: wrapper.layer for name, wrapper in model.hybrid_layers.items()}
        else:
            main_layers_block = self.emm._get_main_layers(model)
            if main_layers_block:
                arch_key = get_architecture_key(model.config)
                target_layers = {f"{arch_key}_layer_{i}": layer for i, layer in enumerate(main_layers_block)}
        
        if not target_layers:
            return {}, {}

        temp_activations = {}
        hooks = setup_hooks(target_layers, temp_activations)
        outputs, loss = forward_and_loss()
        remove_hooks(hooks)

        if outputs is None: return {}, {}

        if loss is not None:
            try:
                loss.backward()
            except Exception as e:
                print(f"  -> WARNING (Grad Collection): Main loss backward() failed: {e}. Trying agnostic loss.")
                loss = None
        
        if loss is None:
            try:
                hooks_agnostic = setup_hooks(target_layers, temp_activations)
                output_tensor = outputs
                if hasattr(outputs, 'last_hidden_state'):
                    output_tensor = outputs.last_hidden_state
                
                if isinstance(output_tensor, torch.Tensor):
                    agnostic_loss = output_tensor.abs().mean()
                    agnostic_loss.backward()
                else:
                    print("    -> Agnostic loss failed: output is not a tensor.")
                    
                remove_hooks(hooks_agnostic)
            except Exception as e_agnostic:
                 print(f"    -> Agnostic loss backward also failed: {e_agnostic}")
                 if 'hooks_agnostic' in locals():
                    remove_hooks(hooks_agnostic)

        activations = {k: v.cpu().numpy() for k, v in temp_activations.items()}
        return activations, out_grads

    def _calculate_pair_similarity(self, nameA, nameB, sigsA, sigsB, acts_A, acts_B, layersA, layersB):
        """Helper function for calculating the similarity of one pair of layers."""
        sigA = sigsA.get(nameA, {})
        sigB = sigsB.get(nameB, {})
        
        s_struct = structural_similarity(sigA, sigB)
        if sigA.get("in_dim") != sigB.get("in_dim") or sigA.get("out_dim") != sigB.get("out_dim"):
            s_struct = 0.2
        elif sigA.get("type") != sigB.get("type"):
            s_struct = 0.5
        
        act_metrics = ensemble_layer_similarity(acts_A.get(nameA), acts_B.get(nameB), get_layer_type(layersA.get(nameA)))
        s_acts = act_metrics["score"]
        
        W_STRUCT = self.config['structural_similarity_weight']
        W_ACT = self.config['functional_similarity_weight']
        
        total_score = W_STRUCT * s_struct + W_ACT * s_acts
        meta = {'total_score': total_score, 'struct_score': s_struct, 'act_score': s_acts, 'details': act_metrics}
        
        return total_score, meta

    def _global_layer_matching(self, donor_model: nn.Module, acts_A: Dict, acts_B: Dict):
        """
        Matches donor layers with layers in the EMM using a hybrid strategy:
        1. Direct index matching for layers of identical architectures.
        2. Global matching (Hungarian algorithm) for layers of new architectures.
        """
        layersA = {name: mod.layer for name, mod in self.emm.hybrid_layers.items()}
        main_layers_donor = self.emm._get_main_layers(donor_model)
        if main_layers_donor is None:
            print("  -> WARNING: Could not extract layers from donor model for matching.")
            return [], {}
        
        arch_key_donor_global = get_architecture_key(donor_model.config)
        layersB = {f"{arch_key_donor_global}_layer_{i}": mod for i, mod in enumerate(main_layers_donor)}

        sigsA = {name: extract_layer_signature(mod) for name, mod in layersA.items()}
        sigsB = {name: extract_layer_signature(mod) for name, mod in layersB.items()}

        # Step 1: Grouping Layers by Architecture
        layersA_by_arch = defaultdict(list)
        for name in layersA: layersA_by_arch[name.split('_layer_')[0]].append(name)
        
        layersB_by_arch = defaultdict(list)
        for name in layersB: layersB_by_arch[name.split('_layer_')[0]].append(name)

        direct_matches = []
        unmatched_B_layers = list(layersB.keys())

        # Step 2: Direct mapping for matching architectures
        for arch_key, b_layers in layersB_by_arch.items():
            if arch_key in layersA_by_arch:
                print(f"  -> Found matching architecture group: '{arch_key}'. Performing direct index-based matching.")
                a_layers = layersA_by_arch[arch_key]
                
                # Sort to ensure correct order
                a_layers.sort(key=lambda x: int(x.split('_')[-1]))
                b_layers.sort(key=lambda x: int(x.split('_')[-1]))
                
                for i in range(min(len(a_layers), len(b_layers))):
                    nameA, nameB = a_layers[i], b_layers[i]
                    score, meta = self._calculate_pair_similarity(nameA, nameB, sigsA, sigsB, acts_A, acts_B, layersA, layersB)
                    direct_matches.append((nameA, nameB, score, meta))
                    if nameB in unmatched_B_layers:
                        unmatched_B_layers.remove(nameB)

        # Step 3: Global mapping for the remaining layers (from the new architectures)
        global_matches = []
        if unmatched_B_layers and layersA:
            print(f"  -> Performing global similarity matching for {len(unmatched_B_layers)} layer(s) from new architectures.")
            
            groupA_global = list(layersA.keys())
            groupB_global = unmatched_B_layers
            
            S_total = np.zeros((len(groupA_global), len(groupB_global)))
            meta_matrix = [[{} for _ in range(len(groupB_global))] for _ in range(len(groupA_global))]

            for i, nameA in enumerate(groupA_global):
                for j, nameB in enumerate(groupB_global):
                    score, meta = self._calculate_pair_similarity(nameA, nameB, sigsA, sigsB, acts_A, acts_B, layersA, layersB)
                    S_total[i, j] = score
                    meta_matrix[i][j] = meta
            
            row_ind, col_ind = linear_sum_assignment(-S_total)
            global_matches = [(groupA_global[r], groupB_global[c], S_total[r, c], meta_matrix[r][c]) for r, c in zip(row_ind, col_ind)]
        
        # Step 4: Combine and sort the results
        all_matches = direct_matches + global_matches
        all_matches.sort(key=lambda x: x[2], reverse=True)
        
        return all_matches, layersB

    def _get_params(self, layer_id: str, model_source: Union['ElasticMemoryModel', nn.Module]) -> Dict[str, np.ndarray]:
        module = None
        if isinstance(model_source, ElasticMemoryModel):
            if layer_id in model_source.hybrid_layers:
                module = model_source.hybrid_layers[layer_id].layer
        else:
            arch, _, idx_str = layer_id.rpartition('_layer_')
            idx = int(idx_str)
            main_layers = self.emm._get_main_layers(model_source)
            if main_layers and idx < len(main_layers):
                module = main_layers[idx]
        
        if module is None:
            return {}

        return {name: p.data.cpu().numpy() for name, p in module.named_parameters()}

    def _replace_params(self, layer_id: str, params: Dict[str, np.ndarray]):
        if layer_id not in self.emm.hybrid_layers:
            return

        module = self.emm.hybrid_layers[layer_id].layer
        device = next(module.parameters()).device
        
        with torch.no_grad():
            param_dict = dict(module.named_parameters())
            for name, value in params.items():
                if name in param_dict and value is not None:
                    tensor_value = torch.from_numpy(value).to(device).type_as(param_dict[name].data)
                    param_dict[name].data.copy_(tensor_value)

    def _merge_layer_blocks_granularly(self, moduleA: nn.Module, moduleB: nn.Module, perm_indices: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
        paramsA = {name: p.data.cpu().numpy() for name, p in moduleA.named_parameters()}
        paramsB = {name: p.data.cpu().numpy() for name, p in moduleB.named_parameters()}
        merged_params = {}
        
        P = np.zeros((len(perm_indices), len(perm_indices)), dtype=np.float32)
        P[np.arange(len(perm_indices)), perm_indices] = 1.0

        for name, pA in paramsA.items():
            if name not in paramsB:
                merged_params[name] = pA
                continue
            
            pB = paramsB[name]
            if pA.shape != pB.shape:
                merged_params[name] = pA
                continue

            pB_permuted = pB
            if any(s in name for s in ['output.dense.weight', 'attention.output.LayerNorm.weight', 'output.LayerNorm.weight']):
                pB_permuted = P @ pB
            elif any(s in name for s in ['output.dense.bias', 'attention.output.LayerNorm.bias', 'output.LayerNorm.bias']):
                pB_permuted = P @ pB
            
            elif 'intermediate.dense.weight' in name:
                pB_permuted = pB @ P.T
            
            elif any(s in name for s in ['query.weight', 'key.weight', 'value.weight']):
                pB_permuted = pB @ P.T

            merged_params[name] = (1.0 - alpha) * pA + alpha * pB_permuted
            
        return merged_params

    def _build_dependency_graph(self) -> Dict:
        """Builds a dependency graph based on layer names in EMM."""
        dep_graph = defaultdict(lambda: {"producers": [], "connection_type": "direct"})
        for arch, layers in self.emm.arch_key_to_layers.items():
            for i, layer_id in enumerate(layers):
                if i > 0:
                    producer = layers[i-1]
                    dep_graph[layer_id]["producers"].append(producer)
        return dep_graph
        
    def run(self, donor_model: nn.Module, donor_tokenizer: Any, donor_dataloader: DataLoader, donor_task_name: str):
        """
        Visualizer method calls have been added to display the unification process, create cognitive links, and add unique layers.
        Safe deep model copying has been implemented by temporarily removing the reference to the non-copyable visualizer object.
        """
        print("\n" + "#"*80)
        print(f" UNIFIED ASSIMILATION ENGINE: Assimilating '{donor_task_name}' ".center(80))
        print("#"*80)

        print("  -> Step 1: Creating a unified agnostic dataloader using the donor's tokenizer and model type...")
        agnostic_dataloader = self._create_agnostic_dataloader(donor_tokenizer, model_for_task_type=donor_model)
        
        print("  -> Step 2: Collecting activations using consistent agnostic data...")
        acts_A, _ = self._get_full_grad_info(self.emm, agnostic_dataloader, task_name_for_emm=donor_task_name)
        acts_B, _ = self._get_full_grad_info(donor_model, agnostic_dataloader)
        
        print("  -> Step 3: Global layer matching between EMM and Donor...")
        matched_pairs, donor_layers_map = self._global_layer_matching(donor_model, acts_A, acts_B)
        
        print("  -> Step 4: Evaluating pre-merge quality of EMM on all known tasks...")
        current_scores = {}
        for task_name, task_loader in self.emm.dataloaders_for_quality_check.items():
            metric_fn = self.emm.task_metrics.get(task_name)
            if metric_fn:
                model_to_eval = self.emm
                if task_name == donor_task_name:
                    model_to_eval = donor_model
                score = metric_fn(model_to_eval, task_loader, self.device, task_name=task_name)
                current_scores[task_name] = score
                print(f"     - Task '{task_name}': {score:.6f}")

        print("\n  -> Step 5: Attempting sequential layer unification...")
        
        donor_arch_key = get_architecture_key(donor_model.config)
        original_donor_path = sorted(donor_layers_map.keys(), key=lambda x: int(x.split('_layer_')[-1]))
        new_task_path = list(original_donor_path)
        donor_layers_to_assimilate = set(original_donor_path)

        for la, lb, score, meta in matched_pairs:
            if lb not in donor_layers_to_assimilate:
                continue

            sigA = extract_layer_signature(self.emm.hybrid_layers[la].layer)
            sigB = extract_layer_signature(donor_layers_map[lb])
            are_compatible = sigA.get("type") == sigB.get("type") and sigA.get("in_dim") == sigB.get("in_dim")
            
            if score >= self.config["direct_merge_threshold"] and are_compatible:
                print(f"\n    Attempting merge (Score: {score:.4f}):\n      (EMM) '{la}'\n      (Donor) '{lb}'")
                
                visualizer_backup = self.emm.visualizer
                self.emm.visualizer = None 
                model_before = copy.deepcopy(self.emm)
                self.emm.visualizer = visualizer_backup
                model_before.visualizer = visualizer_backup
                
                try:
                    moduleA = self.emm.hybrid_layers[la].layer
                    moduleB = donor_layers_map[lb]
                    cost_mat = neuron_cost_matrix(acts_A.get(la), acts_B.get(lb))
                    _, col_ind = find_permutation_matrix_from_cost(cost_mat)
                    print(f"      -> Aligning {len(col_ind)} output neurons before granular merge.")
                    merged_params = self._merge_layer_blocks_granularly(moduleA, moduleB, col_ind, alpha=0.5)
                    self._replace_params(la, merged_params)
                    
                    for task_name_check, current_score in current_scores.items():
                        metric_fn_check = self.emm.task_metrics.get(task_name_check)
                        if metric_fn_check:
                            loader = self.emm.dataloaders_for_quality_check[task_name_check]
                            score_after = metric_fn_check(self.emm, loader, self.device, task_name=task_name_check)
                            
                            quality_loss = 0
                            if metric_fn_check in [accuracy_metric, f1_metric, mIoU_metric]:
                                quality_loss = (current_score - score_after) / (abs(current_score) + 1e-9)
                            else: # For loss-based metrics (MSE)
                                quality_loss = (score_after - current_score) / (abs(current_score) + 1e-9)

                            if quality_loss > self.config["post_merge_quality_loss_tolerance"]:
                                raise ValueError(f"Quality on task '{task_name_check}' dropped by {quality_loss*100:.2f}% (from {current_score:.4f} to {score_after:.4f})")

                    print(f"      -> SUCCESS. Layer '{la}' is now a unified layer.")
                    if self.visualizer:
                        self.visualizer.log_unify_layer(la, lb)
                    idx = new_task_path.index(lb)
                    new_task_path[idx] = la
                    donor_layers_to_assimilate.remove(lb)

                    print(f"      -> Pruning obsolete cognitive links pointing to '{lb}'.")
                    if self.visualizer and hasattr(self.visualizer, 'log_remove_cognitive_links_to_target'):
                        self.visualizer.log_remove_cognitive_links_to_target(lb)
                    for layer_wrapper in self.emm.hybrid_layers.values():
                        if hasattr(layer_wrapper, 'remove_cognitive_link'):
                            layer_wrapper.remove_cognitive_link(lb)

                except Exception as e:
                    print(f"      -> ROLLBACK. Reason: {e}")
                    self.emm.load_state_dict(model_before.state_dict())
                    la_arch_key = la.split('_layer_')[0]
                    lb_arch_key = lb.split('_layer_')[0]
                    if score > self.config["cognitive_link_threshold"] and la_arch_key != lb_arch_key:
                         print(f"      -> High functional similarity detected. Creating cognitive link between '{la}' and '{lb}'.")
                         if la in self.emm.hybrid_layers: self.emm.hybrid_layers[la].add_cognitive_link(lb)
                         if self.visualizer:
                             self.visualizer.log_cognitive_link(la, lb, donor_arch_key_full=donor_arch_key)

            elif self.config["cognitive_link_threshold"] <= score < self.config["direct_merge_threshold"]:
                la_arch_key = la.split('_layer_')[0]
                lb_arch_key = lb.split('_layer_')[0]
                if la_arch_key != lb_arch_key:
                    print(f"\n    Functional similarity detected (Score: {score:.4f}). Creating cognitive link between '{la}' and '{lb}'.")
                    if la in self.emm.hybrid_layers: self.emm.hybrid_layers[la].add_cognitive_link(lb)
                    if self.visualizer:
                        self.visualizer.log_cognitive_link(la, lb, donor_arch_key_full=donor_arch_key)
        
        print("\n  -> Step 6: Assimilating remaining unique layers from donor...")
        unique_layers_path = []
        for layer_id in sorted(list(donor_layers_to_assimilate), key=lambda x: int(x.split('_layer_')[-1])):
             if layer_id not in self.emm.hybrid_layers:
                 print(f"    -> Adding new unique layer '{layer_id}' to EMM graph.")
                 unique_layers_path.append(layer_id)
                 layer_module = donor_layers_map[layer_id]
                 
                 sample_in_dim = -1
                 first_linear = next((m for m in layer_module.modules() if isinstance(m, nn.Linear)), None)
                 if first_linear:
                     sample_in_dim = first_linear.in_features
                 if sample_in_dim == -1: sample_in_dim = 768

                 shapes = self.emm._infer_layer_shapes(layer_module, torch.randn(2, 8, sample_in_dim, device=self.device))
                 layer_copy = copy.deepcopy(layer_module)
                 wrapped_layer = HybridEncoderLayerWrapper(layer_copy, layer_id, shapes['in'], device=self.device).to(self.device)
                 self.emm.hybrid_layers[layer_id] = wrapped_layer
                 self.emm.layer_io_shapes[layer_id] = shapes
                 self.emm.arch_key_to_layers[donor_arch_key].append(layer_id)
                 
                 # Sorting the list of layers for architecture after adding a new one
                 self.emm.arch_key_to_layers[donor_arch_key].sort(key=lambda x: int(x.split('_layer_')[-1]))

                 for la, lb, score, _ in matched_pairs:
                     if lb == layer_id and la in self.emm.hybrid_layers and score > self.config["cognitive_link_threshold"]:
                         print(f"    -> Creating post-assimilation cognitive link from '{la}' to new layer '{layer_id}'.")
                         wrapped_layer.add_cognitive_link(la)
                         if self.visualizer:
                             self.visualizer.log_cognitive_link(la, layer_id, donor_arch_key_full=donor_arch_key)
        
        if self.visualizer and (unique_layers_path or donor_arch_key in self.emm.task_to_arch_key.values()):
            print("  -> VISUALIZER: Logging assimilation of new unique encoder structure.")
            parser_key = self.emm.task_to_parser_key[donor_task_name]
            
            full_arch_layer_ids = self.emm.arch_key_to_layers.get(donor_arch_key, [])
            if full_arch_layer_ids:
                self.visualizer.log_assimilate_encoder(
                    arch_key=donor_arch_key,
                    layer_ids=full_arch_layer_ids,
                    parser_key=parser_key
                )

        print("\n  -> Step 7: Finalizing assimilation...")
        self.emm.task_native_paths[donor_task_name] = new_task_path
        print(f"    -> New native path for '{donor_task_name}': {new_task_path}")
        self.emm._update_all_bridges()

        if self.visualizer:
             head_key = self.emm.task_to_head_key[donor_task_name]
             last_layer_id = new_task_path[-1]
             self.visualizer.log_add_task(donor_task_name, head_key, last_layer_id)

        print("#"*80)
        print(" UNIFIED ASSIMILATION ENGINE: FINISHED ".center(80))
        print("#"*80)

# Must be imported if not already imported in the file
from scipy.spatial.distance import cosine
import torch.nn.functional as F

# --- Helper class for hierarchical projectors ---
class HierarchicalProjector(nn.Module):
    """It represents a projector divided into a shared and several specific parts. This allows for the consolidation of knowledge without losing it."""
    def __init__(self, core_d_model: int, projection_size: int, device: str):
        super().__init__()
        # The general part, storing average knowledge
        self.shared_proj = nn.Linear(core_d_model, projection_size, bias=False).to(device)
        # A dictionary for storing "deltas" - unique knowledge of each model
        self.specific_projs = nn.ModuleDict()

    def add_specific_branch(self, name: str, delta_weights: torch.Tensor):
        """Adds a new specific "delta" to the projector."""
        core_d_model = delta_weights.shape[1]
        projection_size = delta_weights.shape[0]
        device = self.shared_proj.weight.device
        
        specific_proj = nn.Linear(core_d_model, projection_size, bias=False).to(device)
        specific_proj.weight.data.copy_(delta_weights)
        self.specific_projs[name] = specific_proj

    def forward(self, x: torch.Tensor, sub_router_weights: Dict[str, float]) -> torch.Tensor:
        """Performs a forward pass by combining a general part with weighted specific parts."""
        # Basic exit from the common area
        output = self.shared_proj(x)
        
        # Adding weighted outputs from specific deltas
        for name, weight in sub_router_weights.items():
            if name in self.specific_projs and weight > 1e-4:
                output = output + self.specific_projs[name](x) * weight
        return output

# --- The main class UnifiedMultiModalDecoder ---
class UnifiedMultiModalDecoder(nn.Module):
    """
    The unified "voice" of the EMM. This version implements hierarchical refactoring
    to intelligently consolidate knowledge from teacher models without destructive averaging.
    """
    def __init__(self, generative_model_name: str, core_d_model: int, device: str, cache_dir: str):
        super().__init__()
        self.device = device
        self.core_d_model = core_d_model
        
        print(f"\n--- Initializing HIERARCHICAL Unified Decoder with generator: '{generative_model_name}' ---")
        
        from transformers import AutoModelForCausalLM
        self.generator = AutoModelForCausalLM.from_pretrained(
            generative_model_name, cache_dir=cache_dir, local_files_only=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            generative_model_name, cache_dir=cache_dir, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        config = self.generator.config
        def get_config_attr(cfg, keys):
            for key in keys:
                if hasattr(cfg, key): return getattr(cfg, key)
            raise AttributeError(f"Could not find any of these attributes in config: {keys}")

        self.num_layers = get_config_attr(config, ['num_hidden_layers', 'n_layer'])
        self.num_heads = get_config_attr(config, ['num_attention_heads', 'n_head'])
        self.hidden_size = get_config_attr(config, ['hidden_size', 'n_embd'])
        self.head_dim = self.hidden_size // self.num_heads
        
        # Projector storage (can be nn.Linear or HierarchicalProjector)
        self.thought_to_kv_projectors = nn.ModuleDict()
        self.assimilated_archs = set()
        self.projector_metadata = {} # Stores information about which architectures are merged

        self.router = None
        self._update_router()

        print("--- HIERARCHICAL Unified Decoder Initialized Successfully ---")

    def _update_router(self):
        """Recreates the router so that its output dimension matches the number of projectors.."""
        num_projectors = len(self.thought_to_kv_projectors)
        if num_projectors > 0:
            print(f"  -> Updating decoder router for {num_projectors} projector groups.")
            self.router = nn.Sequential(
                nn.Linear(self.core_d_model, num_projectors),
                nn.Softmax(dim=-1)
            ).to(self.device)
        else:
            self.router = None
            
    def _extract_kv_projection_weights(self, teacher_model: nn.Module) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extracts K/V projection weights from a variety of Transformer
        architectures, supporting different naming conventions.
        Returns tensors with .data.detach() without .clone() to save memory.
        """
        kv_weights = []
        # Use _get_main_layers from MultiModalEncoder, creating a temporary instance
        # In EMMv23 code, this function is located within the class itself, so the call is correct
        main_layers_block = self._get_main_layers(teacher_model)
        
        # If the main unit cannot be found, we scan all the modules of the model as a backup option.
        modules_to_scan = main_layers_block if main_layers_block is not None else list(teacher_model.modules())

        for layer in modules_to_scan:
            # 1. Finding the attention module inside the transformer layer
            attention_module = None
            # Extended list of possible names for the attention module
            possible_attn_names = ['attention', 'attn', 'self_attn', 'self', 'multi_head_attention']
            for name in possible_attn_names:
                if hasattr(layer, name):
                    candidate = getattr(layer, name)
                    # Let's make sure that this is not just a tensor, but a module nn.Module
                    if isinstance(candidate, nn.Module):
                        attention_module = candidate
                        break
            
            if attention_module is None:
                continue

            # 2. Search for weights within the identified attention module
            
            # Strategy 1: Finding separate K and V projections (most common in HF)
            k_proj, v_proj = None, None
            # Dictionary of name pairs: ('model_key_name', 'model_value_name')
            possible_kv_pairs = [('k_proj', 'v_proj'), ('key', 'value'), ('k_lin', 'v_lin')]
            for k_name, v_name in possible_kv_pairs:
                if hasattr(attention_module, k_name) and hasattr(attention_module, v_name):
                    k_proj = getattr(attention_module, k_name)
                    v_proj = getattr(attention_module, v_name)
                    if isinstance(k_proj, nn.Linear) and isinstance(v_proj, nn.Linear):
                        kv_weights.append((
                            k_proj.weight.data.detach(),
                            v_proj.weight.data.detach()
                        ))
                        break  # Found a pair, exit the inner loop
            
            if k_proj and v_proj:
                continue # Let's move on to the next layer of the transformer.

            # Strategy 2: Finding a joint QKV projection (as in nn.MultiheadAttention or GPT-2)
            qkv_proj_weight = None
            possible_qkv_names = ['qkv_proj', 'in_proj_weight', 'query_key_value', 'c_attn']
            for name in possible_qkv_names:
                if hasattr(attention_module, name):
                    potential_proj = getattr(attention_module, name)
                    if isinstance(potential_proj, nn.Linear):
                        qkv_proj_weight = potential_proj.weight.data
                        break
                    # For standard nn.MultiheadAttention and GPT-2 Conv1D, weights are stored as nn.Parameter
                    elif isinstance(potential_proj, nn.Parameter):
                        qkv_proj_weight = potential_proj.data
                        break
            
            if qkv_proj_weight is not None:
                try:
                    # Standard layout (Q, K, V)
                    if qkv_proj_weight.shape[0] % 3 == 0:
                        _, w_k, w_v = torch.chunk(qkv_proj_weight, 3, dim=0)
                        kv_weights.append((w_k.detach(), w_v.detach()))
                    # Processing for GPT-2, where weights have a different shape
                    elif qkv_proj_weight.shape[1] % 3 == 0 and hasattr(self, 'hidden_size'):
                         _, w_k, w_v = torch.chunk(qkv_proj_weight, 3, dim=1)
                         # The weights in Conv1D need to be transposed
                         kv_weights.append((w_k.T.detach(), w_v.T.detach()))
                    else:
                         print(f"  -> WARNING: Found QKV weight '{name}' but could not determine how to split it. Shape: {qkv_proj_weight.shape}")
                except Exception as e:
                    print(f"  -> WARNING: Error splitting QKV weight '{name}': {e}")
        
        return kv_weights
    
    def _compare_projectors(self, proj1: nn.Linear, proj2: nn.Module) -> float:
        """Compares two projectors (one is always Linear, the other can be hierarchical)."""
        with torch.no_grad():
            w1 = proj1.weight.flatten().cpu().numpy()
            
            if isinstance(proj2, nn.Linear):
                w2 = proj2.weight.flatten().cpu().numpy()
            elif isinstance(proj2, HierarchicalProjector):
                # Compare with the "common" part of the hierarchical projector
                w2 = proj2.shared_proj.weight.flatten().cpu().numpy()
            else:
                return 0.0
                
            # Add a check in case the vectors are zero.
            if np.all(w1 == 0) or np.all(w2 == 0):
                return 0.0

            return 1.0 - cosine(w1, w2)

    def assimilate_kv_weights(self, teacher_model: nn.Module, arch_key: str, refactor_threshold: float = 0.95):
        """Assimilates K/V weights. If a sufficiently similar existing projector is found, it performs a hierarchical refactoring. Otherwise, it creates a new one."""
        if arch_key in self.assimilated_archs:
            print(f"  -> Arch '{arch_key}' has already contributed to a projector. Skipping.")
            return

        print(f"\n--- Assimilating K/V weights from '{arch_key}' with refactor_threshold={refactor_threshold} ---")
        teacher_kv_weights = self._extract_kv_projection_weights(teacher_model)
        if not teacher_kv_weights:
            print(f"  -> WARNING: Could not extract K/V weights for '{arch_key}'.")
            return
            
        # Step 1: Create a temporary projector for the new model
        with torch.no_grad():
            projection_size = self.num_layers * 2 * self.hidden_size
            temp_projector = nn.Linear(self.core_d_model, projection_size, bias=False).to(self.device)
            # Code for filling temp_projector with teacher weights with dimension adaptation
            decoder_hidden_size = self.hidden_size
            for i in range(self.num_layers):
                if i >= len(teacher_kv_weights): break
                W_k_teacher, W_v_teacher = teacher_kv_weights[i]
                teacher_hidden_size = W_k_teacher.shape[1]

                W_k_adapted = W_k_teacher; W_v_adapted = W_v_teacher
                if teacher_hidden_size != decoder_hidden_size:
                    W_k_adapted = F.pad(W_k_teacher, (0, 0, 0, decoder_hidden_size - teacher_hidden_size)) if teacher_hidden_size < decoder_hidden_size else W_k_teacher[:decoder_hidden_size, :]
                    W_v_adapted = F.pad(W_v_teacher, (0, 0, 0, decoder_hidden_size - teacher_hidden_size)) if teacher_hidden_size < decoder_hidden_size else W_v_teacher[:decoder_hidden_size, :]
                
                target_core_dim = self.core_d_model
                final_W_k, final_W_v = W_k_adapted, W_v_adapted
                if W_k_adapted.shape[1] != target_core_dim:
                    final_W_k = F.pad(W_k_adapted, (0, target_core_dim - W_k_adapted.shape[1])) if W_k_adapted.shape[1] < target_core_dim else W_k_adapted[:, :target_core_dim]
                    final_W_v = F.pad(W_v_adapted, (0, target_core_dim - W_v_adapted.shape[1])) if W_v_adapted.shape[1] < target_core_dim else W_v_adapted[:, :target_core_dim]

                offset = i * 2 * self.hidden_size
                temp_projector.weight.data[slice(offset, offset + self.hidden_size), :] = final_W_k
                temp_projector.weight.data[slice(offset + self.hidden_size, offset + 2 * self.hidden_size), :] = final_W_v
                del W_k_teacher, W_v_teacher, W_k_adapted, W_v_adapted, final_W_k, final_W_v

        # Step 2: Finding the best candidate for refactoring
        best_match_key, best_similarity = None, -1.0
        for key, existing_proj in self.thought_to_kv_projectors.items():
            similarity = self._compare_projectors(temp_projector, existing_proj)
            print(f"    -> Comparing with group '{key}': similarity = {similarity:.4f}")
            if similarity > best_similarity:
                best_similarity, best_match_key = similarity, key

        # Step 3: Decide whether to refactor or create new
        if best_match_key and best_similarity >= refactor_threshold:
            print(f"  -> DECISION: High similarity ({best_similarity:.4f}) with '{best_match_key}'. Refactoring.")
            existing_proj = self.thought_to_kv_projectors[best_match_key]
            W_new = temp_projector.weight.data

            if isinstance(existing_proj, nn.Linear): # First refactoring
                W_existing = existing_proj.weight.data
                W_shared = (W_existing + W_new) / 2.0
                W_delta_existing, W_delta_new = W_existing - W_shared, W_new - W_shared
                
                h_proj = HierarchicalProjector(self.core_d_model, projection_size, self.device)
                h_proj.shared_proj.weight.data.copy_(W_shared)
                
                old_name = self.projector_metadata[best_match_key]['contributors'][0]
                h_proj.add_specific_branch(old_name, W_delta_existing)
                h_proj.add_specific_branch(arch_key, W_delta_new)

                self.thought_to_kv_projectors[best_match_key] = h_proj
                self.projector_metadata[best_match_key]['contributors'].append(arch_key)
            
            elif isinstance(existing_proj, HierarchicalProjector): # Adding to an existing hierarchy
                W_shared = existing_proj.shared_proj.weight.data
                W_delta_new = W_new - W_shared
                existing_proj.add_specific_branch(arch_key, W_delta_new)
                self.projector_metadata[best_match_key]['contributors'].append(arch_key)
        else: # Creating a new
            decision_reason = f"Max similarity ({best_similarity:.4f}) is below threshold" if best_match_key else "No existing projectors"
            print(f"  -> DECISION: {decision_reason}. Creating new projector group.")
            self.thought_to_kv_projectors[arch_key] = temp_projector
            self.projector_metadata[arch_key] = {'contributors': [arch_key]}
            self._update_router()

        self.assimilated_archs.add(arch_key)
        print("\n--- Current Projector Groups in Decoder ---")
        for key, meta in self.projector_metadata.items(): print(f"  - Group '{key}': Contributed by {meta['contributors']}")
    
    def resize_input_projection(self, new_core_d_model: int):
        old_core_d_model = self.core_d_model
        if new_core_d_model <= old_core_d_model: return
        print(f"\n--- Resizing HIERARCHICAL Decoder's Input Projectors from {old_core_d_model} to {new_core_d_model} ---")
        for arch_key, proj_module in self.thought_to_kv_projectors.items():
            if isinstance(proj_module, nn.Linear):
                new_proj = nn.Linear(new_core_d_model, proj_module.out_features, bias=False).to(self.device)
                with torch.no_grad(): new_proj.weight.data[:, :old_core_d_model] = proj_module.weight.data
                self.thought_to_kv_projectors[arch_key] = new_proj
            elif isinstance(proj_module, HierarchicalProjector):
                proj_module.shared_proj = self._resize_linear(proj_module.shared_proj, new_core_d_model)
                for name, spec_proj in proj_module.specific_projs.items():
                    proj_module.specific_projs[name] = self._resize_linear(spec_proj, new_core_d_model)
        self.core_d_model = new_core_d_model
        self._update_router()
        print("--- Decoder's Projectors Resized Successfully ---")

    def _resize_linear(self, layer: nn.Linear, new_in_features: int) -> nn.Linear:
        """Helper function for expanding the Linear layer."""
        old_in_features = layer.in_features
        new_layer = nn.Linear(new_in_features, layer.out_features, bias=False).to(self.device)
        with torch.no_grad():
            new_layer.weight.data.zero_()
            new_layer.weight.data[:, :old_in_features] = layer.weight.data
        return new_layer

    def generate(self, thought_vector: torch.Tensor, prompt_text: str, **kwargs) -> str:
        """
        Generates text by conditioning the internal generator on the EMM's thought_vector.
        It uses a router to dynamically weigh the outputs of the specialized projectors.
        Version 3.1: Fixes past_key_values shape to be 4D.
        """
        self.generator.eval()
        with torch.no_grad():
            if not self.thought_to_kv_projectors:
                print("WARNING: Cannot generate, no K/V projectors have been assimilated. Returning empty string.")
                return ""

            # Step 1: Dynamically weight projectors using a router
        
            # The router expects one vector per batch element. We average if the input is a sequence.
            thought_vector_for_router = thought_vector.mean(dim=1) if thought_vector.dim() > 2 else thought_vector
        
            # Check if the router exists (it may not exist if there is only one projector)
            if self.router and len(self.thought_to_kv_projectors) > 1:
                 router_weights = self.router(thought_vector_for_router.detach()) # Shape: [batch_size, num_projectors]
            else:
                # If there is no router or only one projector, simply use a weight of 1.0
                router_weights = torch.ones(thought_vector_for_router.shape[0], 1, device=self.device)

            projector_keys = list(self.thought_to_kv_projectors.keys())
        
            # Calculate the weighted sum of the outputs of all projectors
            final_projected_kv = 0.0
            for i, key in enumerate(projector_keys):
                # Extract the weight for the current projector and add the dimension for broadcast
                weight = router_weights[:, i].unsqueeze(-1) # Shape: [batch_size, 1]
                proj_module = self.thought_to_kv_projectors[key]
            
                projection = 0.0
                if isinstance(proj_module, nn.Linear):
                    projection = proj_module(thought_vector)
                elif isinstance(proj_module, HierarchicalProjector):
                    # Simple solution: activate all deltas belonging to this group.
                    sub_router_weights = {name: 1.0 for name in proj_module.specific_projs.keys()}
                    projection = proj_module(thought_vector, sub_router_weights)
            
                final_projected_kv += projection * weight
        
            # Step 2: Generate past_key_values ​​of the correct dimension

            # 1. Initial shape: (batch_size, num_layers, 2 (K/V), num_heads, head_dim)
            kv_reshaped = final_projected_kv.view(
                -1, self.num_layers, 2, self.num_heads, self.head_dim
            )

            # 2. Add the missing dimension for seq_len=1
            # New form: (batch_size, num_layers, 2, num_heads, 1, head_dim)
            kv_reshaped_with_seqlen = kv_reshaped.unsqueeze(4)

            # 3. Change the order of measurements to match the HF format
            # New form: (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
            past_key_values_tensor = kv_reshaped_with_seqlen.permute(1, 2, 0, 3, 4, 5)
        
            # 4. Create a tuple of tuples, where each K/V tensor now has a regular 4D shape
            past_key_values = tuple(
                # key: [batch_size, num_heads, 1, head_dim]
                # value: [batch_size, num_heads, 1, head_dim]
                (past_key_values_tensor[i, 0], past_key_values_tensor[i, 1]) 
                for i in range(self.num_layers)
            )

            # Step 3: Final Call to the Generator
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
            # Set default generation parameters, allowing them to be overridden via kwargs
            generation_config = {
                "max_new_tokens": 80,
                "do_sample": True, "top_k": 40, "top_p": 0.9, "temperature": 0.75,
                "no_repeat_ngram_size": 3, "pad_token_id": self.tokenizer.eos_token_id
            }
            generation_config.update(kwargs)

            generated_ids = self.generator.generate(
                **inputs,
                past_key_values=past_key_values,
                **generation_config
            )
        
            # Step 4: Decoding the result
            input_length = inputs.input_ids.shape[1]
            generated_tokens = generated_ids[0, input_length:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

class HierarchicalBranch(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, shared_rank: int, device: str = "cpu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        
        self.shared_branch = AdaptiveBranch(in_dim, out_dim, rank=shared_rank, device=device)
        self.specific_branches = nn.ModuleDict()

    def set_shared_branch(self, rank: int, weights: Tuple[torch.Tensor, torch.Tensor]):
        self.shared_branch = AdaptiveBranch(self.in_dim, self.out_dim, rank=rank, device=self.device)
        with torch.no_grad():
            self.shared_branch.slices[0].data.copy_(weights[0])
            self.shared_branch.slices[1].data.copy_(weights[1])

    def add_specific_branch(self, name: str, rank: int, weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if name not in self.specific_branches:
            branch = AdaptiveBranch(self.in_dim, self.out_dim, rank=rank, device=self.device)
            if weights is not None:
                with torch.no_grad():
                    branch.slices[0].data.copy_(weights[0])
                    branch.slices[1].data.copy_(weights[1])
            self.specific_branches[name] = branch

    def forward(self, x: torch.Tensor, router_weights: Dict[str, float]):
        # 'x' now has the correct d_model_branch dimension
        shared_output = self.shared_branch(x)
        total_output = shared_output
    
        # This logic for router_weights is not currently used, but we will leave it for compatibility.
        for name, branch in self.specific_branches.items():
            weight = router_weights.get(name, torch.tensor(0.0, device=x.device))
            if (weight > 1e-4).any():
                specific_output = branch(x) # Pass the same 'x'
                total_output += specific_output * weight.view(-1, 1)
            
        return total_output, {}

    def get_full_weight_matrix(self, sub_branch_name: str) -> torch.Tensor:
        """Reconstructs and returns the full (Shared + Specific) weight matrix for the specified sub-branch."""
        with torch.no_grad():
            W_shared = self.shared_branch.slices[0] @ self.shared_branch.slices[1]
            if sub_branch_name in self.specific_branches:
                branch = self.specific_branches[sub_branch_name]
                W_specific = branch.slices[0] @ branch.slices[1]
                return W_shared + W_specific
            # If there is no specific sub-branch, return only the general part
            return W_shared

def gram_linear(x: torch.Tensor) -> torch.Tensor:
    # Computes the linear kernel (X @ X.T).
    return x @ x.T

def center_gram(gram: torch.Tensor) -> torch.Tensor:
    # Centers the Gram matrix.
    if not torch.allclose(gram, gram.T, atol=1e-6):
        # Relaxing the check for numerical stability
        pass
    
    means = torch.mean(gram, dim=0, keepdim=True)
    means_t = means.T
    gram = gram - means - means_t + torch.mean(means)
    return gram

def cka_score(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Calculates Centered Kernel Alignment (CKA)."""
    # Reshape 3D tensors (Batch, Seq, Dim) to 2D (Batch*Seq, Dim)
    if X.dim() > 2:
        X = X.reshape(-1, X.shape[-1])
    if Y.dim() > 2:
        Y = Y.reshape(-1, Y.shape[-1])

    # For stability, convert to float32
    gram_x = center_gram(gram_linear(X.float()))
    gram_y = center_gram(gram_linear(Y.float()))
    
    hsic = torch.sum(gram_x * gram_y)
    var1 = torch.sqrt(torch.sum(gram_x * gram_x))
    var2 = torch.sqrt(torch.sum(gram_y * gram_y))
    
    cka = hsic / (var1 * var2 + 1e-8)
    return cka.item()

import itertools
# Basic Elastic Memory Model (EMM)
class ElasticMemoryModel(nn.Module):
    def __init__(self, device: str = "cpu", initial_backbone_blocks: int = 2):
        super().__init__()
        self.device = device
        self.unified_decoder = None
        self.teacher_tokenizers = {}
        self.teacher_embeddings = nn.ModuleDict()
        self.hybrid_layers = nn.ModuleDict()
        self.legacy_encoders = nn.ModuleDict()
        self.layer_io_shapes = {}
        self.task_native_paths = {}
        self.task_to_arch_key = {}
        self.arch_key_to_layers = defaultdict(list)
        self.head_id2label = {}
        self.hybrid_heads = nn.ModuleDict()
        self.task_to_head_key = {}
        # Dictionary for storing parser keys
        self.task_to_parser_key = {}
        # ---------------------------------------------------
        self.lm_biases = nn.ParameterDict()

        self.mod_registry = MultiModalEncoder(device=device)

        # --- Heterogeneous Backbone ---
        self.backbone_by_dim = nn.ModuleDict()
        self.registered_dims = set()

        # --- Branches and their components ---
        self.branched_block = BranchedBlock(device=device)
        
        # --- Metadata and supporting systems ---
        self.task_d_models = {}
        self.task_output_types = {}
        self.task_metrics = {}
        self.task_losses = {}
        self.branch_info = {}
        self.parser_merge_stats = defaultdict(int)
        
        # Save the parameter
        self.initial_backbone_blocks = initial_backbone_blocks 
        self.seen_arch_keys = set()

    def _prepare_initial_activations(self, multi_input: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Correctly handles BatchEncoding from the tokenizer
        and complex embedding layers (e.g., BertEmbeddings) that require
        named arguments.
        """
        all_initial_activations = {}
        for parser_key, x in multi_input.items():
            if parser_key not in self.teacher_embeddings:
                print(f"  -> WARNING: Parser key '{parser_key}' not found in teacher_embeddings. Skipping.")
                continue

            try:
                embedding_layer = self.teacher_embeddings[parser_key].to(self.device)
                
                # Step 1: Prepare input data for the embedding layer
                
                # `x` can be a tensor, dictionary, or BatchEncoding
                if isinstance(x, dict) or hasattr(x, 'keys'): # The check also works for BatchEncoding
                    input_kwargs = {k: v.to(self.device) for k, v in x.items()}
                else: # If it's just a tensor
                    input_kwargs = {'input_ids': x.to(self.device)}
                
                # Step 2: Calling a layer based on its type
                embedding_output = None
                if isinstance(embedding_layer, BertEmbeddings):
                    # BertEmbeddings expects named arguments, at least input_ids
                    # Add token_type_ids if it is not present, as it is often required.
                    if 'token_type_ids' not in input_kwargs and 'input_ids' in input_kwargs:
                        input_kwargs['token_type_ids'] = torch.zeros_like(input_kwargs['input_ids'])
                    
                    embedding_output = embedding_layer(**input_kwargs)
                
                elif isinstance(embedding_layer, nn.Embedding):
                    # A regular nn.Embedding only expects a tensor with indices
                    if 'input_ids' not in input_kwargs:
                        raise ValueError(f"Input for nn.Embedding for parser '{parser_key}' is missing 'input_ids'.")
                    embedding_output = embedding_layer(input_kwargs['input_ids'])
                
                else:
                    # Fallback for other layer types (e.g., ViT)
                    # Trying to pass a single tensor or the entire dictionary
                    if 'pixel_values' in input_kwargs:
                         embedding_output = embedding_layer(input_kwargs['pixel_values'])
                    elif 'input_ids' in input_kwargs:
                         embedding_output = embedding_layer(input_kwargs['input_ids'])
                    else:
                         embedding_output = embedding_layer(**input_kwargs)
                
                activation_key = f"{parser_key}_embedding"
                all_initial_activations[activation_key] = embedding_output

            except Exception as e:
                import traceback
                print(f"  -> WARNING: Failed to create initial activation for parser '{parser_key}'. Error: {e}")
                # traceback.print_exc()  # For detailed debugging
                continue
        
        if not all_initial_activations:
            raise ValueError("Could not create any initial activations from the provided multi_input.")
            
        return all_initial_activations

    def _propagate_through_graph(self, initial_activations: Dict[str, torch.Tensor], trace: bool = False):
        if not self.task_to_arch_key:
            raise RuntimeError("Cannot propagate through graph: No tasks have been assimilated yet.")
        any_task_name = next(iter(self.task_to_arch_key))

        _, _, all_activations, activation_trace = self.mod_registry.forward(
            x={}, task_name=any_task_name, task_to_arch_key=self.task_to_arch_key,
            task_native_paths=self.task_native_paths, embedding_layers=self.teacher_embeddings,
            hybrid_layers=self.hybrid_layers, legacy_encoders=self.legacy_encoders,
            external_activations=initial_activations, trace=trace
        )
        
        all_final_layer_ids = {path[-1] for path in self.task_native_paths.values() if path}
        final_layer_activations = {
            layer_id: tensor for layer_id, tensor in all_activations.items() if layer_id in all_final_layer_ids
        }
        
        # Always return a tuple of 2 elements
        return final_layer_activations, activation_trace

    @torch.no_grad()
    def generate_thought_vector(self, multi_input: Dict[str, Any], trace: bool = False):
        self.eval()
        activation_trace = []
        
        initial_activations = self._prepare_initial_activations(multi_input)
        if trace:
            for act_key in initial_activations.keys():
                parser_key = act_key.replace("_embedding", "")
                activation_trace.append({'type': 'node', 'id': f'parser_{parser_key}'})
        
        final_layer_activations, trace_from_propagate = self._propagate_through_graph(initial_activations, trace=True)
        if trace:
            activation_trace.extend(trace_from_propagate)

        aggregated_thoughts = defaultdict(list)
        for layer_id, h_encoded in final_layer_activations.items():
            native_dim = h_encoded.shape[-1]
            dim_key = str(native_dim)
            h_backbone = h_encoded
            if dim_key in self.backbone_by_dim:
                for block in self.backbone_by_dim[dim_key]:
                    h_backbone = h_backbone + block(h_backbone)
            
            h_vector = h_backbone.mean(dim=1) if h_backbone.dim() > 2 else h_backbone
            aggregated_thoughts[native_dim].append(h_vector)

        final_thought_vectors = {}
        for dim, vectors in aggregated_thoughts.items():
            if vectors:
                final_thought_vectors[dim] = torch.stack(vectors, dim=0).mean(dim=0)
        
        if trace:
            return final_thought_vectors, activation_trace
        return final_thought_vectors

    def _infer_layer_shapes(self, layer: nn.Module, sample_input: torch.Tensor) -> Dict[str, Tuple]:
        """Determines the input and output shapes of a layer using a test run."""
        try:
            with torch.no_grad():
                # Make sure the layer and input are on the same device
                target_device = next(layer.parameters(), torch.tensor(0)).device
                sample_input = sample_input.to(target_device)
                
                raw_output = layer(sample_input)
                
                # Standard processing of the output of transformer layers (often a tuple)
                output_tensor = raw_output[0] if isinstance(raw_output, tuple) else raw_output
                
                return {'in': tuple(sample_input.shape), 'out': tuple(output_tensor.shape)}
        except Exception as e:
            print(f"    -> Warning: The trial run failed for {type(layer).__name__}: {e}")
            return {'in': None, 'out': None}

    def _get_main_layers(self, model_or_encoder: nn.Module) -> Optional[List[nn.Module]]:
        """A universal function for reliably extracting the main layer block from various Transformer architectures and other models."""
        if hasattr(model_or_encoder, 'encoder') and hasattr(model_or_encoder.encoder, 'layer'):
            print("  -> Found layers in .encoder.layer (BERT/RoBERTa-style)")
            return model_or_encoder.encoder.layer
        if hasattr(model_or_encoder, 'h') and isinstance(model_or_encoder.h, nn.ModuleList):
            print("  -> Found layers in .h (GPT-2/Neo-style)")
            return model_or_encoder.h
        if hasattr(model_or_encoder, 'layers') and isinstance(model_or_encoder.layers, nn.ModuleList):
            print("  -> Found layers in .layers (LLaMA/OPT/Falcon/BioGPT-style)")
            return model_or_encoder.layers
        if hasattr(model_or_encoder, 'transformer') and hasattr(model_or_encoder.transformer, 'layer'):
            print("  -> Found layers in .transformer.layer (DistilBERT-style)")
            return model_or_encoder.transformer.layer
        if hasattr(model_or_encoder, 'encoder') and hasattr(model_or_encoder.encoder, 'block'):
            print("  -> Found layers in .encoder.block (T5/BART Encoder-style)")
            return model_or_encoder.encoder.block
        if hasattr(model_or_encoder, 'layer') and isinstance(model_or_encoder.layer, nn.ModuleList):
            print("  -> Found layers in .layer (XLNet-style)")
            return model_or_encoder.layer

        # Recursive search
        for name, child in model_or_encoder.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                # Check if the first element is similar to a transformer block
                first_block = child[0]
                has_attention = any(n in ['self_attn', 'attention', 'attn'] for n, _ in first_block.named_children())
                has_ffn = any(n in ['mlp', 'feed_forward'] for n, _ in first_block.named_children())
                if has_attention and has_ffn:
                    print(f"  -> Found potential transformer block in '{name}'")
                    return child
            
            found_layers = self._get_main_layers(child)
            if found_layers is not None:
                return found_layers

        if isinstance(model_or_encoder, nn.Sequential):
            return list(model_or_encoder.children())
        
        return None

    def _get_embedding_layer(self, model_or_encoder: nn.Module) -> Optional[nn.Module]:
        """A universal function for finding the main embedding layer."""
        if hasattr(model_or_encoder, 'get_input_embeddings'):
            embeds = model_or_encoder.get_input_embeddings()
            if embeds is not None:
                print("  -> Found embedding layer via .get_input_embeddings()")
                return embeds

        common_names = ['embeddings', 'embed_tokens', 'wte', 'wpe']
        for name in common_names:
            if hasattr(model_or_encoder, name):
                layer = getattr(model_or_encoder, name)
                if isinstance(layer, (nn.Embedding, BertEmbeddings)):
                    print(f"  -> Found embedding layer in attribute: '{name}'")
                    return layer

        print("  -> Direct search for embeddings failed. Starting recursive search...")
        for child in model_or_encoder.children():
             if isinstance(child, nn.Embedding):
                 return child
             if hasattr(child, 'word_embeddings') and isinstance(child.word_embeddings, nn.Embedding):
                 return child
             
             found_embed = self._get_embedding_layer(child)
             if found_embed is not None:
                 return found_embed
        
        print(f"  -> WARNING: Could not find embedding layer in {type(model_or_encoder)}.")
        return None

    def _register_dimension(self, dim: int):
        """Registers a new dimension in EMM and creates backbone blocks for it."""
        if dim in self.registered_dims:
            return
        
        print(f"\n--- Registering new dimension in EMM: {dim} ---")
        dim_key = str(dim)
        
        # Use a parameter from the constructor instead of the hard value '2'
        num_blocks = self.initial_backbone_blocks
        print(f"  -> Creating {num_blocks} initial backbone blocks for this dimension.")
        
        new_backbone_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ).to(self.device)
            with torch.no_grad():
                # Decrease std as depth increases for stability
                block[3].weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * (i + 1)))
                if block[3].bias is not None:
                    nn.init.zeros_(block[3].bias)
            new_backbone_blocks.append(block)
            
        self.backbone_by_dim[dim_key] = new_backbone_blocks
        self.registered_dims.add(dim)
        print(f"  -> Created backbone pathway for dimension {dim}.")
        
        # Inform all existing layers about the appearance of a new space
        self._update_all_bridges()

    def _update_all_bridges(self):
        """Tells each layer about all existing dimensions in the EMM."""
        print("  -> Updating cross-dimensional attention bridges...")
        for layer in self.hybrid_layers.values():
            if isinstance(layer, HybridEncoderLayerWrapper):
                layer.update_bridges(self.registered_dims)
    
    def _assimilate_tokenizer_and_embedding(self, 
                                            tokenizer_new: Any, 
                                            embedding_new: nn.Module, 
                                            arch_key: str,
                                            merge_similarity_threshold: float,
                                            alpha_base: float):
        """
        Intelligently assimilates the tokenizer and embedding layer.
        Thresholds and fusion coefficients are now passed as arguments.
        """
        print(f"  -> Assimilating parser for base architecture key '{arch_key}'...")

        def _create_embedding_copy(embedding_layer: nn.Module, device: str) -> nn.Embedding:
            """Helper function for creating an economical copy of an embedding layer."""
            with torch.no_grad():
                embedding_layer_cpu = embedding_layer.to('cpu')
                new_embedding = nn.Embedding(
                    embedding_layer_cpu.num_embeddings,
                    embedding_layer_cpu.embedding_dim,
                    padding_idx=embedding_layer_cpu.padding_idx
                ).to(device)
                new_embedding.weight.data.copy_(embedding_layer_cpu.weight.data)
                return new_embedding

        if not isinstance(tokenizer_new, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            print(f"    -> WARNING: Tokenizer for '{arch_key}' is not a standard Hugging Face PreTrainedTokenizer. "
                  "Using basic copy mechanism.")
            if arch_key not in self.teacher_embeddings:
                self.teacher_tokenizers[arch_key] = tokenizer_new
                self.teacher_embeddings[arch_key] = _create_embedding_copy(embedding_new, self.device)
            return

        if arch_key in self.teacher_embeddings:
            print(f"    -> Found existing master parser for '{arch_key}'. Merging and expanding...")
        
            master_tokenizer = self.teacher_tokenizers[arch_key]
            master_embedding = self.teacher_embeddings[arch_key]

            vocab_master_before_expand = master_tokenizer.get_vocab()
            vocab_new = tokenizer_new.get_vocab()
        
            common_tokens = set(vocab_master_before_expand.keys()) & set(vocab_new.keys())
            new_tokens = list(set(vocab_new.keys()) - set(vocab_master_before_expand.keys()))
        
            print(f"    -> Vocabulary analysis: Common tokens ratio with new model: {len(common_tokens)/len(vocab_new):.2%}, New tokens to add: {len(new_tokens)}")

            with torch.no_grad():
                current_embedding = master_embedding
                if new_tokens:
                    print(f"    -> Expanding vocabulary with {len(new_tokens)} new tokens.")
                    master_tokenizer.add_tokens(new_tokens)
                
                    old_vocab_size, embedding_dim = current_embedding.weight.shape
                    new_vocab_size = len(master_tokenizer)

                    target_device = current_embedding.weight.device
                    expanded_embedding = nn.Embedding(new_vocab_size, embedding_dim, padding_idx=current_embedding.padding_idx).to(target_device)
                    expanded_embedding.weight.data[:old_vocab_size, :] = current_embedding.weight.data

                    old_embedding_ref = self.teacher_embeddings[arch_key]
                    self.teacher_embeddings[arch_key] = expanded_embedding
                    current_embedding = expanded_embedding
                    del old_embedding_ref
                
                    updated_master_vocab = master_tokenizer.get_vocab()
                    for token_str in new_tokens:
                        target_id = updated_master_vocab.get(token_str)
                        source_id = vocab_new.get(token_str)
                    
                        if target_id is not None and source_id is not None:
                            source_vector = embedding_new.weight.data[source_id]
                            current_embedding.weight.data[target_id, :] = source_vector.to(current_embedding.weight.device)
                    
                    print("    -> Re-tying associated LM heads to the new expanded embedding matrix...")
                    tied_heads_count = 0
                    # Looking for all heads that could be connected to this parser.
                    for head_key, head_module in self.hybrid_heads.items():
                        # The LM-head key has the format 'lm_transform_for_{parser_key}'
                        if head_key == f"lm_transform_for_{arch_key}":
                            decoder = self._find_lm_decoder_in_head(head_module, new_vocab_size)
                            if decoder:
                                # Forcefully replace the weights in the decoder with a reference to the new, expanded weights.
                                print(f"      -> Updating decoder weights for head '{head_key}'.")
                                decoder.weight = current_embedding.weight
                                tied_heads_count += 1
                    print(f"    -> Re-tying complete. {tied_heads_count} heads updated.")
            
                print(f"    -> Merging {len(common_tokens)} common tokens using adaptive strategy (threshold={merge_similarity_threshold})...")
                merged_count = 0
                skipped_count = 0
            
                for token_str in common_tokens:
                    id_in_master = vocab_master_before_expand.get(token_str) 
                    id_in_new = vocab_new.get(token_str)

                    if id_in_master is None or id_in_new is None: continue
                
                    vec_old = current_embedding.weight.data[id_in_master]
                    vec_new = embedding_new.weight.data[id_in_new].to(current_embedding.weight.device)
                
                    sim = F.cosine_similarity(vec_old.unsqueeze(0), vec_new.unsqueeze(0)).item()
                
                    if sim > merge_similarity_threshold:
                        alpha_adaptive = alpha_base + (sim - merge_similarity_threshold) * (1.0 - alpha_base) / (1.0 - merge_similarity_threshold)
                        alpha_adaptive = min(alpha_adaptive, 0.95)
                    
                        merged_vec = (1 - alpha_adaptive) * vec_old + alpha_adaptive * vec_new
                        current_embedding.weight.data[id_in_master, :] = merged_vec
                        merged_count += 1
                    else:
                        skipped_count += 1

                print(f"    -> Merge complete. Merged: {merged_count} tokens, Skipped (low similarity): {skipped_count} tokens.")
                if merged_count > 0:
                    self.parser_merge_stats[arch_key] += merged_count
        else:
            print(f"    -> No existing parser found. Creating a new master parser for '{arch_key}'.")
            self.teacher_tokenizers[arch_key] = copy.deepcopy(tokenizer_new) 
            self.teacher_embeddings[arch_key] = _create_embedding_copy(embedding_new, self.device)
        
        if hasattr(self, 'visualizer'):
            self.visualizer.log_assimilate_parser(
                parser_key=arch_key,
                vocab_size=len(self.teacher_tokenizers[arch_key]),
                embed_dim=self.teacher_embeddings[arch_key].embedding_dim
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _neuter_encoder_head(self, module_to_clean: nn.Module, get_head_fn: callable) -> bool:
        """Finds and "neutralizes" (replaces with Identity) the head module within a larger module to avoid duplication."""
        try:
            # Find the target head to be removed
            head_module = get_head_fn(module_to_clean)
            if head_module is None or isinstance(head_module, nn.Identity):
                return True # The head is missing or has already been neutralized.

            # Looking for the parent of this module to replace it.
            for parent_module in module_to_clean.modules():
                for child_name, child_module in parent_module.named_children():
                    if child_module is head_module:
                        setattr(parent_module, child_name, nn.Identity())
                        print(f"      -> Neutered redundant head '{child_name}' from the monolithic encoder module.")
                        return True
            
            # If the parent is not found (unlikely, but possible), report this
            print(f"      -> WARNING: Found head of type {type(head_module).__name__}, but could not find its parent to neuter it.")
            return False
        except Exception as e:
            # If get_head_fn fails with an error, it's not critical.
            print(f"      -> INFO: Could not neuter head due to an error: {e}")
            return False

    def _assimilate_encoder_layers(self, encoder_module: nn.Module, arch_key: str, task_name: str):
        """
        Parses the encoder into layers and saves copies of them in the EMM.
        Now returns True on success and False if unable to parse the encoder to avoid duplication in legacy_encoders.
        """
        import copy

        main_layers = self._get_main_layers(encoder_module)
    
        # If layers are not found, it is a failure of assimilation.
        # Inform the calling function about this so that it uses the legacy path.
        if main_layers is None:
            print(f"  -> INFO: Could not extract main layers for arch '{arch_key}'. Assimilation into hybrid graph failed.")
            return False

        # If we are here, it means the layers have been found and assimilation is possible.
        print(f"  -> Found {len(main_layers)} layers for arch '{arch_key}'. Proceeding with assimilation into hybrid graph.")
        
        found_parser_key = next((p_key for p_key in self.teacher_embeddings.keys() if p_key.startswith(arch_key)), None)
        
        if not found_parser_key:
            if "vit" not in arch_key:
                 raise ValueError(f"Could not find a matching embedding layer for arch key '{arch_key}'.")
            
            d_model_config = encoder_module.config if hasattr(encoder_module, 'config') else encoder_module
            d_model = get_d_model_from_config(d_model_config)
            sample_input = torch.randn(2, 197, d_model, device=self.device)
        else:
            embedding_layer = self.teacher_embeddings[found_parser_key]
            sample_input = torch.randn(2, 8, embedding_layer.embedding_dim, device=self.device)
        
        # Create a list to store the path
        native_path = []
        
        for i, layer in enumerate(main_layers):
            layer_id = f"{arch_key}_layer_{i}"
            # Adding layer ID to path
            native_path.append(layer_id)

            if layer_id not in self.hybrid_layers:
                layer.to('cpu')
                shapes = self._infer_layer_shapes(layer, sample_input)

                if shapes['in'] is None: 
                    print(f"    -> Skipping layer {layer_id} due to shape inference failure.")
                    continue
        
                print(f"    -> Registering new hybrid layer: '{layer_id}' with input shape {shapes['in']}")
                layer_copy = copy.deepcopy(layer)
                wrapped_layer = HybridEncoderLayerWrapper(layer_copy, layer_id, shapes['in'], device=self.device).to(self.device)
            
                self.hybrid_layers[layer_id] = wrapped_layer
                self.layer_io_shapes[layer_id] = shapes
                self.arch_key_to_layers[arch_key].append(layer_id)
    
            output_shape = self.layer_io_shapes.get(layer_id, {}).get('out')
            if output_shape:
                sample_input = torch.randn(*output_shape, device=self.device)
            else:
                print(f"    -> WARNING: Could not determine output shape for {layer_id}. Further shape inference may fail.")
                break
    
        # Save the constructed path
        self.task_native_paths[task_name] = native_path
        self._update_all_bridges()
        
        # If we have reached the end, then assimilation has been successful.
        return True

    @torch.no_grad()
    def reasoning_cascade(self, raw_input: str, coherence_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Implements graph propagation through a common knowledge space.
        1. Activates parsers and projects their outputs into the common space.
        2. Propagates the signal layer by layer:
            a. For each layer, collects all available inputs from the common space.
            b. Filters them by coherence and averages them.
            c. Passes the averaged vector to the layer (which itself performs the common->native projection).
            d. Receives the layer's output (in native space) and projects it back into the common space.
        3. Returns a dictionary of final activations in the common space.
        """
        self.eval()
        print("\n" + "#"*80)
        print(" EMM REASONING CASCADE (VIA COMMON KNOWLEDGE SPACE) ".center(80))
        print("#"*80)

        # all_activations will now store tensors EXCLUSIVELY in the shared space
        all_activations = {}
        print("--- Initial Signal Injection & Projection to Common Space ---")
        
        # Step 1: Signal injection and projection into common space
        for arch_key, tokenizer in self.teacher_tokenizers.items():
            if arch_key not in self.teacher_embeddings or not hasattr(tokenizer, 'batch_encode_plus'):
                continue
            
            embedding_layer = self.teacher_embeddings[arch_key]
            proj_key = f"{arch_key}_embedding_native_to_common"
            if proj_key not in self.knowledge_projectors:
                continue

            inputs = tokenizer(raw_input, return_tensors='pt', truncation=True, max_length=128).to(self.device)
            native_embedding = embedding_layer(inputs['input_ids'])
            
            projector = self.knowledge_projectors[proj_key]
            common_space_embedding = projector(native_embedding)
            
            all_activations[f"{arch_key}_embedding"] = common_space_embedding
        
        if not all_activations:
            raise ValueError("Could not generate any initial common space embeddings.")
        print(f"  -> Injected and projected signals from {len(all_activations)} parsers.")
        
        # Step 2: Layer-by-layer cascade over the graph
        max_depth = max((len(p) for p in self.task_native_paths.values()), default=0)
        level_activations = all_activations.copy()

        for depth in range(max_depth):
            print(f"--- Reasoning Level {depth} ---")
            next_level_activations = {}
            
            target_layers_at_depth = {
                layer_id: wl for layer_id, wl in self.hybrid_layers.items() if f'_layer_{depth}' in layer_id
            }
            if not target_layers_at_depth:
                break

            for layer_id, wrapped_layer in target_layers_at_depth.items():
                # Collect ALL possible inputs for this layer from the general activation pool
                potential_inputs = []
                # "Native" entrance
                arch_key = layer_id.split('_layer_')[0]
                native_source_key = f"{arch_key}_layer_{depth-1}" if depth > 0 else f"{arch_key}_embedding"
                if native_source_key in all_activations:
                    potential_inputs.append(all_activations[native_source_key])

                # Cross-inputs
                for source_id in wrapped_layer.cross_input_sources:
                    if source_id in all_activations:
                        potential_inputs.append(all_activations[source_id])
                
                if not potential_inputs:
                    continue

                # Filtering and averaging right here in the "conductor"
                max_len = max(seq.shape[1] for seq in potential_inputs)
                padded_sequences = [F.pad(s, (0, 0, 0, max_len - s.shape[1])) for s in potential_inputs]
                
                final_input_to_layer = None
                if len(padded_sequences) == 1:
                    final_input_to_layer = padded_sequences[0]
                else:
                    rep_vectors = torch.stack([seq.mean(dim=(0, 1)) for seq in padded_sequences])
                    sim_matrix = F.cosine_similarity(rep_vectors.unsqueeze(1), rep_vectors.unsqueeze(0), dim=-1)
                    coherence_scores = (sim_matrix.sum(dim=1) - 1) / (len(padded_sequences) - 1)
                    coherent_indices = torch.where(coherence_scores > coherence_threshold)[0]
                    
                    if coherent_indices.numel() > 0:
                        coherent_sequences = [padded_sequences[i] for i in coherent_indices]
                        final_input_to_layer = torch.stack(coherent_sequences, dim=0).mean(dim=0)

                if final_input_to_layer is None:
                    continue
                    
                # Calling a layer with a common -> native projection
                in_proj = self.knowledge_projectors[f"{layer_id}_common_to_native"]
                native_output_tuple = wrapped_layer(final_input_to_layer, in_proj)
                native_output = native_output_tuple[0] if isinstance(native_output_tuple, tuple) else native_output_tuple

                # Native -> common projection for the next level
                out_proj = self.knowledge_projectors[f"{layer_id}_native_to_common"]
                common_output = out_proj(native_output)
                next_level_activations[layer_id] = common_output

            if not next_level_activations:
                print(f"  -> Natural propagation stopped at level {depth}.")
                break
                
            all_activations.update(next_level_activations)
            level_activations = next_level_activations
            print(f"  -> Activated and projected {len(level_activations)} layers at this level.")

        # Step 3: Forming Thought Vectors
        print("\n--- Final Thought Vector Generation Stage ---")
        final_thought_vectors = {}
        final_layer_ids = list(level_activations.keys())
        if not final_layer_ids:
            return {}

        for layer_id in final_layer_ids:
            common_space_activation = all_activations[layer_id]
            h_core = common_space_activation.mean(dim=1) if common_space_activation.dim() > 2 else common_space_activation
            
            # The h_core vector is already in the right space for the backbone!
            h_final = h_core
            for block in self.backbone: h_final = h_final + block(h_final)
            
            arch_key = layer_id.split('_layer_')[0]
            branch_out = self.branched_block(h_final, self.projectors, arch_key)
            final_thought_vectors[layer_id] = h_final + branch_out
            
        return final_thought_vectors

    def _align_and_merge_linear(self, linear_ex: nn.Linear, linear_new: nn.Linear, 
                                next_linear_ex: Optional[nn.Linear], next_linear_new: Optional[nn.Linear], 
                                threshold=0.7, alpha=0.5):
        """Performs a merger of two Linear layers with preliminary alignment of signs."""
        with torch.no_grad():
            W_ex, W_new = linear_ex.weight, linear_new.weight
            W_new_aligned = W_new.clone()
            
            signs = torch.sign(torch.sum(W_ex * W_new, dim=1))
            W_new_aligned *= signs.unsqueeze(1)
            
            if next_linear_new is not None and next_linear_ex is not None:
                if next_linear_new.weight.shape[1] == len(signs):
                    next_linear_new.weight.data *= signs.unsqueeze(0)

            sim = F.cosine_similarity(W_ex.flatten(), W_new_aligned.flatten(), dim=0).item()
            if sim > threshold:
                adaptive_alpha = alpha + (sim - threshold) * (1.0 - alpha) / (1.0 - threshold)
                linear_ex.weight.data = (1 - adaptive_alpha) * W_ex + adaptive_alpha * W_new_aligned
                if linear_ex.bias is not None and linear_new.bias is not None:
                    b_new_aligned = linear_new.bias.clone() * signs
                    linear_ex.bias.data = (1 - adaptive_alpha) * linear_ex.bias.data + adaptive_alpha * b_new_aligned
                print(f"      -> Aligned and averaged weights with similarity {sim:.2f}")
            else:
                print(f"      -> Skipped (low similarity after alignment: {sim:.2f})")

    def _merge_heads(self, existing_head: nn.Module, new_head: nn.Module, threshold: float, alpha: float):
        """
        Recursively merges weights, finding pairs of Linear layers to align the signs.
        Thresholds and fusion coefficients are now passed as arguments.
        """
        ex_linears = [m for m in existing_head.modules() if isinstance(m, nn.Linear)]
        new_linears = [m for m in new_head.modules() if isinstance(m, nn.Linear)]
        
        if len(ex_linears) != len(new_linears):
            print("    -> WARNING: Head structures differ in Linear layers. Skipping advanced merge.")
            return

        for i in range(len(ex_linears)):
            linear_ex = ex_linears[i]
            linear_new = new_linears[i]
            
            next_linear_ex = ex_linears[i+1] if i + 1 < len(ex_linears) else None
            next_linear_new = new_linears[i+1] if i + 1 < len(new_linears) else None
            
            if linear_ex.weight.shape == linear_new.weight.shape:
                 self._align_and_merge_linear(linear_ex, linear_new, next_linear_ex, next_linear_new, threshold, alpha)
            else:
                 print(f"    -> Skipping merge for layer pair due to shape mismatch: {linear_ex.weight.shape} vs {linear_new.weight.shape}")

    def add_task(self,
                 task_name: str,
                 dataloader: DataLoader,
                 teacher_model: nn.Module,
                 get_encoder_fn: callable,
                 get_head_fn: callable,
                 arch_type: str,
                 task_d_model: int,
                 arch_key: str,
                 parser_key: str,
                 config: Dict[str, Any],
                 _assimilate_encoder: bool = True, # Flag for encoder assimilation control
                 task_class: Optional[type] = None,
                 num_classes: Optional[int] = None,
                 metric_fn: Optional[callable] = None,
                 loss_fn: Optional[nn.Module] = None,
                 output_handler: Optional[callable] = None):
    
        from transformers import (AutoModelForSequenceClassification, AutoModelForTokenClassification, 
                                  AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForQuestionAnswering,
                                  AutoModelForSeq2SeqLM, AutoModelForImageClassification, AutoModelForMultipleChoice)
        import copy
    
        if arch_key not in self.seen_arch_keys:
            self.seen_arch_keys.add(arch_key)
            print(f"  -> Discovered new architecture: '{arch_key}'. Total unique archs: {len(self.seen_arch_keys)}.")

        print(f"\n--- Registering Task Components for: '{task_name}' ---")
    
        self._register_dimension(task_d_model)

        # --- ENCODER ASSIMILATION LOGIC ---
        # This block is now executed only if the _assimilate_encoder flag is set,
        # which in the new architecture will only happen for the first model.
        if _assimilate_encoder:
            print(f"  -> Natively assimilating encoder for '{task_name}' as a base structure.")
            encoder_module = get_encoder_fn(teacher_model)
            if encoder_module is None: 
                raise ValueError(f"get_encoder_fn failed for initial task '{task_name}'. Cannot build EMM base.")
        
            assimilation_successful = self._assimilate_encoder_layers(encoder_module, arch_key, task_name)
        
            if assimilation_successful:
                print(f"  -> Encoder for '{arch_key}' successfully integrated into the hybrid graph.")
                if hasattr(self, 'visualizer') and self.task_native_paths.get(task_name):
                    self.visualizer.log_assimilate_encoder(
                        arch_key=arch_key,
                        layer_ids=self.task_native_paths[task_name],
                        parser_key=parser_key
                    )
            else:
                # If couldn't disassemble the encoder even for the first model (unlikely), save it as legacy.  
                print(f"  -> Storing encoder for '{arch_key}' as a monolithic legacy block.")
                self.legacy_encoders[arch_key] = copy.deepcopy(encoder_module)
        else:
            print(f"  -> Encoder assimilation for '{task_name}' will be handled by UnifiedAssimilationEngine.")

        # Registration of keys and metadata always occurs
        self.task_to_arch_key[task_name] = arch_key
        self.task_to_parser_key[task_name] = parser_key
    
        head_module_orig = get_head_fn(teacher_model)
        if head_module_orig is None: 
            raise ValueError(f"get_head_fn failed for task '{task_name}'.")
    
        is_lm_task = task_class in [AutoModelForMaskedLM, AutoModelForCausalLM]
    
        # Defining a key for a group of heads
        if is_lm_task:
            head_type_key = f"lm_transform_for_{parser_key}"
        else:
            # More reliable determination of the number of classes
            final_linear_layer = next(reversed([m for m in head_module_orig.modules() if isinstance(m, nn.Linear)]), None)
            real_num_classes = final_linear_layer.out_features if final_linear_layer is not None else None
        
            if real_num_classes is None:
                final_num_classes = num_classes if num_classes is not None else analyze_dataset(dataloader.dataset).get('num_classes', 1)
            else:
                final_num_classes = real_num_classes
                if num_classes is not None and num_classes != real_num_classes:
                    print(f"    -> WARNING: Config num_labels ({num_classes}) mismatches real head size ({real_num_classes}). Prioritizing real size.")
        
            num_classes = final_num_classes
            output_type = 'sequence' if task_class in [AutoModelForTokenClassification] else 'vector'
            modality_prefix = "image" if "vit" in arch_key else "text"
            head_type_key = f"{modality_prefix}-{task_d_model}-{output_type}-{num_classes}"

        self.task_to_head_key[task_name] = head_type_key
        print(f"  -> Task '{task_name}' assigned to Hybrid Head group: '{head_type_key}'")

        if head_type_key not in self.head_id2label and hasattr(teacher_model.config, 'id2label'):
            self.head_id2label[head_type_key] = teacher_model.config.id2label

        # The logic of merging or creating a new head
        if head_type_key not in self.hybrid_heads:
            print(f"    -> Creating new hybrid head for group '{head_type_key}'.")
            head_module_orig.to('cpu') # Saving GPU memory
        
            if is_lm_task:
                if isinstance(head_module_orig, nn.Linear):
                    # Simple case: the head is one Linear layer (decoder)
                    print("    -> LM Head is a Linear decoder. Storing nn.Identity() as transform and separating bias.")
                    if head_module_orig.bias is not None:
                        bias_to_store = copy.deepcopy(head_module_orig.bias.data)
                        self.lm_biases[head_type_key] = nn.Parameter(bias_to_store)
                    head_to_store = nn.Identity()
                else:
                    # Complex case: the head has an internal structure (transformations)
                    print(f"    -> Processing as a complex Language Model head. Removing final decoder matrix.")
                    head_to_store = copy.deepcopy(head_module_orig)
                
                    if parser_key not in self.teacher_embeddings:
                        raise ValueError(f"Cannot process LM Head. Parser '{parser_key}' was not found.")
                
                    vocab_size = self.teacher_embeddings[parser_key].weight.shape[0]
                    decoder_layer = self._find_lm_decoder_in_head(head_to_store, vocab_size)

                    if decoder_layer:
                        # If find a decoder, separate its bias and replace the layer itself with Identity
                        print(f"      -> Found potential decoder layer: '{type(decoder_layer).__name__}'")
                        if hasattr(decoder_layer, 'bias') and decoder_layer.bias is not None:
                            print("      -> Storing decoder bias separately.")
                            bias_to_store = copy.deepcopy(decoder_layer.bias.data)
                            self.lm_biases[head_type_key] = nn.Parameter(bias_to_store)
                    
                        # Looking for a parent and replacing it
                        found_and_replaced = False
                        for parent_module in head_to_store.modules():
                            for child_name, child_module in parent_module.named_children():
                                if child_module is decoder_layer:
                                    print(f"      -> Replacing '{child_name}' in '{type(parent_module).__name__}' with nn.Identity().")
                                    setattr(parent_module, child_name, nn.Identity())
                                    found_and_replaced = True
                                    break
                            if found_and_replaced: break
                        if not found_and_replaced:
                             print("      -> WARNING: Could not find parent of decoder layer to replace it.")
                    else:
                        print("      -> WARNING: Could not find a suitable decoder layer to remove. Storing head as is.")
            else:
                # For non-LM tasks, just copy the entire head
                head_to_store = copy.deepcopy(head_module_orig)

            self.hybrid_heads[head_type_key] = head_to_store.to(self.device)
        else:
            # If the head already exists
            if not is_lm_task:
                print(f"    -> Merging new head into existing hybrid head '{head_type_key}'.")
                existing_head = self.hybrid_heads[head_type_key]
                # Use merge parameters from the global config
                self._merge_heads(existing_head, head_module_orig, **config["MERGE_PARAMS"]["HEADS"])
            else:
                print(f"    -> Reusing existing LM transform head '{head_type_key}'.")

        # Registering task metadata
        self.task_output_types[task_name] = 'sequence' if task_class in [AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForTokenClassification] else 'vector'
        self.task_d_models[task_name] = task_d_model
        print(f"  -> Task '{task_name}' (d_model={task_d_model}) registered for direct native processing.")

        # Define default metric and loss if not provided
        if loss_fn is None: 
            loss_fn = nn.CrossEntropyLoss() if num_classes > 1 else nn.MSELoss()
    
        if metric_fn is None:
            if num_classes <= 1: 
                metric_fn = mse_metric
            elif task_class in [AutoModelForSequenceClassification, AutoModelForImageClassification, AutoModelForMultipleChoice]: 
                metric_fn = accuracy_metric
            elif task_class in [AutoModelForTokenClassification, AutoModelForQuestionAnswering]: 
                metric_fn = f1_metric
            else: 
                metric_fn = None # For LM problems, perplexity is usually used, but we will not implement it here.
    
        self.task_metrics[task_name] = metric_fn
        self.task_losses[task_name] = loss_fn
        if output_handler: 
            self.task_output_handlers[task_name] = output_handler
    
        # Logging to the visualizer
        if hasattr(self, 'visualizer') and self.task_native_paths.get(task_name):
            self.visualizer.log_add_task(
                task_name=task_name,
                head_key=head_type_key,
                last_layer_id=self.task_native_paths[task_name][-1]
            )

        print(f"--- Task '{task_name}' and its components registered successfully ---")

    def _find_lm_decoder_in_head(self, head_module: nn.Module, vocab_size: int, arch_key: Optional[str] = None) -> Optional[nn.Linear]:
        """
        A helper method for finding the output Linear layer in the head of a language model (LM).
        Checks typical architectures (CausalLM, MLM) and their specific structures.
        Doesn't throw an error if the size mismatch occurs, but simply warns.
        """
        print(f"    -> Finding LM decoder for head_module: {type(head_module).__name__}, vocab_size: {vocab_size}, arch_key: {arch_key}")

        if not isinstance(head_module, nn.Module):
            print(f"    -> WARNING: head_module is not an nn.Module: {type(head_module)}. Returning None.")
            return None

        # We find a candidate and check its size, but don't fail with an error,
        # instead we let the caller decide what to do with the mismatch..
        
        candidate_layer = None
        
        # 1. For CausalLM models (BioGPT, GPT-Neo, GPT-2 etc.)
        if hasattr(head_module, 'lm_head') and isinstance(head_module.lm_head, nn.Linear):
            candidate_layer = head_module.lm_head
        # 2. For BERT/RoBERTa-like MLM heads (e.g., predictions.decoder)
        elif hasattr(head_module, 'predictions') and hasattr(head_module.predictions, 'decoder') and isinstance(head_module.predictions.decoder, nn.Linear):
            candidate_layer = head_module.predictions.decoder
        # 3. For other BERT-like models where the decoder is located directly in the head
        elif hasattr(head_module, 'decoder') and isinstance(head_module.decoder, nn.Linear):
            candidate_layer = head_module.decoder
        # 4. For cases where the head itself is a Linear layer (e.g. GPT-2, GPT-Neo)
        elif isinstance(head_module, nn.Linear):
            candidate_layer = head_module
        # 5. For ESM models (e.g. esm2_t30_150M_UR50D)
        elif hasattr(head_module, 'cls') and hasattr(head_module.cls, 'decoder') and isinstance(head_module.cls.decoder, nn.Linear):
            candidate_layer = head_module.cls.decoder
        # 6. As a backup option, we look for the last Linear layer
        else:
            linear_layers = [m for m in head_module.modules() if isinstance(m, nn.Linear)]
            if linear_layers:
                candidate_layer = linear_layers[-1]

        if candidate_layer:
            if candidate_layer.out_features != vocab_size:
                print(f"    -> WARNING: Found LM decoder, but its out_features ({candidate_layer.out_features}) "
                      f"mismatches the master parser's vocab_size ({vocab_size}). "
                      "The calling function will handle resizing.")
            else:
                print(f"    -> Found matching LM decoder with out_features={candidate_layer.out_features}")
            return candidate_layer

        print(f"    -> WARNING: No suitable nn.Linear layer found in head_module for arch_key={arch_key}")
        return None

    def inspect_cross_connections(self):
        """Displays information about potential interactions between spaces."""
        print("\n" + "="*80)
        print(" EMM CROSS-DIMENSIONAL INTERACTION INSPECTION ".center(80))
        print("="*80)

        dims = self.registered_dims
        print(f"Registered Dimensions: {sorted(list(dims))}")
        if len(dims) <= 1:
            print("No cross-dimensional interactions possible (only one dimension registered).")
            return
            
        print("\nPotential interactions are handled by CrossDimensionalAttentionBridges in each layer.")
        for layer_id, layer in self.hybrid_layers.items():
            if isinstance(layer, HybridEncoderLayerWrapper):
                native_dim = layer.native_dim
                other_dims = sorted([d for d in dims if d != native_dim])
                if other_dims:
                    print(f"  - Layer '{layer_id}' (dim={native_dim}) can receive information from dimensions: {other_dims}")
        print("="*80)
        
    def add_branch(self, name: str, architecture_type: str, rank: int = 8, d_model_branch: Optional[int] = None, peft_id: Optional[str] = None):
        """Adds a branch without creating projectors."""
        if d_model_branch is None:
            raise ValueError(f"d_model_branch must be specified for new branch '{name}'")
        
        print(f"Adding new native branch '{name}' (arch: '{architecture_type}', d_model: {d_model_branch})")
        new_branch = self.branched_block.add_branch(name, architecture_type, rank, d_model_branch)
        self.branch_info[name] = {'peft_id': peft_id, 'architecture_type': architecture_type}
        return new_branch

    def forward(self, 
                inputs: Union[Any, Dict[str, Any]], 
                task_name: Optional[str] = None,
                trace: bool = False
               ) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple]:
        
        # --- "PARALLEL CASCADE" MODE (MULTI-OUTPUT / THOUGHT VECTOR) ---
        if isinstance(inputs, dict) and task_name is None:
            multi_input = inputs
            activation_trace = []
            
            # 1. Activate all parsers for which there is input data
            initial_activations = self._prepare_initial_activations(multi_input)
            if trace:
                for act_key in initial_activations.keys():
                    parser_key = act_key.replace("_embedding", "")
                    activation_trace.append({'type': 'node', 'id': f'parser_{parser_key}'})

            # 2. Run the universal graph propagation engine in parallel mode
            _, _, all_activations, trace_from_propagate = self.mod_registry.forward(
                x={}, task_name="__parallel_cascade__", # The task name is not important, since we are working with external_activations
                task_to_arch_key=self.task_to_arch_key,
                task_native_paths=self.task_native_paths,
                embedding_layers=self.teacher_embeddings,
                hybrid_layers=self.hybrid_layers,
                legacy_encoders=self.legacy_encoders,
                external_activations=initial_activations,
                trace=trace
            )
            if trace:
                activation_trace.extend(trace_from_propagate)
            
            # 3. Collect results from the final layers for each task
            all_results = {}
            final_layer_activations = {
                layer_id: tensor for layer_id, tensor in all_activations.items() if "_layer_" in layer_id
            }
            
            for t_name, native_path in self.task_native_paths.items():
                if not native_path: continue
                final_layer_id = native_path[-1]

                if final_layer_id in final_layer_activations:
                    h_encoded = final_layer_activations[final_layer_id]
                    
                    native_dim = h_encoded.shape[-1]
                    dim_key = str(native_dim)
                    h_backbone = h_encoded
                    if dim_key in self.backbone_by_dim:
                        for block in self.backbone_by_dim[dim_key]:
                            h_backbone = h_backbone + block(h_backbone)
                    
                    task_arch_key = self.task_to_arch_key[t_name]
                    h_branched = self.branched_block(h_backbone, task_arch_key)
                    h_final_for_head = h_backbone + h_branched
                    
                    output_type = self.task_output_types.get(t_name, 'vector')
                    final_input_for_head = h_final_for_head.mean(dim=1) if output_type == 'vector' and h_final_for_head.dim() > 2 else h_final_for_head
                    
                    head_key = self.task_to_head_key.get(t_name)
                    if not head_key or head_key not in self.hybrid_heads: continue
                    
                    hybrid_head = self.hybrid_heads[head_key]
                    transformed_output = hybrid_head(final_input_for_head)
                    
                    if head_key.startswith("lm_transform_for_"):
                        parser_key = head_key.replace("lm_transform_for_", "")
                        embedding_matrix = self.teacher_embeddings[parser_key].weight
                        logits = F.linear(transformed_output, embedding_matrix)
                        if head_key in self.lm_biases:
                            logits = logits + self.lm_biases[head_key].to(logits.device)
                        all_results[t_name] = logits
                    else:
                        if isinstance(transformed_output, torch.Tensor): all_results[t_name] = transformed_output
                        elif hasattr(transformed_output, 'logits'): all_results[t_name] = transformed_output.logits
                        elif isinstance(transformed_output, (list, tuple)): all_results[t_name] = transformed_output[0]
            
                    if trace:
                        activation_trace.append({'type': 'edge', 'id': f"layer_{final_layer_id}->head_{head_key}"})
                        activation_trace.append({'type': 'node', 'id': f"head_{head_key}"})
            
            if trace:
                return all_results, activation_trace
            return all_results

        # --- "GUIDED PATH" MODE (SINGLE TASK) ---
        elif task_name is not None:
            x = inputs
            
            # 1. Create a "poor" initial set of activations - only for one task
            parser_key = self.task_to_parser_key.get(task_name)
            if not parser_key: raise ValueError(f"No parser key found for task {task_name}")
            initial_activations = self._prepare_initial_activations({parser_key: x})
            
            # 2. Call the same "engine." It will limit the distribution itself, since there is only one input.
            _, _, all_activations, activation_trace = self.mod_registry.forward(
                x=x, task_name=task_name,
                task_to_arch_key=self.task_to_arch_key,
                task_native_paths=self.task_native_paths,
                embedding_layers=self.teacher_embeddings,
                hybrid_layers=self.hybrid_layers,
                legacy_encoders=self.legacy_encoders,
                external_activations=initial_activations,
                trace=trace
            )
            
            # 3. Extracting the result from the final layer of the native path
            native_path = self.task_native_paths.get(task_name, [])
            if not native_path:
                # Processing for legacy models or models without a graph
                return next(iter(all_activations.values()))

            h_encoded = all_activations.get(native_path[-1])
            if h_encoded is None:
                 raise RuntimeError(f"Propagation failed for task '{task_name}', no final activation produced.")
            
            # 4. Applying Backbone, Branches and Head to the final result
            native_dim = h_encoded.shape[-1]
            dim_key = str(native_dim)
            h_backbone = h_encoded
            if dim_key in self.backbone_by_dim:
                for block in self.backbone_by_dim[dim_key]:
                    h_backbone = h_backbone + block(h_backbone)
            
            task_arch_key = self.task_to_arch_key[task_name]
            h_branched = self.branched_block(h_backbone, task_arch_key)
            h_final_for_head = h_backbone + h_branched
            
            output_type = self.task_output_types.get(task_name, 'vector')
            final_input_for_head = h_final_for_head.mean(dim=1) if output_type == 'vector' and h_final_for_head.dim() > 2 else h_final_for_head
            
            head_key = self.task_to_head_key.get(task_name)
            if not head_key or head_key not in self.hybrid_heads:
                raise ValueError(f"No hybrid head found for task '{task_name}' with key '{head_key}'.")
            
            if trace:
                final_layer_id = self.task_native_paths[task_name][-1]
                activation_trace.append({'type': 'edge', 'id': f"layer_{final_layer_id}->head_{head_key}"})
                activation_trace.append({'type': 'node', 'id': f"head_{head_key}"})

            hybrid_head = self.hybrid_heads[head_key]
            transformed_output = hybrid_head(final_input_for_head)
            
            logits = None
            if head_key.startswith("lm_transform_for_"):
                parser_key = head_key.replace("lm_transform_for_", "")
                embedding_matrix = self.teacher_embeddings[parser_key].weight
                logits = F.linear(transformed_output, embedding_matrix)
                if head_key in self.lm_biases:
                    logits = logits + self.lm_biases[head_key].to(logits.device)
            else:
                raw_output = transformed_output
                if isinstance(raw_output, torch.Tensor): logits = raw_output
                elif hasattr(raw_output, 'logits'): logits = raw_output.logits
                elif isinstance(raw_output, (list, tuple)): logits = raw_output[0]
                else: raise TypeError(f"Could not extract logits from head output for task '{task_name}'")
            
            if trace:
                return logits, activation_trace
            return logits
        else:
            raise ValueError("Invalid arguments for forward pass. Provide either a task_name or a dict of inputs.")

    def log_structure(self, standalone_stats: Optional[Dict[str, Dict]] = None):
        if standalone_stats is None:
            standalone_stats = {}
            
        print("\n" + "="*80)
        print(" EMMv23 FINAL STRUCTURE ANALYSIS ".center(80))
        print("="*80)
        
        dims_str = ", ".join(map(str, sorted(list(self.registered_dims))))
        print(f"Registered EMM Dimensions: [{dims_str}]")
        
        header_tasks = (
            f"{'Task':<28} | {'d_model':<7} | {'Head Key':<40} | {'Encoder Arch Key':<30}"
        )
        print("\n[1. Assimilated Tasks & Component Mapping]")
        print(header_tasks)
        print("-" * len(header_tasks))

        for task_name in sorted(self.task_to_arch_key.keys()):
            task_dim = self.task_d_models.get(task_name, "N/A")
            arch_key = self.task_to_arch_key.get(task_name, "N/A")
            head_key = self.task_to_head_key.get(task_name, "N/A")
            
            print(f"{task_name:<28} | {str(task_dim):<7} | {str(head_key):<40} | {str(arch_key):<30}")
        print("-" * len(header_tasks))
        
        # --- Unique Parsers ---
        header_parsers = (
            f"{'Parser Arch Key':<35} | {'Current Vocab Size':<20} | {'Embedding Dim':<15} | {'Total Merged Tokens':<21}"
        )
        print("\n[2. Unique Parsers (Tokenizers & Embeddings)]")
        print(header_parsers)
        print("-" * len(header_parsers))
        if not self.teacher_tokenizers:
            print("   (No parsers assimilated yet)")
        else:
            for arch_key in sorted(self.teacher_tokenizers.keys()):
                tokenizer = self.teacher_tokenizers[arch_key]
                
                embedding = self.teacher_embeddings[arch_key] if arch_key in self.teacher_embeddings else None
                
                vocab_size = len(tokenizer)
                embed_dim = embedding.embedding_dim if embedding else "N/A"
                merged_tokens = self.parser_merge_stats.get(arch_key, 0)
                print(f"{arch_key:<35} | {vocab_size:<20,} | {str(embed_dim):<15} | {merged_tokens:<21,}")
        print("-" * len(header_parsers))

        # --- Hybrid Encoder Graph ---
        total_orig_layers = sum(s.get('layers', 0) for s in standalone_stats.values() if isinstance(s.get('layers'), int))
        print("\n[3. Hybrid Encoder Graph]")
        print(f"  - Total Original Encoder Layers (Sum of all experts): {total_orig_layers}")
        print(f"  - Unique Layers in EMM's Hybrid Encoder Graph:      {len(self.hybrid_layers)}")
        # It is possible to add the output of information about bridges
        cross_dims = len(self.registered_dims) > 1
        print(f"  - Cross-Dimensional Bridges (Attention): {'ACTIVE' if cross_dims else 'INACTIVE'}")

        # --- Branch Structure ---
        print("\n[4. Branched Block Structure (Native Skills)]")
        if not self.branched_block.branches_by_arch:
            print("   (No branches assimilated yet)")
        else:
            for arch in sorted(self.branched_block.branches_by_arch.keys()):
                branches = self.branched_block.branches_by_arch[arch]
                print(f"   - Architecture Group '{arch}':")
                for b_name in sorted(branches.keys()):
                    branch = branches[b_name]
                    b_dim = getattr(branch, 'in_dim', 'N/A')
                    if isinstance(branch, HierarchicalBranch):
                        print(f"     - [Hierarchical] {b_name} (d_model: {b_dim})")
                        print(f"       - Shared Trunk (Rank: {branch.shared_branch.total_rank})")
                        for spec_name in sorted(branch.specific_branches.keys()):
                            spec_branch = branch.specific_branches[spec_name]
                            print(f"       - Specific: {spec_name} (Rank: {spec_branch.total_rank})")
                    else:
                        rank = getattr(branch, 'total_rank', 'N/A')
                        print(f"     - [Standalone] {b_name} (d_model: {b_dim}, Rank: {rank})")
        
        # --- Backbone structure ---
        print("\n[5. EMM Heterogeneous Backbone Structure]")
        if not self.backbone_by_dim:
            print("   (No backbone pathways created yet)")
        else:
            for dim_key in sorted(self.backbone_by_dim.keys(), key=int):
                pathway = self.backbone_by_dim[dim_key]
                print(f"   - Pathway for d_model={dim_key}: {len(pathway)} blocks")
        
        # --- Hybrid Head Structure ---
        header_heads = (
            f"{'Hybrid Head Key':<45} | {'d_model (In)':<12} | {'Output Dim':<12} | {'Parameters (Own)':<20} | {'Serviced Tasks'}"
        )
        print("\n[6. Hybrid Head Structure]")
        print(header_heads)
        print("-" * (len(header_heads) + 20))
        if not self.hybrid_heads:
            print("   (No heads assimilated yet)")
        else:
            for head_key in sorted(self.hybrid_heads.keys()):
                head = self.hybrid_heads[head_key]
                serviced_tasks = [task for task, key in self.task_to_head_key.items() if key == head_key]
                
                # Logic for calculating parameters taking into account tied weights
                is_tied_head = False
                tied_weight_id = None
                if head_key.startswith('lm-'):
                    lm_arch_key = head_key.split('lm-', 1)[1]
                    if lm_arch_key in self.teacher_embeddings:
                        is_tied_head = True
                        tied_weight_id = id(self.teacher_embeddings[lm_arch_key].weight.data)
                
                num_params = sum(
                    p.numel() for p in head.parameters() 
                    if not (is_tied_head and id(p.data) == tied_weight_id)
                )
                
                # Determining the input/output dimension
                d_model_in_str, out_features_str = "N/A", "N/A"
                try:
                    first_linear = next((m for m in head.modules() if isinstance(m, nn.Linear)), None)
                    last_linear = next(reversed([m for m in head.modules() if isinstance(m, nn.Linear)]), None)
                    if first_linear:
                        d_model_in_str = str(first_linear.in_features)
                    if last_linear:
                        out_features_str = str(last_linear.out_features)
                except:
                    pass

                tasks_str = ", ".join(sorted(serviced_tasks)[:3])
                if len(serviced_tasks) > 3:
                    tasks_str += f", ... (+{len(serviced_tasks) - 3})"
                    
                params_str = f"{num_params:,}"
                if is_tied_head:
                    params_str += " (+ tied)"

                print(f"{head_key:<45} | {d_model_in_str:<12} | {out_features_str:<12} | {params_str:<20} | {tasks_str}")
        print("-" * (len(header_heads) + 20))
        print("="*80)

    def _get_h_vector(self, x: Any, task_name: str, return_features=False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """
        Gets the representation from the encoder. Does NOT manage the activation cache.
        If return_features=True, returns (tensor_features, log_paths).
        Otherwise, returns only the final vector/tensor.
        """
        # mod_registry returns (dict, log, activations). We only need the first two.
        encoder_output_dict, path_log, _ = self.mod_registry(x, task_name)
        
        h_features = encoder_output_dict['last_hidden_state']

        if return_features:
            return h_features, path_log
            
        task_output_type = self.task_output_types.get(task_name)
        is_sequence_task = (task_output_type == 'sequence')

        if h_features.dim() > 2 and not is_sequence_task:
            h_vector = h_features.mean(dim=1)
        else:
            h_vector = h_features
            
        return h_vector

    def absorb_branches(self,
                        new_branch_name: str,
                        new_branch: nn.Module,
                        new_branch_info: Dict[str, Any],
                        dataloaders_main: Dict[str, DataLoader],
                        dataloader_mini: DataLoader,
                        ga_params: Dict[str, Any],
                        energy_threshold: float,
                        threshold_relevance: float,
                        threshold_task_arithmetic: float,
                        threshold_hierarchical_refactor: float
                       ) -> List[Dict[str, Any]]:
        """
        VERSION 2.0: Generates context for analysis by running the signal
        through only ONE native encoder, rather than through the entire hybrid graph.
        This creates a "clean" but sufficiently complex stimulus that allows
        branches to express their individual characteristics, resulting in
        correct and high CKA/RSA/SVCCA values ​​and proper hierarchy formation.
        """
        print(f"\n--- Starting Native Assimilation for branch '{new_branch_name}' ---")

        branch_arch_key = new_branch_info['architecture_type']
        branch_d_model = new_branch.in_dim

        self.branch_info[new_branch_name] = new_branch_info
        self._register_dimension(branch_d_model)

        def decompose(W, max_rank, energy_thresh):
            try:
                U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                energy = S.cumsum(0) / (S.sum() + 1e-9)
                adaptive_rank = (energy < energy_thresh).sum().item() + 1
                final_rank = max(1, min(max_rank, adaptive_rank, U.shape[1], Vh.shape[0]))
                A = (U[:, :final_rank] @ torch.diag(torch.sqrt(S[:final_rank] + 1e-8))).to(W.dtype)
                B = (torch.diag(torch.sqrt(S[:final_rank] + 1e-8)) @ Vh[:final_rank, :]).to(W.dtype)
                return A, B
            except Exception as e:
                print(f"    -> CRITICAL WARNING: SVD failed: {e}. Cannot decompose matrix of shape {W.shape}.")
                rank = max(1, min(max_rank, W.shape[0], W.shape[1]))
                return torch.zeros(W.shape[0], rank, device=W.device, dtype=W.dtype), torch.zeros(rank, W.shape[1], device=W.device, dtype=W.dtype)

        H_context_for_analysis = None
        try:
            # Find a task with the same dimension as the branch to generate the correct context
            compatible_task = next(
                (name for name, d_model in self.task_d_models.items() 
                 if d_model == branch_d_model and name in dataloaders_main),
                None
            )
            if not compatible_task:
                raise ValueError(f"No compatible task with d_model={branch_d_model} found to generate context.")
            
            dataloader_s = dataloaders_main[compatible_task]
            batch = next(iter(dataloader_s))
            x_common = {k: v.to(self.device) for k, v in batch.items() if k not in ['labels', 'task_name']}

            print(f"    -> Using data and native path of task '{compatible_task}' (dim={branch_d_model}) to generate a clean stimulus.")

            with torch.no_grad():
                self.eval()
                # Step 1: Run the signal ONLY through one native pathway, WITHOUT cross-enrichment.
                # This creates a clean but complex signal, specific to a single expert model.

                # Obtain initial activation only for the compatible task.
                parser_key = self.task_to_parser_key[compatible_task]
                current_activations = self._prepare_initial_activations({parser_key: x_common})
                
                # We follow the native path, activating layers one by one
                native_path = self.task_native_paths[compatible_task]
                for i, layer_id in enumerate(native_path):
                    wrapped_layer = self.hybrid_layers[layer_id]
                    
                    # Feed ONLY the output of the previous native layer to the input of the layer.
                    native_source_id = f"{parser_key}_embedding" if i == 0 else native_path[i-1]
                    inputs_for_layer = {native_source_id: current_activations[native_source_id]}
                    
                    # Calling the layer in "clean" mode
                    output = wrapped_layer(inputs_for_layer, native_source_id)
                    current_activations[layer_id] = output[0] if isinstance(output, tuple) else output

                h_encoded = current_activations[native_path[-1]]
                
                # Let's use Backbone since branches work after it.
                dim_key = str(branch_d_model)
                h_backbone = h_encoded
                if dim_key in self.backbone_by_dim:
                    for block in self.backbone_by_dim[dim_key]:
                        h_backbone = h_backbone + block(h_backbone)
                
                H_context_for_analysis = h_backbone
                print(f"    -> Generated clean native context tensor of shape: {H_context_for_analysis.shape}")
        
        except Exception as e:
            print(f"  -> CRITICAL WARNING: Could not generate context for analysis: {e}. Comparison will be based on weights only.")
            import traceback
            traceback.print_exc()
            H_context_for_analysis = None
        
        print("\n  -> Generating a new hierarchical refactoring action plan...")
        action_plan = []
        potential_partners = []
        
        if branch_arch_key in self.branched_block.branches_by_arch:
            for name_s, branch_s in self.branched_block.branches_by_arch[branch_arch_key].items():
                
                if self.branched_block.branch_d_models.get(name_s) != branch_d_model:
                    continue

                targets_for_comparison = []
                if isinstance(branch_s, HierarchicalBranch):
                    for spec_name, spec_branch in branch_s.specific_branches.items():
                        targets_for_comparison.append({
                            'name': spec_name, 'branch_obj': spec_branch,
                            'info': self.branch_info.get(spec_name, {}),
                            'is_hierarchical_part': True, 'parent_name': name_s
                        })
                else:
                    targets_for_comparison.append({
                        'name': name_s, 'branch_obj': branch_s,
                        'info': self.branch_info.get(name_s, {}),
                        'is_hierarchical_part': False, 'parent_name': None
                    })

                for target in targets_for_comparison:
                    target_branch_obj = target['branch_obj']
                    similarities = {}
                    if H_context_for_analysis is not None:
                        with torch.no_grad():
                            if target['is_hierarchical_part']:
                                acts_s = branch_s.shared_branch(H_context_for_analysis) + target_branch_obj(H_context_for_analysis)
                            else:
                                acts_s = target_branch_obj(H_context_for_analysis)
                            
                            acts_m = new_branch(H_context_for_analysis)
                            
                            similarities['cka'] = cka_score(acts_s, acts_m)
                            similarities['rsa'] = abs(rsa(acts_s, acts_m) or 0.0)
                            similarities['svcca'] = svcca(acts_s, acts_m)

                    print(f"    -> Comparing '{target['name']}' (main) vs '{new_branch_name}' (new) | CKA: {similarities.get('cka', 0.0):.3f}, RSA: {similarities.get('rsa', 0.0):.3f}, SVCCA: {similarities.get('svcca', 0.0):.3f}")
                    
                    valid_scores = [v for v in similarities.values() if v is not None and not np.isnan(v)]
                    base_similarity_score = np.mean(valid_scores) if valid_scores else 0.0
                    
                    info_s = target['info']
                    metadata_bonus = 0.1 if info_s.get('base_model') and new_branch_info.get('base_model') and info_s.get('base_model') == new_branch_info.get('base_model') else 0.0
                    final_score = min(base_similarity_score + metadata_bonus, 1.0)
                
                    if final_score >= threshold_relevance:
                        partner_info = target.copy()
                        partner_info.update({'score': final_score, 'branch_s_full': branch_s})
                        potential_partners.append(partner_info)

        if not potential_partners:
             action_plan.append({'action': 'absorb_new', 'name': new_branch_name, 'arch_type': branch_arch_key, 'branch_m': new_branch})
        else:
             best_match = max(potential_partners, key=lambda x: x['score'])
             score_max, name_s, branch_s = best_match['score'], best_match['name'], best_match['branch_obj']
             
             print(f"    -> Best partner found: '{name_s}' with similarity score: {score_max:.4f}")

             is_related = best_match['info'].get('base_model') and new_branch_info.get('base_model') and best_match['info'].get('base_model') == new_branch_info.get('base_model')
             
             if is_related and score_max >= threshold_task_arithmetic:
                 action_plan.append({'action': 'merge_ta', 'name_s': name_s, 'name_m': new_branch_name, 'branch_s': branch_s, 'branch_m': new_branch})
             elif score_max >= threshold_hierarchical_refactor:
                 action_plan.append({'action': 'refactor_into_hierarchical', 'name_m': new_branch_name, 'branch_m': new_branch, 'name_s': name_s, 'branch_s': branch_s, 'parent_name': best_match.get('parent_name'), 'branch_s_full': best_match['branch_s_full']})
             else:
                 action_plan.append({'action': 'absorb_new', 'name': new_branch_name, 'arch_type': branch_arch_key, 'branch_m': new_branch})

        print("\n  -> Executing the adaptive refactoring action plan...")
        final_action_item = action_plan[0] if action_plan else None
        if not final_action_item:
            print("    -> No action to execute.")
            return []

        action = final_action_item['action']
        arch_type = branch_arch_key

        if action == 'absorb_new':
            name_m = final_action_item['name']
            branch_m = final_action_item['branch_m']
            print(f"    -> ACTION: absorb_new for '{name_m}'.")
            
            added_branch = self.add_branch(name=name_m, architecture_type=arch_type, rank=branch_m.total_rank, d_model_branch=branch_m.in_dim, peft_id=new_branch_info.get('peft_id'))
            if added_branch:
                added_branch.load_state_dict(branch_m.state_dict())
                self.branch_info[name_m] = new_branch_info

        elif action == 'refactor_into_hierarchical':
            name_s = final_action_item['name_s']
            branch_s_full = final_action_item['branch_s_full']
            name_m, branch_m = final_action_item['name_m'], final_action_item['branch_m']
            
            print(f"    -> ACTION: refactor_into_hierarchical for '{name_s}' <--- '{name_m}'")

            with torch.no_grad():
                if isinstance(branch_s_full, HierarchicalBranch):
                    W_s = branch_s_full.get_full_weight_matrix(name_s)
                    if W_s is None: raise ValueError(f"Could not reconstruct matrix for sub-branch '{name_s}'")
                else:
                    W_s = branch_s_full.slices[0] @ branch_s_full.slices[1]
                
                W_m = branch_m.slices[0] @ branch_m.slices[1]
        
                P = find_permutation_matrix(W_s, W_m)
                W_m_permuted = P @ W_m
                
                ga = SimpleGA(num_params=1, **ga_params)
                W_target_ga = (W_s + W_m_permuted) / 2.0
                alpha_ga = ga.run(W_s, W_m_permuted, W_target_ga)[0].item()
                
                W_shared = (1 - alpha_ga) * W_s + alpha_ga * W_m_permuted
                W_specific_s = W_s - W_shared
                W_specific_m = W_m_permuted - W_shared

                max_rank = min(branch_d_model, 16)

                A_shared, B_shared = decompose(W_shared, max_rank, energy_threshold)
                shared_rank_actual = A_shared.shape[1]
                
                A_spec_s, B_spec_s = decompose(W_specific_s, max_rank, energy_threshold)
                spec_s_rank_actual = A_spec_s.shape[1]
                
                A_spec_m, B_spec_m = decompose(W_specific_m, max_rank, energy_threshold)
                spec_m_rank_actual = A_spec_m.shape[1]
                
                print(f"      -> Ranks: Shared={shared_rank_actual}, Spec_S={spec_s_rank_actual}, Spec_M={spec_m_rank_actual}")

                new_h_branch = HierarchicalBranch(branch_d_model, branch_d_model, shared_rank=shared_rank_actual, device=self.device)
                new_h_branch.set_shared_branch(shared_rank_actual, (A_shared, B_shared))
                
                name_s_spec = f"{name_s}_specific" if not name_s.endswith("_specific") else name_s
                new_h_branch.add_specific_branch(name_s_spec, spec_s_rank_actual, (A_spec_s, B_spec_s))
                
                name_m_spec = f"{new_branch_name}_specific"
                new_h_branch.add_specific_branch(name_m_spec, spec_m_rank_actual, (A_spec_m, B_spec_m))

                parent_name_to_replace = final_action_item.get('parent_name') or name_s
                if parent_name_to_replace is None:
                    raise ValueError("Could not determine a valid parent name for branch replacement.")
                
                print(f"      -> Replacing branch '{parent_name_to_replace}' with hierarchical structure.")
                original_s_info = self.branch_info.get(name_s, {}).copy()
                
                self.branched_block.branches_by_arch[arch_type][parent_name_to_replace] = new_h_branch
                
                if name_s in self.branch_info:
                    self.branch_info[name_s_spec] = self.branch_info.pop(name_s)
                self.branch_info[name_m_spec] = new_branch_info

                parent_info = original_s_info
                parent_info['children'] = list(new_h_branch.specific_branches.keys())
                self.branch_info[parent_name_to_replace] = parent_info
                
                if isinstance(branch_s_full, HierarchicalBranch):
                    for old_spec_name, old_spec_branch in branch_s_full.specific_branches.items():
                        if old_spec_name != name_s:
                            new_h_branch.add_specific_branch(old_spec_name, old_spec_branch.total_rank)
                            new_h_branch.specific_branches[old_spec_name].load_state_dict(old_spec_branch.state_dict())

        elif action == 'merge_ta':
            name_s, branch_s = final_action_item['name_s'], final_action_item['branch_s']
            branch_m = final_action_item['branch_m']
            print(f"    -> ACTION: merge_ta on '{name_s}' by assimilating '{new_branch_name}'")
            
            with torch.no_grad():
                W_s = branch_s.slices[0] @ branch_s.slices[1]
                W_m = branch_m.slices[0] @ branch_m.slices[1]
                
                P = find_permutation_matrix(W_s, W_m)
                W_m_permuted = P @ W_m
                
                ga = SimpleGA(num_params=1, **ga_params)
                W_target = (W_s + W_m_permuted) / 2.0
                alpha_ga = ga.run(W_s, W_m_permuted, W_target)[0].item()
                
                W_merged = (1 - alpha_ga) * W_s + alpha_ga * W_m_permuted
                
                A, B = decompose(W_merged, branch_s.total_rank, energy_threshold)
                branch_s.slices[0].data.copy_(A)
                branch_s.slices[1].data.copy_(B)
            
            task_name_m = new_branch_info.get('task_name')
            if name_s in self.branch_info and task_name_m:
                if 'associated_tasks' not in self.branch_info[name_s]:
                    old_task = self.branch_info[name_s].get('task_name')
                    self.branch_info[name_s]['associated_tasks'] = [t for t in [old_task, task_name_m] if t]
                elif task_name_m not in self.branch_info[name_s]['associated_tasks']:
                    self.branch_info[name_s]['associated_tasks'].append(task_name_m)

        if hasattr(self, 'visualizer'):
            self.visualizer.log_absorb_branch(final_action_item)

        print("--- Native Branch Assimilation Finished ---")
        return [final_action_item] if final_action_item else []

    def remove_branch(self, name: str):
        """Completely removes a branch and its associated components from the model."""
        print(f"Attempting to remove branch '{name}'...")
        
        # 1. Remove from BranchedBlock
        was_removed = self.branched_block.remove_branch(name)
        
        if was_removed:
            
            # 4. Removing linked projectors
            if name in self.projectors:
                del self.projectors[name]
                
            # 5. Removing meta-information
            if name in self.branch_info:
                del self.branch_info[name]
                
            print(f"Branch '{name}' and its components were successfully removed.")
        else:
            print(f"Warning: Branch '{name}' not found for removal.")

def get_model_stats(model: nn.Module, model_name: str, simulate_tying: bool = False):
    """
    Counts the number of UNIQUE parameters.
    If simulate_tying=True, attempts to find and "link" embeddings and the LM head.
    """
    from transformers.modeling_utils import PreTrainedModel

    seen_ids = set()
    num_params = 0
    
    # Simulation of weight binding
    tied_head_id = -1
    if simulate_tying and isinstance(model, PreTrainedModel) and model.config.tie_word_embeddings:
        try:
            # Finding the embedding tensor ID
            embeddings = model.get_input_embeddings()
            if embeddings is not None:
                embedding_id = id(embeddings.weight)
                seen_ids.add(embedding_id) # Add it immediately to count it
                num_params += embeddings.weight.numel()

                # Find the LM tensor of the head
                output_embeddings = model.get_output_embeddings()
                if output_embeddings is not None and hasattr(output_embeddings, 'weight'):
                    tied_head_id = id(output_embeddings.weight)
                    print(f"  -> Simulating weight tying for '{model_name}'.")
        except Exception as e:
            print(f"  -> Could not simulate tying for '{model_name}': {e}")

    for name, tensor in model.state_dict().items():
        tensor_id = id(tensor)
        if tensor_id not in seen_ids:
            # If this is the ID of a linked head that we have already "processed", skip it.
            if tensor_id == tied_head_id:
                continue
            
            num_params += tensor.numel()
            seen_ids.add(tensor_id)
    
    size_bytes = num_params * 4
    size_mb = size_bytes / (1024 ** 2)
    
    print(f"Statistics for '{model_name}':")
    print(f"  - Number of UNIQUE parameters and buffers: {num_params:,}")
    print(f"  - Approximate total size in memory: {size_mb:.2f} MB")
    
    return num_params, size_mb

def build_consolidated_emm(expert_tasks: Dict[str, Dict],
                           extractor_map: Dict[str, callable],
                           device: str,
                           config: Dict[str, Any],
                           cache_dir: Optional[str] = None) -> Tuple[ElasticMemoryModel, Dict[str, Dict]]:
    """Creates and trains a consolidated ElasticMemoryModel by sequentially assimilating a set of expert models in a heterogeneous architecture."""
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoConfig, AutoTokenizer, DataCollatorWithPadding, DataCollatorForTokenClassification,
        AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForTokenClassification,
        AutoModelForCausalLM, AutoModelForImageClassification
    )
    from datasets import Dataset as HFDataset
    import pandas as pd
    import torch
    import random
    import gc
    from safetensors.torch import load_file
    import time

    # --- Helper functions (nested for encapsulation) ---
    def _load_teacher_expert(task_name: str, expert_config: dict):
        """
        Implements smart loading that uses the standard from_pretrained method for .safetensors and manual loading for
        .bin files to bypass safety checks in transformers on older versions of PyTorch.
        """
        def create_dummy_text_dataset(num_samples=16):
            data = {'text': [f"This is sample sentence number {i}." for i in range(num_samples)], 'labels': [random.randint(0, 1) for _ in range(num_samples)]}
            return HFDataset.from_pandas(pd.DataFrame(data))
        
        def create_dummy_ner_dataset(num_samples=16, seq_len=12, num_classes=3):
            data = {'tokens': [[f"word{j}" for j in range(seq_len)] for _ in range(num_samples)],'labels': [[random.randint(0, num_classes - 1) for _ in range(seq_len)] for _ in range(num_samples)]}
            return HFDataset.from_dict(data)
        
        print("\n" + "="*50); print(f"Loading teacher expert for: {task_name} ({expert_config['model_id']})"); print("="*50)
        model_id = expert_config["model_id"]
        try:
            local_model_path = snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_files_only=True)
            tokenizer = None
            if expert_config.get('task_class') != AutoModelForImageClassification:
                tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True, add_prefix_space=True)
                if tokenizer and tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

            safetensors_path = os.path.join(local_model_path, "model.safetensors")
            bin_path = os.path.join(local_model_path, "pytorch_model.bin")

            if os.path.exists(safetensors_path):
                # The safe and preferred way: using the standard method
                print(f"  -> Found 'model.safetensors'. Loading with standard 'from_pretrained'.")
                teacher_model = expert_config["task_class"].from_pretrained(
                    local_model_path,
                    local_files_only=True
                )
            elif os.path.exists(bin_path):
                # Backup path for .bin files: Manual download to bypass verification
                print(f"  -> Found 'pytorch_model.bin'. Manually loading to bypass transformers' security check.")
                model_config_hf = AutoConfig.from_pretrained(local_model_path)
                teacher_model = expert_config["task_class"].from_config(model_config_hf)
                state_dict = torch.load(bin_path, map_location="cpu")
                teacher_model.load_state_dict(state_dict, strict=False)
            else:
                # If weights are not found in any of the formats
                raise FileNotFoundError(f"Could not find model weights ('model.safetensors' or 'pytorch_model.bin') in {local_model_path}")
            
            # General logic after loading the model
            if tokenizer and hasattr(teacher_model.config, 'vocab_size'):
                if len(tokenizer) > teacher_model.config.vocab_size:
                    print(f"  -> INFO: Tokenizer vocab size ({len(tokenizer)}) > model vocab size ({teacher_model.config.vocab_size}). Resizing model embeddings.")
                    teacher_model.resize_token_embeddings(len(tokenizer))

            teacher_model.to(device)
            teacher_model.eval()
            print("  -> Model loaded and configured successfully onto device.")

            num_labels = expert_config.get("num_labels")
            loader = None
            if expert_config.get('task_class') == AutoModelForTokenClassification:
                raw_dataset = create_dummy_ner_dataset(num_classes=num_labels or 3)
                def tokenize_and_align_labels(examples):
                    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)
                    labels = []
                    for i, label_list in enumerate(examples["labels"]):
                        word_ids = tokenized_inputs.word_ids(batch_index=i)
                        previous_word_idx = None; label_ids = []
                        for word_idx in word_ids:
                            if word_idx is None or word_idx == previous_word_idx: label_ids.append(-100)
                            else: label_ids.append(label_list[word_idx])
                            previous_word_idx = word_idx
                        labels.append(label_ids)
                    tokenized_inputs["labels"] = labels
                    return tokenized_inputs
                tokenized_dataset = raw_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=raw_dataset.column_names)
                data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
                loader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True, collate_fn=data_collator)
            elif expert_config.get('task_class') == AutoModelForImageClassification:
                dummy_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                dummy_images = [torch.rand(3, 256, 256) for _ in range(16)]
                processed_images = [dummy_transforms(img) for img in dummy_images]
                hf_dataset = HFDataset.from_dict({'pixel_values': processed_images, 'labels': [random.randint(0, (num_labels or 2) - 1) for _ in range(16)]})
                def collate_fn(examples): return {"pixel_values": torch.stack([ex["pixel_values"] for ex in examples]), "labels": torch.tensor([ex["labels"] for ex in examples])}
                loader = DataLoader(hf_dataset, batch_size=4, collate_fn=collate_fn)
            else:
                raw_dataset = create_dummy_text_dataset()
                tokenize_fn = lambda exs: tokenizer(exs["text"], truncation=True, max_length=64)
                tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
                loader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True, collate_fn=data_collator)
            return teacher_model, tokenizer, loader
        except Exception as e:
            print(f"CRITICAL ERROR loading expert '{task_name}': {e}")
            import traceback; traceback.print_exc()
            return None, None, None
    EMM_Main = ElasticMemoryModel(device=device, initial_backbone_blocks=config["INITIAL_BACKBONE_BLOCKS"])
    EMM_Main.dataloaders_for_quality_check = {}

    # First, create and assign a visualizer
    if HAS_FLASK:
        visualizer = EMM_Visualizer(EMM_Main)
        EMM_Main.visualizer = visualizer
        visualizer.run()
        time.sleep(2)
    
    # And only then we create the assimilation engine
    assimilation_engine = UnifiedAssimilationEngine(EMM_Main, config, device)

    dataloaders_main = {}
    standalone_model_stats = {}

    for i, (task_name, expert_config) in enumerate(expert_tasks.items()):

        teacher_model, tokenizer, loader = _load_teacher_expert(task_name, expert_config)
        
        if not teacher_model:
            print(f"Skipping task '{task_name}' due to loading error.")
            continue
        
        EMM_Main.dataloaders_for_quality_check[task_name] = loader
        
        if task_name not in standalone_model_stats:
            params, mb = get_model_stats(teacher_model, f"Standalone_{task_name}", simulate_tying=True)
            orig_head_out = 'N/A'
            try:
                head_module_stats = extractor_map[expert_config["head_extractor"]](teacher_model)
                final_linear_stats = next(reversed([m for m in head_module_stats.modules() if isinstance(m, nn.Linear)]), None)
                if final_linear_stats:
                    orig_head_out = final_linear_stats.out_features
            except Exception: pass
            orig_layers = 'N/A'
            try:
                encoder_module_stats = extractor_map[expert_config["encoder_extractor"]](teacher_model)
                main_layers = EMM_Main._get_main_layers(encoder_module_stats)
                orig_layers = len(main_layers) if main_layers is not None else 1
            except Exception: pass

            standalone_model_stats[task_name] = {
                'params': params, 'mb': mb, 'head_out': orig_head_out, 'layers': orig_layers
            }

        model_config = teacher_model.config
        task_d_model = get_d_model_from_config(model_config)
        encoder_key = get_architecture_key(model_config, for_parser=False)
        parser_key = get_architecture_key(model_config, for_parser=True)

        get_encoder_fn = extractor_map[expert_config["encoder_extractor"]]
        encoder_module_for_embeds = get_encoder_fn(teacher_model)
        embedding_layer = EMM_Main._get_embedding_layer(encoder_module_for_embeds)
        if tokenizer and embedding_layer:
            EMM_Main._assimilate_tokenizer_and_embedding(
                tokenizer_new=tokenizer, 
                embedding_new=embedding_layer, 
                arch_key=parser_key,
                **config["MERGE_PARAMS"]["TOKENIZER"]
            )        
        is_first_model = (i == 0)
        EMM_Main.add_task(
            task_name=task_name, dataloader=loader, teacher_model=teacher_model,
            get_encoder_fn=get_encoder_fn,
            get_head_fn=extractor_map[expert_config["head_extractor"]],
            arch_type=expert_config["arch_type"],
            task_d_model=task_d_model,
            task_class=expert_config["task_class"],
            num_classes=expert_config.get("num_labels"),
            arch_key=encoder_key,
            parser_key=parser_key,
            config=config,
            _assimilate_encoder=is_first_model # Use the flag
        )
        
        if not is_first_model:
            assimilation_engine.run(
                donor_model=teacher_model,
                donor_tokenizer=tokenizer, 
                donor_dataloader=loader,
                donor_task_name=task_name
            )
        
        new_branch = AdaptiveBranch(in_dim=task_d_model, out_dim=task_d_model, rank=config["DEFAULT_BRANCH_RANK"], device=device)
        new_branch_info = {
            'architecture_type': expert_config["arch_type"], 'task_name': task_name, 
            'domain': expert_config.get('domain'), 'base_model': expert_config.get('base_model'), 
            'model_id': expert_config['model_id'], 'peft_id': expert_config.get('peft_id'),
            'orig_vocab_size': len(tokenizer) if tokenizer else 'N/A'
        }

        if i > 0:
            EMM_Main.absorb_branches(
                new_branch_name=expert_config["branch_name"],
                new_branch=new_branch,
                new_branch_info=new_branch_info,
                dataloaders_main=EMM_Main.dataloaders_for_quality_check,
                dataloader_mini=loader,
                ga_params=config["GA_PARAMS"],
                **config["BRANCH_ASSIMILATION_PARAMS"]
            )
        else:
            print("\n--- Initializing with the first expert branch (natively) ---")
            EMM_Main.add_branch(
                name=expert_config["branch_name"],
                architecture_type=expert_config["arch_type"],
                d_model_branch=task_d_model,
                rank=config["DEFAULT_BRANCH_RANK"],
                peft_id=expert_config.get('peft_id')
            )
            arch_type = expert_config["arch_type"]
            branch_name = expert_config["branch_name"]
            if arch_type in EMM_Main.branched_block.branches_by_arch and branch_name in EMM_Main.branched_block.branches_by_arch[arch_type]:
                 EMM_Main.branched_block.branches_by_arch[arch_type][branch_name].load_state_dict(new_branch.state_dict())
            
            EMM_Main.branch_info[expert_config["branch_name"]].update(new_branch_info)
        
        print(f"  -> Assimilation of '{task_name}' complete. Releasing teacher model from memory.")
        del teacher_model, tokenizer, loader, embedding_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n--- EMM State After Assimilating '{task_name}' ---")
        EMM_Main.log_structure(standalone_stats=standalone_model_stats)
        
    return EMM_Main, standalone_model_stats

def run_interactive_session(emm_model: ElasticMemoryModel):
    """Launches an interactive session, allowing the user to interact with EMM through the web interface."""
    print("\n\n" + "#"*80)
    print(" INTERACTIVE SESSION STARTED ".center(80))
    print("#"*80)
    print("\nEMM model is built and the visualizer is running.")
    print("Open http://127.0.0.1:5001 in your browser to interact with the model.")
    print("\nThe application will remain active to serve the web interface.")
    print("Press Ctrl+C in this terminal to stop the server and exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down the application...")
    finally:
        print("Interactive session finished.")

if __name__ == "__main__":
    # Standard library imports
    import os
    import copy
    import pandas as pd

    # Third-party imports
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForMaskedLM,
        AutoModelForTokenClassification,
        AutoModelForCausalLM,
        DataCollatorWithPadding,
        DataCollatorForTokenClassification,
        AutoModelForImageClassification
    )
    from datasets import Dataset as HFDataset

    # --- 1. INITIAL SETUP (uses EMM_CONFIG) ---
    set_seed(EMM_CONFIG["SEED"])
    ssl._create_default_https_context = ssl._create_unverified_context
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(script_dir, EMM_CONFIG["CACHE_DIR_NAME"])
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- 2. DEFINITION OF EXTRACTOR FUNCTIONS AND THEIR MAPS ---
    def get_hf_base_model(model): return model.base_model
    def get_hf_bert_encoder(model): return model.bert
    def get_hf_roberta_encoder(model): return model.roberta
    def get_hf_esm_encoder(model): return model.esm
    def get_hf_biogpt_encoder(model): return model.biogpt
    def get_hf_distilbert_encoder(model): return model.distilbert
    def get_hf_classifier_head(model): return model.classifier
    def get_hf_self_as_head(model): return model
    def get_hf_lm_head(model: nn.Module) -> Optional[nn.Module]:
        """
        Generic extracts the head module for language models (LM).
        Checks standard attributes as well as model-specific ones, such as BioGPT ('output_projection') and ESM ('cls').
        """
        # 1. Standard name for many CausalLM/MaskedLM
        if hasattr(model, 'lm_head'):
            return model.lm_head
        
        # 2. Standard structure for BERT/RoBERTa-like MLMs
        if hasattr(model, 'cls'):
            return model.cls
        
        # 3. Specific to BioGPT, where projection is a separate attribute
        if hasattr(model, 'output_projection'):
            return model.output_projection
        
        # 4. As a backup if nothing is found
        print(f"  -> WARNING: Could not find a standard LM head ('lm_head', 'cls', 'output_projection') on {type(model).__name__}. Returning None.")
        return None

    hf_extractor_map = {
        "base_model": get_hf_base_model, "bert": get_hf_bert_encoder,
        "roberta": get_hf_roberta_encoder, "esm": get_hf_esm_encoder,
        "biogpt": get_hf_biogpt_encoder, "distilbert": get_hf_distilbert_encoder,
        "classifier": get_hf_classifier_head, "self": get_hf_self_as_head,
        "lm_head": get_hf_lm_head,
    }

    from transformers import ViTModel, DebertaModel

    def get_hf_vit_encoder(model): return model.vit
    def get_hf_deberta_encoder(model): return model.deberta

    hf_extractor_map.update({
        "vit": get_hf_vit_encoder,
        "deberta": get_hf_deberta_encoder,
    })

    # --- 3. DEFINITION OF CONFIGURATION OF EXPERTS (declarative approach) ---
    biomed_expert_tasks = {
        "biobert": {
            "model_id": "dmis-lab/biobert-v1.1", 
            "task_class": AutoModelForSequenceClassification, 
            "task_d_model": 768, 
            "num_labels": 2, 
            "encoder_extractor": "bert", 
            "head_extractor": "classifier", 
            "branch_name": "biomed_text_classification",  
            "domain": "biomed_text", 
            "base_model": "bert-base-cased"
        },
        "pubmed_bert": { 
            "model_id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
            "task_class": AutoModelForMaskedLM, 
            "task_d_model": 768, 
            "num_labels": 30522, 
            "encoder_extractor": "bert", 
            "head_extractor": "lm_head", 
            "branch_name": "pubmed_abstract_lm", 
            "domain": "biomed_abstracts", 
            "base_model": "bert-base-uncased"
        },
        "clinicalbert": {
            "model_id": "emilyalsentzer/Bio_ClinicalBERT", 
            "task_class": AutoModelForSequenceClassification, 
            "task_d_model": 768, 
            "num_labels": 2, 
            "encoder_extractor": "bert", 
            "head_extractor": "classifier", 
            "branch_name": "clinical_notes_analysis", 
            "domain": "clinical_text", 
            "base_model": "bert-base-uncased"
        },
        "chemberta": {
            "model_id": "DeepChem/ChemBERTa-77M-MTR", 
            "task_class": AutoModelForMaskedLM, 
            "task_d_model": 384, 
            "num_labels": 600, 
            "encoder_extractor": "roberta", 
            "head_extractor": "lm_head", 
            "branch_name": "chemical_smiles_model", 
            "domain": "chemistry", 
            "base_model": "roberta-base"
        },
        "esm2": {
            "model_id": "facebook/esm2_t30_150M_UR50D", 
            "task_class": AutoModelForMaskedLM, 
            "task_d_model": 640, 
            "num_labels": 33, 
            "encoder_extractor": "esm", 
            "head_extractor": "lm_head", 
            "branch_name": "protein_sequence_model", 
            "domain": "proteomics", 
            "base_model": "esm2"
        },
        "biomed_ner": {
            "model_id": "alvaroalon2/biobert_diseases_ner", 
            "task_class": AutoModelForTokenClassification, 
            "task_d_model": 768, 
            "num_labels": 3, 
            "encoder_extractor": "bert", 
            "head_extractor": "classifier", 
            "branch_name": "disease_ner", 
            "domain": "biomed_ner", 
            "base_model": "bert-base-cased"
        },
        "scibert": {
            "model_id": "allenai/scibert_scivocab_uncased", 
            "task_class": AutoModelForSequenceClassification, 
            "task_d_model": 768, 
            "num_labels": 2, 
            "encoder_extractor": "bert", 
            "head_extractor": "classifier", 
            "branch_name": "scientific_paper_classification",  
            "domain": "scientific_text", 
            "base_model": "bert-base-uncased"
        },
        "gatortron": {
            "model_id": "ufnlp/gatortron-base", 
            "task_class": AutoModelForMaskedLM, 
            "task_d_model": 768, 
            "num_labels": 30522, 
            "encoder_extractor": "bert", 
            "head_extractor": "lm_head", 
            "branch_name": "clinical_mega_lm", 
            "domain": "clinical_text", 
            "base_model": "bert-base-uncased"
        },
        "biogpt": {
            "model_id": "microsoft/biogpt", 
            "task_class": AutoModelForCausalLM, 
            "task_d_model": 1024, 
            "num_labels": 42384, 
            "encoder_extractor": "biogpt", 
            "head_extractor": "lm_head",
            "branch_name": "biomedical_generative_qa", 
            "domain": "generative_biomed", 
            "base_model": "biogpt"
        },
        "biomed_ner_comprehensive": {
            "model_id": "d4data/biomedical-ner-all", 
            "task_class": AutoModelForTokenClassification, 
            "task_d_model": 768, 
            "num_labels": 25, 
            "encoder_extractor": "distilbert", 
            "head_extractor": "classifier", 
            "branch_name": "comprehensive_biomed_ner",  
            "domain": "biomed_ner", 
            "base_model": "distilbert-base-cased"
        },
        "biomed_roberta": {
            "model_id": "allenai/biomed_roberta_base", 
            "task_class": AutoModelForSequenceClassification, 
            "task_d_model": 768, 
            "num_labels": 2, 
            "encoder_extractor": "roberta", 
            "head_extractor": "classifier", 
            "branch_name": "biomed_roberta_classification",  
            "domain": "biomed_text", 
            "base_model": "roberta-base"
        },
        "radbert": {
            "model_id": "zzxslp/RadBERT-RoBERTa-4m", 
            "task_class": AutoModelForSequenceClassification, 
            "task_d_model": 768, 
            "num_labels": 2, 
            "encoder_extractor": "roberta", 
            "head_extractor": "classifier", 
            "branch_name": "radiology_report_analysis",  
            "domain": "radiology_text", 
            "base_model": "roberta-base"
        },
        "s_biobert": {
            "model_id": "pritamdeka/S-BioBert-snli-multinli-stsb", 
            "task_class": AutoModelForSequenceClassification, 
            "task_d_model": 768, 
            "num_labels": 3,
            "encoder_extractor": "bert", 
            "head_extractor": "classifier", 
            "branch_name": "biomed_sentence_similarity", 
            "domain": "biomed_text", 
            "base_model": "biobert"
        },
        "gpt_neo_125m": {
            "model_id": "EleutherAI/gpt-neo-125M", 
            "task_class": AutoModelForCausalLM, 
            "task_d_model": 768, 
            "num_labels": 50257, 
            "encoder_extractor": "base_model", 
            "head_extractor": "lm_head", 
            "branch_name": "generative_small_neo", 
            "domain": "generative_general", 
            "base_model": "gpt-neo"
        },
        "prot_bert": {
            "model_id": "Rostlab/prot_bert", 
            "task_class": AutoModelForMaskedLM, 
            "task_d_model": 1024, 
            "num_labels": 30, 
            "encoder_extractor": "bert", 
            "head_extractor": "lm_head", 
            "branch_name": "protein_lm_protbert", 
            "domain": "proteomics", 
            "base_model": "bert-base"
        },
    }

    for k in biomed_expert_tasks: 
        biomed_expert_tasks[k]["arch_type"] = "transformer"

    # --- 4. LAUNCHING THE CONSOLIDATION PROCESS ---
    print("\n\n" + "#"*80); print(" SCENARIO: SEQUENTIAL BIOMEDICAL MODEL ASSIMILATION ".center(80)); print("#"*80)

    EMM_Main, standalone_model_stats = build_consolidated_emm(
        expert_tasks=biomed_expert_tasks,
        extractor_map=hf_extractor_map,
        device=device,
        config=EMM_CONFIG,
        cache_dir=CACHE_DIR
    )
    
    # --- A NEW CHALLENGE TO CHECK THE COUNT ---
    EMM_Main.inspect_cross_connections()

    # --- 5. ANALYSIS AND RESULTS ---
    print("\n\n" + "="*80); print(" POST-ASSIMILATION ANALYSIS ".center(80)); print("="*80)
    params_after, size_after = get_model_stats(EMM_Main, "EMM_Main (Final)")
    
    total_standalone_params = 0
    total_standalone_mb = 0.0
    for stats in standalone_model_stats.values():
        if 'params' in stats and isinstance(stats['params'], int):
            total_standalone_params += stats['params']
        if 'mb' in stats and isinstance(stats['mb'], float):
            total_standalone_mb += stats['mb']
            
    print("\n--- Final Consolidated EMM Structure ---")
    EMM_Main.log_structure(standalone_stats=standalone_model_stats)
    
    print("\n\n" + "#"*80); print(" ASSIMILATION RESULTS SUMMARY ".center(80)); print("#"*80)
    
    param_reduction = total_standalone_params - params_after
    param_reduction_percent = (param_reduction / total_standalone_params) * 100 if total_standalone_params > 0 else 0
    mb_reduction = total_standalone_mb - size_after
    mb_reduction_percent = (mb_reduction / total_standalone_mb) * 100 if total_standalone_mb > 0 else 0
    
    print(f"Total Parameters of ALL standalone models: {total_standalone_params:,}")
    print(f"Total Parameters of FINAL consolidated EMM: {params_after:,}")
    print(f"Parameter Reduction vs. Standalone Models: {param_reduction:,} ({param_reduction_percent:.2f}%)")
    print("-" * 40)
    print(f"Sum of Standalone Model Sizes (MB): {total_standalone_mb:.2f}")
    print(f"EMM Size (MB): {size_after:.2f}")
    print(f"Memory Reduction vs. Standalone Models (MB): {mb_reduction:.2f} ({mb_reduction_percent:.2f}%)")
    
    print("-" * 40)
    total_standalone_layers = 0
    for stats in standalone_model_stats.values():
        if 'layers' in stats and isinstance(stats['layers'], int):
            total_standalone_layers += stats['layers']
    final_unique_layers = len(EMM_Main.hybrid_layers)
    layer_reduction = total_standalone_layers - final_unique_layers
    layer_reduction_percent = (layer_reduction / total_standalone_layers) * 100 if total_standalone_layers > 0 else 0

    print(f"Total Encoder Layers of ALL standalone models: {total_standalone_layers:,}")
    print(f"Total UNIQUE Layers in FINAL hybrid encoder: {final_unique_layers:,}")
    print(f"Layer Reduction vs. Standalone Models: {layer_reduction:,} ({layer_reduction_percent:.2f}%)")

    run_interactive_session(EMM_Main)
