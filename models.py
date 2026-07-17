
from __future__ import annotations
from typing import Dict, List, Tuple
import esm as esm_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .chem import clean_seq
from .config import DEVICE


def masked_mean(H: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
     
    if mask is None:
        return H.mean(dim=1)
    m = mask.unsqueeze(-1).to(H.dtype)
    return (H * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)


class AlphaAdapter(nn.Module):
   
    def __init__(self, d_in: int, m: int, alpha_init: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_in)
        self.down = nn.Linear(d_in, m, bias=False)
        self.up = nn.Linear(m, d_in, bias=False)
        self.ln2 = nn.LayerNorm(d_in)
        self.alpha = nn.Parameter(torch.full([], float(alpha_init)))
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.normal_(self.up.weight, std=1e-3)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        Z = self.ln2(self.up(F.gelu(self.down(self.ln1(H)))))
        return H + torch.tanh(self.alpha) * Z


class _EmbedCache:
    
    def __init__(self, enabled: bool, max_entries: int):
        self.enabled = enabled
        self.max_entries = max_entries
        self.store: Dict[str, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    def get(self, k: str):
        if not self.enabled:
            return None
        v = self.store.get(k)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, k: str, v: torch.Tensor) -> None:
        if not self.enabled:
            return
        if len(self.store) >= self.max_entries:
            self.store.pop(next(iter(self.store)))
        self.store[k] = v.detach().to("cpu")


def _rebuild_padded(cached: List[torch.Tensor], d: int):
    """Re-assemble variable-length cached embeddings into a padded batch."""
    L = max(c.size(0) for c in cached)
    H = torch.zeros(len(cached), L, d)
    mask = torch.zeros(len(cached), L, dtype=torch.bool)
    for i, c in enumerate(cached):
        H[i, :c.size(0)] = c
        mask[i, :c.size(0)] = True
    return H.to(DEVICE), mask.to(DEVICE)


class DrugEncoder(nn.Module):
    
    def __init__(self, cfg: Dict, use_adapter: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["chemberta_name"])
        backbone = AutoModel.from_pretrained(cfg["chemberta_name"])
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone
        self.d_out = int(backbone.config.hidden_size)
        self.use_adapter = use_adapter
        self.adapter = (AlphaAdapter(self.d_out, cfg["adapter_m"], cfg["alpha_init"])
                        if use_adapter else None)
        self.max_len = cfg["max_drug_len"]
        self.cache = _EmbedCache(cfg["cache_embeddings"], cfg["max_cache_entries"])

    @torch.no_grad()
    def _embed(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = [self.cache.get(s) for s in smiles_list]
        if self.cache.enabled and all(c is not None for c in cached):
            return _rebuild_padded(cached, self.d_out)

        enc = self.tokenizer(smiles_list, padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt").to(DEVICE)
        H = self.backbone(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].bool()
        if self.cache.enabled:
            for i, s in enumerate(smiles_list):
                self.cache.put(s, H[i, :int(mask[i].sum().item())])
        return H, mask

    def forward(self, smiles_list: List[str], return_mask: bool = False):
        H, mask = self._embed(smiles_list)      # frozen, no grad
        if self.use_adapter:
            H = self.adapter(H)                 # adapter is trainable
        return (H, mask) if return_mask else H


class ProteinEncoder(nn.Module):
   

    def __init__(self, cfg: Dict, use_adapter: bool = True):
        super().__init__()
        loader = getattr(esm_lib.pretrained, cfg["esm_name"])
        backbone, alphabet = loader()
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone
        self.alphabet = alphabet
        self.converter = alphabet.get_batch_converter()
        self.repr_layer = cfg["esm_repr_layer"]
        self.d_out = int(backbone.embed_dim)        # 1280 for the 650M model
        self.use_adapter = use_adapter
        self.adapter = (AlphaAdapter(self.d_out, cfg["adapter_m"], cfg["alpha_init"])
                        if use_adapter else None)
        self.max_len = cfg["max_prot_len"]
        self.cache = _EmbedCache(cfg["cache_embeddings"], cfg["max_cache_entries"])

    @torch.no_grad()
    def _embed(self, seq_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = [clean_seq(s, self.max_len) for s in seq_list]
        cached = [self.cache.get(k) for k in keys]
        if self.cache.enabled and all(c is not None for c in cached):
            return _rebuild_padded(cached, self.d_out)

        data = [(f"p{i}", k) for i, k in enumerate(keys)]
        _, _, tokens = self.converter(data)
        tokens = tokens.to(DEVICE)
        out = self.backbone(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        H_full = out["representations"][self.repr_layer].float()
        a = self.alphabet
        valid = (tokens != a.padding_idx) & (tokens != a.cls_idx) & (tokens != a.eos_idx)
        H, mask = H_full[:, 1:-1, :], valid[:, 1:-1]
        if self.cache.enabled:
            for i, k in enumerate(keys):
                self.cache.put(k, H[i, :int(mask[i].sum().item())])
        return H, mask

    def forward(self, seq_list: List[str], return_mask: bool = False):
        H, mask = self._embed(seq_list)
        if self.use_adapter:
            H = self.adapter(H)
        return (H, mask) if return_mask else H


class PerceiverIO(nn.Module):
    
    def __init__(self, cfg: Dict, d_drug: int, d_prot: int):
        super().__init__()
        D, M = cfg["d_shared"], cfg["perceiver_M"]
        L, H = cfg["perceiver_L"], cfg["perceiver_heads"]
        self.proj_drug = nn.Linear(d_drug, D)
        self.proj_prot = nn.Linear(d_prot, D)
        self.mod_emb = nn.Embedding(2, D)
        self.latents = nn.Parameter(torch.randn(1, M, D) * 0.02)
        self.cross_in = nn.MultiheadAttention(D, H, dropout=0.1, batch_first=True)
        self.cross_in_ln = nn.LayerNorm(D)
        layer = nn.TransformerEncoderLayer(d_model=D, nhead=H, dim_feedforward=D * 4,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, L)
        self.out_query = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.cross_out = nn.MultiheadAttention(D, H, dropout=0.1, batch_first=True)
        self.out_ln = nn.LayerNorm(D)

    def forward(self, H_drug: torch.Tensor, H_prot: torch.Tensor) -> torch.Tensor:
        B, dev = H_drug.size(0), H_drug.device
        d = self.proj_drug(H_drug) + self.mod_emb(
            torch.zeros(1, H_drug.size(1), dtype=torch.long, device=dev))
        p = self.proj_prot(H_prot) + self.mod_emb(
            torch.ones(1, H_prot.size(1), dtype=torch.long, device=dev))
        X = torch.cat([d, p], dim=1)
        Z = self.latents.expand(B, -1, -1)
        Z2, _ = self.cross_in(Z, X, X)
        Z2 = self.cross_in_ln(Z + Z2)
        Z3 = self.transformer(Z2)
        q = self.out_query.expand(B, -1, -1)
        y, _ = self.cross_out(q, Z3, Z3)
        return self.out_ln(y).squeeze(1)


class MLPHead(nn.Module):
   
    def __init__(self, cfg: Dict, d_in: int | None = None):
        super().__init__()
        D = cfg["d_shared"] if d_in is None else int(d_in)
        H = cfg["mlp_hidden"]
        self.net = nn.Sequential(
            nn.Linear(D, H), nn.BatchNorm1d(H), nn.ReLU(),
            nn.Dropout(cfg["mlp_dropout"]), nn.Linear(H, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class SimCFDTBA(nn.Module):
    """Full proposed model: adapters + Perceiver IO fusion + MLP head."""

    variant = "simcf"

    def __init__(self, cfg: Dict):
        super().__init__()
        self.drug = DrugEncoder(cfg, use_adapter=True)
        self.prot = ProteinEncoder(cfg, use_adapter=True)
        self.fusion = PerceiverIO(cfg, self.drug.d_out, self.prot.d_out)
        self.head = MLPHead(cfg)
        print(f"[model] SimCF-DTBA | d_drug={self.drug.d_out} "
              f"d_prot={self.prot.d_out} d_shared={cfg['d_shared']}")

    def forward(self, smiles: List[str], seqs: List[str]) -> torch.Tensor:
        return self.head(self.fusion(self.drug(smiles), self.prot(seqs)))


class VanillaLMMLP(nn.Module):
   
    variant = "vanilla"

    def __init__(self, cfg: Dict):
        super().__init__()
        self.drug = DrugEncoder(cfg, use_adapter=False)
        self.prot = ProteinEncoder(cfg, use_adapter=False)
        d_in = self.drug.d_out + self.prot.d_out
        self.head = MLPHead(cfg, d_in=d_in)
        print(f"[model] VanillaLM+MLP (ablation) | d_drug={self.drug.d_out} "
              f"d_prot={self.prot.d_out} concat={d_in}")

    def forward(self, smiles: List[str], seqs: List[str]) -> torch.Tensor:
        H_d, m_d = self.drug(smiles, return_mask=True)
        H_p, m_p = self.prot(seqs, return_mask=True)
        return self.head(torch.cat([masked_mean(H_d, m_d), masked_mean(H_p, m_p)], dim=-1))


def build_model(variant: str, cfg: Dict) -> nn.Module:
    if variant == "simcf":
        return SimCFDTBA(cfg).to(DEVICE)
    if variant == "vanilla":
        return VanillaLMMLP(cfg).to(DEVICE)
    raise ValueError(f"unknown variant: {variant} (expected 'simcf' or 'vanilla')")
