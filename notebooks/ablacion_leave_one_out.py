"""
Ablación leave-one-out sobre las tres modalidades de la red de fusión.

CÓMO USARLO
-----------
Se ejecuta como celda al final de `10_fusion.ipynb`, DESPUÉS de que estén
definidos: CrossModalAttentionBlock, fusion_data, DEVICE, TFT_DIM, BERT_DIM,
evaluate() y el pos_weight usado en train_fusion().

Reutiliza las representaciones ya extraídas y congeladas, así que no hay que
reentrenar ningún codificador. Coste estimado: pocos minutos por configuración.

QUÉ MIDE
--------
Quita una modalidad completa y mantiene todo lo demás idéntico (mecanismo de
atención, hiperparámetros, semilla, particiones). Responde: ¿aporta cada
modalidad información que las otras no tengan ya?

Con solo 2 modalidades, K/V es una secuencia de UN token, de modo que el
softmax de la atención devuelve peso 1,0 por construcción: el bloque deja de
seleccionar y se reduce a una mezcla residual aprendida. Es una ablación
válida de MODALIDAD, no de MECANISMO.

`has_note` se elimina junto con la rama de texto (ver USE_HAS_NOTE abajo):
indica si existe informe de radiología, así que mantenerlo dejaría un proxy
de la modalidad textual dentro de la variante "sin texto" y contaminaría la
comparación. Con USE_HAS_NOTE = "always" se aísla solo el CONTENIDO del
informe y no su disponibilidad.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from metricas import brier, ece

# "auto"   -> has_note solo si la modalidad de texto participa  (por defecto)
# "always" -> has_note siempre presente en la rama tabular
USE_HAS_NOTE = "auto"

SEED = 42
BATCH_SIZE = 512


# ==========================================================================
# Modelo
# ==========================================================================
class LeaveOneOutFusion(nn.Module):
    """TrueCrossModalFusion restringida a un subconjunto de modalidades.

    modalities: subconjunto de {"tab", "tft", "bert"} (2 o 3 elementos).
    Cada modalidad genera su query y atiende a las restantes como K/V,
    exactamente igual que el modelo completo.
    """

    def __init__(self, modalities, tab_dim, tft_dim=64, bert_dim=768,
                 d_model=128, n_heads=4, dropout=0.2, use_has_note=None):
        super().__init__()
        self.modalities = list(modalities)
        assert 2 <= len(self.modalities) <= 3, "se requieren 2 o 3 modalidades"

        if use_has_note is None:
            use_has_note = "bert" in self.modalities
        self.use_has_note = bool(use_has_note)

        in_dim = {
            "tab":  tab_dim + (1 if self.use_has_note else 0),
            "tft":  tft_dim,
            "bert": bert_dim,
        }
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(in_dim[m], d_model),
                             nn.LayerNorm(d_model), nn.ReLU())
            for m in self.modalities
        })
        self.ca = nn.ModuleDict({
            m: CrossModalAttentionBlock(d_model, n_heads, dropout)  # noqa: F821
            for m in self.modalities
        })
        self.classifier = nn.Sequential(
            nn.Linear(len(self.modalities) * d_model, d_model),
            nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, X_tab, X_tft, X_bert, has_note):
        raw = {"tab": X_tab, "tft": X_tft, "bert": X_bert}
        if self.use_has_note:
            raw["tab"] = torch.cat([X_tab, has_note.unsqueeze(-1)], dim=-1)

        h = {m: self.proj[m](raw[m]) for m in self.modalities}

        enriched, attn = [], {}
        for m in self.modalities:
            others = [h[o] for o in self.modalities if o != m]
            out, w = self.ca[m](h[m], others)
            enriched.append(out)
            attn[m] = w.mean(1).squeeze(1)          # [B, n_fuentes]

        logit = self.classifier(torch.cat(enriched, dim=-1)).squeeze(-1)
        return logit, attn


# ==========================================================================
# Entrenamiento (réplica exacta de los hiperparámetros de train_fusion)
# ==========================================================================
def _as_tensor(x):
    return x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))


def _split_tensors(fusion_data, split):
    X_tab, X_tft, X_bert, has_note, y = fusion_data[split]
    return tuple(_as_tensor(t).float() for t in (X_tab, X_tft, X_bert, has_note, y))


@torch.no_grad()
def _predict(model, fusion_data, split):
    model.eval()
    X_tab, X_tft, X_bert, has_note, y = _split_tensors(fusion_data, split)
    probs = []
    for i in range(0, len(y), 4096):
        sl = slice(i, i + 4096)
        logit, _ = model(X_tab[sl].to(DEVICE), X_tft[sl].to(DEVICE),      # noqa: F821
                         X_bert[sl].to(DEVICE), has_note[sl].to(DEVICE))
        probs.append(torch.sigmoid(logit).cpu().numpy())
    return y.numpy(), np.concatenate(probs)


def sens_at_spec(y_true, y_prob, spec=0.90):
    """Sensibilidad a la especificidad indicada, igual que evaluate() del cuaderno."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    mask = fpr <= (1.0 - spec)
    return float(tpr[mask].max()) if mask.any() else 0.0


def train_loo(fusion_data, modalities, pos_weight_val,
              n_epochs=80, patience=12, lr=1e-3, seed=SEED, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = _split_tensors(fusion_data, "train")
    tab_dim = Xtr[0].shape[1]

    use_hn = None if USE_HAS_NOTE == "auto" else True
    model = LeaveOneOutFusion(modalities, tab_dim, TFT_DIM, BERT_DIM,   # noqa: F821
                              use_has_note=use_hn).to(DEVICE)           # noqa: F821

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5)
    pos_w = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)  # noqa: F821
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    n = len(Xtr[4])
    best_auroc, best_state, waited = 0.0, None, 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xt, xf, xb, hn, yy = (t[idx].to(DEVICE) for t in Xtr)        # noqa: F821
            opt.zero_grad()
            logit, _ = model(xt, xf, xb, hn)
            loss = crit(logit, yy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        y_val, p_val = _predict(model, fusion_data, "val")
        auroc = roc_auc_score(y_val, p_val)
        sched.step(auroc)

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break

        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"    época {epoch:3d}  AUROC val = {auroc:.4f}"
                  f"  (mejor {best_auroc:.4f})")

    model.load_state_dict(best_state)
    return model, best_auroc


# ==========================================================================
# Driver
# ==========================================================================
CONFIGS = [
    ("Completa (tab + tft + bert)", ["tab", "tft", "bert"]),
    ("Sin texto (bert)",            ["tab", "tft"]),
    ("Sin vitales (tft)",           ["tab", "bert"]),
    ("Sin tabular (tab)",           ["tft", "bert"]),
]


def run_leave_one_out(fusion_data, pos_weight_val, horizon_label=""):
    rows = []
    for name, mods in CONFIGS:
        print(f"\n=== {name} {horizon_label} ===")
        model, auroc_val = train_loo(fusion_data, mods, pos_weight_val)

        y_te, p_te = _predict(model, fusion_data, "test")
        rows.append({
            "configuracion": name,
            "modalidades":   "+".join(mods),
            "auroc_val":     round(auroc_val, 4),
            "auroc_test":    round(roc_auc_score(y_te, p_te), 4),
            "auprc_test":    round(average_precision_score(y_te, p_te), 4),
            "sens_esp90":    round(sens_at_spec(y_te, p_te, 0.90), 4),
            "brier_test_sin_calibrar": round(brier(y_te, p_te), 4),
            "ece_test_sin_calibrar":   round(ece(y_te, p_te), 4),
            "n_params":      sum(p.numel() for p in model.parameters()),
        })
        print(f"  -> test AUROC {rows[-1]['auroc_test']:.4f} | "
              f"AUPRC {rows[-1]['auprc_test']:.4f} | "
              f"Sens@Esp90 {rows[-1]['sens_esp90']:.4f} | "
              f"Brier {rows[-1]['brier_test_sin_calibrar']:.4f} | "
              f"ECE {rows[-1]['ece_test_sin_calibrar']:.4f}")

    import pandas as pd
    df = pd.DataFrame(rows)

    base = df.loc[df.configuracion.str.startswith("Completa")].iloc[0]
    for col in ("auroc_test", "auprc_test", "sens_esp90",
                "brier_test_sin_calibrar",
                "ece_test_sin_calibrar"):
        df[f"delta_{col}"] = (df[col] - base[col]).round(4)

    return df


# --------------------------------------------------------------------------
# EJECUTAR  (nombres reales del cuaderno: fusion_6h / fusion_12h, get_pw)
# --------------------------------------------------------------------------
# df_loo_6h  = run_leave_one_out(fusion_6h,  get_pw(fusion_6h),  "[6h]")
# df_loo_12h = run_leave_one_out(fusion_12h, get_pw(fusion_12h), "[12h]")
# df_loo_6h.to_csv(OUT_DIR / "ablacion_loo_6h.csv", index=False)
# df_loo_12h.to_csv(OUT_DIR / "ablacion_loo_12h.csv", index=False)
# print(df_loo_6h.to_string(index=False)); print(df_loo_12h.to_string(index=False))
