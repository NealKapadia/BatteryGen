"""Inverse-design engine shared by the local app (app.py) and the production server
(server.py).

A request like "possible additives for a zinc aqueous battery" becomes:
  LLM parse -> property/chemistry spec  (+ UI slider overrides: novelty + targets)
  -> novelty-controlled latent optimization
  -> RDKit validate + 3D
  -> electrolyte conductivity/coordination readout for the cation
  -> optional LLM mechanistic explanation.

De-novo: molecules are decoded from the VAE prior/latent, not retrieved.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import config
import data
import infer
import llm
from membership import MolportIndex


def molblock_3d(smiles: str) -> Optional[str]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 1
    if AllChem.EmbedMolecule(mol, p) != 0 and AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        AllChem.Compute2DCoords(mol)
    else:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
    return Chem.MolToMolBlock(mol)


class DesignEngine:
    def __init__(self, ckpt=None, device="cpu"):
        self.device = torch.device(device)
        self.net, self.vocab, self.ck, _ = infer.load_model(ckpt, device=self.device)
        self.mean, self.std = data.load_stats()
        self.std = np.where(self.std > 1e-8, self.std, 1.0)
        self.index = MolportIndex()
        self.state: Dict = {}
        self._load_electrolyte()
        self._load_dft()

    # ---- electrolyte property model (optional) ----------------------------
    def _load_electrolyte(self):
        self.elec = None
        path = config.CKPT_DIR / "electrolyte_model.pt"
        if not path.exists():
            return
        try:
            import electrolyte as E

            ck = torch.load(path, map_location=self.device)
            net = E.PropNet(ck["stats"]["dim"], len(ck["stats"]["targets"])).to(self.device)
            net.load_state_dict(ck["model"]); net.eval()
            self.elec = {"net": net, "stats": ck["stats"], "E": E}
        except Exception as e:
            print(f"[design] electrolyte model not loaded: {e}")

    def electrolyte_predict(self, smiles, cation="Li", source="oedb", conc=1.0, temp=298.0,
                            anion=None) -> Optional[Dict]:
        if self.elec is None:
            return None
        E = self.elec["E"]; st = self.elec["stats"]
        lat = E.latents_for(self.net, self.vocab, [smiles], self.device)
        if smiles not in lat:
            return None
        feat = E.Featurizer(config.LATENT_DIM, st["use_anion"])
        anion_lat = None
        if st["use_anion"] and anion:
            anion_lat = next(iter(E.latents_for(self.net, self.vocab, [anion], self.device).values()), None)
        vec = feat.vector([lat[smiles]], [1.0], anion_lat, cation, source, conc, temp)
        if vec is None:
            return None
        xm = torch.tensor(st["xm"], device=self.device); xs = torch.tensor(st["xs"], device=self.device)
        x = (torch.tensor(vec, device=self.device) - xm) / xs
        with torch.no_grad():
            pred = self.elec["net"](x.unsqueeze(0)).cpu().numpy()[0] * np.asarray(st["ys"]) + np.asarray(st["ym"])
        if st["log_target"]:
            pred = 10 ** pred
        return {t: round(float(v), 4) for t, v in zip(st["targets"], pred)}

    # ---- DFT / electrochemical head (optional) ----------------------------
    def _load_dft(self):
        """Load dft_latest.pt (HOMO/LUMO/gap/dipole/ip/ea/window head) if present.
        It carries its own encoder weights (end-to-end fine-tuned), so it predicts
        from SMILES self-consistently rather than reusing the base latents."""
        self.dft = None
        path = config.CKPT_DIR / "dft_latest.pt"
        if not path.exists():
            return
        try:
            import model as M
            import finetune_dft as FT

            ck = torch.load(path, map_location=self.device)
            net = M.SelfiesVAE(len(self.vocab), config.N_PROPS).to(self.device)
            net.load_state_dict(ck["model"]); net.eval()
            head = FT.DftHead(config.LATENT_DIM, len(ck["dft_targets"])).to(self.device)
            head.load_state_dict(ck["dft_head"]); head.eval()
            self.dft = {"net": net, "head": head, "targets": ck["dft_targets"],
                        "mean": np.asarray(ck["dft_mean"]), "std": np.asarray(ck["dft_std"]), "FT": FT}
        except Exception as e:
            print(f"[design] DFT head not loaded: {e}")

    @torch.no_grad()
    def dft_predict(self, smiles_list: List[str]) -> Dict[str, Dict[str, float]]:
        """{smiles: {target: value}} for HOMO/LUMO/gap/dipole/ip/ea/window. Empty if no head."""
        if self.dft is None:
            return {}
        FT = self.dft["FT"]
        mu = FT._encode_mols(self.dft["net"], self.vocab, smiles_list, self.device)  # {smi: mu}
        out = {}
        for smi, m in mu.items():
            z = torch.tensor(m, device=self.device).unsqueeze(0)
            pred = self.dft["head"](z).cpu().numpy()[0] * self.dft["std"] + self.dft["mean"]
            out[smi] = {t: round(float(v), 4) for t, v in zip(self.dft["targets"], pred)}
        return out

    # ---- encode a known molecule to its latent mean -----------------------
    @torch.no_grad()
    def _encode_smiles(self, smiles: str) -> Optional[torch.Tensor]:
        import selfies as sf

        canon = data.canonical_smiles(smiles)
        if not canon:
            return None
        try:
            sel = sf.encoder(canon)
        except Exception:
            return None
        if not sel or sf.len_selfies(sel) > config.MAX_SELFIES_LEN - 2:
            return None
        ids, length = self.vocab.encode(sel)
        toks = torch.tensor([ids], dtype=torch.long, device=self.device)
        mu, _ = self.net.encode(toks, torch.tensor([length]))
        return mu  # [1, latent]

    # ---- core generation with novelty control -----------------------------
    # NOTE: not @torch.no_grad — the latent optimization below needs gradients on z.
    # The decode step (net.sample) disables grad internally.
    def _sample_pool(self, spec, novelty, start_z, pop):
        target = torch.from_numpy(data.spec_to_condition(spec)).float().to(self.device)
        mask = torch.from_numpy(infer.specified_mask(spec)).to(self.device)
        spread = 0.1 + 0.5 * novelty
        if start_z is not None:
            z0 = start_z.to(self.device).repeat(pop, 1) + spread * torch.randn(pop, self.net.latent_dim, device=self.device)
        else:
            z0 = torch.randn(pop, self.net.latent_dim, device=self.device) * (1.0 + 0.4 * novelty)
        z = z0.detach()
        if bool(mask.any()):
            z = z0.detach().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=0.1)
            for _ in range(180):
                opt.zero_grad()
                loss = ((self.net.prop_head(z) - target)[:, mask] ** 2).mean() + 1e-3 * (z ** 2).mean()
                loss.backward(); opt.step()
            z = z.detach()
        temperature = 0.5 + 0.8 * novelty
        cond = target.unsqueeze(0).repeat(pop, 1)
        seqs = self.net.sample(pop, cond, self.vocab.bos, self.vocab.eos, z=z,
                               temperature=temperature, device=self.device)
        return seqs, z

    def optimize(self, spec, novelty=0.5, start_z=None):
        pop = int(64 + 96 * novelty)
        seqs, z = self._sample_pool(spec, novelty, start_z, pop)
        name_to_idx = {p: i for i, p in enumerate(config.PROPERTIES)}
        desired_novel = novelty >= 0.5
        lean = abs(novelty - 0.5) * 2.0
        best = None
        for i, s in enumerate(seqs.tolist()):
            smi = self.vocab.decode_to_smiles(s)
            if not smi:
                continue
            props = infer.score_smiles(smi)
            if props is None:
                continue
            mid = self.index.get_id(smi, already_canonical=True) if self.index.available else None
            is_novel = mid is None
            err = 0.0
            for p, tv in spec.items():
                pr = config.PROPERTY_ALIASES.get(str(p).lower().strip(), p)
                if pr not in name_to_idx:
                    continue
                j = name_to_idx[pr]
                if isinstance(tv, (list, tuple)) and len(tv) == 2:  # range: 0 error inside [lo,hi]
                    lo, hi = sorted((float(tv[0]), float(tv[1])))
                    d = 0.0 if lo <= props[pr] <= hi else min(abs(props[pr] - lo), abs(props[pr] - hi))
                    err += (d / self.std[j]) ** 2
                else:
                    err += ((props[pr] - float(tv)) / self.std[j]) ** 2
            score = err + (0.0 if is_novel == desired_novel else 0.6 * lean)
            if best is None or score < best[0]:
                best = (score, smi, z[i:i + 1].detach().cpu(), props, mid, is_novel)
        return best

    # ---- public API -------------------------------------------------------
    def design(self, prompt: str, sliders: Optional[Dict] = None, refine: bool = False) -> Dict:
        sliders = sliders or {}
        novelty = float(sliders.get("novelty", 0.5))
        cation = sliders.get("cation")
        if refine and self.state.get("spec") is not None:
            spec = llm.parse_relative(prompt, self.state["spec"], self.state.get("props") or {})
            parsed = {"summary": "refined", "cation": self.state.get("cation"),
                      "aqueous": self.state.get("aqueous"), "role": self.state.get("role")}
            start_z = self.state.get("z")
        else:
            parsed = llm.parse_design(prompt) if prompt else {"spec": {}, "summary": ""}
            spec = dict(parsed.get("spec", {}))
            cation = cation or parsed.get("cation")
            start_z = None
        # slider property overrides (any config.PROPERTIES key present in sliders)
        for k, v in sliders.items():
            kk = config.PROPERTY_ALIASES.get(str(k).lower().strip(), k)
            if kk in config.PROPERTIES and v not in (None, ""):
                spec[kk] = float(v)

        best = self.optimize(spec, novelty=novelty, start_z=start_z)
        if best is None:
            return {"ok": False, "error": "no valid molecule found — adjust sliders or prompt"}
        _, smi, z, props, mid, is_novel = best
        elec = None
        if cation:
            elec = self.electrolyte_predict(
                smi, cation=cation, source=sliders.get("source", "oedb"),
                conc=float(sliders.get("conc", 1.0)), temp=float(sliders.get("temp", 298.0)),
                anion=sliders.get("anion"))
        self.state = {"smiles": smi, "z": z, "spec": spec, "props": props, "request": prompt or self.state.get("request", ""),
                      "cation": cation, "aqueous": parsed.get("aqueous"), "role": parsed.get("role"),
                      "elec": elec, "history": (self.state.get("history", []) + [smi])[-12:]}
        def _fmt(v):  # scalar or [lo,hi] range, JSON-friendly
            if isinstance(v, (list, tuple)):
                return [round(float(x), 2) for x in v]
            return round(float(v), 2)
        return {
            "ok": True, "smiles": smi, "molblock": molblock_3d(smi),
            "properties": {k: round(v, 2) for k, v in props.items()},
            "target": {k: _fmt(v) for k, v in spec.items()},
            "molport_id": mid, "novel": is_novel, "novelty": novelty,
            "cation": cation, "role": parsed.get("role"), "aqueous": parsed.get("aqueous"),
            "summary": parsed.get("summary", ""), "electrolyte": elec,
            "history": self.state["history"],
        }

    # ---- mass generation (range-aware) ------------------------------------
    def _resolve_spec(self, prompt, sliders, spec):
        """Build (point_targets, ranges, cation, summary). spec/slider values may be a
        scalar (point target) or a [lo, hi] pair (range -> midpoint target + filter)."""
        sliders = sliders or {}
        cation = sliders.get("cation")
        summary = ""
        if spec is None:
            parsed = llm.parse_design(prompt) if prompt else {"spec": {}, "summary": ""}
            spec = dict(parsed.get("spec", {}))
            cation = cation or parsed.get("cation")
            summary = parsed.get("summary", "")
        raw = dict(spec)
        for k, v in sliders.items():
            kk = config.PROPERTY_ALIASES.get(str(k).lower().strip(), k)
            if kk in config.PROPERTIES and v not in (None, ""):
                raw[kk] = v
        point, ranges = {}, {}
        for k, v in raw.items():
            kk = config.PROPERTY_ALIASES.get(str(k).lower().strip(), k)
            if kk not in config.PROPERTIES:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 2:
                lo, hi = sorted((float(v[0]), float(v[1])))
                ranges[kk] = (lo, hi); point[kk] = 0.5 * (lo + hi)
            else:
                point[kk] = float(v)
        return point, ranges, cation, summary

    def batch(self, prompt: Optional[str] = None, sliders: Optional[Dict] = None,
              n: int = 100, spec: Optional[Dict] = None, pool_mult: int = 12,
              want_electrolyte: bool = False, want_3d: bool = False,
              source: str = "oedb", conc: float = 1.0, temp: float = 298.0,
              anion: Optional[str] = None) -> Dict:
        """Generate up to n unique, valid, spec-matching molecules as a ranked table.

        Over-generates a large latent-optimized pool (cheap at ~100% validity), filters
        to any [lo, hi] ranges, dedups, scores (RDKit props + novelty + optional
        electrolyte + DFT head), and returns the closest-to-target first. This is the
        robust way to honor sliders/ranges given that point-conditioning is soft."""
        sliders = sliders or {}
        novelty = float(sliders.get("novelty", 0.5))
        point, ranges, cation, summary = self._resolve_spec(prompt, sliders, spec)
        cation = cation or sliders.get("cation")
        pop = int(min(max(n * pool_mult, 256), 8000))
        seqs, z = self._sample_pool(point, novelty, None, pop)
        name_to_idx = {p: i for i, p in enumerate(config.PROPERTIES)}
        desired_novel = novelty >= 0.5
        rows, seen, zsel = [], set(), []
        for i, s in enumerate(seqs.tolist()):
            smi = self.vocab.decode_to_smiles(s)
            if not smi or smi in seen:
                continue
            seen.add(smi)
            props = infer.score_smiles(smi)
            if props is None:
                continue
            if any(not (lo <= props[k] <= hi) for k, (lo, hi) in ranges.items()):
                continue
            mid = self.index.get_id(smi, already_canonical=True) if self.index.available else None
            err = sum(((props[k] - tv) / self.std[name_to_idx[k]]) ** 2 for k, tv in point.items())
            rows.append({"smiles": smi, "molport_id": mid or "", "novel": mid is None,
                         "score": round(float(err), 4),
                         "properties": {k: round(v, 3) for k, v in props.items()}})
            zsel.append(z[i:i + 1])
        rows.sort(key=lambda r: (0 if r["novel"] == desired_novel else 1, r["score"]))
        rows = rows[:n]
        # optional enrichments only on the returned set (keep it cheap)
        if want_electrolyte and cation:
            for r in rows:
                r["electrolyte"] = self.electrolyte_predict(
                    r["smiles"], cation=cation, source=source, conc=conc, temp=temp, anion=anion)
        if self.dft is not None:
            dft = self.dft_predict([r["smiles"] for r in rows])
            for r in rows:
                r["dft"] = dft.get(r["smiles"])
        if want_3d:
            for r in rows:
                r["molblock"] = molblock_3d(r["smiles"])
        return {"ok": True, "n_requested": n, "n_returned": len(rows), "pool": pop,
                "spec": {k: round(v, 2) for k, v in point.items()},
                "ranges": {k: [lo, hi] for k, (lo, hi) in ranges.items()},
                "cation": cation, "summary": summary, "molecules": rows}

    # ---- design from a user's own tested molecules ------------------------
    @torch.no_grad()
    def suggest_similar(self, tested: List, n: int = 50, spread: float = 0.4,
                        temperature: float = 0.7, want_3d: bool = False) -> Dict:
        """Given molecules the user already tested (optionally with a performance score),
        seed the latent at the best performer(s) and sample nearby novel molecules with
        matching properties. tested = ["CCO", ...] or [{"smiles":..., "score":...}, ...]
        (higher score = better). Returns suggestions ranked by similarity to a top seed."""
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        items = []
        for t in tested:
            if isinstance(t, str):
                items.append((t, 0.0))
            elif isinstance(t, dict) and t.get("smiles"):
                items.append((t["smiles"], float(t.get("score", t.get("performance", 0.0)))))
        items = [(data.canonical_smiles(s), sc) for s, sc in items]
        items = [(s, sc) for s, sc in items if s]
        if not items:
            return {"ok": False, "error": "provide >=1 valid tested molecule"}
        items.sort(key=lambda x: x[1], reverse=True)
        seeds = items[:min(3, len(items))]
        known = {s for s, _ in items}
        per_seed = max(8, (n * 8) // len(seeds))

        def fp(smi):
            m = Chem.MolFromSmiles(smi)
            return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None

        rows, seen = [], set(known)
        for smi0, sc0 in seeds:
            mu = self._encode_smiles(smi0)
            if mu is None:
                continue
            fp0 = fp(smi0)
            props0 = infer.score_smiles(smi0) or {}
            cond = (torch.tensor((np.asarray(data.compute_descriptors(Chem.MolFromSmiles(smi0)),
                    np.float32) - self.mean) / self.std, device=self.device).unsqueeze(0).repeat(per_seed, 1))
            z = mu.repeat(per_seed, 1) + spread * torch.randn(per_seed, self.net.latent_dim, device=self.device)
            seqs = self.net.sample(per_seed, cond, self.vocab.bos, self.vocab.eos, z=z,
                                   temperature=temperature, device=self.device)
            for s in seqs.tolist():
                smi = self.vocab.decode_to_smiles(s)
                if not smi or smi in seen:
                    continue
                seen.add(smi)
                props = infer.score_smiles(smi)
                if props is None:
                    continue
                fpi = fp(smi)
                sim = DataStructs.TanimotoSimilarity(fp0, fpi) if (fp0 and fpi) else 0.0
                mid = self.index.get_id(smi, already_canonical=True) if self.index.available else None
                rows.append({"smiles": smi, "molport_id": mid or "", "novel": mid is None,
                             "like_seed": smi0, "seed_performance": sc0,
                             "similarity": round(float(sim), 3),
                             "properties": {k: round(v, 3) for k, v in props.items()}})
        # most similar to a high performer first
        rows.sort(key=lambda r: (r["seed_performance"], r["similarity"]), reverse=True)
        rows = rows[:n]
        if want_3d:
            for r in rows:
                r["molblock"] = molblock_3d(r["smiles"])
        return {"ok": True, "n_returned": len(rows), "seeds": [s for s, _ in seeds],
                "molecules": rows}

    def explain(self) -> Dict:
        s = self.state
        if not s.get("smiles"):
            return {"ok": False, "error": "design a molecule first"}
        text = llm.explain(s.get("request", ""), s["smiles"], s.get("props", {}), s.get("elec"))
        return {"ok": True, "explanation": text}

    def finetune(self, smiles_list: List[str], steps: int = 150, lr: float = 1e-4,
                 max_mols: int = 200) -> Dict:
        """Light reconstruction fine-tune on a user's molecules (low LR, few steps) so
        generation leans toward their chemistry. Saves user.pt. Deliberately small to
        avoid catastrophic forgetting."""
        import selfies as sf
        import model as M
        from rdkit import Chem

        rows = []
        for smi in smiles_list[:max_mols]:
            canon = data.canonical_smiles(smi)
            if not canon:
                continue
            try:
                sel = sf.encoder(canon)
            except Exception:
                continue
            if not sel or sf.len_selfies(sel) > config.MAX_SELFIES_LEN - 2:
                continue
            ids, length = self.vocab.encode(sel)
            mol = Chem.MolFromSmiles(canon)
            cond = (np.asarray(data.compute_descriptors(mol), np.float32) - self.mean) / self.std
            rows.append((ids, length, cond))
        if len(rows) < 2:
            return {"ok": False, "error": "need >=2 valid molecules"}
        toks = torch.tensor([r[0] for r in rows], dtype=torch.long, device=self.device)
        lens = torch.tensor([r[1] for r in rows], dtype=torch.long)
        cond = torch.tensor(np.stack([r[2] for r in rows]), dtype=torch.float32, device=self.device)
        maxlen = int(lens.max())
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.net.train()
        loss_v = 0.0
        for _ in range(steps):
            opt.zero_grad()
            logits, mu, logvar, prop = self.net(toks[:, :maxlen], lens, cond)
            loss, _ = M.vae_loss(logits, toks[:, :maxlen][:, 1:], mu, logvar, prop, cond,
                                 self.vocab.pad, beta=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), config.GRAD_CLIP)
            opt.step()
            loss_v = float(loss.item())
        self.net.eval()
        torch.save({"model": self.net.state_dict(), "hparams": self.ck.get("hparams", {})},
                   config.CKPT_DIR / "user.pt")
        return {"ok": True, "n": len(rows), "loss": round(loss_v, 3)}
