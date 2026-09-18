"""
Logit Lens analysis — unified for MiniCPM-o-4.5 and Qwen3-Omni-30B-A3B-Instruct.

Algorithm (applied per layer l, per token position p):
    logits_l_p = lm_head( final_norm( hidden_state_l[:, p, :] ) )

MiniCPM-o-4.5 architecture:
    model.llm.model.embed_tokens / .layers[0..35] / .norm / .lm_head
    model.vpm  – vision encoder
    model.apm  – audio encoder

Qwen3-Omni-30B-A3B-Instruct architecture:
    model.thinker.model.embed_tokens / .layers[0..47] / .norm / .lm_head
    model.thinker.visual      – vision encoder
    model.thinker.audio_tower – audio encoder
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Arch helpers
# ---------------------------------------------------------------------------

def _attr(obj, *path):
    for k in path:
        obj = getattr(obj, k, None)
        if obj is None:
            return None
    return obj


def _find_llm(model):
    # Qwen3-Omni → thinker;  MiniCPM → llm;  generic fallback
    for name in ("thinker", "llm", "language_model"):
        sub = getattr(model, name, None)
        if sub is not None and hasattr(sub, "lm_head"):
            return sub
    if hasattr(model, "lm_head"):
        return model
    raise ValueError(f"Cannot find CausalLM inside {type(model).__name__}.")


def _find_layers(llm):
    for path in (("model", "layers"), ("layers",)):
        v = _attr(llm, *path)
        if v is not None:
            return v
    raise ValueError("Cannot find transformer layers.")


def _find_norm(llm):
    for path in (("model", "norm"), ("norm",)):
        v = _attr(llm, *path)
        if v is not None:
            return v
    raise ValueError("Cannot find final LayerNorm.")


def _find_embed(llm):
    for path in (("model", "embed_tokens"), ("embed_tokens",)):
        v = _attr(llm, *path)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Position-type tracker
# ---------------------------------------------------------------------------

_MODALITY_UNKNOWN = "text"
_MODALITY_IMAGE   = "image"
_MODALITY_AUDIO   = "audio"
_MODALITY_VIDEO   = "video"


def _build_pos_types(
    input_ids: torch.Tensor,
    image_token_id, audio_token_id, video_token_id=None,
) -> List[str]:
    ids = input_ids[0].tolist()
    labels = []
    for tid in ids:
        if image_token_id is not None and tid == image_token_id:
            labels.append(_MODALITY_IMAGE)
        elif audio_token_id is not None and tid == audio_token_id:
            labels.append(_MODALITY_AUDIO)
        elif video_token_id is not None and tid == video_token_id:
            labels.append(_MODALITY_VIDEO)
        else:
            labels.append(_MODALITY_UNKNOWN)
    return labels


def _pos_label(pos: int, pos_types: List[str], input_ids: List[int], tokenizer) -> str:
    if pos >= len(pos_types):
        return f"pos{pos}"
    mod = pos_types[pos]
    if mod == _MODALITY_IMAGE:
        return "<img_patch>"
    if mod == _MODALITY_AUDIO:
        return "<aud_patch>"
    if mod == _MODALITY_VIDEO:
        return "<vid_patch>"
    tok = tokenizer.decode([input_ids[pos]], skip_special_tokens=False)
    return repr(tok)


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class LogitLensAnalyzer:
    """
    Logit-lens analysis — works with both MiniCPM-o-4.5 and Qwen3-Omni.

    Quick start:
        analyzer = LogitLensAnalyzer(model, tokenizer)
        res, ids, pos = analyzer.analyze("The Eiffel Tower is in")
        res, ids, pos = analyzer.analyze("Describe:", images=[pil_img])
        res, ids, pos = analyzer.analyze("Transcribe:", audio=[audio_arr])
        analyzer.print_table(res, ids, pos)
        analyzer.plot(res, ids, pos, "out.png")
    """

    def __init__(self, model: torch.nn.Module, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer

        self._llm     = _find_llm(model)
        self._layers  = _find_layers(self._llm)
        self._norm    = _find_norm(self._llm)
        self._lm_head = self._llm.lm_head
        self._embed   = _find_embed(self._llm)

        # Token IDs: prefer llm/thinker config, then top-level model config
        cfg = getattr(self._llm, "config", None) or getattr(model, "config", None)
        self._img_tok_id = getattr(cfg, "image_token_id",
                           getattr(cfg, "im_token_id", None))
        self._aud_tok_id = getattr(cfg, "audio_token_id", None)
        self._vid_tok_id = getattr(cfg, "video_token_id", None)

        n = len(self._layers)
        print(f"[LogitLens] {n} transformer layers | "
              f"norm={type(self._norm).__name__} | "
              f"lm_head={type(self._lm_head).__name__}")
        has_vision = hasattr(self._llm, "visual") or hasattr(model, "vpm")
        has_audio  = hasattr(self._llm, "audio_tower") or hasattr(model, "apm")
        print(f"[LogitLens] Multimodal encoders — vision={has_vision}, audio={has_audio}")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _register_hooks(self) -> Tuple[Dict[int, torch.Tensor], list]:
        buf, handles = {}, []

        if self._embed is not None:
            def _emb(_, __, out):
                buf[-1] = out.detach()
            handles.append(self._embed.register_forward_hook(_emb))

        for i, layer in enumerate(self._layers):
            def _lay(_, __, out, idx=i):
                h = out[0] if isinstance(out, tuple) else out
                buf[idx] = h.detach()
            handles.append(layer.register_forward_hook(_lay))

        return buf, handles

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _project(self, hidden: torch.Tensor, top_k: int):
        target_dtype = next(self._lm_head.parameters()).dtype
        normed  = self._norm(hidden.to(target_dtype))
        logits  = self._lm_head(normed)
        probs   = torch.softmax(logits[0], dim=-1)
        top_p, top_id = torch.topk(probs, top_k)
        top_tok = [
            self.tokenizer.decode([i.item()], skip_special_tokens=False)
            for i in top_id
        ]
        return top_tok, top_p.tolist()

    # ------------------------------------------------------------------
    # Build results from captured buffer
    # ------------------------------------------------------------------

    def _apply_logit_lens(
        self,
        buf: Dict[int, torch.Tensor],
        valid_pos: List[int],
        top_k: int,
    ) -> Dict[int, Dict[int, Tuple]]:
        def _display(raw):
            return 0 if raw == -1 else raw + 1

        results: Dict[int, Dict[int, Tuple]] = {}
        for raw in sorted(buf.keys()):
            dk = _display(raw)
            h  = buf[raw]
            results[dk] = {}
            for pos in valid_pos:
                if pos >= h.shape[1]:
                    continue
                results[dk][pos] = self._project(h[:, pos, :], top_k)
        return results

    # ------------------------------------------------------------------
    # Text-only
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        positions: Optional[List[int]] = None,
        top_k: int = 5,
        images=None,
        audio=None,
        videos=None,
    ) -> Tuple[Dict, List[int], List[int]]:
        if images is not None or audio is not None or videos is not None:
            return self.analyze_multimodal(
                text, images=images, audio=audio, videos=videos,
                positions=positions, top_k=top_k,
            )

        if positions is None:
            positions = list(range(1, 8))

        enc       = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"]
        seq_len   = input_ids.shape[1]
        valid_pos = [p for p in positions if 0 <= p < seq_len]
        if not valid_pos:
            raise ValueError(f"No valid positions. seq_len={seq_len}, asked={positions}")

        device    = next(self._llm.parameters()).device
        input_ids = input_ids.to(device)

        buf, handles = self._register_hooks()
        try:
            self._llm(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        results = self._apply_logit_lens(buf, valid_pos, top_k)
        return results, input_ids[0].tolist(), valid_pos

    # ------------------------------------------------------------------
    # Multimodal
    # ------------------------------------------------------------------

    def analyze_multimodal(
        self,
        text: str,
        images=None,
        audio=None,
        videos=None,
        positions: Optional[List[int]] = None,
        top_k: int = 5,
    ) -> Tuple[Dict, List[int], List[int]]:
        if positions is None:
            positions = list(range(1, 8))

        images = images or []
        audio  = audio  or []
        videos = videos or []

        # Structured content list (works for both Qwen3-Omni and MiniCPM paths)
        content: list = []
        for img in images:
            content.append({"type": "image", "image": img})
        for arr in audio:
            content.append({"type": "audio", "audio": arr})
        for vid in videos:
            content.append({"type": "video", "video": vid})
        content.append({"type": "text", "text": text})
        msgs = [{"role": "user", "content": content}]

        device = next(self._llm.parameters()).device

        try:
            inputs = self._preprocess_with_processor(msgs, device)
            return self._run_with_inputs(inputs, positions, top_k)
        except Exception as proc_err:
            pass

        try:
            return self._run_via_model_hook(msgs, positions, top_k)
        except Exception as hook_err:
            raise RuntimeError(
                "Multimodal forward failed via both processor and hook paths.\n"
                f"  processor error : {proc_err}\n"
                f"  hook error      : {hook_err}"
            ) from None

    # ------------------------------------------------------------------
    # Processor path — tries Qwen3-Omni style, falls back to MiniCPM style
    # ------------------------------------------------------------------

    def _get_model_path(self):
        return (
            getattr(getattr(self.model, "config", None), "_name_or_path", None)
            or getattr(getattr(self._llm, "config", None), "_name_or_path", None)
        )

    def _preprocess_with_processor(self, msgs, device):
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(
            self._get_model_path(), trust_remote_code=True
        )

        pil_images, audio_arrays, video_list = [], [], []
        for msg in msgs:
            for item in msg.get("content", []):
                if not isinstance(item, dict):
                    continue
                t = item.get("type", "")
                if t == "image":
                    pil_images.append(item["image"])
                elif t == "audio":
                    audio_arrays.append(item["audio"])
                elif t == "video":
                    video_list.append(item["video"])

        # Try Qwen3-Omni style: apply_chat_template inserts <image>/<audio> tokens
        try:
            text_formatted = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            kwargs = dict(text=text_formatted, return_tensors="pt")
            if pil_images:
                kwargs["images"] = pil_images
            if audio_arrays:
                kwargs["audio"] = audio_arrays
            if video_list:
                kwargs["videos"] = video_list
            inputs = proc(**kwargs)
            return {k: v.to(device) if hasattr(v, "to") else v
                    for k, v in inputs.items()}
        except Exception:
            pass

        # Fall back to MiniCPM style: flat text + AutoProcessor kwargs
        text_parts = [
            item["text"] for msg in msgs
            for item in msg.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        flat_text = " ".join(text_parts)
        kwargs = dict(text=flat_text, return_tensors="pt")
        if pil_images:
            kwargs["images"] = pil_images
        if audio_arrays:
            kwargs["audios"] = audio_arrays
        inputs = proc(**kwargs)
        return {k: v.to(device) if hasattr(v, "to") else v
                for k, v in inputs.items()}

    def _run_with_inputs(self, inputs: dict, positions: List[int], top_k: int):
        seq_len   = inputs["input_ids"].shape[1]
        valid_pos = [p for p in positions if 0 <= p < seq_len]
        if not valid_pos:
            raise ValueError(f"No valid positions (seq_len={seq_len}).")

        buf, handles = self._register_hooks()
        try:
            with torch.no_grad():
                self._llm(**inputs)
        finally:
            for h in handles:
                h.remove()

        results  = self._apply_logit_lens(buf, valid_pos, top_k)
        flat_ids = inputs["input_ids"][0].tolist()
        return results, flat_ids, valid_pos

    # ------------------------------------------------------------------
    # Hook fallback — dispatches to model.chat() or thinker.generate()
    # ------------------------------------------------------------------

    def _run_via_model_hook(self, msgs, positions: List[int], top_k: int):
        if hasattr(self.model, "chat"):
            return self._run_via_chat_hook(msgs, positions, top_k)
        return self._run_via_generate_hook(msgs, positions, top_k)

    def _run_via_chat_hook(self, msgs, positions: List[int], top_k: int):
        """MiniCPM: intercept model.chat()'s first forward pass."""

        class _CaptureSignal(Exception):
            pass

        captured_buf: Dict[int, torch.Tensor] = {}
        handles = []

        if self._embed is not None:
            def _emb(_, __, out):
                captured_buf[-1] = out.detach().float()
            handles.append(self._embed.register_forward_hook(_emb))

        for i, layer in enumerate(self._layers):
            def _lay(_, __, out, idx=i):
                h = out[0] if isinstance(out, tuple) else out
                captured_buf[idx] = h.detach().float()
            handles.append(layer.register_forward_hook(_lay))

        def _abort(*_):
            raise _CaptureSignal()
        handles.append(self._lm_head.register_forward_hook(_abort))

        # Convert structured content dicts back to MiniCPM flat format
        minicpm_msgs = []
        for msg in msgs:
            flat_content = []
            for item in msg.get("content", []):
                if isinstance(item, dict):
                    t = item.get("type", "")
                    if t == "image":
                        flat_content.append(item["image"])
                    elif t == "audio":
                        flat_content.append(item["audio"])
                    elif t == "text":
                        flat_content.append(item["text"])
                else:
                    flat_content.append(item)
            minicpm_msgs.append({"role": msg["role"], "content": flat_content})

        try:
            with torch.no_grad():
                self.model.chat(
                    msgs=minicpm_msgs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=1,
                )
        except _CaptureSignal:
            pass
        finally:
            for h in handles:
                h.remove()

        if not captured_buf:
            raise RuntimeError("No hidden states captured during chat() forward.")

        any_h     = next(iter(captured_buf.values()))
        seq_len   = any_h.shape[1]
        valid_pos = [p for p in positions if 0 <= p < seq_len]
        results   = self._apply_logit_lens(captured_buf, valid_pos, top_k)
        flat_ids  = list(range(seq_len))
        return results, flat_ids, valid_pos

    def _run_via_generate_hook(self, msgs, positions: List[int], top_k: int):
        """Qwen3-Omni: intercept thinker.generate()'s first forward pass."""

        class _CaptureSignal(Exception):
            pass

        captured_buf: Dict[int, torch.Tensor] = {}
        handles = []

        if self._embed is not None:
            def _emb(_, __, out):
                captured_buf[-1] = out.detach().float()
            handles.append(self._embed.register_forward_hook(_emb))

        for i, layer in enumerate(self._layers):
            def _lay(_, __, out, idx=i):
                h = out[0] if isinstance(out, tuple) else out
                captured_buf[idx] = h.detach().float()
            handles.append(layer.register_forward_hook(_lay))

        def _abort(*_):
            raise _CaptureSignal()
        handles.append(self._lm_head.register_forward_hook(_abort))

        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(
            self._get_model_path(), trust_remote_code=True
        )
        pil_images, audio_arrays, video_list = [], [], []
        for msg in msgs:
            for item in msg.get("content", []):
                if not isinstance(item, dict):
                    continue
                t = item.get("type", "")
                if t == "image":
                    pil_images.append(item["image"])
                elif t == "audio":
                    audio_arrays.append(item["audio"])
                elif t == "video":
                    video_list.append(item["video"])

        text_formatted = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        kwargs = dict(text=text_formatted, return_tensors="pt")
        if pil_images:
            kwargs["images"] = pil_images
        if audio_arrays:
            kwargs["audio"] = audio_arrays
        if video_list:
            kwargs["videos"] = video_list
        inputs = proc(**kwargs)

        device = next(self._llm.parameters()).device
        model_dtype = next(self._llm.parameters()).dtype
        _moved = {}
        for k, v in inputs.items():
            if not hasattr(v, "to"):
                _moved[k] = v
            elif hasattr(v, "is_floating_point") and v.is_floating_point():
                _moved[k] = v.to(device=device, dtype=model_dtype)
            else:
                _moved[k] = v.to(device)
        inputs = _moved

        try:
            with torch.no_grad():
                self._llm.generate(**inputs, max_new_tokens=1)
        except _CaptureSignal:
            pass
        finally:
            for h in handles:
                h.remove()

        if not captured_buf:
            raise RuntimeError("No hidden states captured during generate() forward.")

        any_h     = next(iter(captured_buf.values()))
        seq_len   = any_h.shape[1]
        valid_pos = [p for p in positions if 0 <= p < seq_len]
        results   = self._apply_logit_lens(captured_buf, valid_pos, top_k)
        flat_ids  = inputs["input_ids"][0].tolist()
        return results, flat_ids, valid_pos

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_table(
        self,
        results: Dict,
        input_ids: List[int],
        valid_pos: List[int],
        top_k: int = 3,
    ) -> None:
        pos_types = _build_pos_types(
            torch.tensor([input_ids]),
            self._img_tok_id, self._aud_tok_id, self._vid_tok_id,
        )
        tok_labels = [
            _pos_label(p, pos_types, input_ids, self.tokenizer)
            for p in valid_pos
        ]

        model_name = type(self.model).__name__
        print("\n" + "=" * 90)
        print(f"  LOGIT LENS — {model_name}")
        print("=" * 90)

        for pos, label in zip(valid_pos, tok_labels):
            print(f"\n  Position {pos}  (input = {label})")
            print(f"  {'Layer':<8}  " +
                  "  ".join(f"{'Top-'+str(i+1):<22}" for i in range(top_k)))
            print("  " + "-" * 78)
            for lk in sorted(results.keys()):
                if pos not in results[lk]:
                    continue
                toks, probs = results[lk][pos]
                cols = [
                    f"{repr(toks[i]):<14} {probs[i]:.3f}"
                    for i in range(min(top_k, len(toks)))
                ]
                print(f"  L{lk:<7}" + "  ".join(cols))

    # ------------------------------------------------------------------
    # Digit-probability probe  (for 1-7 Likert experiments)
    # ------------------------------------------------------------------

    def probe_digit_probs(
        self,
        text: str,
        images=None,
        audio=None,
        videos=None,
        digits: str = "1234567",
    ) -> Dict[int, Dict[str, float]]:
        digit_ids: Dict[str, int] = {}
        for d in digits:
            ids_plain = self.tokenizer.encode(d, add_special_tokens=False)
            ids_space = self.tokenizer.encode(" " + d, add_special_tokens=False)
            if len(ids_plain) == 1:
                digit_ids[d] = ids_plain[0]
            elif len(ids_space) == 1:
                digit_ids[d] = ids_space[0]
            else:
                digit_ids[d] = ids_plain[-1]

        buf, handles = self._register_hooks()

        if images is not None or audio is not None or videos is not None:
            images_  = images  or []
            audio_   = audio   or []
            videos_  = videos  or []
            content: list = []
            for img in images_:
                content.append({"type": "image", "image": img})
            for arr in audio_:
                content.append({"type": "audio", "audio": arr})
            for vid in videos_:
                content.append({"type": "video", "video": vid})
            content.append({"type": "text", "text": text})
            msgs = [{"role": "user", "content": content}]

            device = next(self._llm.parameters()).device
            if hasattr(self.model, "chat"):
                # MiniCPM: self._llm doesn't embed audio through apm; always use model.chat()
                for h in handles:
                    h.remove()
                handles = []
                buf, handles = self._register_hooks()
                self._run_via_model_hook(msgs, positions=[0], top_k=1)
            else:
                try:
                    inputs = self._preprocess_with_processor(msgs, device)
                    with torch.no_grad():
                        self._llm(**inputs)
                except Exception:
                    for h in handles:
                        h.remove()
                    handles = []
                    buf, handles = self._register_hooks()
                    self._run_via_model_hook(msgs, positions=[0], top_k=1)
        else:
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids_t = enc["input_ids"].to(next(self._llm.parameters()).device)
            try:
                with torch.no_grad():
                    self._llm(input_ids=input_ids_t)
            finally:
                for h in handles:
                    h.remove()
                handles = []

        for h in handles:
            h.remove()

        if not buf:
            raise RuntimeError("No hidden states captured.")

        last_pos = next(iter(buf.values())).shape[1] - 1

        def _display(raw):
            return 0 if raw == -1 else raw + 1

        layer_probs: Dict[int, Dict[str, float]] = {}
        with torch.no_grad():
            for raw in sorted(buf.keys()):
                dk = _display(raw)
                h  = buf[raw][:, last_pos, :]
                normed = self._norm(h)
                logits = self._lm_head(normed)
                probs  = torch.softmax(logits[0], dim=-1)
                layer_probs[dk] = {
                    d: probs[tid].item() for d, tid in digit_ids.items()
                }

        return layer_probs

    def plot(
        self,
        results: Dict,
        input_ids: List[int],
        valid_pos: List[int],
        save_path: str = "logit_lens.png",
    ) -> None:
        pos_types = _build_pos_types(
            torch.tensor([input_ids]),
            self._img_tok_id, self._aud_tok_id, self._vid_tok_id,
        )

        layers   = sorted(results.keys())
        n_layers = len(layers)
        n_pos    = len(valid_pos)

        prob_mat = np.zeros((n_layers, n_pos))
        tok_mat  = [[""] * n_pos for _ in range(n_layers)]

        for li, lk in enumerate(layers):
            for pi, pos in enumerate(valid_pos):
                if pos not in results[lk]:
                    continue
                toks, probs = results[lk][pos]
                prob_mat[li, pi] = probs[0]
                label = toks[0].strip().replace("\n", "\\n")
                tok_mat[li][pi] = label if label else "<sp>"

        cell_w = max(1.8, 10.0 / max(n_pos, 1))
        cell_h = max(0.28, 8.0 / max(n_layers, 1))
        fig, ax = plt.subplots(figsize=(n_pos * cell_w + 2.5, n_layers * cell_h + 2.5))

        im = ax.imshow(prob_mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)

        fs = max(5, min(9, int(220 / n_layers)))
        for li in range(n_layers):
            for pi in range(n_pos):
                p  = prob_mat[li, pi]
                fg = "white" if p > 0.55 else "black"
                ax.text(pi, li, tok_mat[li][pi],
                        ha="center", va="center",
                        fontsize=fs, color=fg, fontweight="bold")

        x_labels = []
        for pos in valid_pos:
            mod = pos_types[pos] if pos < len(pos_types) else "text"
            if mod == _MODALITY_IMAGE:
                icon = "IMG"
            elif mod == _MODALITY_AUDIO:
                icon = "AUD"
            elif mod == _MODALITY_VIDEO:
                icon = "VID"
            else:
                raw = (self.tokenizer.decode([input_ids[pos]], skip_special_tokens=False)
                       if pos < len(input_ids) and isinstance(input_ids[pos], int)
                       else "?")
                icon = repr(raw)
            x_labels.append(f"pos {pos}\n{icon}")

        ax.set_xticks(range(n_pos))
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{lk}" for lk in layers], fontsize=fs)

        ax.set_xlabel("Token Position  (IMG=image patch, AUD=audio patch, VID=video patch)", fontsize=11)
        ax.set_ylabel("Layer  (L0 = embedding output)", fontsize=11)
        ax.set_title(
            f"Logit Lens — {type(self.model).__name__}\n"
            "Top-1 predicted token at each layer × position",
            fontsize=13,
        )

        plt.colorbar(im, ax=ax, label="Top-1 softmax probability", shrink=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[LogitLens] Saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    _BACKENDS = {
        "minicpm":    "/work/wilzwork23/models/MiniCPM-o-4_5",
        "qwen3-omni": "/work/wilzwork23/models/Qwen3-Omni-30B-A3B-Instruct",
    }

    def _parse_args():
        p = argparse.ArgumentParser(
            description="Logit Lens — MiniCPM-o-4.5 or Qwen3-Omni-30B-A3B-Instruct",
            epilog=(
                "Examples:\n"
                "  python logit_lens.py --backend minicpm --text 'Paris is'\n"
                "  python logit_lens.py --backend qwen3-omni --image photo.jpg --text 'Describe'\n"
                "  python logit_lens.py --backend qwen3-omni --audio speech.wav --text 'Transcribe:'"
            ),
        )
        p.add_argument("--backend", choices=list(_BACKENDS), default="minicpm")
        p.add_argument("--model",   default=None,
                       help="Override local model path (default: auto from --backend)")
        p.add_argument("--text",     default="The capital of France is Paris.")
        p.add_argument("--image",    nargs="*", metavar="PATH")
        p.add_argument("--audio",    nargs="*", metavar="PATH")
        p.add_argument("--positions", nargs="+", type=int, default=list(range(1, 8)))
        p.add_argument("--top-k",    type=int, default=5)
        p.add_argument("--output",   default="logit_lens.png")
        p.add_argument("--dtype",    choices=["bf16", "fp16", "fp32"], default="bf16")
        p.add_argument("--device",   default="auto")
        return p.parse_args()

    def _load_model(args):
        model_path = args.model or _BACKENDS[args.backend]
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        print(f"Loading [{args.backend}]: {model_path}  (dtype={args.dtype}, device={device})")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if args.backend == "minicpm":
            from transformers import AutoConfig, AutoModel
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(cfg, "tts_config") and cfg.tts_config is not None:
                for _k, _v in [("top_p", 0.9), ("top_k", 50), ("repetition_penalty", 1.0),
                                ("interleaved", False), ("attention_type", "full_attention"),
                                ("recomputed_chunks", 1)]:
                    if not hasattr(cfg.tts_config, _k):
                        setattr(cfg.tts_config, _k, _v)
            model = AutoModel.from_pretrained(
                model_path, config=cfg, trust_remote_code=True,
                torch_dtype=dtype, device_map=device,
            )
        else:
            from transformers import Qwen3OmniMoeForConditionalGeneration
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, device_map=device,
            )
        model.eval()
        return model, tokenizer

    def _load_images(paths):
        if not paths:
            return []
        from PIL import Image
        imgs = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            imgs.append(img)
            print(f"  Loaded image : {path}  ({img.size[0]}x{img.size[1]})")
        return imgs

    def _load_audio(paths, target_sr=16_000):
        if not paths:
            return []
        arrays = []
        for path in paths:
            try:
                import soundfile as sf
                data, sr = sf.read(path, dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != target_sr:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                import librosa
                data, sr = librosa.load(path, sr=target_sr, mono=True)
            arrays.append(data)
            print(f"  Loaded audio : {path}")
        return arrays

    _args             = _parse_args()
    _model, _tokenizer = _load_model(_args)
    _images            = _load_images(_args.image or [])
    _audio             = _load_audio(_args.audio or [])

    _analyzer = LogitLensAnalyzer(_model, _tokenizer)
    _results, _input_ids, _valid_pos = _analyzer.analyze(
        text      = _args.text,
        images    = _images or None,
        audio     = _audio  or None,
        positions = _args.positions,
        top_k     = _args.top_k,
    )

    _analyzer.print_table(_results, _input_ids, _valid_pos, top_k=3)
    _analyzer.plot(_results, _input_ids, _valid_pos, save_path=_args.output)
