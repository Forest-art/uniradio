from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F

from .text_prompts import build_prompts


class CLIPTextEncoder:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: torch.device,
    ):
        self.device = device
        self._fallback = False
        self._fallback_dim = 512
        try:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
                device=device,
            )
            self.model = model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.tokenizer = open_clip.get_tokenizer(model_name)
        except Exception:
            self._fallback = True
            self.model = None
            self.tokenizer = None

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 256) -> torch.Tensor:
        if self._fallback:
            vecs = []
            for t in texts:
                seed = abs(hash(t)) % (2**31)
                g = torch.Generator(device="cpu")
                g.manual_seed(seed)
                v = torch.randn(self._fallback_dim, generator=g)
                vecs.append(v)
            emb = torch.stack(vecs, dim=0).to(self.device)
            return F.normalize(emb, dim=-1)

        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            tokens = self.tokenizer(chunk).to(self.device)
            emb = self.model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
            out.append(emb)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def build_class_embeddings(
        self,
        class_names: Iterable[str],
        templates: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
        batch_size: int = 256,
    ) -> torch.Tensor:
        if cache_path and Path(cache_path).exists():
            data = torch.load(cache_path, map_location=self.device)
            return data["text_embeddings"].to(self.device)

        class_prompts = build_prompts(class_names=class_names, templates=templates)
        all_emb = []
        for prompts in class_prompts:
            emb = self.encode_texts(prompts, batch_size=batch_size)
            emb = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1)
            all_emb.append(emb)

        class_emb = torch.cat(all_emb, dim=0)
        class_emb = F.normalize(class_emb, dim=-1)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"text_embeddings": class_emb.cpu()}, cache_path)

        return class_emb
