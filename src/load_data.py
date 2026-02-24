from __future__ import annotations

from pathlib import Path


def ensure_dirs(base_dir: Path) -> None:
    """Cria a estrutura de diretórios do projeto (idempotente)."""
    (base_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (base_dir / "outputs" / "charts").mkdir(parents=True, exist_ok=True)
    (base_dir / "outputs" / "reports").mkdir(parents=True, exist_ok=True)


def write_readme_seed_hint() -> None:
    """Só um aviso útil pra manter reprodutibilidade."""
    print("[DICA] Dataset simulado é reprodutível (seed=42). Você pode alterar no main.py.")