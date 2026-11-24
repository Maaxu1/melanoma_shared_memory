import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from kax import Kax

log = logging.getLogger("kax_manager")


def _resolve_store_dir(configured_dir: str | None) -> Path:
    candidate = (
        Path(configured_dir).expanduser()
        if configured_dir
        else (Path.home() / "kax_store")
    )
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "kax_store"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


class KaxBufferManager:
    def __init__(self, store_dir: str | None = None):
        env_store = store_dir or os.getenv("KAX_STORE_DIR")
        self.store_dir: Path = _resolve_store_dir(env_store)

        self.client = Kax(
            registry_url=os.getenv("KAX_REGISTRY_URL", "http://localhost:5060/api"),
            storage_url=os.getenv("KAX_STORAGE_URL", "http://localhost:5070/api"),
            broker=os.getenv("KAX_BROKER", "localhost"),
            port=int(os.getenv("KAX_PORT", "1883")),
            workers=int(os.getenv("KAX_WORKERS", "1")),
            chunk_size=int(os.getenv("KAX_CHUNK_SIZE", "2")),
        )

    # ---------- utilidades ----------
    def _write_files(self, prefix: str, raw_files: list[tuple[str, bytes]]) -> Path:
        batch_dir = self.store_dir / f"{prefix}_{uuid.uuid4().hex[:12]}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for i, (name, raw) in enumerate(raw_files):
            suffix = Path(name).suffix.lower() or ".jpg"
            (batch_dir / f"img_{i:05d}{suffix}").write_bytes(raw)
        return batch_dir

    # ---------- modo RAM (Kax) ----------
    def write_ram_buffer(self, raw_files: list[tuple[str, bytes]]):
        batch_dir = self._write_files("ram", raw_files)
        meta = self.client.load_res(str(batch_dir))
        return batch_dir, meta

    def read_ram_buffer(self, handle) -> list[bytes]:
        _, meta = handle
        resources = sorted(meta["resources"], key=lambda r: r["path"])
        return [self.client.getRes(r["hash"]) for r in resources]

    def cleanup_ram_buffer(self, handle) -> None:
        batch_dir, meta = handle
        for r in meta.get("resources", []):
            try:
                self.client.removeShm(r["hash"])
            except Exception as exc:
                log.warning("removeShm falló para %s: %s", r["hash"], exc)
        shutil.rmtree(batch_dir, ignore_errors=True)

    # ---------- modo disco (sin Kax) ----------
    def write_disk_buffer(self, raw_files: list[tuple[str, bytes]]):
        batch_dir = self._write_files("disk", raw_files)
        return batch_dir, sorted(batch_dir.iterdir())

    def read_disk_buffer(self, handle) -> list[bytes]:
        _, paths = handle
        return [p.read_bytes() for p in paths]

    def cleanup_disk_buffer(self, handle) -> None:
        batch_dir, _ = handle
        shutil.rmtree(batch_dir, ignore_errors=True)