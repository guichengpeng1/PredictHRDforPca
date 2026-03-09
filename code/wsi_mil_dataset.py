import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


Image.MAX_IMAGE_PIXELS = None


def detect_base_magnification(slide: openslide.OpenSlide, default: float = 40.0) -> float:
    for key in ["aperio.AppMag", "openslide.objective-power"]:
        value = slide.properties.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            try:
                return float(str(value).rstrip("xX"))
            except Exception:
                continue
    return float(default)


def iter_positions(limit: int, step: int) -> List[int]:
    if limit <= 0:
        return [0]
    positions = list(range(0, limit, step))
    if positions[-1] != limit:
        positions.append(limit)
    return positions


class WSITileBagDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_tiles: int = 64,
        tile_size: int = 224,
        target_magnification: float = 10.0,
        stride: int | None = None,
        transform=None,
        cache_dir: str | Path = "outputs/tile_cache",
        tissue_white_threshold: int = 220,
        tissue_fraction_threshold: float = 0.30,
        training: bool = True,
        seed: int = 42,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.num_tiles = int(num_tiles)
        self.tile_size = int(tile_size)
        self.target_magnification = float(target_magnification)
        self.stride = int(stride) if stride is not None else int(tile_size)
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tissue_white_threshold = int(tissue_white_threshold)
        self.tissue_fraction_threshold = float(tissue_fraction_threshold)
        self.training = bool(training)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.df)

    def _cache_path(self, slide_path: str) -> Path:
        cache_key = (
            f"{slide_path}|{self.target_magnification}|{self.tile_size}|"
            f"{self.stride}|{self.tissue_white_threshold}|{self.tissue_fraction_threshold}"
        )
        cache_name = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{cache_name}.json"

    def _load_or_build_cache(self, slide_path: str) -> Dict:
        cache_path = self._cache_path(slide_path)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        slide = openslide.OpenSlide(slide_path)
        base_mag = detect_base_magnification(slide)
        desired_downsample = max(base_mag / self.target_magnification, 1.0)
        level = int(slide.get_best_level_for_downsample(desired_downsample))
        level_downsample = float(slide.level_downsamples[level])
        level_width, level_height = slide.level_dimensions[level]

        thumb_max = 2048
        scale = min(thumb_max / level_width, thumb_max / level_height, 1.0)
        thumb_size = (
            max(1, int(round(level_width * scale))),
            max(1, int(round(level_height * scale))),
        )
        thumb = slide.get_thumbnail(thumb_size).convert("RGB")
        thumb_arr = np.asarray(thumb)
        tissue_mask = thumb_arr.mean(axis=2) < self.tissue_white_threshold

        max_x = max(level_width - self.tile_size, 0)
        max_y = max(level_height - self.tile_size, 0)
        coords: List[List[int]] = []
        for x in iter_positions(max_x, self.stride):
            for y in iter_positions(max_y, self.stride):
                x0 = min(int(np.floor(x * scale)), tissue_mask.shape[1] - 1)
                y0 = min(int(np.floor(y * scale)), tissue_mask.shape[0] - 1)
                x1 = max(
                    x0 + 1,
                    min(int(np.ceil((x + self.tile_size) * scale)), tissue_mask.shape[1]),
                )
                y1 = max(
                    y0 + 1,
                    min(int(np.ceil((y + self.tile_size) * scale)), tissue_mask.shape[0]),
                )
                tissue_fraction = float(tissue_mask[y0:y1, x0:x1].mean())
                if tissue_fraction < self.tissue_fraction_threshold:
                    continue
                coords.append(
                    [
                        int(round(x * level_downsample)),
                        int(round(y * level_downsample)),
                    ]
                )

        if not coords:
            center_x = max(int(slide.dimensions[0] / 2 - self.tile_size / 2), 0)
            center_y = max(int(slide.dimensions[1] / 2 - self.tile_size / 2), 0)
            coords = [[center_x, center_y]]

        payload = {
            "slide_path": slide_path,
            "level": level,
            "tile_size": self.tile_size,
            "num_candidates": len(coords),
            "coords_level0": coords,
        }
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return payload

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        slide_path = str(row["slide_path"])
        cache = self._load_or_build_cache(slide_path)
        coords = cache["coords_level0"]
        level = int(cache["level"])

        rng_seed = self.seed + index
        rng = np.random.default_rng(rng_seed if not self.training else None)
        sample_indices = rng.choice(
            len(coords),
            size=self.num_tiles,
            replace=len(coords) < self.num_tiles,
        )

        slide = openslide.OpenSlide(slide_path)
        tiles = []
        for coord_idx in sample_indices:
            loc_x, loc_y = coords[int(coord_idx)]
            tile = slide.read_region(
                (int(loc_x), int(loc_y)),
                level,
                (self.tile_size, self.tile_size),
            ).convert("RGB")
            if self.transform is not None:
                tile = self.transform(tile)
            else:
                tile_arr = np.asarray(tile, dtype=np.float32) / 255.0
                tile = torch.from_numpy(tile_arr).permute(2, 0, 1)
            tiles.append(tile)

        bag = torch.stack(tiles, dim=0)
        hrd_score = torch.tensor(float(row["hrd_score"]), dtype=torch.float32)
        hrd_status = torch.tensor(float(row["hrd_status"]), dtype=torch.float32)
        return bag, hrd_score, hrd_status, row["patient_barcode"], slide_path
