from pathlib import Path
from typing import List, Optional

DATA_ROOT = Path("/home/timmy/data")

def list_geo_traits() -> List[str]:
    geo_root = DATA_ROOT / "GEO"
    return sorted([p.name for p in geo_root.iterdir() if p.is_dir()])

def list_geo_cohorts(trait: str) -> List[str]:
    trait_dir = DATA_ROOT / "GEO" / trait
    if not trait_dir.exists():
        raise FileNotFoundError(f"Trait not found: {trait}")
    return sorted([p.name for p in trait_dir.iterdir() if p.is_dir()])

def get_geo_cohort_dir(trait: str, cohort: str) -> Path:
    cohort_dir = DATA_ROOT / "GEO" / trait / cohort
    if not cohort_dir.exists():
        raise FileNotFoundError(f"Cohort not found: {trait}/{cohort}")
    return cohort_dir

def find_series_matrix(cohort_dir: Path) -> Optional[Path]:
    for p in cohort_dir.iterdir():
        if p.name.endswith("_series_matrix.txt.gz"):
            return p
    return None

def find_soft_file(cohort_dir: Path) -> Optional[Path]:
    for p in cohort_dir.iterdir():
        if p.name.endswith("_family.soft.gz"):
            return p
    return None

def find_preprocessed_matrix(cohort_dir: Path) -> Optional[Path]:
    data_dir = cohort_dir / "data"
    if not data_dir.exists():
        return None
    m = data_dir / "Matrix.txt"
    return m if m.exists() else None
