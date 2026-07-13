"""Logopedia scrape, baseline, and shallow CNN pipeline for APS360 progress report."""

from __future__ import annotations

import io
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

API_URL = "https://logos.fandom.com/api.php"
USER_AGENT = "APS360-Project/1.0 (coursework; derek.lau@mail.utoronto.ca)"
YEAR_PATTERN = re.compile(
    r"(?<!\d)(1[89]\d{2}|20[0-2]\d)(?:\s*[–\-—]\s*(?:present|current|now|today|\d{4}))?",
    re.IGNORECASE,
)
IMAGETOC_LINE = re.compile(
    r"^\|([^|]+\.(?:svg|png|jpg|jpeg|gif|webp))\|([^|\n]+)",
    re.IGNORECASE,
)
FILE_PATTERN = re.compile(
    r"\[\[(?:File|Image):([^\]|]+\.(?:svg|png|jpg|jpeg|gif|webp))(?:\|[^\]]*)?\]\]",
    re.IGNORECASE,
)
FILENAME_YEAR = re.compile(r"\((\d{4})\)", re.IGNORECASE)
SECTION_HEADER = re.compile(r"^=+\s*([^=]+?)\s*=+\s*$", re.MULTILINE)

SEED_CATEGORIES = [
    "Category:Technology companies",
    "Category:Retail companies",
    "Category:Food and drink companies",
    "Category:Media companies",
    "Category:Automotive companies",
    "Category:Financial services companies",
    "Category:Clothing companies",
    "Category:Telecommunications companies",
]


@dataclass
class LogoRecord:
    company: str
    filename: str
    year: int
    decade: int
    image_path: str
    split: str


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wiki_get(params: dict, session: requests.Session) -> dict:
    params = {**params, "format": "json"}
    for attempt in range(5):
        response = session.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return payload
    raise RuntimeError("wiki_get failed")


def parse_start_year(text: str) -> int | None:
    match = YEAR_PATTERN.search(text)
    if not match:
        return None
    return int(match.group(1))


def year_to_decade(year: int) -> int:
    return (year // 10) * 10


def extract_logo_entries(wikitext: str) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    for line in wikitext.splitlines():
        toc_match = IMAGETOC_LINE.match(line.strip())
        if toc_match:
            filename = toc_match.group(1).strip()
            year = parse_start_year(toc_match.group(2))
            if year:
                key = (filename.lower(), year_to_decade(year))
                if key not in seen:
                    seen.add(key)
                    entries.append((filename, year))

    for header in SECTION_HEADER.findall(wikitext):
        year = parse_start_year(header)
        if not year:
            continue
        decade = year_to_decade(year)
        header_pattern = re.escape(header)
        block_match = re.search(
            rf"==+\s*{header_pattern}\s*==+(.+?)(?=\n==|\Z)",
            wikitext,
            re.DOTALL | re.IGNORECASE,
        )
        if not block_match:
            continue
        block = block_match.group(1)
        for file_match in FILE_PATTERN.finditer(block):
            filename = file_match.group(1).strip()
            file_year = year
            year_in_name = FILENAME_YEAR.search(filename)
            if year_in_name:
                file_year = int(year_in_name.group(1))
            key = (filename.lower(), year_to_decade(file_year))
            if key not in seen:
                seen.add(key)
                entries.append((filename, file_year))

    for file_match in FILE_PATTERN.finditer(wikitext):
        filename = file_match.group(1).strip()
        year = None
        year_in_name = FILENAME_YEAR.search(filename)
        if year_in_name:
            year = int(year_in_name.group(1))
        else:
            start = max(0, file_match.start() - 120)
            end = min(len(wikitext), file_match.end() + 120)
            year = parse_start_year(wikitext[start:end])
        if year:
            key = (filename.lower(), year_to_decade(year))
            if key not in seen:
                seen.add(key)
                entries.append((filename, year))

    deduped: dict[tuple[str, int], tuple[str, int]] = {}
    for filename, year in entries:
        deduped[(filename.lower(), year_to_decade(year))] = (filename, year)
    return list(deduped.values())


def list_category_companies(
    categories: Iterable[str],
    session: requests.Session,
    max_companies: int,
) -> list[str]:
    companies: list[str] = []
    seen: set[str] = set()
    for category in categories:
        continue_token: str | None = None
        while len(companies) < max_companies:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": 50,
            }
            if continue_token:
                params["cmcontinue"] = continue_token
            payload = wiki_get(params, session)
            members = payload.get("query", {}).get("categorymembers", [])
            for member in members:
                if member.get("ns") != 0:
                    continue
                title = member["title"]
                if title not in seen:
                    seen.add(title)
                    companies.append(title)
                    if len(companies) >= max_companies:
                        break
            if len(companies) >= max_companies:
                break
            continue_token = payload.get("continue", {}).get("cmcontinue")
            if not continue_token:
                break
            time.sleep(0.1)
    return companies


def fetch_page_wikitext(title: str, session: requests.Session) -> str | None:
    payload = wiki_get(
        {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
        },
        session,
    )
    pages = payload.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    if "missing" in page:
        return None
    revisions = page.get("revisions", [])
    if not revisions:
        return None
    slots = revisions[0].get("slots", {})
    main = slots.get("main", revisions[0])
    return main.get("*")


def resolve_image_url(filename: str, session: requests.Session) -> str | None:
    payload = wiki_get(
        {
            "action": "query",
            "titles": f"File:{filename}",
            "prop": "imageinfo",
            "iiprop": "url|mime",
        },
        session,
    )
    pages = payload.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    infos = page.get("imageinfo", [])
    if not infos:
        return None
    url = infos[0]["url"]
    mime = infos[0].get("mime", "")
    if mime == "image/svg+xml" or filename.lower().endswith(".svg"):
        return url.replace("/revision/latest", "/revision/latest/scale-to-width-down/512")
    return url


def download_logo_image(url: str, session: requests.Session) -> Image.Image | None:
    response = session.get(url, timeout=30)
    if response.status_code != 200:
        return None
    try:
        image = Image.open(io.BytesIO(response.content))
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            alpha = image.split()[-1] if image.mode in ("RGBA", "LA") else None
            background.paste(image, mask=alpha)
            image = background
        else:
            image = image.convert("RGB")
        return image
    except Exception:
        return None


def preprocess_image(image: Image.Image, size: int = 224) -> Image.Image:
    image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def assign_company_splits(
    companies: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, str]:
    companies = sorted(set(companies))
    rng = random.Random(42)
    rng.shuffle(companies)
    n = len(companies)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    splits: dict[str, str] = {}
    for idx, company in enumerate(companies):
        if idx < train_end:
            splits[company] = "train"
        elif idx < val_end:
            splits[company] = "val"
        else:
            splits[company] = "test"
    return splits


def build_dataset(
    root: Path,
    max_companies: int = 250,
    request_delay: float = 0.15,
) -> tuple[list[LogoRecord], dict[str, str], dict]:
    root.mkdir(parents=True, exist_ok=True)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    companies = list_category_companies(SEED_CATEGORIES, session, max_companies)
    company_logos: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    download_errors = 0
    pages_without_years = 0

    for company in tqdm(companies, desc="Scraping companies"):
        wikitext = fetch_page_wikitext(company, session)
        time.sleep(request_delay)
        if not wikitext:
            continue
        entries = extract_logo_entries(wikitext)
        if not entries:
            pages_without_years += 1
            continue
        decade_seen: set[int] = set()
        for filename, year in entries:
            decade = year_to_decade(year)
            if decade in decade_seen:
                continue
            decade_seen.add(decade)
            company_logos[company].append((filename, year, decade))

    all_companies = [c for c, logos in company_logos.items() if logos]
    splits = assign_company_splits(all_companies)
    records: list[LogoRecord] = []

    for company, logos in tqdm(company_logos.items(), desc="Downloading logos"):
        if company not in splits:
            continue
        split = splits[company]
        for filename, year, decade in logos:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", company)[:60]
            out_path = image_dir / f"{safe_name}_{decade}_{Path(filename).stem}.jpg"
            if not out_path.exists():
                url = resolve_image_url(filename, session)
                time.sleep(request_delay)
                if not url:
                    download_errors += 1
                    continue
                image = download_logo_image(url, session)
                time.sleep(request_delay)
                if image is None:
                    download_errors += 1
                    continue
                preprocess_image(image).save(out_path, format="JPEG", quality=92)
            records.append(
                LogoRecord(
                    company=company,
                    filename=filename,
                    year=year,
                    decade=decade,
                    image_path=str(out_path.relative_to(root.parent)),
                    split=split,
                )
            )

    decade_counts = Counter(record.decade for record in records)
    split_counts = Counter(record.split for record in records)
    stats = {
        "companies_indexed": len(companies),
        "companies_with_logos": len(all_companies),
        "total_images": len(records),
        "decades": len(decade_counts),
        "decade_counts": {str(k): v for k, v in sorted(decade_counts.items())},
        "split_counts": dict(split_counts),
        "pages_without_years": pages_without_years,
        "download_errors": download_errors,
    }
    return records, splits, stats


def color_features(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.resize((64, 64)), dtype=np.float32) / 255.0
    features: list[float] = []
    for channel in range(3):
        values = arr[:, :, channel].ravel()
        features.extend(np.histogram(values, bins=16, range=(0.0, 1.0))[0].tolist())
        features.extend([float(values.mean()), float(values.std())])
    hsv = np.asarray(image.resize((64, 64)).convert("HSV"), dtype=np.float32) / 255.0
    for channel in range(3):
        values = hsv[:, :, channel].ravel()
        features.extend(np.histogram(values, bins=16, range=(0.0, 1.0))[0].tolist())
        features.extend([float(values.mean()), float(values.std())])
    return np.asarray(features, dtype=np.float32)


def load_records(root: Path) -> list[LogoRecord]:
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    return [LogoRecord(**item) for item in payload]


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def train_baseline(records: list[LogoRecord], root: Path) -> dict:
    label_values = sorted({record.decade for record in records})
    label_to_idx = {label: idx for idx, label in enumerate(label_values)}

    def subset(split: str) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for record in records:
            if record.split != split:
                continue
            image = Image.open(root.parent / record.image_path).convert("RGB")
            xs.append(color_features(image))
            ys.append(label_to_idx[record.decade])
        return np.asarray(xs), np.asarray(ys)

    x_train, y_train = subset("train")
    x_val, y_val = subset("val")
    x_test, y_test = subset("test")

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(x_train_s, y_train)

    def metrics(x_s: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = model.predict(x_s)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }

    val_metrics = metrics(x_val_s, y_val)
    test_metrics = metrics(x_test_s, y_test)
    y_test_pred = model.predict(x_test_s)
    cm = confusion_matrix(y_test, y_test_pred, labels=list(range(len(label_values))))

    return {
        "labels": label_values,
        "val": val_metrics,
        "test": test_metrics,
        "confusion_matrix": cm.tolist(),
        "y_test": y_test.tolist(),
        "y_test_pred": y_test_pred.tolist(),
    }


class LogoDataset(Dataset):
    def __init__(
        self,
        records: list[LogoRecord],
        root: Path,
        label_to_idx: dict[int, int],
        augment: bool = False,
    ) -> None:
        self.records = records
        self.root = root
        self.label_to_idx = label_to_idx
        self.augment = augment
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _maybe_augment(self, image: Image.Image) -> Image.Image:
        if not self.augment:
            return image
        if random.random() < 0.5:
            image = image.rotate(random.uniform(-10, 10), fillcolor=(255, 255, 255))
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        return image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        image = Image.open(self.root.parent / record.image_path).convert("RGB")
        image = self._maybe_augment(image)
        return self.base_transform(image), self.label_to_idx[record.decade]


class ShallowCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_cnn(records: list[LogoRecord], root: Path, device: torch.device) -> dict:
    label_values = sorted({record.decade for record in records})
    label_to_idx = {label: idx for idx, label in enumerate(label_values)}

    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]
    test_records = [r for r in records if r.split == "test"]

    # Oversample rare decades so each batch sees a balanced class mix.
    train_labels = [label_to_idx[r.decade] for r in train_records]
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_records), replacement=True
    )
    train_loader = DataLoader(
        LogoDataset(train_records, root, label_to_idx, augment=True),
        batch_size=32,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        LogoDataset(val_records, root, label_to_idx, augment=False),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        LogoDataset(test_records, root, label_to_idx, augment=False),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = ShallowCNN(num_classes=len(label_values)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # The balanced sampler already corrects imbalance; weighting the loss as
    # well over-corrects and pushes predictions onto the rarest decades.
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_state = None
    best_f1 = -1.0
    patience = 15
    stale = 0
    max_epochs = 60

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        scheduler.step()
        val_loss /= max(len(val_loader.dataset), 1)
        val_f1 = float(f1_score(y_true, y_pred, average="macro"))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_test_true, y_test_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_test_true.extend(labels.tolist())
            y_test_pred.extend(preds.cpu().tolist())

    test_acc = float(accuracy_score(y_test_true, y_test_pred))
    test_f1 = float(f1_score(y_test_true, y_test_pred, average="macro"))
    cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(len(label_values))))

    return {
        "labels": label_values,
        "param_count": count_parameters(model),
        "history": history,
        "val_macro_f1": best_f1,
        "test": {"accuracy": test_acc, "macro_f1": test_f1},
        "confusion_matrix": cm.tolist(),
        "y_test": y_test_true,
        "y_test_pred": y_test_pred,
        "test_records": test_records,
        "architecture": [
            {"layer": "Conv2d + BN + ReLU + MaxPool", "channels": "3→32", "kernel": 3},
            {"layer": "Conv2d + BN + ReLU + MaxPool", "channels": "32→64", "kernel": 3},
            {"layer": "Conv2d + BN + ReLU + MaxPool", "channels": "64→128", "kernel": 3},
            {"layer": "Conv2d + BN + ReLU + GAP", "channels": "128→128", "kernel": 3},
            {"layer": "Dropout(0.3) + Linear", "channels": f"128→{len(label_values)}", "kernel": "-"},
        ],
    }


def _plot_decade_distribution(stats: dict, out_path: Path) -> None:
    decades = [int(k) for k in stats["decade_counts"].keys()]
    counts = [stats["decade_counts"][str(d)] for d in decades]
    labels = [f"{d}s" for d in decades]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(labels, counts, color="#4C78A8")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Images")
    ax.set_title("Logo images per decade")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_logos(records: list[LogoRecord], root: Path, out_path: Path) -> None:
    by_decade: dict[int, LogoRecord] = {}
    for record in sorted(records, key=lambda r: (r.decade, r.company)):
        if record.decade not in by_decade:
            by_decade[record.decade] = record
    sample = list(by_decade.values())[:8]
    cols = 4
    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 2.2))
    axes = np.array(axes).reshape(-1)
    for ax, record in zip(axes, sample):
        image = Image.open(root.parent / record.image_path)
        ax.imshow(image)
        ax.set_title(f"{record.company[:18]}\n{record.decade}s", fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(cm: list[list[int]], labels: list[int], title: str, out_path: Path) -> None:
    cm_arr = np.asarray(cm)
    tick_labels = [f"{label}s" for label in labels]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(labels)), tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_predictions(
    test_records: list[LogoRecord],
    y_true: list[int],
    y_pred: list[int],
    labels: list[int],
    root: Path,
    out_path: Path,
) -> None:
    """Qualitative panel: correct predictions on top, mistakes on the bottom."""
    correct = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
    wrong = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    picks = correct[:4] + wrong[:4]
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.8))
    axes = np.array(axes).reshape(-1)
    for ax, idx in zip(axes, picks):
        record = test_records[idx]
        image = Image.open(root.parent / record.image_path)
        ax.imshow(image)
        true_lbl = f"{labels[y_true[idx]]}s"
        pred_lbl = f"{labels[y_pred[idx]]}s"
        ok = y_true[idx] == y_pred[idx]
        ax.set_title(
            f"{record.company[:16]}\ntrue {true_lbl} / pred {pred_lbl}",
            fontsize=8,
            color="#2E7D32" if ok else "#C62828",
        )
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle("CNN test predictions (top: correct, bottom: errors)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_architecture(architecture: list[dict], param_count: int, out_path: Path) -> None:
    """Horizontal block diagram of the CNN."""
    blocks = [("Input\n224$\\times$224$\\times$3", "#E8EAF6")]
    for item in architecture:
        layer = item["layer"].replace(" + ", "\n")
        blocks.append((f"{layer}\n{item['channels']}", "#C5CAE9"))
    blocks.append(("Decade\nlogits", "#9FA8DA"))
    fig, ax = plt.subplots(figsize=(10.5, 2.3))
    n = len(blocks)
    width, gap = 1.35, 0.42
    for i, (text, color) in enumerate(blocks):
        x = i * (width + gap)
        ax.add_patch(plt.Rectangle((x, 0), width, 1, facecolor=color, edgecolor="#3F51B5"))
        ax.text(x + width / 2, 0.5, text, ha="center", va="center", fontsize=7.5)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + width + gap, 0.5),
                xytext=(x + width, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#3F51B5"},
            )
    ax.set_xlim(-0.15, n * (width + gap) - gap + 0.15)
    ax.set_ylim(-0.2, 1.2)
    ax.set_title(f"Shallow CNN ({param_count:,} trainable parameters)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_curve(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(epochs, history["train_loss"], label="Train loss", color="#E45756")
    ax1.plot(epochs, history["val_loss"], label="Val loss", color="#F58518")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_macro_f1"], label="Val macro-F1", color="#54A24B")
    ax2.set_ylabel("Macro-F1")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    ax1.set_title("CNN training curve")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def export_figures(
    records: list[LogoRecord],
    stats: dict,
    baseline: dict,
    cnn: dict,
    root: Path,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_decade_distribution(stats, figures_dir / "decade_distribution.pdf")
    _plot_sample_logos(records, root, figures_dir / "sample_logos.pdf")
    _plot_confusion(
        baseline["confusion_matrix"],
        baseline["labels"],
        "Baseline test confusion matrix",
        figures_dir / "baseline_confusion.pdf",
    )
    _plot_confusion(
        cnn["confusion_matrix"],
        cnn["labels"],
        "CNN test confusion matrix",
        figures_dir / "cnn_confusion.pdf",
    )
    _plot_learning_curve(cnn["history"], figures_dir / "cnn_learning_curve.pdf")
    _plot_sample_predictions(
        cnn["test_records"],
        cnn["y_test"],
        cnn["y_test_pred"],
        cnn["labels"],
        root,
        figures_dir / "cnn_sample_predictions.pdf",
    )
    _plot_architecture(
        cnn["architecture"],
        cnn["param_count"],
        figures_dir / "cnn_architecture.pdf",
    )


def run_pipeline(
    project_root: Path,
    max_companies: int = 250,
    skip_scrape: bool = False,
) -> dict:
    set_seed(42)
    data_dir = project_root / "data"
    figures_dir = project_root / "figures"

    if skip_scrape and (data_dir / "manifest.json").exists():
        records = load_records(data_dir)
        splits = json.loads((data_dir / "splits.json").read_text())
        stats = json.loads((data_dir / "stats.json").read_text())
    else:
        records, splits, stats = build_dataset(data_dir, max_companies=max_companies)
        save_json(data_dir / "manifest.json", [asdict(r) for r in records])
        save_json(data_dir / "splits.json", splits)
        save_json(data_dir / "stats.json", stats)

    baseline = train_baseline(records, data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = train_cnn(records, data_dir, device)
    results = {
        "baseline": {
            "val": baseline["val"],
            "test": baseline["test"],
        },
        "cnn": {
            "val_macro_f1": cnn["val_macro_f1"],
            "test": cnn["test"],
            "param_count": cnn["param_count"],
            "architecture": cnn["architecture"],
        },
        "feasibility": {
            "cnn_beats_baseline_f1": cnn["test"]["macro_f1"] > baseline["test"]["macro_f1"],
            "cnn_test_macro_f1": cnn["test"]["macro_f1"],
            "baseline_test_macro_f1": baseline["test"]["macro_f1"],
        },
    }
    save_json(data_dir / "results.json", results)
    export_figures(records, stats, baseline, cnn, data_dir, figures_dir)
    return {
        "records": records,
        "splits": splits,
        "stats": stats,
        "baseline": baseline,
        "cnn": cnn,
        "results": results,
    }
