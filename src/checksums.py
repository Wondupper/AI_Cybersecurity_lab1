"""Работа с файлами SHA256SUMS (формат sha256sum: «хэш  путь»)."""
import hashlib
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sums(sums_path):
    """Возвращает {путь: хэш}. Строки в неверном формате считаются ошибкой."""
    out = {}
    for n, line in enumerate(Path(sums_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"{sums_path}:{n}: неверная строка")
        out[parts[1].strip().lstrip("*")] = parts[0].lower()
    return out


def verify(sums_path, base="."):
    """Сверяет файлы из SHA256SUMS. Возвращает список нарушений (пустой - всё совпало)."""
    problems = []
    try:
        sums = read_sums(sums_path)
    except (OSError, ValueError) as exc:
        return [f"не удалось прочитать {sums_path}: {exc}"]
    if not sums:
        return [f"{sums_path} пустой"]
    for name, expected in sums.items():
        p = Path(base) / name
        if not p.is_file():
            problems.append(f"{name}: файла нет")
        elif sha256_file(p) != expected:
            problems.append(f"{name}: SHA256 не совпадает с эталоном")
    return problems
