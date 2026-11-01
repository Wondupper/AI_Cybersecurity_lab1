"""Security gate: проверка отчётов pip-audit и Trivy по политике configs/security_policy.json.

Политика pip-audit строгая: отказ при любой находке, кроме перечисленных в allow (с причиной).
Политика Trivy: отказ при находках с severity из block_severities (по умолчанию CRITICAL и UNKNOWN).
Пустой, повреждённый или отсутствующий отчёт и ошибка самого сканера - отказ, а не «уязвимостей нет».

Запуск:
  python src/security_gate.py pip-audit reports/pip-audit.json --tool-exit-code $rc \\
      --requirements requirements-runtime.txt requirements-dev.txt
  python src/security_gate.py trivy reports/trivy-image.json
"""
import argparse
import json
import re
import sys
from pathlib import Path


def load_report(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, f"отчёт не прочитан: {exc}"


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned_names(req_files):
    """Имена пакетов из requirements-файлов (строки вида имя==версия; -r и комментарии пропускаются)."""
    names = set()
    for f in req_files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line and not line.startswith("-"):
                names.add(norm(re.split(r"[=<>!~\[; ]", line, maxsplit=1)[0]))
    return names


def allowed_ids(entries):
    return {e["id"] for e in entries}


def check_pip_audit(report, policy):
    """Возвращает список блокирующих находок (строки) и список допущенных."""
    deps = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(deps, list) or not deps:
        return ["в отчёте pip-audit нет списка зависимостей"], []
    allow = allowed_ids(policy["pip_audit"]["allow"])
    blocking, accepted = [], []
    for d in deps:
        name, ver = d.get("name"), d.get("version")
        if d.get("skip_reason"):
            blocking.append(f"{name} {ver}: пакет пропущен ({d['skip_reason']})")
        for v in d.get("vulns", []):
            ids = {v.get("id"), *v.get("aliases", [])}
            line = f"{name} {ver}: {v.get('id')} (исправление: {', '.join(v.get('fix_versions', [])) or 'нет'})"
            (accepted if ids & allow else blocking).append(line)
    return blocking, accepted


def check_trivy(report, policy):
    if not isinstance(report, dict) or "ArtifactName" not in report:
        return ["это не отчёт Trivy"], []
    results = report.get("Results")
    if not isinstance(results, list) or not results:
        return ["в отчёте Trivy нет результатов сканирования"], []
    cfg = policy["trivy"]
    block, allow = {s.upper() for s in cfg["block_severities"]}, allowed_ids(cfg["allow"])
    blocking, accepted = [], []
    for r in results:
        for v in r.get("Vulnerabilities") or []:
            sev = str(v.get("Severity", "UNKNOWN")).upper()
            line = (f"{r.get('Target')}: {v.get('PkgName')} {v.get('InstalledVersion')} "
                    f"{v.get('VulnerabilityID')} [{sev}] исправление: {v.get('FixedVersion') or 'нет'}")
            if sev not in block:
                continue
            (accepted if v.get("VulnerabilityID") in allow else blocking).append(line)
    return blocking, accepted


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["pip-audit", "trivy"])
    ap.add_argument("report")
    ap.add_argument("--policy", default="configs/security_policy.json")
    ap.add_argument("--requirements", nargs="+", default=[],
                    help="requirements-файлы: каждый их пакет должен быть в отчёте pip-audit (иначе он не проверен)")
    ap.add_argument("--tool-exit-code", type=int, default=None,
                    help="код возврата сканера; для pip-audit допустимы 0 и 1 (1 - есть находки)")
    args = ap.parse_args(argv)
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))

    problems = []
    ok_codes = (0, 1) if args.kind == "pip-audit" else (0,)
    if args.tool_exit_code is not None and args.tool_exit_code not in ok_codes:
        problems.append(f"сканер завершился с ошибкой (код {args.tool_exit_code})")
    report, err = load_report(args.report)
    if err:
        problems.append(err)
    else:
        check = check_pip_audit if args.kind == "pip-audit" else check_trivy
        blocking, accepted = check(report, policy)
        if args.kind == "pip-audit" and args.requirements:
            audited = {norm(d.get("name", "")) for d in report.get("dependencies", [])}
            blocking += [f"{n}: пакет есть в requirements, но не проверен pip-audit"
                         for n in sorted(pinned_names(args.requirements) - audited)]
        problems += blocking
        for a in accepted:
            print("допущено по политике:", a)
    for p in problems:
        print("БЛОКИРУЕТ:", p)
    print(f"security gate ({args.kind}):", "ОТКАЗ" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
