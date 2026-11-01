import json
from pathlib import Path

import security_gate

POLICY = {
    "pip_audit": {"allow": [{"id": "PYSEC-1", "reason": "тест"}]},
    "trivy": {"block_severities": ["CRITICAL", "UNKNOWN"], "allow": []},
}


def dep(name, vulns=(), **extra):
    return {"name": name, "version": "1.0", "vulns": list(vulns), **extra}


def vuln(vid, aliases=()):
    return {"id": vid, "aliases": list(aliases), "fix_versions": ["2.0"]}


def test_pip_audit_clean_and_allowed_pass():
    report = {"dependencies": [dep("a"), dep("b", [vuln("PYSEC-1")])]}
    blocking, accepted = security_gate.check_pip_audit(report, POLICY)
    assert blocking == [] and len(accepted) == 1


def test_pip_audit_new_finding_blocks_even_if_severity_unknown():
    blocking, _ = security_gate.check_pip_audit({"dependencies": [dep("a", [vuln("PYSEC-9")])]}, POLICY)
    assert len(blocking) == 1


def test_pip_audit_alias_of_allowed_id_is_accepted():
    blocking, _ = security_gate.check_pip_audit({"dependencies": [dep("a", [vuln("GHSA-x", ["PYSEC-1"])])]}, POLICY)
    assert blocking == []


def test_pip_audit_skipped_package_and_empty_report_block():
    assert security_gate.check_pip_audit({"dependencies": [dep("a", skip_reason="not found")]}, POLICY)[0]
    assert security_gate.check_pip_audit({"dependencies": []}, POLICY)[0]
    assert security_gate.check_pip_audit({}, POLICY)[0]


def trivy(*vulns):
    return {"ArtifactName": "img", "Results": [{"Target": "os", "Vulnerabilities": list(vulns)}]}


def tv(vid, sev):
    return {"VulnerabilityID": vid, "PkgName": "p", "InstalledVersion": "1", "Severity": sev}


def test_trivy_blocks_critical_and_unknown_not_low():
    report = trivy(tv("CVE-1", "CRITICAL"), tv("CVE-2", "UNKNOWN"), tv("CVE-3", "LOW"))
    blocking, _ = security_gate.check_trivy(report, POLICY)
    assert len(blocking) == 2


def test_trivy_clean_ok_but_empty_or_foreign_report_blocks():
    assert security_gate.check_trivy({"ArtifactName": "i", "Results": [{"Target": "os"}]}, POLICY) == ([], [])
    assert security_gate.check_trivy({"ArtifactName": "i"}, POLICY)[0]
    assert security_gate.check_trivy({"x": 1}, POLICY)[0]


def run(tmp_path, kind, report, **kw):
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps(POLICY))
    rep = tmp_path / "r.json"
    if report is not None:
        rep.write_text(report if isinstance(report, str) else json.dumps(report))
    args = [kind, str(rep), "--policy", str(pol)]
    if "code" in kw:
        args += ["--tool-exit-code", str(kw["code"])]
    return security_gate.main(args)


def test_cli_simulated_critical_report_fails(tmp_path):
    """Имитация: тестовый отчёт Trivy с критичной находкой должен остановить gate."""
    assert run(tmp_path, "trivy", trivy(tv("CVE-FAKE-0001", "CRITICAL"))) == 1


def test_cli_ok_and_tool_errors(tmp_path):
    assert run(tmp_path, "pip-audit", {"dependencies": [dep("a")]}, code=0) == 0
    assert run(tmp_path, "pip-audit", {"dependencies": [dep("a")]}, code=2) == 1
    assert run(tmp_path, "pip-audit", "not json") == 1
    (tmp_path / "r.json").unlink()
    assert run(tmp_path, "pip-audit", None) == 1


def test_real_policy_allows_only_documented_setuptools_ids():
    p = Path(__file__).parent.parent / "configs" / "security_policy.json"
    pol = json.loads(p.read_text(encoding="utf-8"))
    assert {e["id"] for e in pol["pip_audit"]["allow"]} == {"PYSEC-2026-3447", "GHSA-xmj3-9gf7-h52w"}
    assert all(e["reason"] for e in pol["pip_audit"]["allow"])


def test_package_missing_from_report_blocks(tmp_path):
    req = tmp_path / "req.txt"
    req.write_text("# comment\n-r other.txt\nnumpy==2.5.4\nboolean.py==5.0\nTyping_Extensions==4.16.0\n")
    assert security_gate.pinned_names([req]) == {"numpy", "boolean-py", "typing-extensions"}
    report = {"dependencies": [dep("numpy"), dep("boolean.py")]}
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps(report))
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps(POLICY))
    args = ["pip-audit", str(rep), "--policy", str(pol), "--tool-exit-code", "0", "--requirements", str(req)]
    assert security_gate.main(args) == 1
    report["dependencies"].append(dep("typing_extensions"))
    rep.write_text(json.dumps(report))
    assert security_gate.main(args) == 0
