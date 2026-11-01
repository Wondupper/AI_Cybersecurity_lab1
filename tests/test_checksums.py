import checksums


def test_sha256_known_value(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    assert checksums.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_verify_ok_and_changed(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  a.txt\n")
    assert checksums.verify(sums, tmp_path) == []
    p.write_bytes(b"abd")
    assert checksums.verify(sums, tmp_path) == ["a.txt: SHA256 не совпадает с эталоном"]


def test_verify_bad_sums_file(tmp_path):
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("not a hash line\n")
    assert checksums.verify(sums, tmp_path)
    assert checksums.verify(tmp_path / "nope", tmp_path)
