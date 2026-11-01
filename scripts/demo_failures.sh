#!/usr/bin/env bash
# Демонстрации отказа CI. Каждый сценарий создаёт отдельную ветку от текущего коммита,
# вносит одно нарушение, делает коммит и отправляет ветку на GitHub; затем возвращает исходную ветку.
# Запуск из корня репозитория при чистом `git status`: scripts/demo_failures.sh test|data|model|security
set -euo pipefail

case "${1:-}" in
  test)     branch=demo/test-failure ;;
  data)     branch=demo/data-gate-failure ;;
  model)    branch=demo/model-gate-failure ;;
  security) branch=demo/security-gate-failure ;;
  *) echo "использование: $0 test|data|model|security"; exit 2 ;;
esac
[ -z "$(git status --porcelain)" ] || { echo "рабочее дерево не чистое"; exit 1; }
start=$(git rev-parse --abbrev-ref HEAD)
git switch -c "$branch"

case "$1" in
  test)
    # неверная константа в формуле MCD: тест test_known_value должен упасть
    sed -i 's|^K = 10.0 / np.log(10.0) \* np.sqrt(2.0)|K = 10.0 / np.log(10.0)|' src/mcd.py
    git add src/mcd.py
    git commit -q -m "demo: ломаем формулу MCD (должен упасть тест)"
    ;;
  data)
    # в копии контрольной выборки появляется NaN; эталонный хэш обновлён, чтобы сработало именно правило схемы
    python - <<'PY'
import numpy as np
p = "data/control/control_val.npz"
with np.load(p, allow_pickle=False) as f:
    d = {k: f[k] for k in f.files}
d["X"][0, 0] = np.nan
np.savez_compressed(p, **d)
PY
    echo "$(sha256sum data/control/control_val.npz | cut -d' ' -f1)  data/control/control_val.npz" > data/control/SHA256SUMS
    git add data/control
    git commit -q -m "demo: NaN в контрольной выборке (должен сработать data gate)"
    ;;
  model)
    # заведомо слабая константная модель; хэш модели обновлён, чтобы дошло до model gate
    mkdir -p out
    python src/make_weak_model.py models/gmm_M16.npz out/weak_model.npz
    cp out/weak_model.npz models/gmm_M16.npz
    ( cd models && sha256sum gmm_M16.npz f0_stats.json > SHA256SUMS )
    git add models
    git commit -q -m "demo: константная модель (должен сработать model gate)"
    ;;
  security)
    # заведомо уязвимая версия urllib3 в инструментах проверки (реальная находка pip-audit)
    sed -i 's/^urllib3==.*/urllib3==1.26.18/' requirements-dev.txt
    git add requirements-dev.txt
    git commit -q -m "demo: уязвимая версия urllib3 (должен сработать аудит зависимостей)"
    ;;
esac

git push -u origin "$branch"
echo "ветка $branch отправлена; commit: $(git rev-parse --short HEAD)"
git switch "$start"
