# Контрольная выборка для CI

- **Источник:** CMU ARCTIC v0.95, дикторы bdl и slt (http://festvox.org/cmu_arctic/), выборка val из `data/splits.json`.
- **Состав:** 20 фраз из val, выбраны случайно (`numpy.random.default_rng(42)`), скрипт `src/make_control.py`.
- **Файл:** `control_val.npz`, контрольная сумма в `SHA256SUMS` (эталон хранится отдельно от файла).
- **Схема:** `X` и `Y` - выровненные пары кадров (N, 24), float32: кепстр WORLD диктора bdl и slt;
  `utt` - номер фразы для каждой пары (int32, по возрастанию); `ids` - имена фраз.
- **Правила проверки:** `configs/data_gate.json`.
- **Как используется:** data gate и model gate в CI. Это не тестовая выборка: финальная оценка модели делается на test отдельно.
- Файл нельзя пересоздавать без обновления SHA256, `configs/data_gate.json` и `metrics/ci_baseline.json`.
