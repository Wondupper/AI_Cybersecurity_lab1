"""Заведомо слабая модель для демонстрации model gate: всегда выдаёт средний вектор цели.

Берёт формат модели gmm_vc, обнуляет линейные предсказания и средние цели по компонентам.
Запуск: python src/make_weak_model.py models/gmm_M16.npz out/weak_model.npz
"""
import sys

import numpy as np

import gmm_vc

model = gmm_vc.load(sys.argv[1])
model["A"] = np.zeros_like(model["A"])
model["muy"] = np.zeros_like(model["muy"])
gmm_vc.save(model, sys.argv[2])
print("записана константная модель:", sys.argv[2])
