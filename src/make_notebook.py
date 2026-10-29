import nbformat as nbf

cells = [
("code", """import os, sys, json, hashlib
os.chdir('..'); sys.path.insert(0, 'src')
import numpy as np, soundfile as sf, matplotlib.pyplot as plt
from collections import Counter
import gmm_vc
from mcd import mcd"""),
("markdown", "## 1. Проверка данных\nФормат, NaN/Inf, дубликаты по SHA256 файлов, утечка между выборками."),
("code", """res = {}
for spk in ('bdl', 'slt'):
    files = sorted(f'data/raw/cmu_us_{spk}_arctic/wav/{n}' for n in os.listdir(f'data/raw/cmu_us_{spk}_arctic/wav'))
    hashes, bad = [], 0
    for f in files:
        x, fs = sf.read(f)
        bad += int(fs != 16000 or x.ndim != 1 or not np.all(np.isfinite(x)))
        hashes.append(hashlib.sha256(open(f, 'rb').read()).hexdigest())
    res[spk] = {'files': len(files), 'bad': bad, 'duplicate_files': len(hashes) - len(set(hashes))}
print(res)
s = json.load(open('data/splits.json'))
print({k: len(s[k]) for k in ('train', 'val', 'test')})
print('пересечения:', len(set(s['train']) & set(s['test'])), len(set(s['train']) & set(s['val'])), len(set(s['val']) & set(s['test'])))"""),
("markdown", "## 2. Метрики на тесте (MCD, дБ, меньше - лучше)"),
("code", """m = json.load(open('metrics/baseline.json'))
names = ['среднее цели', 'копия источника', 'JD-GMM M=16']
vals = [m['value_mean_target'], m['value_copy_source'], m['value_gmm']]
plt.bar(names, vals); plt.ylabel('MCD, дБ')
for i, v in enumerate(vals): plt.text(i, v + 0.15, f'{v:.2f}', ha='center')
plt.title('Тест, 113 фраз'); os.makedirs('docs', exist_ok=True)
plt.savefig('docs/mcd_test.png', dpi=120, bbox_inches='tight'); plt.show()"""),
("markdown", "## 3. Ошибки по фразам"),
("code", """te = np.load('data/aligned/test.npz', allow_pickle=False)
model = gmm_vc.load('models/gmm_M16.npz')
Yh = gmm_vc.convert(model, te['X'])
per = []
for k, name in enumerate(te['ids']):
    idx = te['utt'] == k
    per.append((float(mcd(Yh[idx], te['Y'][idx])), str(name), int(idx.sum())))
per.sort()
print('лучшие:', per[:3])
print('худшие:', per[-5:])
plt.hist([p[0] for p in per], bins=20); plt.xlabel('MCD фразы, дБ'); plt.ylabel('фраз')
plt.savefig('docs/mcd_per_phrase.png', dpi=120, bbox_inches='tight'); plt.show()"""),
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_code_cell(c) if t == 'code' else nbf.v4.new_markdown_cell(c) for t, c in cells]
nbf.write(nb, 'notebooks/02_baseline.ipynb')
