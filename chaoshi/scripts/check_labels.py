from pathlib import Path
import yaml

root = Path('c:/chaoshi/detect_dataset')
# try common yaml locations
yaml_paths = [root / 'data.yaml', root / 'dataset_auto.yaml', root / 'coco_dataset_auto.yaml']
nc = None
for p in yaml_paths:
    if p.exists():
        try:
            d = yaml.safe_load(p.read_text())
            if isinstance(d, dict) and 'nc' in d:
                nc = int(d['nc'])
                break
        except Exception:
            pass
if nc is None:
    print('[WARN] 未在 data.yaml 中读取到 nc，默认使用 nc=1')
    nc = 1

errors = []
count_files = 0
count_boxes = 0

for labels_dir in [root / 'train' / 'labels', root / 'labels' / 'train']:
    if not labels_dir.exists():
        continue
    for f in sorted(labels_dir.rglob('*.txt')):
        count_files += 1
        text = f.read_text(encoding='utf-8').strip()
        if not text:
            errors.append(f"EMPTY:{f}")
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"BAD_FORMAT:{f}:{i}:{line}")
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:])
            except Exception:
                errors.append(f"PARSE_ERR:{f}:{i}:{line}")
                continue
            count_boxes += 1
            if cls < 0 or cls >= nc:
                errors.append(f"BAD_CLASS:{f}:{i}:cls={cls}")
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                errors.append(f"BAD_CENTER:{f}:{i}:{cx},{cy}")
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                errors.append(f"BAD_WH:{f}:{i}:{w},{h}")

print(f"Checked label dirs. files={count_files} boxes={count_boxes} nc={nc}")
if not errors:
    print('No errors found')
else:
    print('Errors sample (first 50):')
    for e in errors[:50]:
        print('  ', e)
    print('Total errors:', len(errors))
