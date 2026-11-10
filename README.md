# segmentation3d — 학습

컨테이너 안 `/app/segmentation3d`에서 실행한다. 오래 걸리므로 `tmux new -s train` 안에서 돌린다.

```bash
# 1) Teeth3DS 원본 -> npz 캐시 (한 번만, 이미 만든 스캔은 건너뛴다)
python training/preprocess.py

# 2) semantic 사전학습
python training/train.py --stage semantic --out runs/semantic

# 3) 인스턴스 학습 (semantic 백본에서 시작)
python training/train.py --stage instance --init-from runs/semantic/best.pt --out runs/instance --val-crop bridge

# 끊긴 학습 이어서
python training/train.py --stage instance --out runs/instance --resume runs/instance/last.pt
```
