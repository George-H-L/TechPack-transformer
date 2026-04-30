import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

HERE = Path(__file__).parent

# all 4 variants
VARIANTS = [
    ('V1 (50 ep, combined)', 'training_history_combined.json', '#2196F3', '#64B5F6'),
    ('V2 (25 ep, deep)',     'training_history_v2.json',       '#4CAF50', '#81C784'),
    ('V3 (25 ep, wide)',     'training_history_v3.json',       '#9C27B0', '#CE93D8'),
    ('V4 (25 ep, asymm)',    'training_history_v4.json',       '#F44336', '#EF9A9A'),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('TechPack Model Training Curves: V1 / V2 / V3 / V4',
             fontsize=14, fontweight='bold', y=1.02)

legend_handles = []

for label, filename, train_col, val_col in VARIANTS:
    path = HERE / filename
    if not path.exists():
        print(f"  [skip] {filename} not found")
        continue

    with open(path, encoding='utf-8') as f:
        history = json.load(f)

    epochs     = [e['epoch'] for e in history]
    train_loss = [e['train_loss'] for e in history]
    val_loss   = [e['val_loss']   for e in history]

    axes[0].plot(epochs, train_loss, color=train_col, linewidth=2,    label=f'{label} train')
    axes[0].plot(epochs, val_loss,   color=val_col,   linewidth=2,
                 linestyle='--', label=f'{label} val')

    axes[1].plot(epochs, val_loss, color=val_col, linewidth=2.5, label=f'{label}  (final {val_loss[-1]:.4f})')

    legend_handles.append(mpatches.Patch(color=train_col, label=label))

# Left: train vs val
axes[0].set_title('Train Loss vs Validation Loss', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend(fontsize=7.5, ncol=2)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

# Right: val loss only (cleaner comparison)
axes[1].set_title('Validation Loss Comparison', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Cross-Entropy Loss')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

plt.tight_layout()

out = HERE / 'training_curve_v1v2v3v4.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved -> {out}")
plt.show()
