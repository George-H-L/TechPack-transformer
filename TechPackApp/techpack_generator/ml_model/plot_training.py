import json
import matplotlib.pyplot as plt
from pathlib import Path
from .config import ModelConfig

config = ModelConfig()

VARIANTS = [
    ('v1', 'training_history_combined.json', '#2196F3', '#64B5F6'),
    ('v2', 'training_history_v2.json',       '#4CAF50', '#81C784'),
    ('v3', 'training_history_v3.json',       '#9C27B0', '#CE93D8'),
    ('v4', 'training_history_v4.json',       '#F44336', '#EF9A9A'),
]


def plot_all(save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training History: v1 / v2 / v3 / v4', fontsize=14, fontweight='bold')

    for variant, filename, train_col, val_col in VARIANTS:
        path = Path(config.model_dir) / filename
        if not path.exists():
            continue
        with open(path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        epochs     = [e['epoch'] for e in history]
        train_loss = [e['train_loss'] for e in history]
        val_loss   = [e['val_loss']   for e in history]

        axes[0].plot(epochs, train_loss, color=train_col, label=f'{variant} train')
        axes[0].plot(epochs, val_loss,   color=val_col,   label=f'{variant} val', linestyle='--')
        axes[1].plot(epochs, val_loss,   color=val_col,   label=f'{variant} val loss')

    axes[0].set_title('Train vs Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation Loss Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    out = save_path or str(Path(config.model_dir) / 'training_curve_all.png')
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == '__main__':
    import sys
    plot_all(save_path=sys.argv[1] if len(sys.argv) > 1 else None)
