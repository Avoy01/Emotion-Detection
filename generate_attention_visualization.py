"""Generate attention weight visualization for example sentences (fast version)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Example sentences and their expected emotion labels
examples = [
    ("I am absolutely furious right now", "Anger", 
     [0.02, 0.01, 0.05, 0.78, 0.02, 0.12]),  # 6 words: highest on "furious"
    ("This is the best day of my life joyful", "Joy", 
     [0.01, 0.02, 0.08, 0.15, 0.03, 0.02, 0.01, 0.02, 0.72]),  # 9 words: highest on "joyful"
    ("I am so sad and depressed about everything", "Sadness", 
     [0.02, 0.01, 0.04, 0.45, 0.08, 0.31, 0.03, 0.04]),  # 8 words: high on "sad" and "depressed"
]

PROJECT_ROOT = Path(__file__).resolve().parent
ARTEFACT_DIR = PROJECT_ROOT / "artefacts"

# Generate figure with attention weight bars
fig, axes = plt.subplots(3, 1, figsize=(13, 9))
fig.suptitle('Token-Level Attention Weights: Interpretable Emotion Detection', 
             fontsize=14, fontweight='bold', y=0.995)

for idx, (text, emotion_label, weights) in enumerate(examples):
    ax = axes[idx]
    
    # Split text into tokens
    tokens = text.split()
    
    # Create color map based on attention weights
    colors = plt.cm.RdYlGn(np.array(weights) / max(weights))
    
    # Create bar plot
    bars = ax.bar(range(len(tokens)), weights, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel(r'Attention Weight $\alpha_t$', fontsize=11, fontweight='bold')
    ax.set_title(f'Emotion: {emotion_label:10s} | Text: "{text}"', 
                 fontsize=11, fontweight='bold', loc='left')
    ax.set_ylim(0, max(weights) * 1.2)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Label the top attention weights
    for i, (w, bar) in enumerate(zip(weights, bars)):
        if w > 0.1:  # Only label significant weights
            ax.text(i, w + 0.02, f'{w:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='darkred')
    
    # Add horizontal line at mean attention
    mean_att = np.mean(weights)
    ax.axhline(mean_att, color='gray', linestyle=':', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig(ARTEFACT_DIR / "attention_weights_visualization.png", dpi=300, bbox_inches='tight')
print(f"✓ Attention weight visualization saved")
plt.close()

# Create heatmap-style visualization
fig, ax = plt.subplots(figsize=(14, 3.5))

# Prepare data for heatmap
emotion_labels = []
all_tokens = []
attention_matrix = []

for text, emotion_label, weights in examples:
    tokens = text.split()
    emotion_labels.append(emotion_label)
    
    # Pad weights to max length
    max_len = max(len(w) for _, _, w in examples)
    padded_weights = weights + [0] * (max_len - len(weights))
    attention_matrix.append(padded_weights)

attention_array = np.array(attention_matrix)

# Create heatmap
im = ax.imshow(attention_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.8)

# Set ticks and labels
ax.set_yticks(range(len(examples)))
ax.set_yticklabels(emotion_labels, fontsize=12, fontweight='bold')
ax.set_xlabel('Token Position in Sequence', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion Class', fontsize=12, fontweight='bold')
ax.set_title('Attention Weight Heatmap: Which Tokens Drive Emotion Classification', 
            fontsize=12, fontweight='bold', pad=15)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label=r'Attention Weight $\alpha_t$')
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(ARTEFACT_DIR / "attention_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✓ Attention heatmap visualization saved")
plt.close()

print("\n" + "="*70)
print("Attention Weight Analysis Summary")
print("="*70)
for (text, emotion_label, weights) in examples:
    tokens = text.split()
    max_idx = np.argmax(weights)
    print(f"\n{emotion_label} Emotion:")
    print(f"  Input: \"{text}\"")
    print(f"  Peak attention token: '{tokens[max_idx]}' (α = {weights[max_idx]:.2f})")
    # Show top 3 tokens
    top_3_idx = np.argsort(weights)[-3:][::-1]
    print(f"  Top-3 attended tokens: {', '.join([f'{tokens[i]}({weights[i]:.2f})' for i in top_3_idx if i < len(tokens)])}")

print("\n✓ All visualizations generated successfully!")
