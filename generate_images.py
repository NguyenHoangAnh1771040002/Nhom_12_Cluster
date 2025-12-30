# -*- coding: utf-8 -*-
"""
Script tạo hình ảnh cho README.md
Chạy: python generate_images.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Thiết lập style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Tạo thư mục images nếu chưa có
os.makedirs("images", exist_ok=True)

# ===========================================================================
# 1. Load data
# ===========================================================================
print("Loading data...")
cluster_df = pd.read_csv("data/processed/customer_clusters_from_rules.csv")
rules_df = pd.read_csv("data/processed/rules_apriori_filtered.csv")
strategy_df = pd.read_csv("data/processed/cluster_strategies.csv")

print(f"Loaded {len(cluster_df)} customers, {len(rules_df)} rules")

# ===========================================================================
# 2. Hình 1: Phân bố cụm (Pie chart)
# ===========================================================================
print("Generating: cluster_distribution.png")
fig, ax = plt.subplots(figsize=(8, 6))
cluster_counts = cluster_df['cluster'].value_counts().sort_index()
colors = ['#3498db', '#e74c3c']
labels = ['Cụm 0: Regular Customers\n(Khách thông thường)', 'Cụm 1: Champions\n(Khách VIP)']
explode = (0, 0.1)

wedges, texts, autotexts = ax.pie(
    cluster_counts.values, 
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    shadow=True,
    startangle=90
)
ax.set_title('Phân Bố Khách Hàng Theo Cụm', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/cluster_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ===========================================================================
# 3. Hình 2: So sánh RFM giữa các cụm (Bar chart)
# ===========================================================================
print("Generating: rfm_comparison.png")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

rfm_cols = ['Recency', 'Frequency', 'Monetary']
titles = ['Recency (ngày)', 'Frequency (đơn hàng)', 'Monetary (£)']
colors = ['#3498db', '#e74c3c']

for i, (col, title) in enumerate(zip(rfm_cols, titles)):
    means = cluster_df.groupby('cluster')[col].mean()
    bars = axes[i].bar(['Cụm 0\nRegular', 'Cụm 1\nChampions'], means.values, color=colors)
    axes[i].set_title(f'TB {title}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel(title)
    
    # Thêm giá trị trên cột
    for bar, val in zip(bars, means.values):
        height = bar.get_height()
        if col == 'Monetary':
            axes[i].text(bar.get_x() + bar.get_width()/2., height, f'£{val:,.0f}',
                        ha='center', va='bottom', fontweight='bold')
        else:
            axes[i].text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}',
                        ha='center', va='bottom', fontweight='bold')

plt.suptitle('So Sánh RFM Giữa Các Cụm', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/rfm_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ===========================================================================
# 4. Hình 3: PCA 2D Projection
# ===========================================================================
print("Generating: pca_clusters.png")

# Chuẩn bị features cho PCA (sử dụng RFM)
rfm_features = cluster_df[['Recency', 'Frequency', 'Monetary']].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm_features)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], 
    c=cluster_df['cluster'], 
    cmap='coolwarm',
    alpha=0.6,
    s=50
)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax.set_title('PCA 2D Projection: Phân Bố Cụm Khách Hàng', fontsize=14, fontweight='bold')

# Legend
legend_elements = [
    plt.scatter([], [], c='#3498db', s=100, label='Cụm 0: Regular'),
    plt.scatter([], [], c='#e74c3c', s=100, label='Cụm 1: Champions')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/pca_clusters.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ===========================================================================
# 5. Hình 4: Top 15 Rules theo Lift
# ===========================================================================
print("Generating: top_rules_lift.png")
top_rules = rules_df.head(15).copy()

fig, ax = plt.subplots(figsize=(12, 8))

# Tạo rule label ngắn gọn
if 'rule_str' in top_rules.columns:
    top_rules['rule_short'] = top_rules['rule_str'].apply(lambda x: x[:50] + '...' if len(str(x)) > 50 else x)
else:
    top_rules['rule_short'] = top_rules['antecedents_str'] + ' → ' + top_rules['consequents_str']
    top_rules['rule_short'] = top_rules['rule_short'].apply(lambda x: x[:50] + '...' if len(str(x)) > 50 else x)

bars = ax.barh(range(len(top_rules)), top_rules['lift'].values, color='steelblue')
ax.set_yticks(range(len(top_rules)))
ax.set_yticklabels(top_rules['rule_short'].values, fontsize=9)
ax.set_xlabel('Lift', fontsize=11)
ax.set_title('Top 15 Luật Kết Hợp (theo Lift)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Thêm giá trị
for bar, val in zip(bars, top_rules['lift'].values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/top_rules_lift.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ===========================================================================
# 6. Hình 5: Rules Scatter Plot (Support vs Confidence, color=Lift)
# ===========================================================================
print("Generating: rules_scatter.png")
fig, ax = plt.subplots(figsize=(10, 8))

top_100_rules = rules_df.head(100)
scatter = ax.scatter(
    top_100_rules['support'],
    top_100_rules['confidence'],
    c=top_100_rules['lift'],
    s=60,
    alpha=0.7,
    cmap='viridis'
)
cbar = plt.colorbar(scatter, ax=ax, label='Lift')
ax.set_xlabel('Support', fontsize=11)
ax.set_ylabel('Confidence', fontsize=11)
ax.set_title('Phân Bố Luật: Support vs Confidence (màu = Lift)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/rules_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ===========================================================================
# 7. Hình 6: RFM Distributions (Histogram)
# ===========================================================================
print("Generating: rfm_distributions.png")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (col, title) in enumerate(zip(rfm_cols, titles)):
    for cluster_id in [0, 1]:
        data = cluster_df[cluster_df['cluster'] == cluster_id][col]
        label = 'Champions' if cluster_id == 1 else 'Regular'
        color = '#e74c3c' if cluster_id == 1 else '#3498db'
        axes[i].hist(data, bins=30, alpha=0.5, label=label, color=color)
    
    axes[i].set_xlabel(title, fontsize=11)
    axes[i].set_ylabel('Số khách hàng', fontsize=11)
    axes[i].set_title(f'Phân Phối {col}', fontsize=12, fontweight='bold')
    axes[i].legend()

plt.suptitle('Phân Phối RFM Theo Cụm', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/rfm_distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ===========================================================================
# 8. Hình 7: Cluster Profile Summary
# ===========================================================================
print("Generating: cluster_profile.png")
fig, ax = plt.subplots(figsize=(10, 6))

# Tạo radar chart data
categories = ['Recency\n(thấp = tốt)', 'Frequency', 'Monetary', 'Số KH (%)']

# Normalize data to 0-1 scale for comparison
cluster_0 = cluster_df[cluster_df['cluster'] == 0]
cluster_1 = cluster_df[cluster_df['cluster'] == 1]

# Inverse recency (lower is better)
max_recency = cluster_df['Recency'].max()
max_frequency = cluster_df['Frequency'].max()
max_monetary = cluster_df['Monetary'].max()

values_0 = [
    1 - cluster_0['Recency'].mean() / max_recency,  # Inverse
    cluster_0['Frequency'].mean() / max_frequency,
    cluster_0['Monetary'].mean() / max_monetary,
    len(cluster_0) / len(cluster_df)
]

values_1 = [
    1 - cluster_1['Recency'].mean() / max_recency,  # Inverse
    cluster_1['Frequency'].mean() / max_frequency,
    cluster_1['Monetary'].mean() / max_monetary,
    len(cluster_1) / len(cluster_df)
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, values_0, width, label='Cụm 0: Regular', color='#3498db')
bars2 = ax.bar(x + width/2, values_1, width, label='Cụm 1: Champions', color='#e74c3c')

ax.set_ylabel('Giá trị chuẩn hóa (0-1)', fontsize=11)
ax.set_title('So Sánh Profile Cụm (Chuẩn Hóa)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend()
ax.set_ylim(0, 1.1)

# Thêm giá trị
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
            ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('images/cluster_profile.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ===========================================================================
# 9. Hình 8: Pipeline Diagram
# ===========================================================================
print("Generating: pipeline.png")
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

# Pipeline boxes
steps = [
    ('1. Tiền xử lý\nDữ liệu', '#3498db'),
    ('2. Khai phá\nLuật (Apriori)', '#2ecc71'),
    ('3. Feature\nEngineering', '#f1c40f'),
    ('4. Phân cụm\nK-Means', '#e74c3c'),
    ('5. Diễn giải\n& Profiling', '#9b59b6'),
    ('6. Chiến lược\nMarketing', '#1abc9c'),
]

box_width = 0.12
box_height = 0.6
y = 0.5
start_x = 0.05

for i, (label, color) in enumerate(steps):
    x = start_x + i * (box_width + 0.04)
    
    # Box
    rect = plt.Rectangle((x, y - box_height/2), box_width, box_height, 
                         facecolor=color, edgecolor='black', linewidth=2,
                         transform=ax.transAxes)
    ax.add_patch(rect)
    
    # Text
    ax.text(x + box_width/2, y, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white',
            transform=ax.transAxes)
    
    # Arrow
    if i < len(steps) - 1:
        ax.annotate('', xy=(x + box_width + 0.03, y), xytext=(x + box_width + 0.01, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2),
                   transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Pipeline: Phân Cụm Khách Hàng Dựa Trên Luật Kết Hợp', fontsize=14, fontweight='bold', y=0.95)

plt.tight_layout()
plt.savefig('images/pipeline.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ===========================================================================
# Done
# ===========================================================================
print("\n✅ Đã tạo xong tất cả hình ảnh trong thư mục 'images/':")
for f in os.listdir('images'):
    print(f"   - images/{f}")
