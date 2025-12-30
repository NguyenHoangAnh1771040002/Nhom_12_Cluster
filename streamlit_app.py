# streamlit run streamlit_app.py
"""
Dashboard Phân Tích Cụm Khách Hàng
==================================
Trực quan hóa và khám phá các phân khúc khách hàng dựa trên luật kết hợp.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Dashboard Phân Cụm Khách Hàng",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("🛍️ Dashboard Phân Tích Cụm Khách Hàng")
st.markdown("**Khám phá các phân khúc khách hàng dựa trên luật kết hợp (Association Rules)**")

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_cluster_data():
    """Tải dữ liệu phân cụm khách hàng."""
    data_dir = "data/processed"
    cluster_file = os.path.join(data_dir, "customer_clusters_from_rules.csv")
    
    if not os.path.exists(cluster_file):
        st.error(f"❌ Không tìm thấy file cụm: {cluster_file}")
        return None
    
    return pd.read_csv(cluster_file)

@st.cache_data
def load_rules_data():
    """Tải dữ liệu luật kết hợp."""
    data_dir = "data/processed"
    rules_file = os.path.join(data_dir, "rules_apriori_filtered.csv")
    
    if not os.path.exists(rules_file):
        return None
    
    return pd.read_csv(rules_file)

@st.cache_data
def load_strategies_data():
    """Tải dữ liệu chiến lược marketing."""
    data_dir = "data/processed"
    strategy_file = os.path.join(data_dir, "cluster_strategies.csv")
    
    if not os.path.exists(strategy_file):
        return None
    
    return pd.read_csv(strategy_file)

# Load data
meta_df = load_cluster_data()
rules_df = load_rules_data()
strategy_df = load_strategies_data()

if meta_df is None:
    st.error("Không thể tải dữ liệu cụm. Vui lòng chạy notebook phân cụm trước.")
    st.stop()

# ============================================================================
# SIDEBAR: LỰA CHỌN CỤM
# ============================================================================

st.sidebar.header("🎯 Chọn Cụm Khách Hàng")

clusters = sorted(meta_df['cluster'].unique())
selected_cluster = st.sidebar.selectbox(
    "Chọn cụm để phân tích:",
    clusters,
    format_func=lambda x: f"Cụm {x}",
)

cluster_data = meta_df[meta_df['cluster'] == selected_cluster]
n_customers = len(cluster_data)
pct_customers = 100 * n_customers / len(meta_df)

st.sidebar.info(
    f"**Cụm đã chọn: {selected_cluster}**\n\n"
    f"👥 Số khách hàng: {n_customers:,} ({pct_customers:.1f}%)\n"
    f"📊 Tổng số khách hàng: {len(meta_df):,}"
)

# Hiển thị chiến lược nếu có
if strategy_df is not None:
    cluster_strategy = strategy_df[strategy_df['cluster_id'] == selected_cluster]
    if not cluster_strategy.empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Hồ Sơ Cụm")
        row = cluster_strategy.iloc[0]
        if 'name_en' in row:
            st.sidebar.write(f"**Tên:** {row.get('name_vi', 'N/A')} ({row.get('name_en', 'N/A')})")
        if 'persona' in row:
            st.sidebar.write(f"**Đặc điểm:** {row['persona']}")

# ============================================================================
# NỘI DUNG CHÍNH: TỔNG QUAN CỤM
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="👥 Số Khách Hàng", value=f"{n_customers:,}", delta=f"{pct_customers:.1f}%")

if 'Recency' in cluster_data.columns:
    with col2:
        avg_recency = cluster_data['Recency'].mean()
        st.metric(label="📅 TB Recency (ngày)", value=f"{avg_recency:.0f}")
    
    with col3:
        avg_frequency = cluster_data['Frequency'].mean()
        st.metric(label="🛒 TB Tần Suất", value=f"{avg_frequency:.1f}")
    
    with col4:
        avg_monetary = cluster_data['Monetary'].mean()
        st.metric(label="💷 TB Chi Tiêu (£)", value=f"£{avg_monetary:.2f}")

st.markdown("---")

# ============================================================================
# TAB 1: THỐNG KÊ CỤM
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Thống Kê", "🎁 Top Luật", "💰 Phân Tích RFM", "🧠 Chiến Lược", "⚙️ Cài Đặt"])

with tab1:
    st.subheader(f"Thống Kê Cụm {selected_cluster}")
    
    # So sánh các cụm
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Phân Bố Số Lượng Khách Hàng Theo Cụm**")
        cluster_sizes = meta_df['cluster'].value_counts().sort_index()
        fig_sizes = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            labels={'x': 'Mã Cụm', 'y': 'Số Khách Hàng'},
            title="Số Khách Hàng Mỗi Cụm",
        )
        fig_sizes.add_hline(y=n_customers, line_dash="dash", line_color="red", 
                            annotation_text=f"Đã chọn: {n_customers}", annotation_position="right")
        st.plotly_chart(fig_sizes, use_container_width=True)
    
    with col2:
        st.write("**So Sánh RFM Giữa Các Cụm**")
        if 'Recency' in cluster_data.columns:
            rfm_stats = []
            for cid in clusters:
                cdata = meta_df[meta_df['cluster'] == cid]
                rfm_stats.append({
                    'Cụm': f"Cụm {cid}",
                    'TB Recency': cdata['Recency'].mean(),
                    'TB Tần Suất': cdata['Frequency'].mean(),
                    'TB Chi Tiêu': cdata['Monetary'].mean(),
                })
            rfm_df = pd.DataFrame(rfm_stats)
            
            # Highlight cụm đã chọn
            st.dataframe(
                rfm_df.assign(Đã_chọn=rfm_df['Cụm'] == f"Cụm {selected_cluster}"),
                hide_index=True,
            )

# ============================================================================
# TAB 2: TOP LUẬT KẾT HỢP
# ============================================================================

with tab2:
    st.subheader(f"Top Luật Kết Hợp Cụm {selected_cluster}")
    
    if rules_df is None:
        st.warning("Không có dữ liệu luật. Vui lòng kiểm tra file rules CSV.")
    else:
        # Hiển thị top 15 luật
        top_rules = rules_df.head(15).copy()
        
        if len(top_rules) > 0:
            display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
            available_cols = [c for c in display_cols if c in top_rules.columns]
            
            st.dataframe(
                top_rules[available_cols],
                hide_index=True,
                use_container_width=True,
            )
            
            # Biểu đồ scatter top luật
            if 'lift' in top_rules.columns and 'confidence' in top_rules.columns:
                fig_scatter = px.scatter(
                    top_rules,
                    x='confidence',
                    y='lift',
                    size='support' if 'support' in top_rules.columns else None,
                    hover_data=['antecedents_str', 'consequents_str'],
                    title="Biểu Đồ Luật: Lift vs Confidence",
                    labels={'confidence': 'Độ Tin Cậy', 'lift': 'Lift'},
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================================
# TAB 3: PHÂN TÍCH RFM CHI TIẾT
# ============================================================================

with tab3:
    if 'Recency' in cluster_data.columns:
        st.subheader(f"Phân Tích RFM Chi Tiết - Cụm {selected_cluster}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Recency (Số Ngày Từ Lần Mua Cuối)**")
            st.metric("Trung Bình", f"{cluster_data['Recency'].mean():.0f} ngày")
            st.metric("Trung Vị", f"{cluster_data['Recency'].median():.0f} ngày")
            st.metric("Độ Lệch Chuẩn", f"{cluster_data['Recency'].std():.0f} ngày")
            
            # Phân phối
            fig_recency = px.histogram(cluster_data, x='Recency', nbins=30,
                                       title="Phân Phối Recency", labels={'Recency': 'Số ngày'})
            st.plotly_chart(fig_recency, use_container_width=True)
        
        with col2:
            st.write("**Frequency (Tần Suất Mua Hàng)**")
            st.metric("Trung Bình", f"{cluster_data['Frequency'].mean():.1f}")
            st.metric("Trung Vị", f"{cluster_data['Frequency'].median():.1f}")
            st.metric("Độ Lệch Chuẩn", f"{cluster_data['Frequency'].std():.1f}")
            
            fig_frequency = px.histogram(cluster_data, x='Frequency', nbins=20,
                                        title="Phân Phối Tần Suất")
            st.plotly_chart(fig_frequency, use_container_width=True)
        
        with col3:
            st.write("**Monetary (Tổng Chi Tiêu £)**")
            st.metric("Trung Bình", f"£{cluster_data['Monetary'].mean():.2f}")
            st.metric("Trung Vị", f"£{cluster_data['Monetary'].median():.2f}")
            st.metric("Độ Lệch Chuẩn", f"£{cluster_data['Monetary'].std():.2f}")
            
            fig_monetary = px.histogram(cluster_data, x='Monetary', nbins=20,
                                       title="Phân Phối Chi Tiêu", labels={'Monetary': 'Chi tiêu (£)'})
            st.plotly_chart(fig_monetary, use_container_width=True)
        
        # Ma trận tương quan RFM
        st.subheader("Phân Tích Tương Quan RFM")
        rfm_cols = ['Recency', 'Frequency', 'Monetary']
        corr = cluster_data[rfm_cols].corr()
        fig_heatmap = px.imshow(corr, text_auto=True, title="Ma Trận Tương Quan RFM", aspect="auto")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Không có dữ liệu RFM trong file cụm.")

# ============================================================================
# TAB 4: CHIẾN LƯỢC MARKETING
# ============================================================================

with tab4:
    st.subheader(f"Chiến Lược Marketing Cụm {selected_cluster}")
    
    # Lấy chiến lược nếu có
    if strategy_df is not None:
        cluster_strategy = strategy_df[strategy_df['cluster_id'] == selected_cluster]
        if not cluster_strategy.empty:
            row = cluster_strategy.iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(
                    f"**Tên Cụm (EN):** {row.get('name_en', 'N/A')}\n\n"
                    f"**Tên Cụm (VN):** {row.get('name_vi', 'N/A')}"
                )
            
            with col2:
                st.success(
                    f"**Đặc Điểm Khách Hàng:** {row.get('persona', 'N/A')}"
                )
            
            st.markdown("---")
            
            # Chiến lược marketing
            if 'strategy' in row and pd.notna(row['strategy']):
                st.subheader("🎯 Chiến Lược Marketing Đề Xuất")
                st.write(row['strategy'])
            
            # Đề xuất dựa trên RFM
            if 'Recency' in cluster_data.columns:
                st.subheader("💡 Đề Xuất Dựa Trên Dữ Liệu")
                
                avg_recency = cluster_data['Recency'].mean()
                avg_frequency = cluster_data['Frequency'].mean()
                avg_monetary = cluster_data['Monetary'].mean()
                
                recommendations = []
                
                if avg_recency > 90:
                    recommendations.append(
                        "⚠️ **Kích Hoạt Khách Ngủ Đông:** Recency cao cho thấy khách hàng đã lâu không mua. "
                        "Cân nhắc chiến dịch win-back, ưu đãi đặc biệt, hoặc email tái kết nối."
                    )
                
                if avg_frequency > 10:
                    recommendations.append(
                        "✅ **Chương Trình Khách Hàng Thân Thiết:** Tần suất cao cho thấy khách hàng trung thành. "
                        "Triển khai quyền lợi VIP, giảm giá độc quyền, hoặc chương trình tích điểm."
                    )
                
                if avg_monetary > meta_df['Monetary'].median() * 1.5:
                    recommendations.append(
                        "💎 **Upsell Sản Phẩm Cao Cấp:** Phân khúc chi tiêu cao. Giới thiệu sản phẩm premium, "
                        "bundle độc quyền, hoặc bộ sưu tập đặc biệt cho nhóm này."
                    )
                
                if avg_frequency < 5 and avg_monetary < meta_df['Monetary'].median():
                    recommendations.append(
                        "🌱 **Nuôi Dưỡng Khách Mới:** Phân khúc chưa gắn bó. Tập trung giáo dục sản phẩm, "
                        "ưu đãi lần mua đầu, hoặc khuyến mãi sản phẩm phổ thông."
                    )
                
                for rec in recommendations:
                    st.write(rec)
        else:
            st.warning("Không có dữ liệu chiến lược cho cụm này.")
    else:
        st.info("Chưa tải được dữ liệu chiến lược. Vui lòng chạy notebook phân cụm để tạo chiến lược.")

# ============================================================================
# TAB 5: CÀI ĐẶT & THÔNG TIN
# ============================================================================

with tab5:
    st.subheader("Thông Tin Dữ Liệu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tóm Tắt Dữ Liệu**")
        st.write(f"Tổng số khách hàng: {len(meta_df):,}")
        st.write(f"Tổng số cụm: {len(clusters)}")
        st.write(f"Các cột dữ liệu: {', '.join(meta_df.columns.tolist())}")
    
    with col2:
        st.write("**Đường Dẫn File**")
        st.code(f"Cụm: data/processed/customer_clusters_from_rules.csv\n"
                f"Luật: data/processed/rules_apriori_filtered.csv\n"
                f"Chiến lược: data/processed/cluster_strategies.csv")
    
    # Xem trước dữ liệu
    st.subheader("Xem Trước Dữ Liệu")
    st.dataframe(cluster_data.head(10), use_container_width=True)
    
    # Nút tải xuống
    st.subheader("📥 Xuất Dữ Liệu")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_cluster = cluster_data.to_csv(index=False)
        st.download_button(
            label="Tải Dữ Liệu Cụm (CSV)",
            data=csv_cluster,
            file_name=f"cum_{selected_cluster}_du_lieu.csv",
            mime="text/csv",
        )
    
    with col2:
        if rules_df is not None:
            csv_rules = rules_df.to_csv(index=False)
            st.download_button(
                label="Tải Luật Kết Hợp (CSV)",
                data=csv_rules,
                file_name="luat_ket_hop.csv",
                mime="text/csv",
            )
    
    with col3:
        if strategy_df is not None:
            csv_strategy = strategy_df.to_csv(index=False)
            st.download_button(
                label="Tải Chiến Lược (CSV)",
                data=csv_strategy,
                file_name="chien_luoc_cum.csv",
                mime="text/csv",
            )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; font-size: 0.8em; color: gray;'>"
    f"Dashboard Phân Tích Cụm Khách Hàng | Nhóm 12 | Tạo lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True,
)
