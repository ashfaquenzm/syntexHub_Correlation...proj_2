import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Streamlit Page Config
st.set_page_config(
    page_title="Correlation & Pairwise Relationships",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Project 3: Correlation Heatmap & Pairwise Relationships")
st.markdown("### Syntecxhub Internship Project")

# ---------------------------------------------------------
# 1. GENERATE / LOAD DATASET
# ---------------------------------------------------------
st.sidebar.header("⚙️ Data Settings")


@st.cache_data
def load_data():
    np.random.seed(42)
    n = 300

    # Generating realistic numeric features with intentional correlations
    marketing_budget = np.random.uniform(10, 100, n)
    website_visits = marketing_budget * 15 + np.random.normal(0, 100, n)
    conversions = website_visits * 0.08 + np.random.normal(0, 5, n)
    discount_rate = np.random.uniform(5, 30, n)
    profit_margin = 40 - (discount_rate * 0.8) + np.random.normal(0, 3, n)
    customer_satisfaction = profit_margin * 0.1 + np.random.normal(5, 1, n)

    df = pd.DataFrame(
        {
            "Marketing Budget ($k)": marketing_budget,
            "Website Visits": website_visits,
            "Conversions": conversions,
            "Discount Rate (%)": discount_rate,
            "Profit Margin (%)": profit_margin,
            "Customer Satisfaction": customer_satisfaction,
        }
    )
    return df.round(2)


df = load_data()

st.sidebar.subheader("Dataset Preview")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(df)

# Key Metrics
st.markdown("#### Overview of Dataset Features")
cols = st.columns(len(df.columns))
for idx, col in enumerate(df.columns):
    cols[idx].metric(label=col, value=f"{df[col].mean():.1f} (Avg)")

st.markdown("---")

# ---------------------------------------------------------
# 2. CORRELATION HEATMAP (MASK UPPER TRIANGLE)
# ---------------------------------------------------------
st.subheader("🔥 Pearson Correlation Heatmap")

# Compute Pearson Correlation
corr_matrix = df.corr(method="pearson")

# Mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    plt.title("Pearson Correlation Matrix (Annotated & Masked)", fontsize=12)
    st.pyplot(fig)

with col2:
    st.markdown("### 💡 Key Takeaways")

    # Find strongest positive and negative correlations
    unstacked = corr_matrix.unstack()
    unstacked_clean = unstacked[unstacked != 1.0]  # Remove self correlations

    max_pos = unstacked_clean.idxmax()
    max_pos_val = unstacked_clean.max()

    min_neg = unstacked_clean.idxmin()
    min_neg_val = unstacked_clean.min()

    st.success(
        f"**Strongest Positive Correlation:**\n\n"
        f"**{max_pos[0]}** ↔ **{max_pos[1]}** (`{max_pos_val:.2f}`)"
    )

    st.error(
        f"**Strongest Negative Correlation:**\n\n"
        f"**{min_neg[0]}** ↔ **{min_neg[1]}** (`{min_neg_val:.2f}`)"
    )

st.markdown("---")

# ---------------------------------------------------------
# 3. PAIRWISE RELATIONSHIPS & SCATTER PLOTS
# ---------------------------------------------------------
st.subheader("📌 Pairwise Relationships (Scatter Matrix / Pairplot)")

selected_features = st.multiselect(
    "Select features to plot pairplot:",
    options=df.columns.tolist(),
    default=[
        "Marketing Budget ($k)",
        "Website Visits",
        "Conversions",
        "Profit Margin (%)",
    ],
)

if len(selected_features) >= 2:
    pairplot_fig = sns.pairplot(
        df[selected_features], corner=True, diag_kind="kde", corner_mask=True
    )
    st.pyplot(pairplot_fig.fig)
else:
    st.warning("Kam se kam 2 features select karein pairplot dekhne ke liye.")

st.markdown("---")

# ---------------------------------------------------------
# 4. SUMMARY & DISCUSSION
# ---------------------------------------------------------
st.subheader("📝 Project Summary & Findings")

st.markdown(
    f"""
1. **Strong Positive Relationships:**
   - **{max_pos[0]}** and **{max_pos[1]}** exhibit a very strong positive correlation (`r = {max_pos_val:.2f}`). Increasing budget directly drives higher website visits and conversions.

2. **Strong Negative Relationships:**
   - **{min_neg[0]}** and **{min_neg[1]}** share a strong inverse relationship (`r = {min_neg_val:.2f}`). Offering higher discount rates significantly reduces overall profit margins.

3. **Chart Choice & Formatting Discussion:**
   - **Heatmap with Masking**: Upper triangle mask is applied to remove redundant duplicate values, making the correlation matrix clean and easy to read.
   - **Annotations**: Values are annotated (`fmt='.2f'`) directly inside each cell to give exact metric precision.
   - **Pairplots**: Provide a quick holistic view of linear/non-linear relationships across key variables alongside univariate KDE distributions.
"""
)