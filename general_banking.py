import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from faker import Faker
import random
from datetime import datetime

# Page config and custom CSS
st.set_page_config(page_title="Banking Analytics Dashboard", layout="wide", page_icon="💳")

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample data generation
faker = Faker()
np.random.seed(42)
random.seed(42)

N = 10000

account_types = ['Checking', 'Savings', 'Credit', 'Loan']
account_weights = [0.2, 0.3, 0.3, 0.2]

transaction_channels = ['ATM', 'Online', 'Mobile App', 'Branch']
channel_category_map = {
    'ATM': ['Utilities', 'Groceries'],
    'Online': ['Travel', 'Entertainment', 'Dining'],
    'Mobile App': ['Groceries', 'Utilities', 'Entertainment'],
    'Branch': ['Rent', 'Utilities', 'Loan Payment']
}

marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
marital_weights = [0.4, 0.45, 0.1, 0.05]

education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
education_weights = [0.4, 0.35, 0.2, 0.05]

employment_statuses = ['Employed', 'Unemployed', 'Self-employed', 'Retired']
employment_weights = [0.6, 0.1, 0.2, 0.1]

loan_statuses = ['Active', 'Paid off', 'Defaulted']
genders = ['Male', 'Female']
currencies = ['USD', 'EUR', 'GBP']
residential_statuses = ['Own', 'Rent', 'Mortgage']

data = []

for i in range(N):
    dob = faker.date_of_birth(minimum_age=18, maximum_age=75)
    account_open_date = faker.date_between(start_date='-10y', end_date='today')
    age = (pd.Timestamp('today') - pd.to_datetime(dob)).days // 365

    gender = random.choice(genders)

    marital_status = random.choices(marital_statuses, weights=marital_weights)[0]
    education_level = random.choices(education_levels, weights=education_weights, k=1)[0]

    employment_status = random.choices(employment_statuses, weights=employment_weights, k=1)[0]

    edu_income_map = {
        'High School': (30000, 10000),
        'Bachelor': (50000, 15000),
        'Master': (65000, 20000),
        'PhD': (80000, 25000)
    }
    mean_income, std_income = edu_income_map[education_level]
    annual_income = max(round(np.random.normal(loc=mean_income, scale=std_income), 2), 1000)

    if employment_status == 'Unemployed':
        annual_income *= 0.5
    elif employment_status == 'Retired':
        annual_income *= 0.6

    credit_score = random.randint(500, 750)

    balance = round(np.random.lognormal(mean=9, sigma=0.8), 2)

    account_type = random.choices(account_types, weights=account_weights, k=1)[0]

    transaction_channel = random.choice(transaction_channels)
    transaction_category = random.choice(channel_category_map[transaction_channel])
    transaction_amount = round(np.random.exponential(scale=200), 2)

    loan_amount = round(np.random.normal(loc=15000, scale=5000), 2)
    loan_amount = max(0, loan_amount)

    data.append({
        'customer_id': i + 1,
        'first_name': faker.first_name(),
        'last_name': faker.last_name(),
        'gender': gender,
        'age': age,
        'marital_status': marital_status,
        'education_level': education_level,
        'employment_status': employment_status,
        'annual_income': round(annual_income, 2),
        'credit_score': credit_score,
        'residential_status': random.choice(residential_statuses),
        'account_type': account_type,
        'account_open_date': account_open_date,
        'balance': balance,
        'currency': random.choice(currencies),
        'transaction_channel': transaction_channel,
        'transaction_category': transaction_category,
        'transaction_amount': transaction_amount,
        'loan_amount': loan_amount,
        'loan_status': random.choice(loan_statuses),
    })

df = pd.DataFrame(data)

# Streamlit page setup
st.title("Banking Analytics Dashboard")

# Sidebar filters
def apply_filters(df):
    st.sidebar.header("🔧 Filter Options")
    
    age_min, age_max = st.sidebar.slider(
        "Age Range:",
        min_value=int(df['age'].min()), 
        max_value=int(df['age'].max()), 
        value=(int(df['age'].min()), int(df['age'].max())),
        step=1
    )
    
    marital_status_options = ['All'] + sorted(df['marital_status'].unique().tolist())
    selected_marital_status = st.sidebar.multiselect("Marital Status:", options=marital_status_options, default=['All'])
    
    education_options = ['All'] + sorted(df['education_level'].unique().tolist())
    selected_education = st.sidebar.multiselect("Education Level:", options=education_options, default=['All'])
    
    loan_status_options = ['All'] + sorted(df['loan_status'].unique().tolist())
    selected_loan_status = st.sidebar.selectbox("Loan Status:", options=loan_status_options, index=0)
    
    # Apply filters
    filtered_df = df.copy()
    
    filtered_df = filtered_df[(filtered_df['age'] >= age_min) & (filtered_df['age'] <= age_max)]
    
    if 'All' not in selected_marital_status:
        filtered_df = filtered_df[filtered_df['marital_status'].isin(selected_marital_status)]
    
    if 'All' not in selected_education:
        filtered_df = filtered_df[filtered_df['education_level'].isin(selected_education)]
    
    if selected_loan_status != 'All':
        filtered_df = filtered_df[filtered_df['loan_status'] == selected_loan_status]
    
    return filtered_df

# KPI calculation
def calculate_kpis(df):
    if len(df) == 0:
        return dict.fromkeys(['total_customers', 'average_income', 'average_balance', 'active_loans'], 0)
    
    kpis = {
        'total_customers': len(df),
        'average_income': df['annual_income'].mean(),
        'average_balance': df['balance'].mean(),
        'active_loans': df[df['loan_status'] == 'Active'].shape[0]
    }
    return kpis

# KPI display
def display_kpis(kpis):
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Customers", f"{kpis['total_customers']:,}")
    with col2:
        st.metric("💰 Avg Income", f"${kpis['average_income']:,.2f}")
    with col3:
        st.metric("💵 Avg Balance", f"${kpis['average_balance']:,.2f}")
    with col4:
        st.metric("🏦 Active Loans", f"{kpis['active_loans']:,}")

# Visualization function
def create_visualizations(df):
    st.markdown("---")
    st.markdown("### 📊 Data Visualizations")
    
    # Plot 1: Average Income by Employment Status (Plotly Bar)
    fig = px.bar(df, x='employment_status', y='annual_income', 
                 title='Average Income by Employment Status', 
                 labels={'annual_income': 'Average Income', 'employment_status': 'Employment Status'},
                 color='employment_status', 
                 barmode='group')
    st.plotly_chart(fig)

    # Plot 2: Distribution of Annual Income (Plotly Histogram)
    fig = px.histogram(df, x='annual_income', nbins=30, 
                       title='Distribution of Annual Income',
                       labels={'annual_income': 'Annual Income'},
                       color_discrete_sequence=['green'])
    fig.update_layout(xaxis_title='Annual Income', yaxis_title='Number of Customers')
    st.plotly_chart(fig)

    # Plot 3: Credit Score Distribution by Loan Status (Plotly Box Plot)
    fig = px.box(df, x='loan_status', y='credit_score', 
                 title="Credit Score Distribution by Loan Status", 
                 color='loan_status',
                 labels={'loan_status': 'Loan Status', 'credit_score': 'Credit Score'})
    st.plotly_chart(fig)

    # Plot 4: Annual Income vs Loan Amount (Plotly Scatter Plot)
    fig = px.scatter(df, x='annual_income', y='loan_amount',
                     title='Annual Income vs Loan Amount',
                     labels={'annual_income': 'Annual Income', 'loan_amount': 'Loan Amount'},
                     opacity=0.6)
    st.plotly_chart(fig)

    # Plot 5: Transaction Channel Distribution (Plotly Pie Chart)
    channel_counts = df['transaction_channel'].value_counts()
    fig = px.pie(values=channel_counts.values, names=channel_counts.index,
                 title='Transaction Channel Distribution')
    st.plotly_chart(fig)

    # Plot 6: Transaction Categories by Frequency (Plotly Bar)
    fig = px.bar(df, y='transaction_category', 
                 title='Transaction Categories by Frequency', 
                 labels={'transaction_category': 'Transaction Category'},
                 category_orders={"transaction_category": df['transaction_category'].value_counts().index.tolist()},
                 color='transaction_category', 
                 orientation='h')
    fig.update_layout(xaxis_title='Number of Transactions')
    st.plotly_chart(fig)

    # Plot 7: Average Balance by Age (Plotly Line Plot)
    age_balance = df.groupby('age')['balance'].mean().reset_index()
    fig = px.line(age_balance, x='age', y='balance', 
                  title='Average Balance by Age', 
                  labels={'age': 'Age', 'balance': 'Average Balance'},
                  markers=True)
    st.plotly_chart(fig)

    # Plot 8: Loan Status by Employment Status (Plotly Bar)
    fig = px.histogram(df, x='loan_status', color='employment_status',
                       title='Loan Status by Employment Status', 
                       labels={'loan_status': 'Loan Status'})
    st.plotly_chart(fig)

    # Plot 9: Transaction Amount Distribution (Plotly Histogram)
    fig = px.histogram(df, x='transaction_amount', nbins=30, 
                       title='Transaction Amount Distribution', 
                       labels={'transaction_amount': 'Transaction Amount'})
    fig.update_layout(xaxis_title='Transaction Amount', yaxis_title='Frequency')
    st.plotly_chart(fig)

    # Plot 10: Credit Score by Education Level (Plotly Box Plot)
    fig = px.box(df, x='education_level', y='credit_score',
                 title='Credit Score by Education Level',
                 color='education_level',
                 labels={'education_level': 'Education Level', 'credit_score': 'Credit Score'})
    st.plotly_chart(fig)


# Main function
def main():
    st.markdown("""
    <div class="main-header">
        <h1>💳 Banking Analytics Dashboard</h1>
        <p>Analyze banking data, customer behavior, and financial transactions.</p>
    </div>
    """, unsafe_allow_html=True)

    filtered_df = apply_filters(df)
    st.success(f"📋 Showing {len(filtered_df):,} customers out of {len(df):,} total customers")

    kpis = calculate_kpis(filtered_df)
    display_kpis(kpis)

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if len(filtered_df) > 0:
            st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False).encode('utf-8'), file_name=f'banking_data_filtered_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')
    with col2:
        st.download_button("📥 Download Full Dataset", df.to_csv(index=False).encode('utf-8'), file_name=f'banking_data_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')

    if st.sidebar.checkbox("Show Raw Data"):
        st.markdown("### 📄 Raw Data Preview")
        if len(filtered_df) > 0:
            st.dataframe(filtered_df, use_container_width=True, height=300)
        else:
            st.info("No data to display with current filters")

    if len(filtered_df) > 0:
        create_visualizations(filtered_df)
    else:
        st.warning("⚠️ No data available with current filters.")

    st.markdown("---")
    st.markdown("### 📊 Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Total Records:** {len(filtered_df):,}")
    with col2:
        st.info(f"**Total Columns:** {len(filtered_df.columns)}")
    with col3:
        st.info(f"**Memory Usage:** {filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    st.markdown("---")
    st.markdown("""<div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Banking Analytics Dashboard v1.0</p>
    </div>""", unsafe_allow_html=True)

# Run
if __name__ == "__main__":
    main()