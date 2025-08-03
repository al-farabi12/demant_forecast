import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.colors import qualitative

st.set_page_config(page_title="Multi-Target Quantity Forecasting", layout="wide")
st.title("Multi-Target Quantity Forecasting")

# ------------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------------
def enforce_nonneg_int(series: pd.Series, how: str = "round") -> pd.Series:
    vals = np.clip(series, 0, None)
    if how == "floor": arr = np.floor(vals)
    elif how == "ceil": arr = np.ceil(vals)
    elif how == "round": arr = np.round(vals)
    else: return pd.Series(vals, index=series.index).astype(float)
    return pd.Series(arr, index=series.index).fillna(0).astype(int)

def safe_parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out = out.sort_values(date_col)
    return out

def aggregate_to_daily(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    temp = df.copy()
    temp[date_col] = temp[date_col].dt.normalize()
    daily = temp.groupby(date_col, as_index=False)[value_col].sum()
    return daily

def reindex_missing_dates(daily: pd.DataFrame, date_col: str, value_col: str, fill_strategy: str = "ffill") -> pd.DataFrame:
    if daily.empty: return daily
    idx = pd.date_range(daily[date_col].min(), daily[date_col].max(), freq="D")
    daily = daily.set_index(date_col).reindex(idx)
    if fill_strategy == "zero": daily[value_col] = daily[value_col].fillna(0)
    elif fill_strategy == "ffill": daily[value_col] = daily[value_col].ffill().fillna(0)
    daily.index.name = date_col
    daily = daily.reset_index()
    return daily

def train_valid_slice(daily: pd.DataFrame, date_col: str, train_days: int, valid_days: int):
    daily = daily.sort_values(date_col)
    n = len(daily)
    if n == 0: return daily.iloc[0:0], daily.iloc[0:0]
    train_days = int(min(train_days, n))
    valid_days = int(min(valid_days, n - train_days)) if n - train_days > 0 else 0
    train_start_idx = max(0, n - (train_days + valid_days))
    train_end_idx = train_start_idx + train_days
    valid_end_idx = train_end_idx + valid_days
    train_df = daily.iloc[train_start_idx:train_end_idx].copy()
    valid_df = daily.iloc[train_end_idx:valid_end_idx].copy()
    return train_df, valid_df

def build_prophet_model(train_df: pd.DataFrame, seasonality_ok: bool) -> Prophet:
    if seasonality_ok: m = Prophet(growth="linear")
    else: m = Prophet(growth="linear", yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    return m

def fit_and_forecast_prophet(train_df: pd.DataFrame, valid_days: int, horizon_days: int, seasonality_ok: bool):
    model = build_prophet_model(train_df, seasonality_ok=seasonality_ok)
    model.fit(train_df)
    periods = valid_days + horizon_days
    future = model.make_future_dataframe(periods=periods, freq="D", include_history=True)
    fcst = model.predict(future)
    return model, fcst

def calc_metrics(actual: pd.Series, pred: pd.Series):
    if len(actual) == 0 or len(pred) == 0: return {k: np.nan for k in ["MAE","RMSE","sMAPE","MAPE"]}
    actual, pred = actual.reset_index(drop=True), pred.reset_index(drop=True)
    err = actual - pred
    mae, rmse = np.mean(np.abs(err)), np.sqrt(np.mean(err**2))
    mask = actual > 0
    mape = np.mean(np.abs(err[mask] / actual[mask])) * 100 if mask.any() else np.nan
    denom = np.abs(actual) + np.abs(pred)
    smape = np.mean(np.where(denom == 0, 0, 2 * np.abs(err) / denom)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}

# ------------------------------------------------------------------------------------
# PLOTTING FUNCTION FOR MULTIPLE TARGETS
# ------------------------------------------------------------------------------------
def plotly_multi_target_forecast(item_name: str, forecast_data: dict, rounding_rule: str):
    """Plots actuals and forecasts for multiple metrics on a single chart."""
    fig = go.Figure()
    colors = qualitative.Plotly
    metrics_results = {}
    
    last_train_date, last_valid_date = None, None

    for i, (metric_name, data) in enumerate(forecast_data.items()):
        color = colors[i % len(colors)]
        daily = data['daily']
        fcst = data['fcst']
        train_df = data['train_df']
        valid_df = data['valid_df']
        
        if i == 0:
            last_train_date = train_df['ds'].max() if not train_df.empty else None
            last_valid_date = valid_df['ds'].max() if not valid_df.empty else last_train_date
            
        fig.add_trace(go.Scatter(
            x=daily['ds'], y=daily['y'], name=f'Actual - {metric_name}',
            mode='lines+markers', line=dict(color=color), marker=dict(size=4),
            legendgroup=metric_name
        ))
        
        fcst_display = fcst.copy()
        fcst_display['yhat_display'] = enforce_nonneg_int(fcst['yhat'], how=rounding_rule)
        
        fig.add_trace(go.Scatter(
            x=fcst_display['ds'], y=fcst_display['yhat_display'], name=f'Forecast - {metric_name}',
            mode='lines', line=dict(color=color, dash='dash'),
            legendgroup=metric_name
        ))
        
        if not valid_df.empty:
            valid_pred = fcst.set_index('ds').loc[valid_df['ds']]
            metrics_results[metric_name] = calc_metrics(valid_df['y'], valid_pred['yhat'])

    if last_train_date:
        fig.add_vline(x=last_train_date, line=dict(color='gray', dash='dot'), name='Train End')
    if last_valid_date and last_valid_date != last_train_date:
        fig.add_vline(x=last_valid_date, line=dict(color='gray', dash='dot'), name='Valid End')

    fig.update_layout(
        title=f"Forecasts for: {item_name}",
        xaxis_title="Date",
        yaxis_title="Quantity",
        legend_title="Metric",
        hovermode='x unified'
    )
    
    metrics_df = pd.DataFrame(metrics_results).T.reset_index()
    if not metrics_df.empty:
        metrics_df = metrics_df.rename(columns={'index': 'Metric'})
        for col in ['MAE', 'RMSE', 'MAPE', 'sMAPE']:
            if col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
    
    return fig, metrics_df


# ------------------------------------------------------------------------------------
# Sidebar / Controls
# ------------------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file using the sidebar to begin.")
    st.stop()

raw = pd.read_csv(uploaded_file)

with st.sidebar:
    st.header("2. Configure Columns")
    column_options = raw.columns.tolist()
    outlet_column = st.selectbox("Select the outlet column:", column_options, index=0)
    product_column = st.selectbox("Select the product column:", column_options, index=1)
    date_column = st.selectbox("Select the date column:", column_options, index=2)
    
    available_value_cols = [c for c in column_options if c not in [outlet_column, product_column, date_column]]
    value_columns = st.multiselect(
        "Select the quantity columns to forecast:", 
        available_value_cols, 
        default=available_value_cols[:2] if len(available_value_cols) >= 2 else available_value_cols,
        help="Select one or more metrics to forecast together on the same plot."
    )

    st.header("3. Forecasting Scope")
    forecast_level = st.selectbox("Forecast level:", ["Product (across all outlets)", "Product by Outlet", "Outlet (all products combined)"], index=0)
    
    st.subheader("Filters")
    selected_outlets = st.multiselect("Select outlets:", sorted(raw[outlet_column].dropna().unique()))
    selected_products = st.multiselect("Select products:", sorted(raw[product_column].dropna().unique()))

    st.header("4. Set Forecast Windows")
    train_days = st.number_input("TRAIN days", 2, 1000, 90)
    valid_days = st.number_input("VALIDATION days", 0, 1000, 14)
    horizon_days = st.number_input("FUTURE days to forecast", 1, 365, 30)

    st.header("5. Model & Display Options")
    rounding_rule = st.selectbox("Rounding rule", ["round","floor","ceil", "none"], 0)
    fill_strategy = st.selectbox("Missing date fill strategy", ["ffill","zero","nan"], 0)
    auto_seasonality = st.checkbox("Enable Prophet seasonality if train history >= 30 days", True)
    
    run_button = st.button("Run All Forecasts", type="primary")

# ------------------------------------------------------------------------------------
# Main Execution Logic
# ------------------------------------------------------------------------------------
if run_button:
    if not value_columns:
        st.error("Please select at least one quantity column to forecast.")
        st.stop()
        
    filtered_data = raw.copy()
    if selected_outlets: filtered_data = filtered_data[filtered_data[outlet_column].isin(selected_outlets)]
    if selected_products: filtered_data = filtered_data[filtered_data[product_column].isin(selected_products)]
    
    if forecast_level == "Product (across all outlets)": group_cols, group_label = [product_column], "Product"
    elif forecast_level == "Product by Outlet": group_cols, group_label = [product_column, outlet_column], "Product-Outlet"
    else: group_cols, group_label = [outlet_column], "Outlet"
    
    unique_groups = filtered_data[group_cols].drop_duplicates().reset_index(drop=True)
    all_results_for_csv = []
    
    st.markdown("---")
    
    for i, row in unique_groups.iterrows():
        filter_condition = pd.Series(True, index=filtered_data.index)
        group_name_parts = []
        for col in group_cols:
            filter_condition &= (filtered_data[col] == row[col])
            group_name_parts.append(f"{row[col]}")
        group_name = " | ".join(group_name_parts)
        
        st.subheader(f"Forecast for Group: {group_name}")
        
        forecasts_for_group = {}
        
        for value_col_to_forecast in value_columns:
            df_group = filtered_data[filter_condition][[date_column, value_col_to_forecast]].copy()
            
            if df_group.empty or df_group[value_col_to_forecast].isnull().all():
                continue

            daily = safe_parse_dates(df_group, date_column)
            daily = aggregate_to_daily(daily, date_column, value_col_to_forecast)
            daily = reindex_missing_dates(daily, date_column, value_col_to_forecast, fill_strategy)
            daily = daily.rename(columns={date_column: 'ds', value_col_to_forecast: 'y'})

            if len(daily) < 2: continue

            train_df, valid_df = train_valid_slice(daily, 'ds', train_days, valid_days)
            if train_df.empty: continue

            try:
                seasonality_ok = auto_seasonality and (len(train_df) >= 30)
                model, fcst = fit_and_forecast_prophet(train_df, len(valid_df), horizon_days, seasonality_ok)
                
                forecasts_for_group[value_col_to_forecast] = {
                    'daily': daily, 'fcst': fcst, 'train_df': train_df, 'valid_df': valid_df
                }
            except Exception as e:
                st.warning(f"Could not generate forecast for metric '{value_col_to_forecast}'. Error: {e}")

        if forecasts_for_group:
            fig, metrics_df = plotly_multi_target_forecast(group_name, forecasts_for_group, rounding_rule)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Validation Metrics:")
            st.dataframe(metrics_df, use_container_width=True)
            
            for metric, data in forecasts_for_group.items():
                fcst_export = data['fcst'].copy()
                
                # ***** THIS IS THE CORRECTED LINE *****
                fcst_export['forecast_qty'] = enforce_nonneg_int(fcst_export['yhat'], rounding_rule)
                
                fcst_export = fcst_export[['ds', 'forecast_qty']]
                
                actuals_export = data['daily'].rename(columns={'y': 'actual_qty'})
                
                final_tbl = pd.merge(fcst_export, actuals_export, on='ds', how='left')
                final_tbl['target_metric'] = metric
                for col in group_cols:
                    final_tbl[col] = row[col]
                all_results_for_csv.append(final_tbl)
        else:
            st.warning("No data found for any selected metrics in this group.")
        
        st.markdown("---")

    if all_results_for_csv:
        st.header("Download All Forecasts")
        all_df = pd.concat(all_results_for_csv, ignore_index=True)
        all_df['actual_qty'] = all_df['actual_qty'].fillna(-1).astype(int).replace(-1, pd.NA)
        
        id_cols = ['target_metric'] + group_cols
        time_cols = ['ds', 'actual_qty', 'forecast_qty']
        final_cols = id_cols + [c for c in time_cols if c in all_df.columns]
        all_df = all_df[final_cols]
        
        st.dataframe(all_df)
        csv_bytes = all_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download All Forecasts as CSV", data=csv_bytes, file_name="multi_target_forecasts.csv", mime="text/csv")