import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

st.set_page_config(page_title="Multi-Outlet Quantity Forecasting", layout="wide")
st.title("Multi-Outlet Quantity Forecasting")

# ------------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------------

def enforce_nonneg_int(series: pd.Series, how: str = "round") -> pd.Series:
    """Return a non-negative Series from numeric input. If 'none', returns float.

    Parameters
    ----------
    series : pd.Series
        Numeric values.
    how : {"round","floor","ceil", "none"}
        Rounding rule applied before clipping. If "none", no rounding is done.
    """
    # Always clip at 0
    vals = np.clip(series, 0, None)

    if how == "floor":
        return pd.Series(np.floor(vals), index=series.index).astype(int)
    elif how == "ceil":
        return pd.Series(np.ceil(vals), index=series.index).astype(int)
    elif how == "round":
        return pd.Series(np.round(vals), index=series.index).astype(int)
    
    # If 'none', just return the clipped float values
    return pd.Series(vals, index=series.index).astype(float)


def safe_parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Safely parse date column and handle errors."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out = out.sort_values(date_col)
    return out


def aggregate_to_daily(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Aggregate data to daily sums."""
    # Ensure date only (no time) and aggregate duplicates by sum (quantity sold)
    temp = df.copy()
    temp[date_col] = temp[date_col].dt.normalize()
    daily = temp.groupby(date_col, as_index=False)[value_col].sum()
    return daily


def reindex_missing_dates(daily: pd.DataFrame, date_col: str, value_col: str, fill_strategy: str = "ffill") -> pd.DataFrame:
    """Ensure a continuous daily date index, filling missing values."""
    if daily.empty:
        return daily
    idx = pd.date_range(daily[date_col].min(), daily[date_col].max(), freq="D")
    daily = daily.set_index(date_col).reindex(idx)
    if fill_strategy == "zero":
        daily[value_col] = daily[value_col].fillna(0)
    elif fill_strategy == "ffill":
        daily[value_col] = daily[value_col].ffill().fillna(0)
    else:  # nan
        pass
    daily.index.name = date_col
    daily = daily.reset_index()
    return daily


def train_valid_slice(daily: pd.DataFrame, date_col: str, train_days: int, valid_days: int):
    """Slice data into training and validation sets from the end of the history."""
    daily = daily.sort_values(date_col)
    n = len(daily)
    if n == 0:
        return daily.iloc[0:0], daily.iloc[0:0]

    # Clip requested windows to available length
    train_days = int(min(train_days, n))
    valid_days = int(min(valid_days, n - train_days)) if n - train_days > 0 else 0

    train_start_idx = n - (train_days + valid_days)
    if train_start_idx < 0:
        train_start_idx = 0
    train_end_idx = train_start_idx + train_days
    valid_end_idx = train_end_idx + valid_days

    train_df = daily.iloc[train_start_idx:train_end_idx].copy()
    valid_df = daily.iloc[train_end_idx:valid_end_idx].copy()
    return train_df, valid_df


def build_prophet_model(train_df: pd.DataFrame, seasonality_ok: bool) -> Prophet:
    """Build a Prophet model with optional seasonality."""
    # Minimal model for tiny datasets: disable seasonality if too short
    if seasonality_ok:
        m = Prophet(growth="linear")
    else:
        m = Prophet(growth="linear", yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    return m


def fit_and_forecast_prophet(train_df: pd.DataFrame, valid_days: int, horizon_days: int, seasonality_ok: bool):
    """Fit a Prophet model and generate a forecast."""
    model = build_prophet_model(train_df, seasonality_ok=seasonality_ok)
    model.fit(train_df)
    # Forecast through validation + future horizon
    periods = valid_days + horizon_days
    future = model.make_future_dataframe(periods=periods, freq="D", include_history=True)
    fcst = model.predict(future)
    return model, fcst


def calc_metrics(actual: pd.Series, pred: pd.Series):
    """Calculate evaluation metrics between actual and predicted values."""
    if len(actual) == 0 or len(pred) == 0:
        return {k: np.nan for k in ["MAE","RMSE","sMAPE","MAPE"]}

    # FIX: Reset indices to ensure alignment for calculations.
    # This prevents the 'Unalignable boolean Series' error.
    actual = actual.reset_index(drop=True)
    pred = pred.reset_index(drop=True)

    err = actual - pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    
    # Safe MAPE: avoid div0 -> use where actual>0
    mask = actual > 0
    if mask.any():
        mape = np.mean(np.abs(err[mask] / actual[mask])) * 100
    else:
        mape = np.nan
        
    # sMAPE
    denom = (np.abs(actual) + np.abs(pred))
    # Avoid division by zero for sMAPE
    smape_vals = np.where(denom == 0, 0, 2 * np.abs(err) / denom)
    smape = np.mean(smape_vals) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}


# ------------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------------

def plot_train_valid_future(outlet_name: str,
                            daily: pd.DataFrame,
                            train_df: pd.DataFrame,
                            valid_df: pd.DataFrame,
                            fcst: pd.DataFrame,
                            rounding: str,
                            show_int: bool = True):
    """Create a custom chart showing actuals, fitted train, validation preds, and future preds."""
    fcst = fcst.copy()
    fcst['ds'] = pd.to_datetime(fcst['ds'])

    # Determine boundaries
    last_train_date = train_df['ds'].max() if not train_df.empty else None
    last_valid_date = valid_df['ds'].max() if not valid_df.empty else last_train_date

    # Extract yhat aligned to history
    fcst_hist = fcst[fcst['ds'] <= daily['ds'].max()].copy()
    fcst_hist = fcst_hist.merge(daily[['ds','y']], on='ds', how='right', suffixes=("","_drop"))

    # Train fit preds
    train_mask = fcst_hist['ds'] <= last_train_date if last_train_date is not None else np.array([False]*len(fcst_hist))
    train_fit = fcst_hist.loc[train_mask, ['ds','yhat','yhat_lower','yhat_upper']].copy()

    # Validation preds
    valid_mask = (fcst_hist['ds'] > last_train_date) & (fcst_hist['ds'] <= last_valid_date) if last_valid_date is not None and last_train_date is not None else np.array([False]*len(fcst_hist))
    valid_pred = fcst_hist.loc[valid_mask, ['ds','yhat','yhat_lower','yhat_upper']].copy()

    # Future preds (after full history)
    fcst_fut = fcst[fcst['ds'] > daily['ds'].max()].copy()

    # Enforce integer predictions for display
    train_fit['yhat_display'] = enforce_nonneg_int(train_fit['yhat'], how=rounding)
    valid_pred['yhat_display'] = enforce_nonneg_int(valid_pred['yhat'], how=rounding)
    fcst_fut['yhat_display'] = enforce_nonneg_int(fcst_fut['yhat'], how=rounding)

    # Metrics on validation window using integer preds
    daily_valid = daily[daily['ds'].isin(valid_pred['ds'])]
    
    # Ensure we only calculate metrics if there is validation data
    if not daily_valid.empty and not valid_pred.empty:
        # Align predictions with actuals for metrics calculation
        preds_for_metrics = valid_pred.set_index('ds').loc[daily_valid['ds'],'yhat_display']
        metrics = calc_metrics(daily_valid['y'], preds_for_metrics)
    else:
        metrics = {k: np.nan for k in ["MAE","RMSE","sMAPE","MAPE"]}


    fig, ax = plt.subplots(figsize=(10,4))

    # Actuals (blue)
    ax.plot(daily['ds'], daily['y'], marker='o', linestyle='-', label='Actual', color='tab:blue', zorder=5)

    # Train fit (orange dashed)
    if not train_fit.empty:
        ax.plot(train_fit['ds'], train_fit['yhat_display'], linestyle='--', label='Train Fit', color='tab:orange', zorder=10)

    # Validation preds (green solid)
    if not valid_pred.empty:
        ax.plot(valid_pred['ds'], valid_pred['yhat_display'], linestyle='-', marker='s', label='Validation Pred', color='tab:green', zorder=10)

    # Future forecast (purple X)
    if not fcst_fut.empty:
        ax.plot(fcst_fut['ds'], fcst_fut['yhat_display'], linestyle='-', marker='x', label='Future Forecast', color='tab:purple', zorder=10)

    # Uncertainty band across validation + future
    if show_int:
        int_df = pd.concat([valid_pred[['ds','yhat_lower','yhat_upper']], fcst_fut[['ds','yhat_lower','yhat_upper']]])
        if not int_df.empty:
            # Enforce non-negativity on uncertainty bounds for plotting
            yhat_lower_display = enforce_nonneg_int(int_df['yhat_lower'], how='floor')
            yhat_upper_display = enforce_nonneg_int(int_df['yhat_upper'], how='ceil')
            ax.fill_between(int_df['ds'], yhat_lower_display, yhat_upper_display, color='tab:blue', alpha=0.1, label='Uncertainty', zorder=1)

    # Vertical lines marking boundaries
    if last_train_date is not None:
        ax.axvline(last_train_date + timedelta(hours=12), color='gray', linestyle=':', linewidth=1)
        ax.text(last_train_date, ax.get_ylim()[1], ' Train End', rotation=90, va='top', ha='right', fontsize=8, color='gray')
    if last_valid_date is not None and last_valid_date != last_train_date:
        ax.axvline(last_valid_date + timedelta(hours=12), color='gray', linestyle=':', linewidth=1)
        ax.text(last_valid_date, ax.get_ylim()[1], ' Valid End', rotation=90, va='top', ha='right', fontsize=8, color='gray')

    # Integer y-axis only if rounding is enabled
    if rounding != 'none':
        ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity Sold")
    ax.set_title(f"Forecast for: {outlet_name}")
    ax.legend(loc='best')
    ax.grid(True, which='major', axis='y', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()

    # Metrics annotation
    txt = (f"Validation Metrics:\n"
           f"MAE: {metrics.get('MAE', 'N/A'):.2f}\n"
           f"RMSE: {metrics.get('RMSE', 'N/A'):.2f}\n"
           f"sMAPE: {metrics.get('sMAPE', 'N/A'):.1f}%")
    ax.text(1.02, 0.5, txt, transform=ax.transAxes, va='center', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for metrics box

    return fig, metrics, train_fit, valid_pred, fcst_fut


# ------------------------------------------------------------------------------------
# Sidebar / Controls
# ------------------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file using the sidebar to begin.")
    st.stop()

# This part of the script runs only if a file is uploaded
raw = pd.read_csv(uploaded_file)
st.write("Uploaded Data Preview:")
st.dataframe(raw.head())

with st.sidebar:
    st.header("2. Configure Columns")
    # Try to auto-detect columns
    column_options = raw.columns.tolist()
    outlet_col_index = 0
    date_col_index = 1
    qty_col_index = 2
    for i, col in enumerate(column_options):
        col_lower = col.lower()
        if 'outlet' in col_lower: outlet_col_index = i
        if 'date' in col_lower: date_col_index = i
        if 'qty' in col_lower or 'quant' in col_lower: qty_col_index = i

    outlet_column = st.selectbox("Select the outlet column:", column_options, index=outlet_col_index)
    date_column = st.selectbox("Select the date column:", column_options, index=date_col_index)
    value_column = st.selectbox("Select the quantity column:", column_options, index=qty_col_index)

    st.header("3. Set Forecast Windows")
    train_days = st.number_input("Number of TRAIN days", min_value=2, max_value=1000, value=7, help="Number of days at the end of the history to use for model training.")
    valid_days = st.number_input("Number of VALIDATION days", min_value=0, max_value=1000, value=2, help="Number of days immediately after the training period to test the model's accuracy.")
    horizon_days = st.number_input("Number of FUTURE days to forecast", min_value=1, max_value=365, value=7, help="Number of days to forecast into the future, beyond all known data.")

    st.header("4. Model & Display Options")
    rounding_rule = st.selectbox("Rounding rule for forecasts", ["round","floor","ceil", "none"], index=0, help="'none' will keep forecast values as decimals.")
    fill_strategy = st.selectbox("Missing date fill strategy", ["ffill","zero","nan"], index=0, help="'ffill' uses the last known value, 'zero' fills with 0.")
    auto_seasonality = st.checkbox("Enable Prophet seasonality if train history >= 30 days", value=True)
    show_int = st.checkbox("Show prediction uncertainty intervals", value=True)

    run_button = st.button("Run Forecasts", type="primary")


if run_button:
    outlets = raw[outlet_column].dropna().unique()
    st.write(f"Found {len(outlets)} unique outlets. Generating forecasts...")

    all_results = []

    for outlet in outlets:
        with st.container():
            st.subheader(f"Forecast for: {outlet}")

            df_o = raw[raw[outlet_column] == outlet][[date_column, value_column]].copy()
            if df_o.empty:
                st.warning("No data for this outlet.")
                continue

            # --- Data Cleaning and Preparation ---
            df_o = safe_parse_dates(df_o, date_column)
            if df_o.empty:
                st.warning("All dates failed to parse; skipping.")
                continue
            daily = aggregate_to_daily(df_o, date_column, value_column)
            daily = reindex_missing_dates(daily, date_column, value_column, fill_strategy=fill_strategy)
            daily = daily.rename(columns={date_column: 'ds', value_column: 'y'})

            if len(daily) < train_days + valid_days:
                st.warning(f"Not enough data for the requested train/validation window ({train_days}+{valid_days} days). Adjusting to available data.")
            
            if len(daily) < 2:
                st.warning("Not enough data to create a forecast; skipping.")
                continue

            # --- Train/Validation Split ---
            train_df, valid_df = train_valid_slice(daily, 'ds', train_days, valid_days)
            if train_df.empty:
                st.warning("Training window is empty after slicing; skipping.")
                continue

            # --- Model Fitting ---
            seasonality_ok = auto_seasonality and (len(train_df) >= 30)
            try:
                model, fcst = fit_and_forecast_prophet(train_df, valid_days=len(valid_df), horizon_days=horizon_days, seasonality_ok=seasonality_ok)
            except Exception as e:
                st.error(f"Prophet model failed to fit for {outlet}: {e}")
                continue

            # --- Plotting and Metrics ---
            fig, metrics, train_fit, valid_pred, fcst_fut = plot_train_valid_future(
                outlet_name=outlet,
                daily=daily,
                train_df=train_df,
                valid_df=valid_df,
                fcst=fcst,
                rounding=rounding_rule,
                show_int=show_int
            )
            st.pyplot(fig, clear_figure=True)

            # --- Prepare Results for Download ---
            actuals_tbl = daily.rename(columns={'y': 'actual_qty'})
            
            train_pred_tbl = train_fit[['ds','yhat_display']].rename(columns={'yhat_display':'forecast_qty'})
            train_pred_tbl['segment'] = 'train_fit'

            valid_pred_tbl = valid_pred[['ds','yhat_display']].rename(columns={'yhat_display':'forecast_qty'})
            valid_pred_tbl['segment'] = 'validation_pred'

            fut_tbl = fcst_fut[['ds','yhat_display']].rename(columns={'yhat_display':'forecast_qty'})
            fut_tbl['segment'] = 'future_forecast'

            res_tbl = pd.concat([train_pred_tbl, valid_pred_tbl, fut_tbl], ignore_index=True)
            
            # Merge with actuals
            final_tbl = pd.merge(res_tbl, actuals_tbl, on='ds', how='left')
            final_tbl.insert(0, 'outlet', outlet)
            all_results.append(final_tbl)

    if all_results:
        st.header("Download All Forecasts")
        all_df = pd.concat(all_results, ignore_index=True)
        # Final cleanup on dtypes
        all_df['forecast_qty'] = enforce_nonneg_int(all_df['forecast_qty'], how=rounding_rule)
        all_df['actual_qty'] = all_df['actual_qty'].fillna(-1).astype(int).replace(-1, pd.NA) # Preserve NaNs for future dates
        
        st.dataframe(all_df)
        
        csv_bytes = all_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecasts as CSV", data=csv_bytes, file_name="multi_outlet_forecasts.csv", mime="text/csv")

