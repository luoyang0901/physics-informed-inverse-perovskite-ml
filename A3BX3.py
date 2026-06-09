import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
except Exception as exc:
    raise ImportError("xgboost is required for this script.") from exc

warnings.filterwarnings("ignore")

TARGETS = ["Voc", "Jsc", "FF", "PCE"]
REPEAT_SEEDS = list(range(10))
MAIN_RANDOM_STATE = 42
TEST_SIZE = 0.2
PCE_CONSISTENCY_ABS_TOL = 0.5
CLEAN_BY_PCE_CONSISTENCY = True
OUTPUT_DIR = "csv_outputs"

BASE_FEATURES = [
    "Abs_Thickness_nm", "Abs_Eg_eV", "Abs_Chi_eV", "Abs_Eps_r", "Abs_mu_n", "Abs_mu_p",
    "Abs_log10_NA", "Abs_log10_Nt", "ETL_Thickness_nm", "ETL_Eg_eV", "ETL_Chi_eV",
    "ETL_Eps_r", "ETL_mu_n", "ETL_mu_p", "ETL_log10_ND", "ETL_log10_Nt",
    "HTL_Thickness_nm", "HTL_Eg_eV", "HTL_Chi_eV", "HTL_Eps_r", "HTL_mu_n", "HTL_mu_p",
    "HTL_log10_NA", "HTL_log10_Nt"
]

DERIVED_FEATURES = [
    "Abs_mu_ratio_log", "ETL_mu_ratio_log", "HTL_mu_ratio_log", "ETL_HTL_transport_balance_log",
    "ETL_Abs_Chi_offset", "Abs_HTL_Chi_offset", "ETL_Abs_Eg_offset", "HTL_Abs_Eg_offset",
    "Total_log10_Nt", "Abs_Nt_minus_ETL_Nt", "Abs_Nt_minus_HTL_Nt", "ETL_Nt_minus_HTL_Nt",
    "Abs_thickness_x_Abs_Nt", "ETL_thickness_x_ETL_Nt", "HTL_thickness_x_HTL_Nt",
    "Abs_ETL_Chi_mismatch_abs", "Abs_HTL_Chi_mismatch_abs", "Transport_mu_min_log"
]

STRUCTURE_MAP = {
    "Structure 1": {"Absorber": "Sr3SbI3", "Device_Structure": "FTO/CdS/Sr3SbI3/SrCu2O2/Au"},
    "Structure 2": {"Absorber": "Ca3NBr3", "Device_Structure": "FTO/PCBM/Ca3NBr3/CBTS/Au"},
    "Structure 3": {"Absorber": "Sr3AsBr3", "Device_Structure": "FTO/PCBM/Sr3AsBr3/CBTS/Au"},
    "Structure 4": {"Absorber": "Sr3NCl3", "Device_Structure": "FTO/TiO2/Sr3NCl3/Cu2O/Au"},
    "Structure 5": {"Absorber": "Ba3NCl3", "Device_Structure": "FTO/IGZO/Ba3NCl3/V2O5/Au"},
    "Structure 6": {"Absorber": "Ba3PCl3", "Device_Structure": "FTO/WS2/Ba3PCl3/CBTS/Au"},
    "Structure 7": {"Absorber": "Ca3NF3", "Device_Structure": "FTO/WS2/Ca3NF3/CuI/Au"},
    "Structure 8": {"Absorber": "Sr3PI3", "Device_Structure": "FTO/SnS2/Sr3PI3/Cu2O/Au"},
    "Structure 9": {"Absorber": "Ca3NI3", "Device_Structure": "FTO/C60/Ca3NI3/CIGSe/Au"},
    "Structure 10": {"Absorber": "Ca3NBr3", "Device_Structure": "FTO/C60/Ca3NBr3/CIS/Au"},
    "Structure 11": {"Absorber": "Ca3NCl3", "Device_Structure": "FTO/CdS/Ca3NCl3/CISe/Au"},
    "Structure 12": {"Absorber": "Ca3NI3", "Device_Structure": "FTO/C60/Ca3NI3/Si/Au"},
    "Structure 13": {"Absorber": "Ca3SbI3", "Device_Structure": "FTO/LBSO/Ca3SbI3/CISSe/Au"},
    "Structure 14": {"Absorber": "Mg3AsBr3", "Device_Structure": "FTO/STO/Mg3AsBr3/CTSe/Au"},
    "Structure 15": {"Absorber": "Sr3SbI3", "Device_Structure": "FTO/ZnO/Sr3SbI3/Spiro-MeOTAD/Au"},
    "Structure 16": {"Absorber": "Ca3AsI3", "Device_Structure": "FTO/ZnO/Ca3AsI3/CuSbS2/Au"},
    "Structure 17": {"Absorber": "Ba3SbI3", "Device_Structure": "FTO/SnS2/Ba3SbI3/CuSCN/Au"},
    "Structure 18": {"Absorber": "Ba3AsBi3", "Device_Structure": "FTO/ZnS/Ba3AsBr3/CZTS/Au"},
    "Structure 19": {"Absorber": "Ba3PBr3", "Device_Structure": "FTO/SnS2/Ba3PBr3/MoO3/Au"},
    "Structure 20": {"Absorber": "Ba3SbBr3", "Device_Structure": "FTO/SnS2/Ba3SbBr3/CuI/Au"},
    "Structure 21": {"Absorber": "Sr3SbI3", "Device_Structure": "FTO/SnS2/Sr3SbI3/V2O5/Au"},
    "Structure 22": {"Absorber": "Sr3SbI3", "Device_Structure": "FTO/CdS/Sr3SbI3/CuSCN/Au"},
    "Structure 23": {"Absorber": "Sr3PBr3", "Device_Structure": "FTO/In2P3/Sr3PBr3/V2O5/Au"},
    "Structure 24": {"Absorber": "Sr3PBr3", "Device_Structure": "FTO/SnS2/Sr3PBr3/NiO/Au"},
    "Structure 25": {"Absorber": "Sr3PCl3", "Device_Structure": "FTO/MZO/Sr3PCl3/Zn2P3/Au"},
    "Structure 26": {"Absorber": "Ca3AsBr3", "Device_Structure": "FTO/MZO/Ca3AsBr3/BiI3/Au"},
    "Structure 27": {"Absorber": "Ca3PBr3", "Device_Structure": "FTO/SnO2/Ca3PBr3/WSe2/Au"},
    "Structure 28": {"Absorber": "Sr3AsI3", "Device_Structure": "FTO/ZnS/Sr3AsI3/CuS/Au"},
    "Structure 29": {"Absorber": "Sr3BiI3", "Device_Structure": "FTO/ZnSe/Sr3BiI3/PTAA/Au"},
    "Structure 30": {"Absorber": "Ca3BiCl3", "Device_Structure": "FTO/PCBM/Ca3BiCl3/CuS/Au"}
}


def read_csv_auto(path):
    last_error = None
    for encoding in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except Exception as exc:
            last_error = exc
    raise last_error


def find_dataset_file():
    for name in ["dataset2.csv", "dataset.csv"]:
        if os.path.exists(name):
            return name
    raise FileNotFoundError("dataset2.csv or dataset.csv was not found in the working directory.")


def detect_structure_column(df):
    if "Name" in df.columns:
        return "Name"
    if "Unnamed: 0" in df.columns:
        return "Unnamed: 0"
    return None


def clean_numeric_columns(df, columns):
    df = df.copy()
    for column in columns:
        df[column] = df[column].astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def add_pce_consistency_columns(df):
    df = df.copy()
    df["PCE_calc_from_Voc_Jsc_FF"] = df["Voc"] * df["Jsc"] * df["FF"] / 100.0
    df["PCE_consistency_abs_error"] = np.abs(df["PCE"] - df["PCE_calc_from_Voc_Jsc_FF"])
    df["PCE_consistency_rel_error"] = df["PCE_consistency_abs_error"] / np.maximum(np.abs(df["PCE"]), 1e-12)
    df["PCE_consistency_is_bad"] = df["PCE_consistency_abs_error"] > PCE_CONSISTENCY_ABS_TOL
    return df


def add_derived_features(df):
    df = df.copy()
    eps = 1e-30
    df["Abs_mu_ratio_log"] = np.log10((df["Abs_mu_n"] + eps) / (df["Abs_mu_p"] + eps))
    df["ETL_mu_ratio_log"] = np.log10((df["ETL_mu_n"] + eps) / (df["ETL_mu_p"] + eps))
    df["HTL_mu_ratio_log"] = np.log10((df["HTL_mu_n"] + eps) / (df["HTL_mu_p"] + eps))
    df["ETL_HTL_transport_balance_log"] = np.log10((df["ETL_mu_n"] + eps) / (df["HTL_mu_p"] + eps))
    df["ETL_Abs_Chi_offset"] = df["ETL_Chi_eV"] - df["Abs_Chi_eV"]
    df["Abs_HTL_Chi_offset"] = df["Abs_Chi_eV"] - df["HTL_Chi_eV"]
    df["ETL_Abs_Eg_offset"] = df["ETL_Eg_eV"] - df["Abs_Eg_eV"]
    df["HTL_Abs_Eg_offset"] = df["HTL_Eg_eV"] - df["Abs_Eg_eV"]
    df["Total_log10_Nt"] = df["Abs_log10_Nt"] + df["ETL_log10_Nt"] + df["HTL_log10_Nt"]
    df["Abs_Nt_minus_ETL_Nt"] = df["Abs_log10_Nt"] - df["ETL_log10_Nt"]
    df["Abs_Nt_minus_HTL_Nt"] = df["Abs_log10_Nt"] - df["HTL_log10_Nt"]
    df["ETL_Nt_minus_HTL_Nt"] = df["ETL_log10_Nt"] - df["HTL_log10_Nt"]
    df["Abs_thickness_x_Abs_Nt"] = df["Abs_Thickness_nm"] * df["Abs_log10_Nt"]
    df["ETL_thickness_x_ETL_Nt"] = df["ETL_Thickness_nm"] * df["ETL_log10_Nt"]
    df["HTL_thickness_x_HTL_Nt"] = df["HTL_Thickness_nm"] * df["HTL_log10_Nt"]
    df["Abs_ETL_Chi_mismatch_abs"] = np.abs(df["ETL_Abs_Chi_offset"])
    df["Abs_HTL_Chi_mismatch_abs"] = np.abs(df["Abs_HTL_Chi_offset"])
    df["Transport_mu_min_log"] = np.log10(np.minimum(df["ETL_mu_n"], df["HTL_mu_p"]) + eps)
    return df


def prepare_dataset(path):
    df = read_csv_auto(path)
    structure_column = detect_structure_column(df)
    if structure_column is not None and structure_column != "Name":
        df = df.rename(columns={structure_column: "Name"})
    required_columns = BASE_FEATURES + TARGETS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))
    df["Original_Row_Index"] = np.arange(len(df))
    numeric_columns = BASE_FEATURES + TARGETS
    df = clean_numeric_columns(df, numeric_columns)
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()
    df = df.dropna(subset=numeric_columns).reset_index(drop=True)
    df = add_pce_consistency_columns(df)
    pce_check = df[["Original_Row_Index"] + (["Name"] if "Name" in df.columns else []) + TARGETS + [
        "PCE_calc_from_Voc_Jsc_FF", "PCE_consistency_abs_error", "PCE_consistency_rel_error", "PCE_consistency_is_bad"
    ]].copy()
    if CLEAN_BY_PCE_CONSISTENCY:
        df = df.loc[~df["PCE_consistency_is_bad"].astype(bool)].copy().reset_index(drop=True)
    df = add_derived_features(df)
    return df, pce_check


def make_bins(y):
    y = pd.Series(np.asarray(y, dtype=float))
    bins_count = min(10, max(2, int(np.sqrt(len(y)))))
    try:
        labels = pd.qcut(y, q=bins_count, duplicates="drop").cat.codes.to_numpy()
        if len(np.unique(labels)) > 1 and np.min(np.bincount(labels[labels >= 0])) >= 2:
            return labels
    except Exception:
        return None
    return None


def train_test_indices(y, random_state):
    indices = np.arange(len(y))
    bins = make_bins(y)
    if bins is None:
        return train_test_split(indices, test_size=TEST_SIZE, random_state=random_state)
    return train_test_split(indices, test_size=TEST_SIZE, random_state=random_state, stratify=bins)


def iqr_filter_train_only(X, y, target):
    if target != "Voc":
        return X, y
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    mask = (y >= lower) & (y <= upper)
    return X[mask], y[mask]


def rmse_value(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae_value(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def metric_row(target, model, y_true, y_pred, extra=None):
    row = {
        "Target": target,
        "Model": model,
        "N_Test": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "RMSE": rmse_value(y_true, y_pred),
        "MAE": mae_value(y_true, y_pred)
    }
    if extra:
        row.update(extra)
    return row


def build_models(random_state):
    models = {
        "RF": RandomForestRegressor(random_state=random_state, n_jobs=1),
        "XGB": XGBRegressor(random_state=random_state, n_jobs=1, objective="reg:squarederror", verbosity=0),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(random_state=random_state, early_stopping=True, validation_fraction=0.1, max_iter=1200))
        ]),
        "DeepMLP": Pipeline([
            ("scaler", StandardScaler()),
            ("deepmlp", MLPRegressor(random_state=random_state, early_stopping=True, validation_fraction=0.1, max_iter=1500))
        ])
    }
    return models


def build_param_grids():
    return {
        "RF": {
            "n_estimators": [400, 800],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt"]
        },
        "XGB": {
            "n_estimators": [400, 800],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_lambda": [1.0, 3.0]
        },
        "MLP": {
            "mlp__hidden_layer_sizes": [(64, 64), (128, 64)],
            "mlp__activation": ["relu"],
            "mlp__alpha": [1e-4, 1e-3],
            "mlp__learning_rate_init": [1e-3, 5e-4]
        },
        "DeepMLP": {
            "deepmlp__hidden_layer_sizes": [(128, 128, 64), (128, 128, 64, 32)],
            "deepmlp__activation": ["relu"],
            "deepmlp__alpha": [1e-3, 1e-2],
            "deepmlp__learning_rate_init": [5e-4, 1e-4]
        }
    }


def build_xgb_fixed(random_state):
    return XGBRegressor(
        random_state=random_state,
        n_jobs=1,
        objective="reg:squarederror",
        verbosity=0,
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0
    )


def fit_with_grid(model, grid, X_train, y_train, random_state):
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    search = GridSearchCV(model, grid, scoring="r2", cv=cv, n_jobs=1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def evaluate_random_split(df, features, random_state):
    models = build_models(random_state)
    grids = build_param_grids()
    rows = []
    predictions = []
    best_params_rows = []
    for target in TARGETS:
        X = df[features].to_numpy(dtype=float)
        y = df[target].to_numpy(dtype=float)
        train_index, test_index = train_test_indices(y, random_state)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        X_train, y_train = iqr_filter_train_only(X_train, y_train, target)
        for model_name, base_model in models.items():
            fitted_model, best_params, cv_score = fit_with_grid(base_model, grids[model_name], X_train, y_train, random_state)
            y_pred = fitted_model.predict(X_test)
            rows.append(metric_row(target, model_name, y_test, y_pred, {
                "Random_State": random_state,
                "R2_CV": cv_score,
                "R2_Independent": float(r2_score(y_test, y_pred)),
                "RMSE_Independent": rmse_value(y_test, y_pred),
                "MAE_Independent": mae_value(y_test, y_pred)
            }))
            best_params_rows.append({
                "Target": target,
                "Model": model_name,
                "Random_State": random_state,
                "Best_Params": str(best_params),
                "R2_CV": cv_score
            })
            meta_columns = ["Original_Row_Index"] + (["Name"] if "Name" in df.columns else [])
            meta = df.iloc[test_index][meta_columns].reset_index(drop=True)
            pred_df = meta.copy()
            pred_df["Target"] = target
            pred_df["Model"] = model_name
            pred_df["Random_State"] = random_state
            pred_df["True_Value"] = y_test
            pred_df["Predicted_Value"] = y_pred
            pred_df["Residual"] = y_pred - y_test
            predictions.append(pred_df)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), pd.DataFrame(best_params_rows)


def summarize_repeat(metrics):
    rows = []
    for (target, model), group in metrics.groupby(["Target", "Model"]):
        rows.append({
            "Target": target,
            "Model": model,
            "N_Seeds": int(group["Random_State"].nunique()),
            "R2_Mean": float(group["R2_Independent"].mean()),
            "R2_Std": float(group["R2_Independent"].std(ddof=1)),
            "RMSE_Mean": float(group["RMSE_Independent"].mean()),
            "RMSE_Std": float(group["RMSE_Independent"].std(ddof=1)),
            "MAE_Mean": float(group["MAE_Independent"].mean()),
            "MAE_Std": float(group["MAE_Independent"].std(ddof=1)),
            "R2_Mean_Std_Text": f"{group['R2_Independent'].mean():.4f} ± {group['R2_Independent'].std(ddof=1):.4f}",
            "RMSE_Mean_Std_Text": f"{group['RMSE_Independent'].mean():.4f} ± {group['RMSE_Independent'].std(ddof=1):.4f}",
            "MAE_Mean_Std_Text": f"{group['MAE_Independent'].mean():.4f} ± {group['MAE_Independent'].std(ddof=1):.4f}"
        })
    return pd.DataFrame(rows).sort_values(["Target", "R2_Mean"], ascending=[True, False]).reset_index(drop=True)


def table_best_models(metrics):
    rows = []
    for target, group in metrics.groupby("Target"):
        best = group.sort_values("R2_Independent", ascending=False).iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows).reset_index(drop=True)


def run_repeat_seed_evaluation(df, features):
    metric_parts = []
    param_parts = []
    for seed in REPEAT_SEEDS:
        metrics, _, params = evaluate_random_split(df, features, seed)
        metric_parts.append(metrics)
        param_parts.append(params)
    all_metrics = pd.concat(metric_parts, ignore_index=True)
    all_params = pd.concat(param_parts, ignore_index=True)
    return all_metrics, summarize_repeat(all_metrics), all_params


def run_structure_lomo(df, features):
    if "Name" not in df.columns:
        raise ValueError("The dataset must contain a Name column or a structure column that can be renamed to Name.")
    rows = []
    predictions = []
    structures = sorted(df["Name"].dropna().unique(), key=lambda x: int(str(x).split()[-1]) if str(x).split()[-1].isdigit() else str(x))
    for structure in structures:
        test_mask = df["Name"].astype(str) == str(structure)
        train_mask = ~test_mask
        if test_mask.sum() < 2 or train_mask.sum() < 5:
            continue
        structure_info = STRUCTURE_MAP.get(str(structure), {})
        for target in TARGETS:
            X_train = df.loc[train_mask, features].to_numpy(dtype=float)
            y_train = df.loc[train_mask, target].to_numpy(dtype=float)
            X_test = df.loc[test_mask, features].to_numpy(dtype=float)
            y_test = df.loc[test_mask, target].to_numpy(dtype=float)
            X_train, y_train = iqr_filter_train_only(X_train, y_train, target)
            model = build_xgb_fixed(MAIN_RANDOM_STATE)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rows.append(metric_row(target, "XGB", y_test, y_pred, {
                "Heldout_Structure": structure,
                "Absorber": structure_info.get("Absorber", ""),
                "Device_Structure": structure_info.get("Device_Structure", ""),
                "N_Train": int(len(y_train))
            }))
            meta = df.loc[test_mask, ["Original_Row_Index", "Name"]].reset_index(drop=True)
            pred_df = meta.copy()
            pred_df["Heldout_Structure"] = structure
            pred_df["Target"] = target
            pred_df["Model"] = "XGB"
            pred_df["True_Value"] = y_test
            pred_df["Predicted_Value"] = y_pred
            pred_df["Residual"] = y_pred - y_test
            predictions.append(pred_df)
    metrics = pd.DataFrame(rows)
    preds = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    summary_rows = []
    for target, group in metrics.groupby("Target"):
        summary_rows.append({
            "Target": target,
            "Number_of_Heldout_Structures": int(group["Heldout_Structure"].nunique()),
            "R2_Mean": float(group["R2"].mean()),
            "R2_Std": float(group["R2"].std(ddof=1)),
            "RMSE_Mean": float(group["RMSE"].mean()),
            "RMSE_Std": float(group["RMSE"].std(ddof=1)),
            "MAE_Mean": float(group["MAE"].mean()),
            "MAE_Std": float(group["MAE"].std(ddof=1)),
            "R2_Mean_Std_Text": f"{group['R2'].mean():.4f} ± {group['R2'].std(ddof=1):.4f}",
            "RMSE_Mean_Std_Text": f"{group['RMSE'].mean():.4f} ± {group['RMSE'].std(ddof=1):.4f}",
            "MAE_Mean_Std_Text": f"{group['MAE'].mean():.4f} ± {group['MAE'].std(ddof=1):.4f}"
        })
    summary = pd.DataFrame(summary_rows)
    best_structure = None
    best_table = pd.DataFrame()
    if not metrics.empty:
        pivot = metrics.pivot_table(index="Heldout_Structure", columns="Target", values="R2", aggfunc="mean")
        available_targets = [target for target in TARGETS if target in pivot.columns]
        pivot["Mean_R2"] = pivot[available_targets].mean(axis=1)
        positive_count = (pivot[available_targets] > 0).sum(axis=1)
        pivot["Positive_Target_Count"] = positive_count
        best_structure = pivot.sort_values(["Positive_Target_Count", "Mean_R2"], ascending=[False, False]).index[0]
        best_table = metrics.loc[metrics["Heldout_Structure"] == best_structure].copy().reset_index(drop=True)
    return metrics, preds, summary, best_table


def save_xgb_feature_importance(df, features):
    rows = []
    for target in TARGETS:
        X = df[features].to_numpy(dtype=float)
        y = df[target].to_numpy(dtype=float)
        X, y = iqr_filter_train_only(X, y, target)
        model = build_xgb_fixed(MAIN_RANDOM_STATE)
        model.fit(X, y)
        importances = np.asarray(model.feature_importances_, dtype=float)
        table = pd.DataFrame({"Target": target, "Feature": features, "Importance": importances})
        table = table.sort_values("Importance", ascending=False).reset_index(drop=True)
        table["Rank"] = np.arange(1, len(table) + 1)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def write_csv(df, filename):
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False, encoding="utf-8-sig")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset_path = find_dataset_file()
    df, pce_check = prepare_dataset(dataset_path)
    features = BASE_FEATURES + DERIVED_FEATURES
    write_csv(pce_check, "pce_consistency_check.csv")
    write_csv(df, "cleaned_dataset_with_derived_features.csv")
    derived_definition = pd.DataFrame({"Derived_Feature": DERIVED_FEATURES})
    write_csv(derived_definition, "derived_feature_names.csv")
    random_metrics, random_predictions, random_params = evaluate_random_split(df, features, MAIN_RANDOM_STATE)
    write_csv(random_metrics, "table_model_performance_random_split.csv")
    write_csv(random_predictions, "independent_test_predictions_random_split.csv")
    write_csv(random_params, "best_hyperparameters_random_split.csv")
    write_csv(table_best_models(random_metrics), "table_best_models_random_split.csv")
    repeat_metrics, repeat_summary, repeat_params = run_repeat_seed_evaluation(df, features)
    write_csv(repeat_metrics, "repeat_seed_metrics.csv")
    write_csv(repeat_summary, "table_repeat_seed_summary.csv")
    write_csv(repeat_params, "repeat_seed_best_hyperparameters.csv")
    lomo_metrics, lomo_predictions, lomo_summary, best_lomo = run_structure_lomo(df, features)
    write_csv(lomo_metrics, "structure_lomo_metrics.csv")
    write_csv(lomo_predictions, "structure_lomo_predictions.csv")
    write_csv(lomo_summary, "table_structure_lomo_summary.csv")
    write_csv(best_lomo, "table_representative_best_structure_lomo.csv")
    importance = save_xgb_feature_importance(df, features)
    write_csv(importance, "xgb_feature_importance_all_targets.csv")
    write_csv(importance.groupby("Target").head(10).reset_index(drop=True), "table_xgb_top10_feature_importance.csv")


if __name__ == "__main__":
    main()
