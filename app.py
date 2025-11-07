import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, auc, classification_report)
import base64
import matplotlib
matplotlib.use('Agg')

st.set_page_config(layout="wide", page_title="Insurance Policy Prediction Dashboard")

st.title("Insurance Policy Status — Dashboard & Modeling")

# Helper functions
def load_sample():
    return pd.read_csv("sample_insurance.csv")

def preprocess_df(df, drop_high_card=True, id_like_keywords=None):
    df = df.copy()
    if id_like_keywords is None:
        id_like_keywords = ["policy_no","policy no","policyno","id","name","no","number"]
    # Drop columns with >50% unique values (likely IDs) or explicit id-like
    n = len(df)
    drop_cols = []
    for c in df.columns:
        try:
            if df[c].nunique() > 0.5 * n and drop_high_card:
                drop_cols.append(c)
        except Exception:
            pass
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in id_like_keywords):
            drop_cols.append(c)
    drop_cols = sorted(set(drop_cols))
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    # Strip column names
    df.columns = [c.strip() for c in df.columns]
    return df, drop_cols

def build_pipeline(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    for c in num_cols.copy():
        if X[c].nunique() <= 10 and X[c].dtype.kind in 'iu':
            cat_cols.append(c); num_cols.remove(c)
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    preprocessor = ColumnTransformer([("num", numeric_transformer, num_cols),
                                      ("cat", categorical_transformer, cat_cols)], remainder='drop')
    return preprocessor, num_cols, cat_cols

def train_models(X, y, random_state=42, n_estimators=100):
    preprocessor, num_cols, cat_cols = build_pipeline(X)
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
        try:
            cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=1)
            cv_mean = cv_scores.mean()
        except Exception:
            cv_mean = np.nan
        pipe.fit(X, y)
        results[name] = {"pipeline": pipe, "cv_mean": float(np.round(cv_mean,4))}
    return results, num_cols, cat_cols

def compute_metrics(pipe, X_train, y_train, X_test, y_test):
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps['clf'], "predict_proba"):
        y_test_proba = pipe.predict_proba(X_test)
    else:
        y_test_proba = np.zeros((len(y_test), len(np.unique(y_test))))
        y_test_proba[np.arange(len(y_test)), y_test_pred] = 1.0
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    avg = "binary" if len(np.unique(y_test))==2 else "macro"
    precision = precision_score(y_test, y_test_pred, average=avg, zero_division=0)
    recall = recall_score(y_test, y_test_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average=avg, zero_division=0)
    try:
        if len(np.unique(y_test))==2:
            auc_val = roc_auc_score(y_test, y_test_proba[:,1])
        else:
            y_test_b = label_binarize(y_test, classes=np.arange(len(np.unique(y_test))))
            auc_val = roc_auc_score(y_test_b, y_test_proba, average="macro", multi_class="ovr")
    except Exception:
        auc_val = np.nan
    return {"train_acc":train_acc,"test_acc":test_acc,"precision":precision,"recall":recall,"f1":f1,"auc":auc_val,
            "y_test_pred":y_test_pred,"y_test_proba":y_test_proba,"y_train_pred":y_train_pred}

def plot_confusion_matrix(cm, class_labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def plot_roc_curve(y_test, y_score, class_labels, label):
    fig, ax = plt.subplots(figsize=(5,4))
    if len(class_labels)==2:
        fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    else:
        y_test_b = label_binarize(y_test, classes=np.arange(len(class_labels)))
        fpr_micro, tpr_micro, _ = roc_curve(y_test_b.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, label=f"{label} (micro AUC={roc_auc_micro:.3f})")
    ax.plot([0,1],[0,1],'k--'); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC"); ax.legend()
    st.pyplot(fig); plt.close(fig)

def to_download_link(df, filename="data.csv"):
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

# UI - load data
st.sidebar.header("Data")
use_sample = st.sidebar.checkbox("Use sample dataset (provided)", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload CSV", type=["csv"])
if use_sample:
    df_raw = load_sample()
elif uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("Select sample or upload a CSV to proceed.")
    st.stop()

df, dropped_cols = preprocess_df(df_raw)
st.sidebar.info(f"Dropped columns: {', '.join(dropped_cols) if dropped_cols else 'None'}")

# Basic display
st.sidebar.markdown("### Quick Data Info")
st.sidebar.write(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(df.head(200))

# tabs
tab1, tab2, tab3 = st.tabs(["Dashboard (Charts)", "Run Models", "Upload & Predict"])

with tab1:
    st.header("Interactive Dashboard")
    st.write("Use filters below to explore the dataset. Charts update with filters.")
    # Filters area
    with st.expander("Filters (applies to all charts)"):
        # Auto-detect categorical columns and create multiselects
        cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_filters = {}
        for c in cat_cols:
            vals = sorted(df[c].dropna().unique().tolist())
            if len(vals) <= 30:
                sel = st.multiselect(f"Filter {c}", options=vals, default=vals, key=f"f_{c}")
                selected_filters[c] = sel
        # Numeric sliders
        num_filter_ranges = {}
        for c in num_cols:
            mn = float(df[c].min()); mx = float(df[c].max())
            rng = st.slider(f"Range for {c}", min_value=mn, max_value=mx, value=(mn,mx), key=f"n_{c}")
            num_filter_ranges[c] = rng
        prob_threshold = st.slider("Probability threshold (requires trained model)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    # Apply filters to dataframe copy
    df_f = df.copy()
    for c, sel in selected_filters.items():
        if sel:
            df_f = df_f[df_f[c].isin(sel)]
    for c, (lo,hi) in num_filter_ranges.items():
        df_f = df_f[(df_f[c]>=lo) & (df_f[c]<=hi)]
    st.markdown(f"**Filtered rows:** {df_f.shape[0]}")

    # Chart 1: Approval rate by STATE (if available)
    st.subheader("Approval rate by State / Region")
    if "PI_STATE" in df_f.columns or "ZONE" in df_f.columns:
        region_col = "PI_STATE" if "PI_STATE" in df_f.columns else "ZONE"
        group = df_f.groupby(region_col)["POLICY_STATUS"].value_counts(normalize=False).unstack(fill_value=0)
        if "Approved Death Claim" in group.columns:
            approval_rate = group["Approved Death Claim"] / group.sum(axis=1)
        else:
            # attempt to detect positive label as most common
            approval_rate = group.iloc[:,0] / group.sum(axis=1)
        approval_rate = approval_rate.sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,4))
        approval_rate.plot(kind="bar", ax=ax)
        ax.set_ylabel("Approval rate"); ax.set_title(f"Approval rate by {region_col}"); st.pyplot(fig); plt.close(fig)
    else:
        st.info("No state/zone column found for region chart.")

    # Chart 2: Sum assured distribution and mean by status
    st.subheader("Sum assured distribution (and mean by policy status)")
    if "SUM_ASSURED" in df_f.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df_f["SUM_ASSURED"].dropna(), bins=30, ax=ax, kde=False)
        ax.set_title("SUM_ASSURED distribution"); st.pyplot(fig); plt.close(fig)
        # mean by status:
        if "POLICY_STATUS" in df_f.columns:
            mean_by_status = df_f.groupby("POLICY_STATUS")["SUM_ASSURED"].mean().sort_values(ascending=False)
            st.bar_chart(mean_by_status)
    else:
        st.info("SUM_ASSURED column not found.")

    # Chart 3: Age distribution by Policy status (violin)
    st.subheader("Age distribution by Policy Status")
    if "PI_AGE" in df_f.columns and "POLICY_STATUS" in df_f.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.violinplot(x="POLICY_STATUS", y="PI_AGE", data=df_f, ax=ax)
        ax.set_title("Age by policy status"); st.pyplot(fig); plt.close(fig)
    else:
        st.info("PI_AGE or POLICY_STATUS missing.")

    # Chart 4: Claim reasons counts (complex insight: top reasons & rejection rates)
    st.subheader("Reasons for claim — counts and rejection rate")
    if "REASON_FOR_CLAIM" in df_f.columns and "POLICY_STATUS" in df_f.columns:
        counts = df_f["REASON_FOR_CLAIM"].value_counts().nlargest(15)
        fig, ax = plt.subplots(figsize=(8,4))
        counts.plot(kind="bar", ax=ax); ax.set_title("Top claim reasons"); st.pyplot(fig); plt.close(fig)
        # rejection rates by reason
        rej = df_f.groupby("REASON_FOR_CLAIM")["POLICY_STATUS"].apply(lambda s: (s=="Repudiate Death").mean())
        rej = rej.loc[counts.index]
        st.line_chart(rej)
    else:
        st.info("REASON_FOR_CLAIM or POLICY_STATUS missing.")

    # Chart 5: Approval vs Annual income (binned)
    st.subheader("Approval rate by Annual Income bins")
    if "PI_ANNUAL_INCOME" in df_f.columns and "POLICY_STATUS" in df_f.columns:
        bins = pd.qcut(df_f["PI_ANNUAL_INCOME"].fillna(0), q=6, duplicates="drop")
        tmp = df_f.groupby(bins)["POLICY_STATUS"].apply(lambda s: (s=="Approved Death Claim").mean())
        fig, ax = plt.subplots(figsize=(8,3))
        tmp.plot(kind="bar", ax=ax)
        ax.set_ylabel("Approval rate"); st.pyplot(fig); plt.close(fig)
    else:
        st.info("PI_ANNUAL_INCOME or POLICY_STATUS missing.")

    st.write("Note: Probability threshold slider will filter predictions if models have been trained in 'Run Models' tab.")

with tab2:
    st.header("Train & Evaluate Models")
    st.write("Click the button to train Decision Tree, Random Forest and Gradient Boosting using stratified 5-fold CV.")
    target_col = st.selectbox("Select target column", options=[c for c in df.columns if df[c].nunique()<=50], index=0)
    if st.button("Train models (cv=5, stratified)"):
        # prepare X,y (drop rows with null target)
        df_train = df.dropna(subset=[target_col]).copy()
        X = df_train.drop(columns=[target_col]).copy()
        y = df_train[target_col].astype(str).copy()
        # encode target into numeric
        le = LabelEncoder(); y_enc = le.fit_transform(y)
        st.session_state['label_encoder'] = le
        # drop high-card cols automatically in pipeline builder
        results, num_cols, cat_cols = train_models(X, y_enc)
        st.session_state['models'] = results
        st.session_state['num_cols'] = num_cols; st.session_state['cat_cols'] = cat_cols
        st.success("Models trained and stored in session state. Showing metrics below...")

        # Evaluate on hold-out set using a stratified split (25% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
        metrics_table = []
        roc_fig, ax = plt.subplots(figsize=(6,5))
        colors = {"DecisionTree":"tab:blue","RandomForest":"tab:green","GradientBoosting":"tab:red"}
        for name, obj in results.items():
            pipe = obj["pipeline"]
            # compute metrics
            metrics = compute_metrics(pipe, X_train, y_train, X_test, y_test)
            metrics_table.append({
                "Algorithm": name,
                "Train Accuracy": round(metrics["train_acc"],4),
                "Test Accuracy": round(metrics["test_acc"],4),
                "Precision": round(metrics["precision"],4),
                "Recall": round(metrics["recall"],4),
                "F1-score": round(metrics["f1"],4),
                "AUC": round(metrics["auc"],4)
            })
            # confusion matrices
            st.subheader(f"{name} — Confusion Matrices")
            st.write("Training set")
            cm_train = confusion_matrix(y_train, metrics["y_train_pred"])
            plot_confusion_matrix(cm_train, le.classes_, title=f"{name} - Train CM")
            st.write("Test set")
            cm_test = confusion_matrix(y_test, metrics["y_test_pred"])
            plot_confusion_matrix(cm_test, le.classes_, title=f"{name} - Test CM")
            # ROC on combined axes for summary plot
            if len(le.classes_)==2:
                fpr, tpr, _ = roc_curve(y_test, metrics["y_test_proba"][:,1])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=colors.get(name))
            else:
                y_test_b = label_binarize(y_test, classes=np.arange(len(le.classes_)))
                fpr_micro, tpr_micro, _ = roc_curve(y_test_b.ravel(), metrics["y_test_proba"].ravel())
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                ax.plot(fpr_micro, tpr_micro, label=f"{name} (micro AUC={roc_auc_micro:.3f})", color=colors.get(name))
            # Feature importance if available
            st.subheader(f"{name} — Feature importance (if available)")
            clf = pipe.named_steps['clf']
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                # map to feature names using pipeline transforms: we approximate by num_cols+cat_cols
                feat_names = num_cols + cat_cols
                if len(importances)==len(feat_names):
                    idx = np.argsort(importances)[::-1][:20]
                    fig, ax2 = plt.subplots(figsize=(6,4))
                    ax2.bar(range(len(idx)), importances[idx])
                    ax2.set_xticks(range(len(idx))); ax2.set_xticklabels(np.array(feat_names)[idx], rotation=45, ha="right")
                    ax2.set_title(f"{name} feature importance")
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.write("Feature importances produced but could not map to feature names reliably.")
            else:
                st.write("Feature importances not available for this estimator.")
        ax.plot([0,1],[0,1],'k--'); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves (All models)")
        ax.legend(); st.pyplot(roc_fig); plt.close(roc_fig)
        st.table(pd.DataFrame(metrics_table).set_index("Algorithm"))

with tab3:
    st.header("Upload new dataset & Predict")
    st.write("Upload a CSV with the same feature columns used for training (or compatible). Choose a model and click Predict to add predicted label and probability to the file, then download.")
    uploaded = st.file_uploader("Upload CSV for prediction", type=["csv"], key="predict_upload")
    model_choice = st.selectbox("Choose model to use for predictions", options=["RandomForest","GradientBoosting","DecisionTree"])
    id_col = st.text_input("If you want an ID column preserved in output, type its name (optional)", value="")
    if st.button("Predict & Download"):
        if uploaded is None:
            st.error("Please upload a CSV to predict.")
        elif 'models' not in st.session_state:
            st.error("No trained models found in session. Please go to 'Run Models' and train models first.")
        else:
            df_new = pd.read_csv(uploaded)
            # Preprocess similarly: drop same high-card columns by running preprocess_df with same settings
            df_proc, dropped = preprocess_df(df_new)
            # load chosen pipeline
            pipe = st.session_state['models'][model_choice]['pipeline']
            le = st.session_state.get('label_encoder', None)
            # Predict (handle missing cols by aligning)
            try:
                preds = pipe.predict(df_proc)
                if hasattr(pipe.named_steps['clf'], "predict_proba"):
                    prob = pipe.predict_proba(df_proc)
                    # if binary assume class 1 prob
                    if prob.shape[1] == 2:
                        prob_pos = prob[:,1]
                    else:
                        # take max prob
                        prob_pos = prob.max(axis=1)
                else:
                    prob_pos = np.zeros(len(preds))
                # decode preds
                if le is not None:
                    pred_label = le.inverse_transform(preds.astype(int))
                else:
                    pred_label = preds.astype(str)
                df_out = df_new.copy()
                df_out['PREDICTED_STATUS'] = pred_label
                df_out['PREDICTED_PROB'] = prob_pos
                # download
                bio = to_download_link(df_out)
                st.download_button("Download predictions CSV", data=bio, file_name="predictions.csv", mime="text/csv")
                st.success("Prediction completed. Rows predicted: " + str(len(df_out)))
            except Exception as e:
                st.error("Prediction failed: " + str(e))

st.sidebar.markdown("---")
st.sidebar.markdown("Built for demo. If you want, train models with sample dataset and then upload your dataset in 'Upload & Predict' tab to get predictions and download.")