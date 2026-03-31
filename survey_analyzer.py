import os
import sys
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QComboBox,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

def load_data(data_path, questions_path):
    df = pd.read_excel(data_path)
    df_pitanja = pd.read_excel(questions_path)
    return df, df_pitanja


def sort_questions(df_pitanja:pd.DataFrame):
    ordinal_vars=[]
    select_one = []
    multi_nominal_cols = []
    binary = []

    for qid, qtype, options in zip(df_pitanja["QuestionID"], df_pitanja["Type"], df_pitanja["Options"]):

        qid = str(int(float(qid))) if str(qid).endswith('.0') else str(qid)
        if qtype == "Scale" and pd.notnull(options):
            choices = [opt.strip() for opt in str(options).split(';') if '=' in opt]
            if len(choices) > 3:   # more than 3 options
                ordinal_vars.append(qid)

            if len(choices) <= 3:   # <= 3 options
                binary.append(qid)

        elif qtype == "SelectOne":
            select_one.append(qid)

        elif qtype == "SelectMultiple":
            multi_nominal_cols.append(qid)

    return ordinal_vars, select_one, multi_nominal_cols, binary

def drop_missing(df: pd.DataFrame, target_question):
    df_missing = df.isnull().sum().sort_values(ascending=False)
    za_drop = []
    x = int(len(df) * 0.3)
    target_question = str(target_question)

    for index, value in df_missing.items():
        col = str(int(float(index))) if str(index).endswith('.0') else str(index)
        if value > x and col != target_question:
            za_drop.append(col)

    return za_drop

def drop_2(df_pitanja: pd.DataFrame, target_question):
    za_drop2 = []
    target_question = str(target_question)

    for qid, qtype in zip(df_pitanja["QuestionID"], df_pitanja["Type"]):
        qid = str(int(float(qid))) if str(qid).endswith('.0') else str(qid)
        if qtype == "Text" and qid != target_question:
            za_drop2.append(qid)

    return za_drop2

def target_cleaning(target_question, df: pd.DataFrame, za_drop, za_drop2):
    df = df.copy()
    col = str(target_question)

    df.columns = df.columns.astype(str)

    if col in df.columns:
        df = df[df[col].notna()]
        df = df.drop(columns=za_drop2, errors='ignore')
        df = df.drop(columns=za_drop, errors='ignore')
    else:
        print("There is no column named", col)

    return df

def filtering(df: pd.DataFrame, ordinal_vars, multi_nominal_cols, select_one, binary):
    df = df.copy()
    ordinal_vars = [col for col in ordinal_vars if col in df.columns]
    for col in ordinal_vars:
        df[col] = df[col].astype('Int64')

    for col in multi_nominal_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
            dummies = df[col].str.get_dummies(sep=';')
            dummies.columns = [f"{col}_{c.strip()}" for c in dummies.columns]
            df[f'{col}_missing_all'] = dummies.sum(axis=1).eq(0).astype(int)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)

    for col in select_one:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
            dummies = df[col].str.get_dummies(sep=',')
            dummies.columns = [f"{col}_{c.strip()}" for c in dummies.columns]
            df[f'{col}_missing_all'] = dummies.sum(axis=1).eq(0).astype(int)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)

    for col in binary:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
            dummies = df[col].str.get_dummies(sep=',')
            dummies.columns = [f"{col}_{c.strip()}" for c in dummies.columns]
            df[f'{col}_missing_all'] = dummies.sum(axis=1).eq(0).astype(int)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)

    return df

def nps_reverse(score):
    if score in [1, 2]:
        return "Promoter"
    elif score in [3, 4]:
        return "Passive"
    else:
        return "Detractor"
    
def satisfaction_label(score):
    mapping = {
        1: "begeistert",
        2: "sehr zufrieden",
        3: "zufrieden",
        4: "unzufrieden",
        5: "sehr unzufrieden",
    }
    return mapping.get(score, None)


def target_building_by_type(df: pd.DataFrame, target_question, target_type: str):
    df = df.copy()
    col = str(target_question)

    if col not in df.columns:
        raise KeyError(f"Target column '{col}' not found.")

    numeric_target = pd.to_numeric(df[col], errors="coerce")

    if target_type == "NPS":
        df["target_category"] = numeric_target.apply(nps_reverse)

    elif target_type == "Satisfaction":
        df["target_category"] = numeric_target.apply(satisfaction_label)

    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    df = df[df["target_category"].notna()].copy()
    df = df.drop(columns=[col])

    return df, "target_category"

def process(df: pd.DataFrame, target_col):
    df_proc = df.copy()

    excluded_columns = [target_col]
    columns_to_process = [
        c for c in df_proc.columns
        if c not in excluded_columns
    ]

    for col in columns_to_process:
        df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')

    X = df_proc.drop(columns=[target_col])
    Y = df_proc[target_col]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )

    columns_to_process = [c for c in columns_to_process if c in X_train.columns]

    imputer = SimpleImputer(strategy="median")
    X_train[columns_to_process] = imputer.fit_transform(X_train[columns_to_process])
    X_test[columns_to_process] = imputer.transform(X_test[columns_to_process])

    return X_train, X_test, Y_train, Y_test

from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score

def classifier(X_train, Y_train, X_test, Y_test):
    rf = RandomForestClassifier(random_state=42)

    params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    min_class_count = Y_train.value_counts().min()

    if min_class_count < 2:
        raise ValueError(
            "At least one target class has fewer than 2 samples in the training set. "
            "Cross-validation cannot run. You need more data or fewer target classes."
        )

    cv_folds = min(5, int(min_class_count))

    rf_cv = GridSearchCV(
        rf,
        param_grid=params,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1
    )

    rf_cv.fit(X_train, Y_train)

    print("Using cv =", cv_folds)
    print("✅ Best parameters:", rf_cv.best_params_)
    print("✅ Best cross-val Accuracy:", rf_cv.best_score_)

    best_rf = rf_cv.best_estimator_
    y_pred = best_rf.predict(X_test)
    test_acc = accuracy_score(Y_test, y_pred)

    print("Test accuracy:", test_acc)

    return best_rf, rf_cv.best_params_, rf_cv.best_score_, test_acc

def shap_analysis(best_rf, X_train, df_pitanja, output_excel="xy.xlsx", plot_path="shap_summary.png", class_idx=0):
    print("Classes:", best_rf.classes_)

    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        sv_class = shap_values[class_idx]
        global_mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    else:
        shap_values = np.array(shap_values)

        if shap_values.ndim == 3:
            sv_class = shap_values[:, :, class_idx]
            global_mean_abs = np.abs(shap_values).mean(axis=(0, 2))

        elif shap_values.ndim == 2:
            sv_class = shap_values
            global_mean_abs = np.abs(shap_values).mean(axis=0)

        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    plt.figure()
    shap.summary_plot(sv_class, X_train, plot_type="dot", max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    global_top = (
        pd.DataFrame({
            "Feature": X_train.columns,
            "Mean |SHAP| (global)": global_mean_abs
        })
        .sort_values("Mean |SHAP| (global)", ascending=False)
        .head(15)
        .copy()
    )

    global_top["QuestionID_extracted"] = (
        global_top["Feature"]
        .astype(str)
        .str.extract(r"^(\d+\.?\d*)")[0]
    )

    global_top["QuestionID_extracted"] = pd.to_numeric(
        global_top["QuestionID_extracted"], errors="coerce"
    )

    pitanja_merge = df_pitanja[["QuestionID", "Text"]].copy()
    pitanja_merge["QuestionID"] = pd.to_numeric(pitanja_merge["QuestionID"], errors="coerce")

    global_top = global_top.merge(
        pitanja_merge,
        left_on="QuestionID_extracted",
        right_on="QuestionID",
        how="left"
    )

    global_top.to_excel(output_excel, index=False)

    return global_top

def shap_multiclass_report(
    best_rf,
    X_train,
    df_pitanja,
    class_idx=0,
    top_n=15,
    max_display=30,
    output_excel="xy.xlsx",
    plot_path="shap_summary.png"
):
    print("Classes:", best_rf.classes_)

    explainer = shap.TreeExplainer(best_rf)
    shap_values_raw = explainer.shap_values(X_train)

    if isinstance(shap_values_raw, list):
        if class_idx >= len(shap_values_raw):
            raise ValueError(f"class_idx must be between 0 and {len(shap_values_raw)-1}")

        sv_class = shap_values_raw[class_idx]
        shap_values_3d = np.stack(shap_values_raw, axis=2)
        global_mean_abs = np.abs(shap_values_3d).mean(axis=(0, 2))

    else:
        shap_values_raw = np.array(shap_values_raw)

        if shap_values_raw.ndim == 3:
            if class_idx >= shap_values_raw.shape[2]:
                raise ValueError(f"class_idx must be between 0 and {shap_values_raw.shape[2]-1}")

            sv_class = shap_values_raw[:, :, class_idx]
            global_mean_abs = np.abs(shap_values_raw).mean(axis=(0, 2))

        elif shap_values_raw.ndim == 2:
            sv_class = shap_values_raw
            global_mean_abs = np.abs(shap_values_raw).mean(axis=0)

        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_values_raw.shape}")

    plt.figure()
    shap.summary_plot(
        sv_class,
        X_train,
        plot_type="dot",
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    global_top = (
        pd.DataFrame({
            "Feature": X_train.columns.astype(str),
            "Mean |SHAP| (global)": global_mean_abs
        })
        .sort_values("Mean |SHAP| (global)", ascending=False)
        .head(top_n)
        .copy()
    )

    global_top["QuestionID_extracted"] = (
        global_top["Feature"]
        .str.extract(r"^(\d+\.?\d*)")[0]
    )

    global_top["QuestionID_extracted"] = pd.to_numeric(
        global_top["QuestionID_extracted"],
        errors="coerce"
    )

    pitanja_merge = df_pitanja[["QuestionID", "Text"]].copy()
    pitanja_merge["QuestionID"] = pd.to_numeric(pitanja_merge["QuestionID"], errors="coerce")

    global_top = global_top.merge(
        pitanja_merge,
        left_on="QuestionID_extracted",
        right_on="QuestionID",
        how="left"
    )

    global_top.to_excel(output_excel, index=False)

    print(global_top)
    print(f"Saved Excel: {output_excel}")
    print(f"Saved plot: {plot_path}")

    return global_top, explainer, shap_values_raw

from pathlib import Path
import contextlib
import io

from PySide6.QtWidgets import QSpinBox


def _normalize_qid_ui(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def _normalize_df_columns_ui(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_qid_ui(c) for c in df.columns]
    return df


def load_data_from_one_workbook(workbook_path: str):
    xls = pd.ExcelFile(workbook_path)

    if len(xls.sheet_names) < 2:
        raise ValueError(
            "The Excel file must contain at least 2 sheets. "
            "First sheet = df, second sheet = df_pitanja."
        )

    df = pd.read_excel(workbook_path, sheet_name=0)
    df_pitanja = pd.read_excel(workbook_path, sheet_name=1)

    df = _normalize_df_columns_ui(df)

    df = df.drop(
        columns=[
            'ResultId', 'Project', 'Survey', 'Language', 'UserGroup',
            'Code', 'AccessId', 'Access', 'Segments', 'CurrentPage',
            'Progress', 'StartTime', 'EndTime', 'Test'
        ],
        errors='ignore'
    )

    return df, df_pitanja


def drop_missing_percent(df: pd.DataFrame, target_question, missing_percent: int):
    df_missing = df.isnull().sum().sort_values(ascending=False)
    za_drop = []
    target_question = str(target_question)

    threshold_count = int(len(df) * (missing_percent / 100.0))

    for index, value in df_missing.items():
        col = _normalize_qid_ui(index)
        if value > threshold_count and col != target_question:
            za_drop.append(col)

    return za_drop


class _LogEmitter(io.StringIO):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        self._buffer = ""

    def write(self, s):
        if not s:
            return 0
        self._buffer += str(s)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self.signal.emit(line)
        return len(s)

    def flush(self):
        if self._buffer.strip():
            self.signal.emit(self._buffer.strip())
        self._buffer = ""


class AnalysisWorker(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, workbook_path: str, target_question: str, target_type: str, missing_percent: int, output_dir: str, top_n: int):
        super().__init__()
        self.workbook_path = workbook_path
        self.target_question = _normalize_qid_ui(target_question)
        self.target_type = target_type
        self.missing_percent = missing_percent
        self.output_dir = output_dir.strip()
        self.top_n = top_n

    @Slot()
    def run(self):
        logger = _LogEmitter(self.log)

        try:
            with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(logger):
                self.progress.emit(5)
                self.log.emit("Loading workbook...")
                df, df_pitanja = load_data_from_one_workbook(self.workbook_path)

                self.progress.emit(15)
                self.log.emit("Sorting question types...")
                ordinal_vars, select_one, multi_nominal_cols, binary = sort_questions(df_pitanja)

                self.progress.emit(25)
                self.log.emit("Calculating columns to drop...")
                za_drop = drop_missing_percent(df, self.target_question, self.missing_percent)
                za_drop2 = drop_2(df_pitanja, self.target_question)

                self.progress.emit(35)
                self.log.emit("Cleaning target and dropping columns...")
                df_clean = target_cleaning(self.target_question, df, za_drop, za_drop2)

                if df_clean.empty:
                    raise ValueError(
                        "No data left after cleaning. Check the target question ID and missing threshold."
                    )

                self.progress.emit(50)
                self.log.emit("Encoding survey columns...")
                df_filtered = filtering(
                    df_clean,
                    ordinal_vars=ordinal_vars,
                    multi_nominal_cols=multi_nominal_cols,
                    select_one=select_one,
                    binary=binary
                )

                self.progress.emit(60)
                self.log.emit(f"Building target column using type: {self.target_type} ...")
                df_model, target_col = target_building_by_type(
                    df_filtered,
                    self.target_question,
                    self.target_type
                )

                self.progress.emit(70)
                self.log.emit("Preparing train / test data...")
                X_train, X_test, Y_train, Y_test = process(df_model, target_col)

                self.progress.emit(80)
                self.log.emit("Training RandomForest with GridSearchCV...")
                best_rf, best_params, best_cv_score, test_acc = classifier(
                    X_train, Y_train, X_test, Y_test
                )

                self.progress.emit(90)
                self.log.emit("Running SHAP analysis...")

                if self.output_dir:
                    out_dir = Path(self.output_dir).resolve()
                else:
                    out_dir = Path(self.workbook_path).resolve().parent

                out_dir.mkdir(parents=True, exist_ok=True)

                output_excel = str(out_dir / f"top_features_target_{self.target_question}.xlsx")
                plot_path = str(out_dir / f"shap_summary_target_{self.target_question}.png")

                global_top, explainer, shap_values_raw = shap_multiclass_report(
                    best_rf=best_rf,
                    X_train=X_train,
                    df_pitanja=df_pitanja,
                    class_idx=0,
                    top_n=self.top_n,
                    max_display=30,
                    output_excel=output_excel,
                    plot_path=plot_path
                )

                self.progress.emit(100)
                self.finished.emit({
                    "rows_final": len(df_model),
                    "features": X_train.shape[1],
                    "best_params": best_params,
                    "best_cv_score": best_cv_score,
                    "test_acc": test_acc,
                    "classes": list(best_rf.classes_),
                    "excel_path": output_excel,
                    "plot_path": plot_path,
                    "save_dir": str(out_dir),
                    "top_preview": global_top.head(10).to_string(index=False)
                })

        except Exception:
            self.failed.emit(traceback.format_exc())


class SurveyAnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Survey Analyzer")
        self.resize(920, 700)

        self.thread = None
        self.worker = None

        self.file_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.target_edit = QLineEdit()
        self.target_type_combo = QComboBox()
        self.target_type_combo.addItems(["NPS", "Satisfaction"])
        self.percent_spin = QSpinBox()
        self.percent_spin.setRange(1, 100)
        self.percent_spin.setValue(30)
        self.percent_spin.setSuffix(" %")
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 1000)
        self.top_n_spin.setValue(15)

        self.run_button = QPushButton("Run")
        self.clear_button = QPushButton("Clear log")

        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.status_label = QLabel("Choose file, output folder, enter target question ID, then click Run.")
        self.status_label.setWordWrap(True)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setOpenExternalLinks(True)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)

        self._build_ui()
        self._connect()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        title = QLabel("<h2>Survey Analyzer</h2>")
        subtitle = QLabel(
            "Excel file must have 2 sheets: first sheet = survey data, second sheet = question metadata."
        )
        subtitle.setWordWrap(True)

        form = QFormLayout()

        file_row = QHBoxLayout()
        file_row.addWidget(self.file_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(browse_btn)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(output_btn)

        form.addRow("Excel workbook:", file_row)
        form.addRow("Save folder:", output_row)
        form.addRow("Target question ID:", self.target_edit)
        form.addRow("Target type:", self.target_type_combo)
        form.addRow("Missing threshold:", self.percent_spin)
        form.addRow("Top values:", self.top_n_spin)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.run_button)
        btn_row.addWidget(self.clear_button)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addWidget(self.result_label)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_box)

    def _connect(self):
        self.run_button.clicked.connect(self._start_run)
        self.clear_button.clicked.connect(self.log_box.clear)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel workbook",
            "",
            "Excel files (*.xlsx *.xls);;All files (*.*)"
        )
        if file_path:
            self.file_edit.setText(file_path)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to save outputs")
        if folder:
            self.output_edit.setText(folder)

    def _set_running(self, running: bool):
        self.run_button.setEnabled(not running)
        self.file_edit.setEnabled(not running)
        self.output_edit.setEnabled(not running)
        self.target_edit.setEnabled(not running)
        self.target_type_combo.setEnabled(not running)
        self.percent_spin.setEnabled(not running)
        self.top_n_spin.setEnabled(not running)

    def _append_log(self, text: str):
        self.log_box.appendPlainText(text)

    def _start_run(self):
        workbook_path = self.file_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        target_question = self.target_edit.text().strip()
        missing_percent = int(self.percent_spin.value())
        target_type = self.target_type_combo.currentText()
        top_n = int(self.top_n_spin.value())

        if not workbook_path:
            QMessageBox.warning(self, "Missing file", "Please choose the Excel workbook.")
            return

        if not target_question:
            QMessageBox.warning(self, "Missing target", "Please enter the target question ID.")
            return

        if not output_dir:
            QMessageBox.warning(self, "Missing save folder", "Please choose where results should be saved.")
            return

        self.log_box.clear()
        self.result_label.setText("")
        self.progress.setValue(0)
        self.status_label.setText("Running...")
        self._set_running(True)

        self.thread = QThread()
        self.worker = AnalysisWorker(
            workbook_path,
            target_question,
            target_type,
            missing_percent,
            output_dir,
            top_n
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_thread)

        self.thread.start()

    def _cleanup_thread(self):
        self._set_running(False)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None

    def _on_finished(self, result: dict):
        excel_uri = Path(result["excel_path"]).resolve().as_uri()
        plot_uri = Path(result["plot_path"]).resolve().as_uri()

        self.status_label.setText("Finished successfully.")

        self.result_label.setText(
            f"<b>Done.</b><br>"
            f"Saved in: {result['save_dir']}<br>"
            f"Rows used: {result['rows_final']}<br>"
            f"Features: {result['features']}<br>"
            f"Classes: {', '.join(map(str, result['classes']))}<br>"
            f"Best CV accuracy: {result['best_cv_score']:.4f}<br>"
            f"Test accuracy: {result['test_acc']:.4f}<br>"
            f"Best params: {result['best_params']}<br><br>"
            f"<a href='{excel_uri}'>Open Excel output</a><br>"
            f"<a href='{plot_uri}'>Open SHAP plot</a>"
        )

        self._append_log("")
        self._append_log(f"Saved in folder: {result['save_dir']}")
        self._append_log("Top features preview:")
        self._append_log(result["top_preview"])

        QMessageBox.information(self, "Finished", f"Analysis completed.\nSaved in:\n{result['save_dir']}")

    def _on_failed(self, error_text: str):
        self.status_label.setText("Failed.")
        self._append_log(error_text)
        QMessageBox.critical(self, "Error", error_text)


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = SurveyAnalyzerWindow()
    window.show()
    sys.exit(app.exec())