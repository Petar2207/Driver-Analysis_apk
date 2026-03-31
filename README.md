# 📊 Driver Analysis Studio (PySide6 + Scikit-learn + SHAP)

A desktop application for analyzing survey drivers from Excel data using Python, PySide6, Random Forest, and SHAP.

The project consists of:

- a native desktop GUI for selecting the workbook, target question, target type, and output folder
- an automated analysis pipeline for cleaning survey data, encoding question types, training a model, and ranking the most influential drivers
- export tools that generate:
  - an Excel file with top features
  - a SHAP summary plot image
  - a PowerPoint slide based on a template

---

## ✨ Features

### General
- 🖥️ Native desktop GUI built with PySide6
- 📂 Load one Excel workbook containing survey data + question metadata
- 🎯 Choose target question ID directly in the app
- 🧠 Supports target analysis for:
  - NPS
  - Satisfaction
- 🧹 Configurable missing-data threshold
- 🌲 RandomForest training with GridSearchCV
- 📈 SHAP-based feature importance analysis
- 📝 Built-in log window and progress bar
- 📊 Export ranked drivers to Excel
- 🖼️ Export SHAP summary plot as PNG
- 📽️ Create a PowerPoint slide from a template

---

## 🧭 How It Works

The application (`survey_analyzer.py`) lets you:

- Select an Excel workbook
- Choose a save folder
- Enter a target question ID
- Select the target type
- Set the missing-value threshold
- Set how many top drivers should be ranked

After running the analysis, the app:

1. Loads the workbook
2. Detects survey question types from the metadata sheet
3. Drops text fields and high-missing columns
4. Encodes survey answers into model-ready features
5. Builds the target class
6. Trains a RandomForest model with GridSearchCV
7. Generates SHAP-based driver importance outputs
8. Lets you choose exactly 8 items for presentation
9. Optionally marks selected negative items with a red outline
10. Creates a PowerPoint slide from a template

---

## 📈 Analysis Pipeline

### 1️⃣ Workbook Loader
- Reads one Excel workbook with **2 sheets**
- **Sheet 1** = survey data
- **Sheet 2** = question metadata

### 2️⃣ Survey Question Processing
Supported metadata-driven question handling includes:

- **Scale**
- **SelectOne**
- **SelectMultiple**
- **Text**

The app automatically separates columns into:
- ordinal variables
- binary variables
- single-choice variables
- multiple-choice variables

### 3️⃣ Data Cleaning
- Removes rows with missing target values
- Drops text columns (except target if needed)
- Drops columns above the selected missing-data threshold
- Normalizes question IDs and column names

### 4️⃣ Model Training
- Train/test split
- Median imputation
- RandomForestClassifier
- Hyperparameter search with GridSearchCV
- Reports:
  - best parameters
  - best cross-validation accuracy
  - test accuracy

### 5️⃣ SHAP Driver Analysis
- Generates a SHAP summary plot
- Ranks the top influencing features
- Matches extracted question IDs back to question text
- Saves the ranked driver table to Excel

### 6️⃣ PowerPoint Slide Generation
- Uses a PowerPoint template file: `impact_template.pptx`
- Replaces placeholders such as title, target, accuracy, note, and driver items
- Requires exactly **8 selected items**
- Optionally adds red outlines to selected “negative” items
- Hides the negative legend automatically if no negative items are selected

---

## 📂 Project Structure

```text
project-root/
├─ survey_analyzer.py      # Main desktop application
├─ requirements.txt
├─ run.bat                 # Windows helper
├─ README.md
├─ LICENSE
└─ impact_template.pptx    # Required PowerPoint template (must be next to the script)
