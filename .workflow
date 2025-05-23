name: Preprocessing Churn Dataset

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  preprocess:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: preprocessing

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.12.7"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy pandas scikit-learn joblib

      - name: Run preprocessing script
        run: |
          python automate_Muhammad_Farkhan_Adhitama.py

      - name: Upload preprocessed dataset
        uses: actions/upload-artifact@v4
        with:
          name: preprocessed-dataset
          path: preprocessing/bank_customer_preprocessed.csv
