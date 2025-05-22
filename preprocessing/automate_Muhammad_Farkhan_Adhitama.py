import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump


def preprocess_data(data=None, target_column=None, save_path=None, file_path=None):
    # Load data jika tidak diberikan langsung
    if data is None:
        if file_path is None:
            raise ValueError("Harap berikan data langsung atau path ke file CSV.")
        data = pd.read_csv(file_path)

    # 1. Hapus missing values dan duplikat
    data = data.dropna()
    data = data.drop_duplicates()

    # 2. Hapus kolom yang tidak dibutuhkan
    data_cleaned = data.drop(
        columns=["RowNumber", "CustomerId", "Surname"], errors="ignore"
    )

    # 3. Label Encoding kolom kategorikal
    le = LabelEncoder()
    for col in data_cleaned.select_dtypes(include="object").columns:
        data_cleaned[col] = le.fit_transform(data_cleaned[col])

    # 4. Standarisasi kolom numerikal (kecuali target)
    numeric_cols = data_cleaned.select_dtypes(
        include=["int64", "float64", "int32"]
    ).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]

    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(data_cleaned[numeric_cols])
    df_scaled = pd.DataFrame(scaled_numeric, columns=numeric_cols)

    # Gabungkan kembali data
    data_scaled = pd.concat(
        [
            df_scaled.reset_index(drop=True),
            data_cleaned[[target_column]].reset_index(drop=True),
        ],
        axis=1,
    )

    # Split fitur dan target
    X = data_scaled.drop(target_column, axis=1)
    y = data_scaled[target_column]

    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42, stratify=y
    )

    # Simpan scaler
    if save_path is not None:
        dump(scaler, save_path)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df_final = preprocess_data(
        file_path="bank_customer.csv",
        target_column="Exited",
        save_path="scaler_churn.joblib",
    )
    # Simpan hasilnya ke file
    df_final.to_csv("bank_customer_preprocessed.csv", index=False)
