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

    # Hapus missing values dan duplikat
    data_cleaned = data.dropna()
    data_cleaned = data.drop_duplicates()

    # Hapus outiers
    Q1 = data_cleaned.quantile(0.25)
    Q3 = data_cleaned.quantile(0.75)
    IQR = Q3 - Q1
    data_cleaned = data_cleaned[
        ~((data_cleaned < (Q1 - 1.5 * IQR)) | (data_cleaned > (Q3 + 1.5 * IQR))).any(
            axis=1
        )
    ]
    # Scaling (kecuali target)
    cols = [col for col in data_cleaned if col != target_column]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_cleaned[cols])
    df_scaled = pd.DataFrame(scaled, columns=cols)

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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Simpan scaler
    if save_path is not None:
        dump(scaler, save_path)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(
        target_column="Outcome",
        file_path="../dataset_raw/diabetes.csv",
    )

# Mengembalikan data yang siap dilatih
df_final = pd.concat(
    [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
)
df_final.to_csv("diabetes_preprocessing.csv", index=False)
