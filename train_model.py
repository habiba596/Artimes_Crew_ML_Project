import os
import subprocess
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

DROP_COLUMNS = [
    "Application_ID",
    "Registration_Code",
    "International",
    "Nacionality",
    "Inflation rate",
    "Previous qualification (grade)",
    "Educational special needs",
]

BOOL_COLUMNS = ["Displaced", "Debtor", "Tuition fees up to date", "Scholarship holder"]

CAT_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
]

NUMERIC_COLUMNS = [
    "Application order",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "GDP",
]

TARGET_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}


def clean_data(df):
    df = df.copy()

    df.fillna(
        {
            "Previous qualification": df["Previous qualification"].mode()[0],
            "Scholarship holder": df["Scholarship holder"].mode()[0],
            "Mother's qualification": "Unknown",
            "Father's qualification": "Unknown",
            "Mother's occupation": "Uknown",
            "Father's occupation": "Unkown",
            "Gender": "Unkown",
            "Admission grade": df["Admission grade"].mean(),
        },
        inplace=True,
    )

    df.drop(columns=DROP_COLUMNS, inplace=True, errors="ignore")
    df.drop_duplicates(inplace=True)

    df["Gender"] = df["Gender"].str.lower().replace({"m": "male", "f": "female"})
    df["Marital status"] = df["Marital status"].str.lower()

    for col in BOOL_COLUMNS:
        df[col] = df[col].str.lower().replace(
            {"n": "no", "y": "yes", "true": "yes", "false": "no"}
        )

    return df


def encode_bool_columns(df):
    bool_map = {"yes": 1, "no": 0}
    for col in BOOL_COLUMNS:
        df[col] = df[col].map(bool_map)
    return df


def build_model(input_dim):
    model = Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            Dense(128, activation="tanh"),
            Dropout(0.5),
            Dense(128, activation="tanh"),
            Dropout(0.5),
            Dense(128, activation="tanh"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="tanh"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="tanh"),
            Dropout(0.4),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(train_path, output_dir, epochs):
    os.makedirs(output_dir, exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_train = clean_data(df_train)

    if "Student_ID" in df_train.columns:
        df_train.drop(columns=["Student_ID"], inplace=True)

    x = df_train.drop(columns=["Target"])
    y = df_train["Target"].map(TARGET_MAP)

    x = encode_bool_columns(x)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(x[CAT_COLUMNS])
    ohe_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(CAT_COLUMNS))

    x = pd.concat([x, ohe_df], axis=1)
    x.drop(columns=CAT_COLUMNS, inplace=True)

    feature_columns = x.columns.tolist()

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    model = build_model(x_train_scaled.shape[1])
    model.fit(x_train_scaled, y_train, epochs=epochs)

    val_loss, val_accuracy = model.evaluate(x_val_scaled, y_val)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    category_options = {
        col: [str(v) for v in cats] for col, cats in zip(CAT_COLUMNS, encoder.categories_)
    }

    model.save(os.path.join(output_dir, "model.keras"))
    joblib.dump(encoder, os.path.join(output_dir, "encoder.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(output_dir, "feature_columns.pkl"))
    joblib.dump(category_options, os.path.join(output_dir, "category_options.pkl"))
    joblib.dump(CAT_COLUMNS, os.path.join(output_dir, "cat_columns.pkl"))
    joblib.dump(NUMERIC_COLUMNS, os.path.join(output_dir, "numeric_columns.pkl"))
    joblib.dump(BOOL_COLUMNS, os.path.join(output_dir, "bool_columns.pkl"))

    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="train.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(args.train_path, args.output_dir, args.epochs)
