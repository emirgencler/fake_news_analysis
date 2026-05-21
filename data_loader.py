import pandas as pd

def load_data(fake_path, true_path):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 1
    true["label"] = 0

    df = pd.concat([fake, true], ignore_index=True)

    return df
