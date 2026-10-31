import os  # used to find the folder this script is saved in
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # stops the harmless joblib cpu warning on windows
import glob  # finds all the csv files in the deauth folder
import datetime  # puts the time on every alert
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw to image files instead of opening a window
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

np.random.seed(42)
n = 5000  # max rows per class used for training in each fold
normal_per_file = 1000  # normal rows sampled from each file (all deauth rows are kept)
folds = 4  # every file is held out for testing exactly once

label_col = "Label"
addr_cols = ["wlan.ta", "wlan.sa", "wlan.ra"]  # ta = the device that sent the frame

# wlan.fc.subtype and wlan.fc.protected are left out on purpose, they give the answer away
# (deauth is subtype 12 and not protected). frame.number and frame.time are left out too,
# they only say where a frame sits in the capture
features = ["frame.len", "frame.time_delta", "radiotap.dbm_antsignal", "wlan.duration", "wlan_radio.data_rate",
            "wlan_radio.duration", "wlan.fc.type", "wlan.fc.retry", "wlan.fc.moredata", "wlan.fc.pwrmgt", "frames_per_sec"]

# for the ablation test: put feature names here to run without them,
# e.g. ["frames_per_sec", "frame.time_delta"]
drop_features = []
features = [f for f in features if f not in drop_features]

needed = [c for c in features if c != "frames_per_sec"] + ["frame.time_epoch"] + addr_cols + [label_col]
if "frame.time_delta" in drop_features or "frames_per_sec" in drop_features:
    needed = list(dict.fromkeys(needed + ["frame.len"]))  # frame.len is still needed to count frames per second

here = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(here, "wids_alerts.log")
png_path = os.path.join(here, "confusion_matrix.png")


def load_files():
    files = sorted(glob.glob(os.path.join(here, "deauth", "*.csv")))
    print("Files found:", len(files))
    if not files:
        raise SystemExit("No csv files found in " + os.path.join(here, "deauth"))

    tables = []
    for f in files:
        name = os.path.basename(f)
        header = pd.read_csv(f, nrows=0).columns.str.strip()  # column names only, fast
        if label_col not in header:  # the cut down selected-columns files have no Label
            print("Skipping", name, "because it has no", label_col, "column")
            continue
        labels = pd.read_csv(f, usecols=lambda c: c.strip() == label_col)[label_col].astype(str).str.lower()
        if not labels.str.contains("deauth").any():  # attack-free files are skipped
            print("Skipping", name, "because it has no deauth frames")
            continue

        df = pd.read_csv(f, usecols=lambda c: c.strip() in needed, low_memory=False)
        df.columns = df.columns.str.strip()

        missing = [c for c in needed if c not in df.columns]
        if missing:  # a file missing features would get filled with medians and distort the data
            print("Skipping", name, "missing columns:", missing)
            continue

        # frames each device sent in each second, worked out on the whole file before sampling
        epoch = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
        sender = df["wlan.ta"].fillna(df["wlan.sa"])
        df["frames_per_sec"] = df["frame.len"].groupby([sender.fillna("none"), np.floor(epoch)]).transform("size").fillna(0)
        df.loc[sender.isna(), "frames_per_sec"] = 0  # acks etc have no sender

        lab = df[label_col].astype(str).str.lower()
        normal_rows = df[lab.str.contains("normal")]
        attack_rows = df[lab.str.contains("deauth")]
        print("Loaded", name, "-", len(normal_rows), "normal,", len(attack_rows), "deauth")

        part = pd.concat([normal_rows.sample(min(len(normal_rows), normal_per_file), random_state=42), attack_rows])
        part["file"] = name  # the group tag, GroupKFold uses it to keep each file on one side
        tables.append(part)

    if not tables:
        raise SystemExit("No usable files found")

    data = pd.concat(tables, ignore_index=True)
    data["label"] = data[label_col].astype(str).str.lower().str.contains("deauth").astype(int)  # 0 normal, 1 attack
    print("\nUsable files:", len(tables), "of", len(files))
    print("Rows loaded - normal:", (data["label"] == 0).sum(), "| attack:", (data["label"] == 1).sum())
    return data


def pick_features(data):
    # keeps only features that exist, are not mostly empty and are not constant
    used = []
    for c in features:
        if c in data.columns:
            v = pd.to_numeric(data[c], errors="coerce")
            if v.notna().mean() > 0.5 and v.nunique() > 1:
                used.append(c)
    print("Features used:", used)
    return used


def run_fold(train, test, used):
    # balance the training part only, the test part keeps its natural mix
    k = min(n, (train["label"] == 0).sum(), (train["label"] == 1).sum())
    train = pd.concat([train[train["label"] == 0].sample(k, random_state=42),
                       train[train["label"] == 1].sample(k, random_state=42)])

    X_train = train[used].apply(pd.to_numeric, errors="coerce")
    med = X_train.median()  # medians come from training data only, so nothing leaks from the test files
    X_train = X_train.fillna(med).fillna(0)
    X_test = test[used].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0)

    scaler = StandardScaler()  # knn uses distances, so every feature needs the same scale
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, train["label"])
    pred = knn.predict(X_test_s)
    return pred, knn, scaler, med, X_test


def wids_alert(source, rate):
    # only alerts, does not block anything (WIDS, not WIPS)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] Possible deauth attack from {} ({:.0f} frames/sec)".format(stamp, source, rate)
    print("[ALERT]", line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def replay(test, X_test, knn, scaler, count=20):
    # test frames are treated like live traffic and checked one by one
    # (it is a replay of dataset frames, not a live capture)
    print("\n--- WIDS replay on held-out test frames ---")
    open(log_path, "w").close()  # clears the log so it only holds this run
    alerts = 0
    for i in test.sample(count, random_state=42).index:
        row = test.loc[i]
        source = "frame #" + str(i)
        for a in addr_cols:
            if a in test.columns and pd.notna(row[a]):
                source = row[a]
                break
        result = knn.predict(scaler.transform(X_test.loc[[i]]))[0]
        if result == 1:
            alerts += 1
            wids_alert(source, row["frames_per_sec"])
        else:
            print("[OK] Traffic from", source, "looks normal.")
    print("\nFrames checked:", count, "| alerts raised:", alerts, "(saved in wids_alerts.log)")


def main():
    data = load_files()
    used = pick_features(data)

    scores = []
    fp_rates = []
    total_cm = np.zeros((2, 2), dtype=int)
    last = None

    print("\n--- File-wise cross validation,", folds, "folds ---")
    splitter = GroupKFold(n_splits=folds)
    for num, (tr, te) in enumerate(splitter.split(data, data["label"], groups=data["file"]), start=1):
        train, test = data.iloc[tr], data.iloc[te]
        pred, knn, scaler, med, X_test = run_fold(train, test, used)

        acc = accuracy_score(test["label"], pred)
        cm = confusion_matrix(test["label"], pred, labels=[0, 1])
        fp = cm[0, 1] / cm[0].sum()  # share of normal frames that wrongly raised an alert
        scores.append(acc)
        fp_rates.append(fp)
        total_cm += cm
        print("Fold {} - test files: {} | accuracy {:.2f}% | false positive rate {:.2f}%".format(
            num, sorted(test["file"].unique()), acc * 100, fp * 100))
        last = (test, X_test, knn, scaler)

    print("\nAccuracy: {:.2f}% +/- {:.2f}%".format(np.mean(scores) * 100, np.std(scores) * 100))
    print("False positive rate: {:.2f}% +/- {:.2f}%".format(np.mean(fp_rates) * 100, np.std(fp_rates) * 100))
    print("\nConfusion matrix (all folds added together):")
    print(total_cm)

    ConfusionMatrixDisplay(total_cm, display_labels=["Normal", "Deauth Attack"]).plot(cmap="Blues")
    plt.title("KNN Confusion Matrix (AWID3, file-wise split)")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print("Confusion matrix saved as confusion_matrix.png")

    # classification report on the last fold only, as an example of precision and recall
    test, X_test, knn, scaler = last
    print("\nClassification report (last fold):")
    print(classification_report(test["label"], knn.predict(scaler.transform(X_test)),
                                target_names=["Normal", "Deauth Attack"]))

    replay(test, X_test, knn, scaler)


if __name__ == "__main__":
    main()