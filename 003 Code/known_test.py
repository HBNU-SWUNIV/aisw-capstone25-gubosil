import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
import sys

# 경고 무시
warnings.filterwarnings('ignore')

def main():
    print("START: NIDS Binary SSL + Finetune (target: label 0/1)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---------- load ----------
    try:
        train_df = pd.read_csv('1_train_dataset.csv')
        test_df  = pd.read_csv('2_test_known_dataset.csv')
    except FileNotFoundError:
        print("[오류] 데이터셋 파일(1_train_dataset.csv, 2_test_known_dataset.csv)을 찾을 수 없습니다.")
        sys.exit()

    if 'attack_cat' not in train_df.columns or 'attack_cat' not in test_df.columns:
         raise ValueError("데이터에 'attack_cat' 컬럼이 없습니다. 'Normal'과 공격을 구분해야 합니다.")

    # ---------- 'label' 컬럼 생성 (0=Normal, 1=Anomaly) ----------
    BENIGN_LABEL_NAME = 'Normal'
    print(f"Creating binary 'label' column... (0 = '{BENIGN_LABEL_NAME}', 1 = Anomaly)")
    train_df['label'] = (train_df['attack_cat'] != BENIGN_LABEL_NAME).astype(int)
    test_df['label']  = (test_df['attack_cat'] != BENIGN_LABEL_NAME).astype(int)
    print("'label' column created.")

    # ---------- numeric features only ----------
    drop_cols = [c for c in ['attack_cat','label','id'] if c in train_df.columns]
    X_train_all = train_df.drop(columns=drop_cols).select_dtypes(include=[np.number]).astype(np.float32)
    X_test_all  = test_df.drop(columns=drop_cols).select_dtypes(include=[np.number]).astype(np.float32)

    # ---------- encode labels ----------
    print("Using binary 'label' column (0=Normal, 1=Anomaly)")
    y_train_label = train_df['label'].values.astype(np.int64)
    y_test_label  = test_df['label'].values.astype(np.int64)

    num_classes = 2
    print(f"Binary classification. num_classes = {num_classes}")

    print("Train size:", len(X_train_all), "Test size:", len(X_test_all))
    print("Label distribution in train (0=Normal, 1=Anomaly):\n", pd.Series(y_train_label).value_counts())

    # ---------- SSL pretraining ----------
    X_pretrain = X_train_all.copy()
    
    if X_pretrain.shape[1] <= 0:
        raise ValueError(f"입력 특징의 개수(input_dim)가 0입니다. CSV 파일을 확인하세요. (shape: {X_pretrain.shape})")

    pretrain_ds = TensorDataset(torch.from_numpy(X_pretrain.values).float())
    pretrain_loader = DataLoader(pretrain_ds, batch_size=256, shuffle=True, drop_last=True)

    class Encoder(nn.Module):
        def __init__(self, input_dim, latent=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256), nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, latent)
            )
        def forward(self, x):
            return self.net(x)

    latent_dim = 128
    lr_enc = 0.0004849231537959607
    lr_clf = 0.009552483274165705
    ft_batch_size = 128
    augment_noise = 0.02037440253989113

    enc = Encoder(X_pretrain.shape[1], latent=latent_dim).to(device)
    clf = nn.Linear(latent_dim, num_classes).to(device)

    def nt_xent(z1, z2, temperature=0.1):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T)
        B = z1.size(0)
        mask = (~torch.eye(2*B, dtype=torch.bool)).to(z.device)
        exp_sim = torch.exp(sim/temperature) * mask.float()
        pos = torch.cat([
            torch.exp((z1*z2).sum(dim=1)/temperature),
            torch.exp((z2*z1).sum(dim=1)/temperature)
        ], dim=0)
        denom = exp_sim.sum(dim=1)
        loss = -torch.log(pos / (denom + 1e-12))
        return loss.mean()

    def augment(x):
        x = x + augment_noise * torch.randn_like(x)
        drop = (torch.rand_like(x) > 0.95).float()
        return x * (1 - drop)

    opt_pre = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-6)
    epochs_pre = 40
    
    print("Start pretraining (SimCLR / NT-Xent)...")
    for ep in range(epochs_pre):
        enc.train()
        running = 0.0; it=0
        for (xb,) in pretrain_loader:
            xb = xb.to(device)
            x1, x2 = augment(xb), augment(xb)
            z1, z2 = enc(x1), enc(x2)
            loss = nt_xent(z1, z2)
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()
            running += loss.item(); it += 1
        print(f"Pretrain {ep+1}/{epochs_pre}, avg loss {running/it:.6f}")
    print("Pretraining done.")

    # ---------- Finetune ----------
    ft_fraction = 0.3
    min_per_class = 200
    df = train_df.copy()
    selected_idx = []

    print(f"Finetuning: {min_per_class}개 또는 {ft_fraction*100}% 샘플링 (Normal/Anomaly 각각)")
    for cls in [0, 1]:
        idxs = df.index[df['label'] == cls].tolist()
        if not idxs:
             print(f"[경고] Finetune: 클래스 {cls}에 대한 샘플이 없습니다. 건너뜁니다.")
             continue

        n = max(int(len(idxs)*ft_fraction), min_per_class)
        if n > len(idxs):
            print(f"[알림] Finetune: 클래스 {cls} 샘플 부족 ({len(idxs)}개). {n}개로 복원 추출(replace=True)합니다.")
            replace = True
        else:
            replace = False

        chosen = np.random.choice(idxs, n, replace=replace).tolist()
        selected_idx.extend(chosen)

    if not selected_idx:
        print("[오류] Finetune용 훈련 데이터가 없습니다. CSV 파일의 'label'을 확인하세요.")
        sys.exit()

    ft_df = df.loc[selected_idx].reset_index(drop=True)
    X_ft = ft_df.drop(columns=drop_cols).select_dtypes(include=[np.number]).astype(np.float32)
    y_ft_label = ft_df['label'].values.astype(np.int64)

    print(f"Finetune 훈련 샘플 개수: {len(y_ft_label)}")
    print(f"Finetune 레이블 분포:\n{pd.Series(y_ft_label).value_counts()}")

    finetune_ds = TensorDataset(torch.from_numpy(X_ft.values).float(), torch.from_numpy(y_ft_label).long())
    finetune_loader = DataLoader(finetune_ds, batch_size=ft_batch_size, shuffle=True)

    for p in enc.parameters():
        p.requires_grad = True

    opt_ft = torch.optim.Adam([
        {'params': enc.parameters(), 'lr': lr_enc},
        {'params': clf.parameters(), 'lr': lr_clf}
    ], weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    print("Start finetuning binary classifier...")
    epochs_ft = 160
    for ep in range(epochs_ft):
        enc.train(); clf.train()
        running = 0.0; it=0
        for xb, yb in finetune_loader:
            xb, yb = xb.to(device), yb.to(device)
            z = enc(xb)
            logits = clf(z)
            loss = criterion(logits, yb)
            opt_ft.zero_grad(); loss.backward(); opt_ft.step()
            running += loss.item(); it += 1
        if (ep+1) % 5 == 0 or ep==0:
            print(f"Finetune {ep+1}/{epochs_ft}, avg loss {running/it:.6f}")
    print("Finetuning done.")

    # ---------- Evaluation ----------
    enc.eval(); clf.eval()
    test_ds = TensorDataset(torch.from_numpy(X_test_all.values).float())
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    pred_labels = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            z = enc(xb)
            logits = clf(z)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().numpy())
    print("\nBinary Classification Report (0=Normal, 1=Anomaly):")
    target_names = ['Normal (0)', 'Anomaly (1)']
    print(classification_report(y_test_label, pred_labels, target_names=target_names, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_label, pred_labels))

    print("\nSaving models...")
    torch.save(enc.state_dict(), "enc.pth")
    torch.save(clf.state_dict(), "clf.pth")
    print("Models saved: 'enc.pth', 'clf.pth'")

if __name__ == "__main__":
    main()