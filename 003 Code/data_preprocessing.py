import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import sys
import os

warnings.filterwarnings('ignore')

FILE_TRAIN = "UNSW_NB15_training-set.csv"
FILE_TEST  = "UNSW_NB15_testing-set.csv"

def main():
    print("START: Data Preprocessing (Concat Train+Test) & 4 Dataset Generation")

    if not os.path.exists(FILE_TRAIN) or not os.path.exists(FILE_TEST):
        print(f"[오류] 파일이 없습니다.")
        print(f"현재 경로에 '{FILE_TRAIN}' 과 '{FILE_TEST}' 파일이 있는지 확인하세요.")
        sys.exit()

    print(f"Loading '{FILE_TRAIN}' and '{FILE_TEST}'...")
    try:
        unsw_train = pd.read_csv(FILE_TRAIN)
        unsw_test  = pd.read_csv(FILE_TEST)
        
        print(f" - Train set shape: {unsw_train.shape}")
        print(f" - Test set shape:  {unsw_test.shape}")
        df = pd.concat([unsw_train, unsw_test], ignore_index=True)
        print(f"Combined df shape: {df.shape}")

    except Exception as e:
        print(f"[오류] 파일을 읽거나 합치는 중 에러 발생: {e}")
        sys.exit()

    df.columns = df.columns.str.strip()
    df = df.drop(['id'], axis=1, errors='ignore')

    # 공격 타입 정의
    known_attack_types = ['Fuzzers', 'Analysis', 'Backdoors', 'DoS', 'Reconnaissance', 'Generic']
    unknown_attack_types = ['Exploits', 'Shellcode', 'Worms']
    if 'attack_cat' not in df.columns:
        print("[오류] 데이터에 'attack_cat' 컬럼이 없습니다.")
        sys.exit()
    
    print("Feature Engineering started...")
    processed_df = df.copy()
    X_features = processed_df.drop(['label', 'attack_cat'], axis=1)
    categorical_features = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'proto' in categorical_features:
        print("'proto' feature found: Applying Label Encoding.")
        le = LabelEncoder()
        X_features['proto'] = le.fit_transform(X_features['proto'])
        categorical_features.remove('proto')
    if categorical_features:
        print(f"Applying One-Hot Encoding to: {categorical_features}")
        X_processed = pd.get_dummies(X_features, columns=categorical_features, drop_first=True)
        X_processed = X_processed.astype(float)
    else:
        X_processed = X_features.astype(float)

    processed_df = pd.concat([X_processed, processed_df[['label', 'attack_cat']]], axis=1)
    numerical_features = X_processed.select_dtypes(include=np.number).columns.tolist()
    skewed_feats = processed_df[numerical_features].skew().loc[lambda x: abs(x) > 2].index
    if not skewed_feats.empty:
        print(f"Applying Log transformation to skewed features (skew > 2): {len(skewed_feats)} features")
        for feat in skewed_feats:
            processed_df[feat] = np.log1p(processed_df[feat])
    else:
        print("No skewed features found.")

    print("Splitting data into Normal / Known Attacks / Unknown Attacks...")

    processed_df['attack_cat'] = processed_df['attack_cat'].astype(str).str.strip()
    
    normal_df = processed_df[processed_df['attack_cat'] == 'Normal']
    known_df = processed_df[processed_df['attack_cat'].isin(known_attack_types)]
    unknown_df = processed_df[processed_df['attack_cat'].isin(unknown_attack_types)]

    if len(normal_df) == 0:
        print("[경고] 'Normal' 데이터가 0개입니다. attack_cat 컬럼 값을 확인하세요.")

    normal_train, normal_test = train_test_split(normal_df, test_size=0.3, random_state=42)
    known_train, known_test = train_test_split(known_df, test_size=0.3, random_state=42)
    unknown_test = unknown_df 

    train_df = pd.concat([normal_train, known_train]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_known_df = pd.concat([normal_test, known_test]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_zeroday_df = pd.concat([normal_test, unknown_test]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_all_df = pd.concat([normal_test, known_test, unknown_test]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Standardizing features...")
    features_to_scale = X_processed.columns.tolist()
    
    scaler = StandardScaler()
    scaler.fit(train_df[features_to_scale])

    train_df[features_to_scale] = scaler.transform(train_df[features_to_scale])
    test_known_df[features_to_scale] = scaler.transform(test_known_df[features_to_scale])
    test_zeroday_df[features_to_scale] = scaler.transform(test_zeroday_df[features_to_scale])
    test_all_df[features_to_scale] = scaler.transform(test_all_df[features_to_scale])

    print("Standardization Done.")

    print("Saving CSV files...")
    train_df.to_csv('1_train_dataset.csv', index=False)
    test_known_df.to_csv('2_test_known_dataset.csv', index=False)
    test_zeroday_df.to_csv('3_test_zeroday_dataset.csv', index=False)
    test_all_df.to_csv('4_test_all_dataset.csv', index=False)

    print("\nDataset generation completed successfully.")
    print(f"1. train_dataset.csv: {train_df.shape}")
    print(f"2. test_known_dataset.csv: {test_known_df.shape}")
    print(f"3. test_zeroday_dataset.csv: {test_zeroday_df.shape}")
    print(f"4. test_all_dataset.csv: {test_all_df.shape}")


if __name__ == "__main__":
    main()