import pyreadstat
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import os

# 1. 데이터 폴더 경로
folder_path = r"C:\Users\knw19\OneDrive\바탕 화면\HN13-23"

# 2. 연도별 파일 불러오기
years = range(2013, 2024)  # 2013 ~ 2023
dfs = []

for y in years:
    file_name = f"HN{str(y)[-2:]}_ALL.sav"
    file_path = os.path.join(folder_path, file_name)

    df_year, meta = pyreadstat.read_sav(file_path)
    df_year["year"] = y  # 연도 변수 추가
    dfs.append(df_year)

# 3. 하나로 합치기
df = pd.concat(dfs, ignore_index=True)

# 4. 그룹 나누기
group1 = df[(df['year'] >= 2013) & (df['year'] <= 2018)]
group2 = df[(df['year'] >= 2019) & (df['year'] <= 2023)]

results = []

# 5. 변수별 가설검정
for col in df.columns:
    if col == 'year':
        continue

    g1 = group1[col].dropna()
    g2 = group2[col].dropna()

    if g1.empty or g2.empty:
        continue

    if np.issubdtype(df[col].dtype, np.number):
        stat, p = ttest_ind(g1, g2, equal_var=False)
        test_type = "t-test"
    else:
        contingency_table = pd.crosstab(group1[col], group2[col])
        try:
            stat, p, _, _ = chi2_contingency(contingency_table)
        except ValueError:
            stat, p = np.nan, np.nan
        test_type = "Chi-square"

    results.append({
        "Variable": col,
        "Test": test_type,
        "Statistic": stat,
        "p-value": p
    })

# 6. 결과 저장
result_df = pd.DataFrame(results)
result_df.to_csv("all_variables_test_results.csv", index=False)

# 7. 유의한 변수만 저장 (p<0.05)
significant_df = result_df[result_df["p-value"] < 0.05]
significant_df.to_csv("significant_variables.csv", index=False)

print("검정 완료! 결과 파일이 저장되었습니다.")
