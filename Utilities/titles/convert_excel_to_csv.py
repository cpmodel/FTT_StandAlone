# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:07:01 2025

@author: Femke Nijsse
"""

import pandas as pd

excel_path = "classification_titles.xlsx"
output_csv = "classification_titles.csv"
cover_sheet = "Cover"

# Load cover sheet with descriptions (assumes short names in col B, descriptions in col D, no header)
cover = pd.read_excel(excel_path, sheet_name=cover_sheet, usecols="B,D", header=None)
desc_map = dict(zip(cover.iloc[:, 0], cover.iloc[:, 1]))

xls = pd.ExcelFile(excel_path)
rows = []

# Add fallback descriptions
fallback_descriptions = {
    "FTTI": "FTT:Fr vehicle classification",
    "C6TI": "FTT:Fr cost components",
    "ITTI": "FTT:IH technology classification",
    "CTTI": "FTT:IH cost components"
}
desc_map.update(fallback_descriptions)

for sheet in xls.sheet_names:
    if sheet == cover_sheet:
        continue  # skip cover sheet
    
    df = xls.parse(sheet, keep_default_na=False)
    full_names = df.iloc[:, 0].tolist()
    short_names = df.iloc[:, 1].tolist()
    
    description = desc_map.get(sheet, "")
    
    row_full = [sheet, description, "", "", "Full name"] + [str(val) for val in full_names]
    row_short = [sheet, description, "", "", "Short name"] + [str(val) for val in short_names]
    
    rows.append(row_full)
    rows.append(row_short)

combined = pd.DataFrame(rows)
combined.to_csv(output_csv, index=False, header=False)