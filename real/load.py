"""Load Uniwork KPI exports by header names; drop teacher names (anonymize)."""
import pandas as pd, numpy as np

def load(path, year):
    raw = pd.read_excel(path, header=None)
    h0 = raw.iloc[0].ffill().astype(str); h1 = raw.iloc[1].ffill().astype(str); h2 = raw.iloc[2].astype(str)
    data = raw.iloc[3:].reset_index(drop=True)
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    out = pd.DataFrame({'year': year,
                        'faculty': data[3].astype(str), 'dept': data[4].astype(str),
                        'rate': num(data[5]), 'total': num(data[2])})
    cats = {}
    for j in range(6, raw.shape[1]):
        if h1[j] == 'Jami' and h2[j] == 'Balli':
            cats[h0[j].strip().replace('\u2018',"'").replace('\u2019',"'")] = num(data[j])
        if h0[j].startswith('Ilmiy daraja/unvon mavjudligi') and h2[j] == 'Soni' and h1[j] != 'Jami':
            out['has_' + h1[j]] = num(data[j]) > 0
    cat = pd.DataFrame(cats)
    out = pd.concat([out, cat], axis=1)
    out['tid'] = ['%d-%04d' % (year, i) for i in range(len(out))]   # anonymous id
    return out, list(cats)
