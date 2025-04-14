import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import lognorm
import pickle

# Stap 1: Laad en voorbewerk de data
conn = sqlite3.connect("data/clean/2020.db")
conn.text_factory = lambda x: x.decode(errors='ignore')
query = conn.execute("SELECT * FROM wide_blood_draws")
cols = [column[0] for column in query.description]
df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
conn.close()

# Selecteer alleen labwaarden
lab_columns = ['AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC', 'NATSE', 'TRISE', 
               'ALBSE', 'ASTSE', 'CRPSE', 'TRC']
data = df[lab_columns]

# Vervang tekst en NaN door -1
data = data.apply(pd.to_numeric, errors='coerce').fillna(-1)
data_values = data.values

# Stap 2: Fit log-normale distributies per labtest
n_features = len(lab_columns)
dist_params = []  # Lijst van (shape, loc, scale) per labtest
for i in range(n_features):
    valid_data = data_values[data_values[:, i] != -1, i]
    if len(valid_data) > 1:
        # Voeg kleine constante toe om negatieve/0 waarden te vermijden
        valid_data = valid_data + 1e-10
        shape, loc, scale = lognorm.fit(valid_data, floc=0)  # loc=0 voor eenvoud
        dist_params.append((shape, loc, scale))
    else:
        dist_params.append(None)
        print(f"Waarschuwing: Geen geldige data voor {lab_columns[i]}")

# Sla distributieparameters op
with open('out/model/lognorm_dist_params.pkl', 'wb') as f:
    pickle.dump(dist_params, f)
print("Distributieparameters opgeslagen als 'lognorm_dist_params.pkl'")

# Stap 3: Bereken correlatiematrix
corr_matrix = np.zeros((n_features, n_features))
for i in range(n_features):
    for j in range(i + 1, n_features):
        # Selecteer samples waar beide tests niet -1 zijn
        mask = (data_values[:, i] != -1) & (data_values[:, j] != -1)
        if np.sum(mask) > 1:
            corr = np.corrcoef(data_values[mask, i], data_values[mask, j])[0, 1]
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
        else:
            corr_matrix[i, j] = 0
            corr_matrix[j, i] = 0

# Sla correlatiematrix op
with open('out/model/corr_matrix.pkl', 'wb') as f:
    pickle.dump(corr_matrix, f)
print("Correlatiematrix opgeslagen als 'corr_matrix.pkl'")

# Stap 4: Functie voor waarschijnlijkheidsscore per labtest
def get_probability_score_per_test(data_new, dist_params, corr_matrix, lab_columns, corr_threshold=0.3):
    """
    Bereken een waarschijnlijkheidsscore per labtest gebaseerd op log-normale distributies en correlaties.
    
    Parameters:
    - data_new: numpy array van vorm (n_samples, n_features) met -1 voor ontbrekende waarden
    - dist_params: lijst van (shape, loc, scale) per labtest
    - corr_matrix: correlatiematrix tussen labtests
    - lab_columns: lijst van labtestnamen
    - corr_threshold: minimale absolute correlatie om mee te nemen
    
    Returns:
    - probability_scores: numpy array van vorm (n_samples, n_features) met waarschijnlijkheden
    """
    n_samples, n_features = data_new.shape
    probability_scores = np.zeros_like(data_new, dtype=float)
    
    for i in range(n_features):
        if dist_params[i] is None:
            probability_scores[:, i] = np.nan
            continue
        
        shape, loc, scale = dist_params[i]
        valid_mask = data_new[:, i] != -1
        if np.sum(valid_mask) == 0:
            probability_scores[:, i] = np.nan
            continue
        
        # Basiswaarschijnlijkheid uit log-normale verdeling
        values = data_new[valid_mask, i] + 1e-10
        lognorm_prob = lognorm.cdf(values, s=shape, loc=loc, scale=scale)
        base_scores = 1 - lognorm_prob  # Hoge score voor waarden dicht bij mediaan
        
        # Aanpassing op basis van correlaties
        corr_scores = np.ones(np.sum(valid_mask))
        for j in range(n_features):
            if i == j or dist_params[j] is None:
                continue
            if abs(corr_matrix[i, j]) > corr_threshold:
                # Check consistentie met labtest j
                valid_pair_mask = (data_new[:, i] != -1) & (data_new[:, j] != -1)
                if np.sum(valid_pair_mask) == 0:
                    continue
                shape_j, loc_j, scale_j = dist_params[j]
                values_j = data_new[valid_pair_mask, j] + 1e-10
                prob_j = lognorm.cdf(values_j, s=shape_j, loc=loc_j, scale=scale_j)
                # Als waarde j onwaarschijnlijk is, verlaag de score voor test i
                corr_weight = abs(corr_matrix[i, j]) / np.sum(abs(corr_matrix[i, corr_matrix[i] > corr_threshold]))
                corr_scores[valid_pair_mask[valid_mask]] *= (1 - prob_j) ** corr_weight
        
        # Combineer basis- en correlatiescores
        probability_scores[valid_mask, i] = base_scores * corr_scores
        probability_scores[~valid_mask, i] = np.nan
    
    probability_scores = np.clip(probability_scores, 0, 1)
    return probability_scores

# Stap 5: Test het model
# Originele data
prob_scores = get_probability_score_per_test(data_values, dist_params, corr_matrix, lab_columns)
print("Waarschijnlijkheidsscores voor sample 0:")
for i, test in enumerate(lab_columns):
    print(f"{test}: {prob_scores[0, i]:.3f}" if not np.isnan(prob_scores[0, i]) else f"{test}: ontbrekend")

# Test met NATSE = 1400
abnormal_data = data_values.copy()
abnormal_data[0, lab_columns.index('NATSE')] = 1400
prob_scores_abnormal = get_probability_score_per_test(abnormal_data, dist_params, corr_matrix, lab_columns)
print("\nWaarschijnlijkheidsscores voor sample 0 (NATSE = 1400):")
for i, test in enumerate(lab_columns):
    print(f"{test}: {prob_scores_abnormal[0, i]:.3f}" if not np.isnan(prob_scores_abnormal[0, i]) else f"{test}: ontbrekend")

# Stap 6: Inspecteer correlaties
print("\nSignificante correlaties (|r| > 0.3):")
for i in range(n_features):
    for j in range(i + 1, n_features):
        if abs(corr_matrix[i, j]) > 0.3:
            print(f"{lab_columns[i]} - {lab_columns[j]}: {corr_matrix[i, j]:.3f}")