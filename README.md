# superai-ss6-mh4.4-heart-disease-prediction

# ❤️ MH4.4 — Heart Disease Prediction

**Super AI Engineer Season 6 | Mini Hackathon 4.4**

Binary classification สำหรับทำนายความเสี่ยงโรคหัวใจ
ด้วย 3-model ensemble + Optuna tuning + OOF threshold optimization

---

## 📊 Result

| Metric | Value |
|--------|-------|
| **Score (F2)** | **~0.540** |
| Rank | 214 |
| **Status** | ✅ Passed baseline |

> **หมายเหตุ:** Metric คือ **F2-score** (recall-weighted) — เหมาะกับ medical use case
> ที่ False Negative มีต้นทุนสูงกว่า False Positive

---

## 🔧 Tech Stack

`Python` · `LightGBM` · `XGBoost` · `CatBoost` · `Optuna` · `scikit-learn` · `Google Colab`

---

## 🏗️ Approach

### Pipeline Overview
```
Data → Preprocessing
    → 5-Fold Stratified CV
    → 3 Models (LGB / XGB / CatBoost) tuned with Optuna
    → OOF Predictions
    → Ensemble Weight Search (grid search)
    → Threshold Optimization (maximize F2 on OOF)
    → Pseudo-labeling on test set
    → Final Submission
```

### Ensemble Strategy
```python
# Grid search ของ weights สำหรับ 3 models
for w_lgb, w_xgb, w_cat in weight_grid:
    oof_blend = w_lgb*oof_lgb + w_xgb*oof_xgb + w_cat*oof_cat
    f2 = fbeta_score(y_train, oof_blend > threshold, beta=2)
    # เก็บ weights ที่ให้ F2 สูงสุด
```

### Threshold Optimization
```python
# ปรับ threshold แทนที่จะใช้ 0.5 default
# เพราะ F2-score ให้น้ำหนัก recall มากกว่า precision
for threshold in np.arange(0.1, 0.9, 0.01):
    f2 = fbeta_score(y_true, proba > threshold, beta=2)
```

### Pseudo-labeling
```python
# รอบที่ 1: train บน train set → predict test
# เลือก high-confidence test samples (proba > 0.85 หรือ < 0.15)
# รอบที่ 2: train ใหม่บน train + pseudo-labeled test
```

---

## 💡 Key Learnings

- **Threshold tuning** สำคัญมากสำหรับ F2-score — อย่าใช้ 0.5 default
- **OOF vs Public LB** ต้องติดตามทั้งคู่เพื่อ detect overfitting
- **เปลี่ยนทีละตัวแปร** — การเปลี่ยนหลายอย่างพร้อมกันทำให้ debug ยากมาก
- **Pseudo-labeling** ต้องระวัง confidence threshold — ต่ำเกินไปทำให้ noise เข้าระบบ

---

## 📁 Files

```
mh4.4-colab.py
```

---
