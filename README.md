# 🛡️ Credit Card Fraud Detection Pipeline

Just wrapped up a comprehensive machine learning project focused on detecting fraudulent credit card transactions. Using real-world data with severe class imbalance (only 0.17% fraud), I built an end-to-end pipeline that identifies 88.78% of fraudulent transactions while reducing false alarms by 51% compared to baseline models.

---

## 📋 Project Overview

**The Challenge:** Credit card fraud is rare but costly. A naive model could achieve 99.83% accuracy by simply flagging all transactions as legitimate—but would catch zero frauds. The real challenge is building a system that detects fraud effectively while minimizing false positives that annoy customers.

**The Solution:** A complete data science pipeline from raw data exploration through model deployment, featuring:
- Systematic handling of severe class imbalance (0.17% fraud rate)
- Feature scaling to prevent bias from different magnitude features
- Multiple model comparison (Logistic Regression vs Random Forest)
- Threshold optimization to balance fraud detection vs false alarms
- Business impact analysis to justify model selection

---

## 🛠️ Workflow Overview

### 1. **Explore** – Understand the Data & Problem
- Loaded 284,807 credit card transactions spanning 2 days
- Discovered severe class imbalance: 492 frauds (0.17%) vs 284,315 legitimate (99.83%)
- Analyzed fraud vs legitimate patterns:
  - Median fraud amount: $9.25 (fraudsters use small amounts to avoid detection)
  - Median legitimate amount: $22
  - No obvious temporal patterns (fraud happens at all times)
- Key insight: Amount alone can't distinguish fraud—need all 30 features

**Tools:** Python (pandas, NumPy), Jupyter Notebook

---

### 2. **Prepare** – Clean & Transform Data
- Verified data quality: zero missing values, zero duplicates
- Split data: 80% training (227,845), 20% testing (56,962)
  - Used stratified split to maintain 0.17% fraud ratio in both sets
- Applied feature scaling (StandardScaler):
  - Normalized Amount (0-25,691) and Time (0-172,792) to match V1-V28 scale
  - Prevented large features from dominating model learning
  - Fit on training data only to avoid data leakage
- Handled class imbalance with undersampling:
  - Balanced training data: 394 frauds, 394 legitimate (50/50)
  - Kept test data imbalanced (0.17%) for realistic evaluation

**Why these steps matter:**
- Stratified split ensures fair evaluation
- Feature scaling prevents bias toward high-magnitude features
- Balancing forces model to learn fraud patterns (not just predict majority class)

**Tools:** scikit-learn (StandardScaler, train_test_split), imbalanced-learn (RandomUnderSampler)

---

### 3. **Model** – Train & Compare Multiple Approaches

#### Baseline: Logistic Regression
- Trained on balanced data (788 transactions)
- Default threshold (0.5): 91.84% recall, 2,258 false positives
- Optimized threshold (0.7): 91.84% recall, 1,378 false positives
  - 39% reduction in false alarms with same fraud detection!

#### Advanced: Random Forest
- Ensemble of 100 decision trees
- Default threshold (0.5): 91.84% recall, 2,036 false positives
- Optimized threshold (0.7): 88.78% recall, 668 false positives
  - **51% reduction in false alarms vs Logistic Regression**
  - Only 3% recall drop (90 frauds caught vs 87)

**Model Selection Decision:**
- Random Forest at threshold 0.7 chosen as final model
- Business impact analysis showed $2,720 savings per 56,962 transactions
- Better customer experience (710 fewer false alarms) worth the 3 missed frauds

**Tools:** scikit-learn (LogisticRegression, RandomForestClassifier)

---

### 4. **Evaluate** – Measure Performance & Business Impact

#### Key Metrics (Random Forest @ 0.7)
- **Recall:** 88.78% (catches 87 out of 98 frauds)
- **Precision:** 11.52% (1 in 9 predictions is correct)
- **False Positives:** 668 (51% fewer than Logistic Regression)
- **False Negatives:** 11 (missed frauds)
- **Accuracy:** 98.73%

#### Why These Metrics Matter
- **Recall** = Priority #1 (missing fraud = $750 loss + customer trust)
- **Precision** = Secondary (false alarm = $7 customer service cost)
- **Accuracy** = Misleading (99.83% accuracy by predicting "all legit" = useless)

#### Confusion Matrix
```
                Predicted
            Legit    Fraud
Actual Legit  56,196   668  (56,864 legitimate)
       Fraud     11    87   (98 frauds)
```

#### ROC & Precision-Recall Curves
- Visualized model performance across all thresholds
- ROC-AUC score: 0.97+ (excellent fraud detection)
- Demonstrated superiority of Random Forest over baseline

**Tools:** scikit-learn (confusion_matrix, classification_report, roc_curve, precision_recall_curve), matplotlib

---

### 5. **Deploy** – Save Model for Production Use
- Serialized Random Forest model (pickle format)
- Saved StandardScaler (critical for preprocessing new data)
- Created deployment guide for using model on new transactions
- Documented limitations and retraining requirements

**Tools:** pickle, Python standard library

---

## ⚙️ Technologies Used

### Core Libraries
- **pandas** – Data manipulation and analysis
- **NumPy** – Numerical operations
- **scikit-learn** – Machine learning models, preprocessing, evaluation
- **imbalanced-learn** – Handling class imbalance (undersampling)
- **matplotlib/seaborn** – Data visualization

### Development Environment
- **Python 3.11** – Programming language
- **Jupyter Notebook** – Interactive development and analysis
- **VSCode** – IDE
- **Git/GitHub** – Version control

### Azure-Compatible Tools
- **Designed for Azure Databricks migration** (pandas → PySpark conversion)
- **Azure Blob Storage** ready (bronze/silver/gold architecture)
- **MLflow** compatible for experiment tracking

---

## 📊 Results & Key Achievements

### Model Performance
✅ **88.78% fraud detection rate** (catches 87 out of 98 frauds)  
✅ **51% reduction in false alarms** (668 vs 1,378 from baseline)  
✅ **98.73% overall accuracy** on realistic, imbalanced test data  
✅ **$2,720 cost savings** per 56,962 transactions vs baseline  

### Technical Accomplishments
✅ Built end-to-end ML pipeline from raw data to deployable model  
✅ Handled severe class imbalance (0.17% fraud) using domain-appropriate techniques  
✅ Implemented proper train-test split with stratification to prevent data leakage  
✅ Applied feature scaling with correct fit/transform methodology  
✅ Optimized decision threshold using business impact analysis  
✅ Compared multiple models systematically (Logistic Regression vs Random Forest)  
✅ Created professional visualizations (ROC curve, Precision-Recall curve, confusion matrix)  
✅ Documented entire process for reproducibility and portfolio presentation  

### Business Impact
✅ **Reduced customer friction:** 710 fewer false alarms per 56,962 transactions  
✅ **Maintained fraud protection:** Still catches 88.78% of fraudulent activity  
✅ **Quantified trade-offs:** Clear cost-benefit analysis justifying model choice  
✅ **Production-ready:** Saved model and preprocessing pipeline for deployment  

---

## 🎓 Key Learnings & Insights

### Technical Lessons
1. **Accuracy is misleading for imbalanced data** – A 99.83% "accurate" model that catches 0 frauds is worthless
2. **Class imbalance requires special handling** – Balancing training data forces models to learn minority class patterns
3. **Feature scaling matters** – Without scaling, high-magnitude features (Amount, Time) drown out V1-V28 signals
4. **Threshold tuning is powerful** – Changed one number (0.5→0.7) and reduced false alarms by 39-67% with no retraining
5. **Data leakage prevention is critical** – Must fit scaler on training only, then transform both train/test

### Domain Knowledge
1. **Fraud patterns are subtle** – Median fraud amount ($9.25) is LOWER than legitimate ($22)—fraudsters stay under the radar
2. **Cost asymmetry drives decisions** – Missing fraud ($750) is far more costly than false alarm ($7)
3. **Recall trumps precision** in fraud detection – Better to be cautious than miss real fraud
4. **Real-world deployment needs continuous monitoring** – Fraud patterns change monthly; models must be retrained

### Process Insights
1. **Start simple, iterate** – Logistic Regression baseline → Random Forest improvement
2. **Visualize everything** – ROC/PR curves reveal insights that metrics alone miss
3. **Document your decisions** – Why you chose each technique matters as much as the code
4. **Think like the business** – Translate ML metrics into dollars and customer experience

---

## 📁 Project Structure

```
fraud-detection-pipeline/
│
├── data/
│   ├── raw/                    # Original dataset (not in repo)
│   │   └── .gitkeep
│   ├── processed/              # Cleaned data (generated)
│   └── README.md               # Data download instructions
│
├── notebooks/
│   └── 01_eda_and_exploration.ipynb  # Main analysis notebook
│
├── models/
│   ├── rf_fraud_detector.pkl   # Saved Random Forest model
│   ├── scaler.pkl              # Saved StandardScaler
│   └── .gitkeep
│
├── visualizations/             # Generated plots
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── confusion_matrix.png
│
├── .gitignore                  # Ignore large files
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip (Python package manager)
- Kaggle account (for dataset download)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraud-detection-pipeline.git
   cd fraud-detection-pipeline
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Download `creditcard.csv` (144 MB)
   - Place in: `data/raw/creditcard.csv`

5. **Run the analysis**
   ```bash
   jupyter notebook
   # Open: notebooks/01_eda_and_exploration.ipynb
   # Run all cells
   ```

---

## 📈 Future Enhancements

### Technical Improvements
- [ ] Try SMOTE (oversampling) instead of undersampling to retain all legitimate data
- [ ] Experiment with XGBoost and ensemble methods
- [ ] Feature engineering: time-based features (hour of day, day of week), amount buckets, transaction velocity
- [ ] Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] Implement SHAP values for model interpretability

### Production Readiness
- [ ] Migrate to Azure Databricks (pandas → PySpark for big data)
- [ ] Deploy model as REST API using FastAPI/Flask
- [ ] Set up MLflow for experiment tracking and model registry
- [ ] Implement model monitoring dashboard (detect model drift)
- [ ] Create automated retraining pipeline (monthly/quarterly)
- [ ] Add real-time prediction endpoint with <100ms latency
- [ ] Integrate with Azure ML for cloud deployment

### Business Features
- [ ] Build fraud risk scoring system (0-100 scale)
- [ ] Create alert prioritization (high/medium/low risk)
- [ ] Implement feedback loop (learn from false positives/negatives)
- [ ] A/B test different thresholds in production
- [ ] Build dashboard for fraud analytics team

---

## 🔒 Important Notes

### Model Limitations
⚠️ **This model ONLY works on:**
- Data with the same 30 features (Time, V1-V28, Amount)
- Transactions with similar fraud patterns to training data
- PCA-transformed V1-V28 features (specific to this dataset)

⚠️ **For production fraud detection:**
- Train on YOUR company's actual transaction data
- Include real features (merchant, location, device, card type, etc.)
- Retrain regularly (fraud patterns evolve monthly)
- Monitor model performance continuously
- Implement human-in-the-loop for high-risk transactions

### Data Privacy
- Dataset is anonymized (V1-V28 are PCA components)
- Original features (card numbers, names, etc.) removed for privacy
- This is research/educational data, not production transaction data

---

## 📄 License

This project is for educational and portfolio purposes. 

**Dataset:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**License:** Database Contents License (DbCL) v1.0

---

## 👤 Author

**Your Name**  
Data Engineer | Machine Learning Enthusiast

📧 Email: your.email@example.com  
💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
🐙 GitHub: [github.com/yourusername](https://github.com/yourusername)  
🌐 Portfolio: [yourportfolio.com](https://yourportfolio.com)

---

## 🙏 Acknowledgments

- **Kaggle** – For providing the Credit Card Fraud Detection dataset
- **scikit-learn & imbalanced-learn** – Excellent open-source ML libraries
- **Machine Learning Community** – F