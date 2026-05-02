# Applied Data Science Project Requirements Checklist

## 1. Project Overview & General Constraints
- [x] Define a clear business problem and stakeholder [cite: 4, 32].
- [x] Acquire and integrate data from at least **TWO** reliable sources [cite: 5, 23].
- [x] Ensure data is real-world, not synthetic (except for oversampling like SMOTE on training split) [cite: 23, 25, 26].
- [x] Dataset size: &ge; 5,000 rows [cite: 20].
- [x] Dataset features: &ge; 10 features [cite: 21].
- [x] Target variable: Clearly defined classification target [cite: 22, 49].
- [x] Follow software engineering practices: clean code, modular structure, logging [cite: 11, 111, 112].

## 2. Team & Administrative Requirements
- [x] Form a team of 4 students [cite: 15].
- [x] Include assigned team number in all submissions [cite: 16].
- [ ] Document individual member contributions in the final report [cite: 17, 105].

## 3. Phase 1: Proposal & Dataset Validation (25%)
- [x] **Proposal Report (PDF)** [cite: 31]:
    - [x] Problem statement with justification for classification and stakeholder impact [cite: 32].
    - [x] Dataset documentation: names, URLs/citations, feature descriptions [cite: 33].
    - [x] Feature engineering plans and class distribution [cite: 33].
    - [x] **Full Data Validation Report**: row/column counts, missing values, duplicates, data types, and quality issues [cite: 34, 49].
    - [x] Cover all validation dimensions discussed in the course [cite: 35].

## 4. Phase 2: Final Delivery (75%)
- [ ] **Submission Package**: One zip file containing final PDF report, runnable code, and Git link [cite: 38].
- [x] **Repository Requirements**: Code, `pyproject.toml`, `poetry.lock`, and `README.md` [cite: 39].

### 4.1 Data Acquisition & Preprocessing
- [ ] Snapshot of acquisition pipeline in report [cite: 44].
- [ ] Document merge strategy between sources with row counts before/after [cite: 46].
- [ ] Cleaning steps with justification and outlier handling [cite: 53, 54].
- [ ] Feature transformation (encoding, normalization) with rationale [cite: 55].
- [ ] Feature selection and engineering details [cite: 56, 57].
- [x] Train/validation/test split procedure and data balancing strategy [cite: 58, 59].

### 4.2 Exploratory Data Analysis (EDA)
- [x] At least five major meaningful visualizations with interpretation [cite: 62].
- [x] Feature-to-target analysis [cite: 63].
- [ ] Insights influencing modeling decisions [cite: 64].
- [ ] Dashboard featuring EDA, model comparisons, and business insights [cite: 65, 124].

### 4.3 Model Development & Experiment Tracking
- [x] Build at least **five classification models**, including one baseline [cite: 8, 67].
- [ ] Provide rationale for model selection [cite: 68].
- [ ] MLflow Integration:
    - [x] Log all runs (metrics, hyperparameters, artifacts, versions) [cite: 9, 70, 71].
    - [x] Log at least 2 standard metrics and 2 business-related metrics per run [cite: 72].
    - [x] Save trained models as MLflow artifacts [cite: 73].
    - [ ] Include screenshot of MLflow experiment comparison in report [cite: 74].

### 4.4 Testing & Automation
- [ ] Develop unit tests for critical functions and modules [cite: 77].
- [ ] Implement integration tests for component interaction and data flow [cite: 80].
- [ ] Report test coverage statistics and results [cite: 81, 82].
- [ ] **Code Automation**:
    - [ ] Use **Makefiles** for preprocessing, training, evaluation, and testing [cite: 92, 115].
    - [x] Use `.env` files for secure configuration (API keys, credentials) [cite: 93, 115].
- [ ] **Continuous Integration (CI)**:
    - [x] Implement **GitHub Actions** for build, test (unit/integration), and linting [cite: 12, 98, 99, 100].

### 4.5 Results, Evaluation & Documentation
- [ ] Model performance on training vs. test data (overfitting check) [cite: 84].
- [ ] Model comparison table or chart [cite: 86].
- [ ] Error analysis of the best model [cite: 87].
- [ ] Business-oriented interpretation of results [cite: 88].
- [ ] Discussion of unsuccessful approaches [cite: 90].
- [ ] Honest discussion of limitations and future work [cite: 102, 103].
- [x] **README**: Team names, project description, setup/run instructions [cite: 113].

## 5. Bonus Opportunities (10%)
- [ ] Deploy model to a public endpoint (5%) [cite: 124].
- [ ] Deliver an interactive dashboard (Streamlit, Power BI, etc.) for stakeholders (5%) [cite: 124].

## 6. Technical Stack Requirements
- [x] Language: **Python** [cite: 107].
- [x] Dependency Management: **Poetry** [cite: 11, 108].
- [x] Version Control: **Git** [cite: 11, 109].
- [x] Structure: **Modular / Cookie-Cutter** folder structure [cite: 111].
