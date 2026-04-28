# FairAI – Bias Detection and Correction Platform

Welcome to the ultimate guide for your Hackathon Project, **FairAI**! This documentation covers everything you need: from concepts to exact setup steps, the technical architecture, and crucial tips for a winning pitch.

---

## 1. Project Overview
**FairAI** is a comprehensive tool designed to tackle Algorithmic Bias. Many real-world machine learning models inadvertently learn prejudice because the historical data they are trained on is biased (e.g., favoring one gender or race). 

This platform allows users to:
1. **Upload Tabular Data**: Load real-world or synthetic datasets.
2. **Detect Imbalances**: Visually identify skewness across sensitive attributes.
3. **Train & Evaluate**: Build a standard Machine Learning model (Logistic Regression) and test it for fairness metrics like Demographic Parity and Selection Rate.
4. **Mitigate Bias (Correction)**: Blind the model to sensitive features to provide a corrected, fairer prediction process.
5. **Explain Decisions**: See exactly which features the mitigated model relies on to make choices.

---

## 2. Architecture Diagram (Text Format)

```text
[ User / End-User ] 
      |
      v (Uploads CSV / Views Dashboard)
+-------------------------------------------------+
|               Streamlit Frontend                |
| (Interactive UI, Sidebar Nav, Charts & Metrics) |
+-------------------------------------------------+
      |                                ^
      v (Transfers Data)               | (Returns Visuals/Results)
+-------------------------------------------------+
|                 Python Backend                  |
|                                                 |
|  [ Pandas & Numpy ] -> Data Processing / EDA    |
|  [ Scikit-Learn ]   -> Logistic Regression      |
|  [ Fairlearn ]      -> Demographic Parity Calc  |
|  [ Matplotlib/Seaborn] -> Explainability Graphs |
+-------------------------------------------------+
```

---

## 3. Folder Structure

```text
neuro fair/
│
├── data/
│   └── dataset.csv          <- The biased synthetic loan approval dataset
│
├── app.py                   <- The main Streamlit web application
├── generate_dataset.py      <- Script used to create the mock dataset
├── requirements.txt         <- List of Python dependencies
└── README.md                <- This documentation file
```

---

## 4. Step-by-Step Implementation

1. **Setup the Environment**: Install all dependencies needed for the frontend, data wrangling, machine learning, and algorithmic fairness.
2. **Create the Mock Data**: Run `generate_dataset.py` to create a scenario where "Gender" acts as the sensitive attribute, and the approval outcome is historically skewed.
3. **Build the Interface:** Set up a sidebar driven Streamlit app in `app.py`.
4. **Data Intake**: Add a file upload component so judges can interact with their own dataset or the default synthetic one.
5. **EDA Graphics**: Use Pandas value counts and Seaborn countplots.
6. **Model Pipeline**: Split the data, fit a standard `LogisticRegression()`, compute `selection_rate` to demonstrate disparity.
7. **Correction Code**: Formulate a new feature subset by deliberately dropping the Sensitive Attribute flag (e.g., leaving only Income/Credit Score).
8. **Analyze Impact**: Extract the newly fitted model `coef_` to show feature importance.

---

## 5. Complete Working Code

The code is compartmentalized securely in the `app.py` and `generate_dataset.py` scripts situated in this directory. Both files are strictly commented and leverage clean syntax to maximize readability. 
*See `app.py` for the Streamlit UI and ML logic.*
*See `generate_dataset.py` for the data generator logic.*

---

## 6. Sample Dataset Reference

The project includes `data/dataset.csv`. It is a 1,000-row tabular dataset for **Loan Approvals**.
- **Features Includes**: Age, Credit_Score, Income.
- **Sensitive Attribute**: Gender (Male, Female)
- **Target**: Approved (0 or 1).
*Note: The dataset was algorithmically injected with systemic bias, meaning the `Approved` column disproportionately favors Males over Females due to synthetic historical biases encoded in the income mapping.*

---

## 7. UI Explanation

- **Sidebar Menu**: Clean navigation breaking down the ML pipeline logically (from EDA to Mitigation).
- **Data Upload View**: Direct Pandas Dataframe inspection mapping exactly what shape/features are present.
- **Bias Detection**: Shows colorful Seaborn Pie/Bar charts alerting users to the imbalance.
- **Model Training**: A 3-column metric layout showcasing Base Accuracy vs Selection Rates separated by demographic.
- **Bias Correction**: A comparative tabular matrix displaying the "Before VS After" effect of fairness mitigation interventions, followed by an Explainability feature importance bar chart.

---

## 8. How to Run the Project (Locally)

1. Open your terminal in this directory.
2. Install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. A web browser will automatically open at `http://localhost:8501`.

---

## 9. How to Deploy (Streamlit Community Cloud)

To deploy this project to the public internet during the hackathon:
1. Initialize a Git repository in this folder and push all the files (`app.py`, `requirements.txt`, etc.) to a new **GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New App"**.
4. Select your GitHub repository, choose the main branch, and specify `app.py` as the Main file path.
5. Click **"Deploy"**. Streamlit will read your `requirements.txt` and launch the app in a few minutes! You will get a shareable URL to give to the judges.

---

## 10. Tips to Present in a Hackathon

- **Focus on the "Why"**: Start your pitch by explaining *why* algorithmic bias matters (e.g., "An AI that denies loans based on gender is illegal and unethical. We built a tool to intercept that.").
- **Don't just show code, show the impact**: When demoing the app, emphasize the "Before vs After Mitigation" table. Point out how the "Demographic Parity Difference" rating shrinks!
- **Highlight the "Explainability"**: Judges love interpretable ML. Show them the Feature Importance chart to prove "the model isn't a black box, it tells us exactly what logic it uses to make legal decisions."
- **Mention Scalability**: State that while this prototype drops sensitive attributes, future versions could use advanced Fairlearn techniques like Demographic Parity Reweighting or Grid Search Reduction. 
- **Keep the theme clean**: Streamlit is powerful—let the simple UI do the talking. Speak slowly and let the graphics validate your claims.
