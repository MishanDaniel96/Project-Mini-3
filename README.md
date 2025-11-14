# Project-Mini-3
End-to-end data science project predicting employee turnover risk. Uses EDA &amp; ML to find key drivers. Features a Streamlit app for real-time risk assessment and retention planning.
The Employee Attrition Predictor is a comprehensive, end-to-end machine learning project focused on mitigating organizational talent loss. The core objective is to identify and understand the factors contributing to employee turnover, and subsequently build a robust predictive model to flag employees at high risk of leaving.

Methodology and Analysis
The project began with an extensive Exploratory Data Analysis (EDA) on a proprietary HR dataset. This phase involved cleaning the data, handling categorical variables, and performing feature engineering to extract meaningful patterns. Key insights from the EDA revealed that factors such as Monthly Income, Job Role, Years at Company, and whether an employee works OverTime were the strongest indicators of attrition.

A variety of classification algorithms were tested, including K-Nearest Neighbors, Support Vector Machines, and ensemble methods. The Logistic Regression model, after hyperparameter tuning and balancing the class weights (due to the inherent class imbalance in attrition data), emerged as the most effective solution for this business problem. The final model achieved a strong performance, validated by an AUC-ROC score of 0.7963 and a high Recall score of 0.71. This high recall ensures that the model correctly identifies most of the employees who will actually leave, minimizing costly false negatives.

Deployment and Impact
The predictive model was encapsulated within an interactive web application built using Streamlit. This dashboard provides HR professionals and managers with an intuitive interface to:

Visualize the key insights from the EDA.

Input employee features to get real-time attrition probability predictions.

Generate a ranked list of at-risk employees.

By providing actionable intelligence directly to decision-makers, this tool allows for the proactive implementation of targeted retention strategies, ultimately reducing recruitment costs and preserving institutional knowledge.
