# Salary Prediction Project
							Salary Prediction Project
**Problem Statement:**

In today's competitive job market, it is crucial for both employers and job seekers to have an accurate understanding of salary expectations. Many factors influence salary levels, including education, experience, job role, age and industry. Therefore, the aim of this project is to develop a salary prediction model that can provide reliable estimates of salary based on various input features.

**Project Objectives:**

**Data Collection:** Gather a comprehensive dataset containing information about job seekers and their corresponding salaries. The dataset should include features such as education, years of experience, job role and age.

**Data Preprocessing:** Clean and preprocess the dataset to handle missing values, outliers, and categorical variables. Perform feature engineering if necessary to create meaningful input features for the prediction model.

**Exploratory Data Analysis (EDA):** Conduct an in-depth analysis of the dataset to gain insights into the relationships between different features and salary levels. Visualize key trends and correlations.

**Model Selection:** Choose appropriate machine learning algorithms for salary prediction. Consider regression techniques since this is a regression problem. Experiment with different algorithms and evaluate their performance using suitable metrics (e.g., Mean Absolute Error, Root Mean Squared Error, R2 score).

**Feature Importance:** Determine which features have the most significant impact on salary predictions. This information can help both job seekers and employers understand the factors driving salary levels.

**Model Interpretability:** Make the model interpretable by providing explanations for salary predictions. This can enhance trust and understanding of the model's predictions.

**Deployment:** Develop a user-friendly interface or application where users can input their information, and the model will provide a salary estimate based on the trained model.

**Testing and Validation:** Thoroughly test the model with different datasets to ensure its robustness and reliability. Validate the model's predictions against real-world salary data.

**Documentation:** Create comprehensive documentation that explains the project's methodology, data sources, model architecture, and how to use the deployed salary prediction tool.

**User Feedback and Improvement:** Collect feedback from users and stakeholders to continuously improve the model and its user interface. Implement updates and enhancements as needed.

**Deliverables:**

- A well-documented and trained salary prediction model.<br>
- A user-friendly interface or application for salary prediction.<br>
- Documentation detailing the project's methodology and findings.<br><br>
By addressing these objectives, this project aims to provide a valuable tool for job seekers and employers to make informed decisions regarding salary expectations, ultimately contributing to a more transparent and efficient job market.

**Data collection:**
I downloaded this dataset from Kaggle.<br>
**Feature Description:**<br>
**Age:** The age of the individual to which the observation belongs to.<br>
**Gender:** The gender of the individual to which the observation belongs to.<br>
**Education:** The educational qualification of the individual.<br>
**Job Title:** The Job title of the individual they are currently in.<br>
**Years of Experience:** This feature tells how many years of experience the individual currently have.<br>
**Salary (Target Feature):** This feature tells the salary of the individual.

**Outline of Data:**
- The dataset has total observations of 6704 and 6 features.
- All 6 features has some null values.
- The dataset has 3 float (including target) and 3 object datatypes.

**Insights from Exploratory Data Analysis:**
 - The most people are in the age range between 25-40.
 - The most people have experience of 0-10 years.
 - The distribution of the age and Years of experience features skewed towards left.
 - The distribution of the Salary (target) column follows a normal distribution.
 - From the scatter plot we can see that:
 `The age Years of Experience and salary have positive relationship. It is obvious that as age increases the experience also increases, so that salary is.`
  - As the Education Level increases the median salary of a person also increases.
  - The median salary of a person with qualification High School is around 40000 whereas a person with PhD degree is 175000
  - The Top 10 roles earning high salaries are:
  
    - Chief Executive Officer
    - Chief Technology Officer
    - Chief Data Officer
    - Director of Data Science
    - Vice President of Finance
    - Operations Director
    - Vice President of Operations
    - Director of Human Resources
    - Marketing Director
<br>

**Summary:** <br>
- Created a Pandas dataframe and loaded the dataset.<br>
- Conducted comprehensive data preprocessing, including checking feature datatypes, handling missing values, addressing duplicates, and ensuring data integrity.<br>
- Performed univariate and bivariate analyses to gain insights into feature distributions and relationships.<br>
- Assessed feature importance using a correlation matrix.<br>
- Applied Feature Encoding to convert categorical variables into numerical format.<br>
- Utilized Feature Scaling to normalize the range of numerical features.<br>
- Developed multiple models: Lasso Regression, Random Forest, XGBoost, Support Vector Machine, and KNeighbors Regressor.<br>
- Conducted hyperparameter tuning for each model to enhance prediction accuracy.<br>
- Evaluated model performance using metrics like Root Mean Square Error, R2 Score, and Mean Absolute Percentage Error on both training and testing data.<br>
- Achieved an impressive **93%** overall accuracy with the XGBoost Regressor in predicting the target variable (Salary) with minimal error.<br>
- Saved the final model as a pickle file for seamless integration into a Flask App.
App Link: https://salary-prediction-app-28e5.onrender.com
