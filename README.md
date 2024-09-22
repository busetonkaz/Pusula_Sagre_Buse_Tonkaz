# Pusula_SagreBuse_Tonkaz
## Project Description ##
In this project, we utilized a dataset that contain detail such as the start dates of medications by patient, patient side effect and informations about patient. The dataset was preprocessed and transformed to improve its usability and efficiency for modelling and analysis.


## Steps ##
- Data Loading: The data is loaded in Excel format using the pandas library.
- Missing Data Analysis: Missing data is detected and filled. For categorical variables, imputation is done using a Decision Tree Classifier, while numerical data is filled with mean values.
- Feature Engineering: New features are created using date columns, such as medicine usage duration, medicine start age and side effect's time.
- Outlier Detection: Outliers are detected using the Interquartile Range (IQR) method but no significant outliers were found in the dataset.
-Transformation of Categorical Variables: Categorical variables are converted into numerical data using Label Encoding methods.
- Data Exploding: Columns with list-formatted data, such as chronic diseases are exploded so that each item becomes a separate row.
- Data Visualization: Frequency distributions of categorical variables and correlation analyses of numerical data are visualized.


## Libraries Used ##
1. pandas
2. numpy
3. matplotlib
4. seaborn
5. scikit-learn
