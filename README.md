# MLCapstoneProject

## Loading and Preprocessing :
The first step involved loading the dataset using the pandas library. After reading the dataset into a DataFrame, an initial exploration was carried out using functions like .head(), .info(), and .describe() to understand the structure, data types, and summary statistics of the features. Missing values were identified and handled—either by removing rows/columns with excessive nulls or by imputing them using appropriate strategies like mean or mode replacement. Categorical variables were encoded using either Label Encoding (for ordinal features) or One-Hot Encoding (for nominal features), enabling the models to interpret non-numeric data. Numerical features were standardized using StandardScaler to ensure that all values lie within the same range, which is especially important for distance-based algorithms like SVR. The data was finally split into training and test sets using train_test_split, ensuring an 80-20 or 70-30 ratio to preserve generalization during evaluation.

## Model Implementation :
Five different regression algorithms were implemented using scikit-learn's powerful library of estimators. The models chosen were:

Linear Regression – A basic model that assumes a linear relationship between independent features and the target (car price). It was used as a baseline model.

Decision Tree Regressor – A non-linear model that splits the data based on feature thresholds to predict target values.

Random Forest Regressor – An ensemble model that constructs multiple decision trees on random subsets of data and averages their predictions to improve accuracy and reduce overfitting.

Gradient Boosting Regressor – Another ensemble model that builds trees sequentially, where each tree tries to correct the errors of the previous one. It typically yields high performance but takes more time to train.

Support Vector Regressor (SVR) – A kernel-based method that finds the optimal hyperplane within a margin of tolerance. It’s useful for high-dimensional data and can capture non-linear relationships with the help of kernels.

Each model was fit on the training data and used to make predictions on the test data. The goal was to compare how well each algorithm could predict car prices based on the input features.

## Model Evaluation:
The predictive performance of all five models was evaluated using three key metrics:

R-squared (R²): Measures the proportion of variance in the dependent variable that is predictable from the independent variables. A value close to 1 indicates a good fit.

Mean Squared Error (MSE): Measures the average of the squares of the errors; penalizes larger errors more than smaller ones.

Mean Absolute Error (MAE): The average of absolute differences between predicted and actual values, providing an interpretable error value in the unit of the target variable (e.g., dollars).

All models were compared using these metrics. Generally, ensemble models like Random Forest and Gradient Boosting performed better due to their ability to capture complex relationships in the data. The best model was selected based on achieving the highest R² and the lowest MSE and MAE, signifying accurate and reliable predictions.

##  Feature Importance Analysis:
To identify which features most significantly affected car prices, feature importance scores were extracted from the tree-based models (Random Forest and Gradient Boosting). These models naturally rank features based on how frequently and effectively they are used in decision splits. Features like car brand, year of manufacture, engine size, mileage, and fuel type were often found to be the most influential. This step is critical in understanding the model’s decision-making process and can also help in reducing dimensionality by removing low-impact variables. Feature selection improves computational efficiency and interpretability of the model.

##  Hyperparameter Tuning :
To further improve model performance, hyperparameter tuning was conducted using GridSearchCV or RandomizedSearchCV. These methods perform cross-validated searches over specified parameter grids to identify the optimal combination.
For example:

For Random Forest, parameters like n_estimators, max_depth, and min_samples_split were tuned.

For Gradient Boosting, learning rate (learning_rate), number of estimators, and depth were varied.

For SVR, kernel type, C, and epsilon were adjusted.

After tuning, the models were re-evaluated on the test data. In most cases, performance improved (higher R² and lower errors), demonstrating that careful tuning of hyperparameters significantly enhances the model’s generalization ability and reduces overfitting.

