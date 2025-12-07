# Final-Project-Template
<!-- Edit the title above with your project title -->

## Project Overview

## Self Assessment and Reflection

<!-- Edit the following section with your self assessment and reflection -->

### Self Assessment
<!-- Replace the (...) with your score -->

| Category          | Score    |
| ----------------- | -------- |
| **Setup**         | ... / 10 |
| **Execution**     | ... / 20 |
| **Documentation** | ... / 10 |
| **Presentation**  | ... / 30 |
| **Total**         | ... / 70 |

### Reflection
<!-- Edit the following section with your reflection -->

#### What went well?
Finding out what questions were best to use
#### What did not go well?
Just not being able to commit properly but found out how
#### What did you learn?
How to find data sources. Based on those data sources, what can you analayze and find the conculsion.
#### What would you do differently next time?
Learn how to commit changes. 
---

## Getting Started
### Installing Dependencies

To ensure that you have all the dependencies installed, and that we can have a reproducible environment, we will be using `pipenv` to manage our dependencies. `pipenv` is a tool that allows us to create a virtual environment for our project, and install all the dependencies we need for our project. This ensures that we can have a reproducible environment, and that we can all run the same code.

```bashgitg
pipenv install
```

This sets up a virtual environment for our project, and installs the following dependencies:

- `ipykernel`
- `jupyter`
- `notebook`
- `black`
  Throughout your analysis and development, you will need to install additional packages. You can can install any package you need using `pipenv install <package-name>`. For example, if you need to install `numpy`, you can do so by running:

```bash
pipenv install numpy
```

This will update update the `Pipfile` and `Pipfile.lock` files, and install the package in your virtual environment.

## Helpful Resources:
* [Markdown Syntax Cheatsheet](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
* [Dataset options](https://it4063c.github.io/guides/datasets)

Machine Learning Plan:

I am planning on using Linear Regression as a Machine Learning Model.
Some challenges I might come across are:
1) Not Understanding terms and definintion like what a pipeline is.
2) Not knowing what to do if I come across some errors.

I plan to address these challenges by:
1) Understanding terms and defintion and understand how to implement by watching course videos again
2) And if I do come across some errors, I will use CO-PILOT to correct errors, look through  websites like stack overflow on what I can do to correct these errors and take help from the professor


Machine Learning Implementation Process:

1) First I will find out what answer I will need to find out.
2) I will then download the data, and upload it.
3) I will find any missing values and handle some erros if I come across any. 
4) Then I will process the data. I will train the data set, then I will build a machine learning pipeline
4) I will then analyze the data by figuring out any patterns and identify it.
5) Finally I will evaluate by looking for the best model.


On previous feedback, I received how I have to make minor refinements in correlation analysis and ML detailing.


here's the implementation: 

# 1. Load the data
spotify_data = pd.read_csv('data/track_data_final.csv')

# 2. Handle missing values
machine_learning_data = spotify_data[['track_popularity', 'artist_popularity', 'artist_followers', 
                      'track_duration_ms', 'explicit', 'album_total_tracks', 
                      'album_type', 'track_age']].copy()
machine_learning_data = machine_learning_data.dropna()

# 3. Prepare features (X) and target (y)
X = machine_learning_data.drop('track_popularity', axis=1)
y = machine_learning_data['track_popularity']

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build a machine learning pipeline
numeric_features = ['artist_popularity', 'artist_followers', 'track_duration_ms', 
                    'album_total_tracks', 'track_age']
categorical_features = ['explicit', 'album_type']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 6. Train the model
pipeline.fit(X_train, y_train)

# 7. Make predictions
y_pred = pipeline.predict(X_test)

# 8. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# 9. Analyze
# The model can predict about 27% of a song's popularity (R² ≈ 0.27)
# Artist popularity is the strongest predictor of track popularity
# Other factors like explicit content and track age have smaller impacts 



