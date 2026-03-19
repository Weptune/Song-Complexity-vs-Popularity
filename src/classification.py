import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def classify_songs(df):
    # Create popularity classes
    df['popularity_class'] = pd.qcut(
        df['popularity'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    X = df.drop(['popularity', 'popularity_class'], axis=1)
    y = df['popularity_class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)

    with open("results/models/classification_report.txt", "w") as f:
        f.write(report)

    return model
