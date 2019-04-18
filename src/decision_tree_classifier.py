from sklearn import tree
# supervised learning (classification/prediction)
# decision tree classification/prediction
# This is done based on most significant attributes (independent variables) to make as distinct groups as possible
def DecisionTree(x_train, y_train, x_test, criterion_name):
    # x_train - train input values
    # y_train - train target values
    # x_test - test input values

    # Default criterion is gini; otherwise, entropy to create Decision Tree
    model = tree.DecisionTreeClassifier(criterion = criterion_name) 

    # Train model
    model.fit(x_train, y_train)

    # Predict
    y_test=model.predict(x_test)
    return y_test
