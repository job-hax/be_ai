from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def SVM_Classifier(kernel_name,x_train,y_train,x_test,x_train_num,x_test_num):
    # x_train - train input values
    # y_train - train target values
    # x_test - test input values
# Create SVM classification object 
    model = svm.SVC(kernel=kernel_name)  
    for i in range(10):
        valid_set_size = 0.10
        # divide the original training set into training set and validation set
        XTrain, XTest, yTrain, yTest = train_test_split(x_train_num, y_train, test_size=valid_set_size)
        # Train the model 
        model.fit(XTrain, yTrain)
        # Use the model to predict on the test set
        yPred = model.predict(XTest)
        print('the validation set size: ' + str(valid_set_size))
        # Get accuracy
        score = accuracy_score(yTest, yPred)
        print('the validation accuracy: ' + str(score))
    y_test = model.predict(x_test_num)
    return y_test
