import data_loader as dl
import data_converter as dc
import svm_classifier as sc
import decision_tree_classifier as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load train data
train_file = input('Enter the training data filename: ')
x_train, y_train = dl.load_data(train_file,'train')

# Load test data
test_file = input('Enter the test data filename: ')
X, x_test = dl.load_data(test_file,'test')

# Convert to numeric
x_train_num = dc.convert_categorical(x_train)
x_test_num = dc.convert_categorical(x_test)

# Apply one of Classfier Algorithms
classifier = input('Enter classifier: dt or svm ')
if classifier=='svm':
    kernel_name = input('Select kernel function for SVM: linear, poly, rbf or sigmoid: ')
    # Predict using SVM classification 
    print('Training and validating via SVM')
    y_test=sc.SVM_Classifier(kernel_name,x_train,y_train,x_test,x_train_num,x_test_num)
elif classifier=='dt':
    # Predict using Decision Tree classification 
    print('Training and validating via Decision Tree')
    y_test=dt.DecisionTree(x_train_num, y_train, x_test_num, 'gini')

# Save results to file
print('The prediction result is saved to file in data folder: admission_pediction_'+classifier+'.csv')
dc.merge_to_file(X, y_test)
