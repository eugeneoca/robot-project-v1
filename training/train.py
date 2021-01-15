import numpy as np
import cv2
from os import walk
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import RandomizedSearchCV

def translate_input(path):
    raw_image = cv2.imread(path)

    resized_image = cv2.resize(raw_image, (400,300))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, binarized_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
    return binarized_image.flatten()

smooth_ds_paths = [] # Class: 0 => Smooth
gravel_ds_paths = [] # Class: 1 => Gravel

dataset_X = []
dataset_y = []
_, _, smooth_ds_paths = next(walk("datasets/smooth"))
_, _, gravel_ds_paths = next(walk("datasets/gravel"))

report = 0

for path in smooth_ds_paths:

    if (report % 100) == 0:
        print("Loaded data: {count}".format(count=report))

    report +=1

    dataset_X.append(translate_input(join("datasets/smooth",path)))
    dataset_y.append(0)

print("Total Loaded data: {count}".format(count=report))

report = 0

for path in gravel_ds_paths:

    if (report % 100) == 0:
        print("Loaded data: {count}".format(count=report))

    report +=1

    dataset_X.append(translate_input(join("datasets/gravel",path)))
    dataset_y.append(1)

print("Total Loaded data: {count}".format(count=report))

is_training = False

if is_training:
    X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size=0.33)

    model = RandomForestClassifier(verbose=True)
    model.fit(X_train, y_train)

    print("Test Predicting:")
    print(X_test)

    test_result = model.predict(X_test)

    print("Result:")
    print(test_result)

    print("Base Accuracy: "+str(accuracy_score(y_test, test_result)*100.00) + "%")

    """
    Hyperparameter Grid Search
    """
    print("---Hyperparameter Grid Search---")
    # Parameters
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    search_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    base_model = RandomForestClassifier(verbose=True)
    hyperparam_result = RandomizedSearchCV(estimator = base_model, param_distributions = search_grid, n_iter = 200, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    hyperparam_result.fit(X_train, y_train)
    model = hyperparam_result.best_estimator_

    print("Test Predicting:")
    print(X_test)

    test_result = model.predict(X_test)

    print("Result:")
    print(test_result)

    print("Tuned Accuracy: "+str(accuracy_score(y_test, test_result)*100.00) + "%")
    print("Best Parameters:")
    print(hyperparam_result.best_params_)

else:
    print("Training...")
    model = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=4, max_features='auto', max_depth=10, bootstrap=True)
    model.fit(dataset_X, dataset_y)
    pickle.dump(model, open("terrain_model", 'wb'))
    print("Saved.")