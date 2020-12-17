import os
import time
import datetime
import warnings
import logging, logging.config
from src.config import settings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


logger = logging.getLogger()
LOGGING_CONF = os.path.join(os.path.dirname(__file__), "logging.ini")
if not os.path.exists(os.path.dirname(__file__)+'/log'):
    os.makedirs(os.path.dirname(__file__)+'/log')
logging.config.fileConfig(LOGGING_CONF)


class WinePredictor:
    """
    Wine Predictor
    ...

    Attributes:
    ----------
    name: str, default "RWP"
        the name of the class

    wine_data: DataFrame
        the pandas DataFrame which reads comma-separated values (csv) files
        it represents data in tabular form with labeled rows and columns

    csv_file: str, default None
        the name of the csv file includes wine data

    accuracies: list
        the list of the classification score accuracy

    Methods:
    -------
    pre_process(self):
        Retrieves the wine data by reading csv file

    comprehending_data_basics(self):
        Viewing and understanding the basic details of our wine dataset

    content_control(self):
        Retrieves the imbalance dollar/tick/volume bar data

    plot_outliers(self):
        Plotting box charts to check if there are outliers

    plot_relations_between_quality(self):
        Plotting bar charts to see relation per feature between the ‘Quality’

    plot_correlation_heat_map(self):
        Plotting the correlations between the attributes

    start_ml_process(self):
        To trigger the machine learning process

    report_result_of_classification(self, method, classifier, predict, x_train, x_test, y_train, y_test):
        To calculate and print the classification report, actual and predicted head, accuracy, confusion matrix,
        f1 and precision score of requested classification result

    """

    def __init__(self, csv_file=None):
        self.name = "RWP"
        self.csv_file = csv_file
        self.wine_data = None
        self.accuracies = list()

    def run(self):
        logging.info("%s | Wine Predictor started" % self.name)
        try:
            self.work()
        except KeyboardInterrupt:
            time.sleep(0.5)
            logging.error("%s | Wine Predictor exiting" % self.name)
        finally:
            logging.info("%s | Wine Predictor finished" % self.name)

    def work(self):
        start_time = datetime.datetime.now()
        try:
            self.pre_process()
            self.comprehending_data_basics()
            self.content_control()
            self.plot_outliers()
            self.plot_relations_between_quality()
            self.plot_correlation_heat_map()
            self.start_ml_process()
        except Exception as exp:
            logging.error("%s | Something broke: %s" % (self.name, str(exp)))
            raise exp
        finally:
            remove_time = datetime.datetime.now() - start_time
            logging.info('%s | Wine Predictor process took %s' % (self.name, remove_time))

    def pre_process(self):
        """
        Retrieve the data by reading csv file before processing
        :return: self
        """
        self.wine_data = pd.read_csv(settings.FILE_DIRECTORY+self.csv_file)

    def comprehending_data_basics(self):
        """
        Domain knowledge, comprehending the data
        Let's check how the data is distributed
        :return: self
        """
        # viewing top 5 rows of dataset
        print('First five rows of the dataset', self.wine_data.head(), sep='\n')
        # Information about the data columns
        print('Data content info', self.wine_data.info(), sep='\n')
        # viewing shape(rows, columns) of dataset
        print('Shape rows and columns ', self.wine_data.shape, sep='\n')
        # describing the features of dataset statistically
        print('Dataset features', self.wine_data.describe(), sep='\n')
        # To find correlation between variables
        print('The variable correlations', self.wine_data.corr().T, sep='\n')

    def content_control(self):
        """
        Check for null values and duplicates if any
        :return: self
        """
        # Controlling the null or missing values
        print('The null values sum', self.wine_data.isnull().sum(), sep='\n')
        print('The count of the values', self.wine_data.quality.value_counts(), sep='\n')

        # Most wines have a pH between 3.0 — 4.0
        print('All unique ph values', self.wine_data['pH'].unique(), sep='\n')
        print('The data set count', self.wine_data.count(), sep='\n')

        # Check for null values if any
        print('Whether is any null value', self.wine_data.isnull().any(), sep='\n')

    def plot_outliers(self):
        """
        Plotting box chart to see if there are any outliers in our data (considering data between 25th and 75th
        percentile as non outlier)
        :return: self
        """
        fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(15, 5))
        ax = ax.flatten()
        index = 0
        for i in self.wine_data.columns:
            if i not in ['quality', 'density']:
                sns.boxplot(y=i, data=self.wine_data, ax=ax[index])
                index += 1
        plt.tight_layout(pad=0.4)
        plt.show()

    def plot_relations_between_quality(self):
        """
        Plotting bar chart to see relation between each independent feature with dependent feature ‘Quality’
        :return: self
        """
        fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(15, 5))
        ax = ax.flatten()
        index = 0
        for i in self.wine_data.columns:
            if i not in ['quality', 'density']:
                sns.barplot(x='quality', y=i, data=self.wine_data, ax=ax[index])
                index += 1
        plt.tight_layout(pad=0.4)
        plt.show()

    def plot_correlation_heat_map(self):
        """
        Plotting correlation heat map to verify the above statements
        :return: self
        """
        wine_correlation = self.wine_data.corr()

        ''' Now check the relationship between all the features with the target (Quality) '''
        plt.figure(figsize=(15, 10))
        sns.heatmap(wine_correlation, annot=True, square=True, annot_kws={'size': 14})
        sns.set_style('whitegrid')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

        # For another view, this method can be used to view correlations
        print('Another view of correlations among features', end='\n')
        wine_correlation.style.background_gradient(cmap="coolwarm")

        print('Correlation of different features of our dataset with quality', end='\n')
        for i in self.wine_data.columns:
            corr, _ = pearsonr(self.wine_data[i], self.wine_data['quality'])
            print('%s : %.4f' % (i, corr))

        # Calculate and order correlations
        correlations = wine_correlation['quality'].sort_values(ascending=False)
        print(correlations)
        correlations.plot(kind='bar', color='#EB984E')
        plt.show()
        print('The correlations greater than zero', abs(correlations) > 0.2, sep='\n')

    def start_ml_process(self):
        """
        Implementing different machine learning algorithms based on classification and selecting the best out of
        them on the basis of evaluation scores
        :return: self
        """
        # ### Create Classification version of target variable, Classify The Quality ### #
        '''
        quality interval: from 3 to 8
        3–4: Table Wine -> 0/table
        5–6: Fine Wine -> 1/fine
        7–8: Premium Wine -> 2/premium
        '''
        try:
            warnings.filterwarnings('ignore')
            print('Wine quality counts', self.wine_data['quality'].value_counts(), sep='\n')
            fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
            ax = ax.flatten()
            self.wine_data['quality'].value_counts().plot(x=0, y=1, kind='pie', figsize=(15, 5), ax=ax[0])
            sns.countplot(self.wine_data['quality'], ax=ax[1])
            plt.show()

            wine_table = []
            for i in self.wine_data.quality:
                if i <= 4:
                    wine_table.append("table")  # Table Wine
                elif i >= 7:
                    wine_table.append("premium")  # Premium Wine
                else:
                    wine_table.append("fine")  # Fine Wine

            self.wine_data['label'] = wine_table

            print(self.wine_data['label'].value_counts())
            fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
            ax = ax.flatten()
            self.wine_data['label'].value_counts().plot(x=0, y=1, kind='pie', figsize=(15, 5), ax=ax[0])
            sns.countplot(self.wine_data['label'], ax=ax[1])
            plt.show()

            # Comparing the Top 4 Features
            # Filtering df for only good quality
            df_table = self.wine_data[self.wine_data['label'] == 'table']
            print('Table wine features', df_table.describe(), sep='\n')
            # Filtering df for only premium quality
            df_fine = self.wine_data[self.wine_data['label'] == 'fine']
            print('Fine wine features', df_fine.describe(), sep='\n')
            # Filtering df for only premium quality
            df_premium = self.wine_data[self.wine_data['label'] == 'premium']
            print('Premium wine features', df_premium.describe(), sep='\n')

            # Creating set of independent and dependent features
            X_features, y_labels = self.wine_data.iloc[:, :-2], self.wine_data.iloc[:, -1]
            X = X_features.values   # Features, as define features X
            y = y_labels.values  # Labels, as define target y

            label_encoder_y = LabelEncoder()
            y = label_encoder_y.fit_transform(y)

            # determining the shape of x and y
            print(X.shape)
            print(y.shape)

            # Training and Testing Data # Creating training and test set
            '''
            RandomState:
            train_test_split splits arrays or matrices into random train and test subsets. 
            That means that every time you run it without specifying random_state, you will get a different result, 
            this is expected behavior
            
            On the other hand if you use random_state=some_number, then you can guarantee that the output of Run 1 
            will be equal to the output of Run 2, i.e. your split will be always the same. It doesn't matter what 
            the actual random_state number is 42, 0, 21, ... 
            
            The important thing is that every time you use 42, you will always get the same output the first time
            you make the split. random_state simply sets a seed to the random generator, so that your train-test splits 
            are always deterministic.
            
            If random_state is None or np.random, then a randomly-initialized RandomState object is returned.
            If random_state is an integer, then it is used to seed a new RandomState object.
            
            random_state to be used; None, 0, 10, 21, 42
            test_size values to be tried: .3, .2, .25
            '''
            # Creating training and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

            # determining the shapes of training and testing sets
            print('X train shape is: %s' % str(X_train.shape))
            print('y train shape is: %s' % str(y_train.shape))
            print('X test shape is: %s' % str(X_test.shape))
            print('y test shape is: %s' % str(y_test.shape))

            # Normalize feature variables, Scaling the data for optimise predictions
            # Applying Standard scaling to get optimized result
            # Feature scaling, but not scaling dependent variable as it has categorical data
            std_sc = StandardScaler()
            X_train = std_sc.fit_transform(X_train)
            X_test = std_sc.fit_transform(X_test)

            # ### Implementing ML - Classification algorithm based Training the Model and Predicting the Test ### #
            model_names = ['Support Vector', 'Random Forest', 'K-Nearest Neighbors', 'Logistic Regression', 'Decision Tree',
                           'Naive Bayes', 'Stochastic Gradient', 'Multi Layer Perceptron', 'AdaBoost', 'Gradient Boosting',
                           'XGBoost']

            ''' The Part related with Support Vector '''
            svc = SVC()
            svc.fit(X_train, y_train)
            pred_svc = svc.predict(X_test)
            self.report_result_of_classification(model_names[0], svc, pred_svc, X_train, X_test,
                                                  y_train, y_test)

            ''' The Part related with Random Forest'''
            '''
                n_estimators : int, default=100, the values to be tried: 50, Default 100, 200, 250
                The number of trees in the forest.
                The default value of ``n_estimators`` changed from 10 to 100
                Estimator that was chosen by the search, i.e. estimator which gave highest score
            '''
            rfc = RandomForestClassifier(n_estimators=200, max_features='log2')
            rfc.fit(X_train, y_train)
            pred_rfc = rfc.predict(X_test)
            self.report_result_of_classification(model_names[1], rfc, pred_rfc, X_train, X_test,
                                                  y_train, y_test)

            '''
            # You have to fit your data before you can get the best parameter combination
            
            params_grid = {
                'n_estimators': [50, 100, 200, 250],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            
            grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=params_grid, cv=5)
            grid_search_rfc.fit(X_train, y_train)
            best_params = grid_search_rfc.best_params_
            '''

            ''' The Part related with K-Nearest Neighbors '''
            knn = KNeighborsClassifier()
            knn.fit(X_train, y_train)
            pred_knn = knn.predict(X_test)
            self.report_result_of_classification(model_names[2], knn, pred_knn, X_train, X_test,
                                                  y_train, y_test)

            ''' The Part related with Logistic Regression '''
            logistic_regression = LogisticRegression(solver='lbfgs', random_state=0)
            logistic_regression.fit(X_train, y_train)
            pred_lr = logistic_regression.predict(X_test)
            self.report_result_of_classification(model_names[3], logistic_regression, pred_lr,
                                                  X_train, X_test, y_train, y_test)

            ''' The Part related with Decision Tree '''
            decision_tree = DecisionTreeClassifier(random_state=1)
            decision_tree.fit(X_train, y_train)
            pred_dt = decision_tree.predict(X_test)
            self.report_result_of_classification(model_names[4], decision_tree, pred_dt, X_train,
                                                  X_test, y_train, y_test)

            ''' The Part related with Naive Bayes '''
            nbc = GaussianNB()
            nbc.fit(X_train, y_train)
            pred_nb = nbc.predict(X_test)
            self.report_result_of_classification(model_names[5], nbc, pred_nb, X_train, X_test, y_train,
                                                 y_test)

            ''' The Part related with Stochastic Gradient Descent '''
            sgd = SGDClassifier()
            sgd.fit(X_train, y_train)
            pred_sgd = sgd.predict(X_test)
            self.report_result_of_classification(model_names[6], sgd, pred_sgd, X_train,
                                                 X_test, y_train, y_test)

            ''' The Part related with Multi Layer Perceptron '''
            # creating the model
            mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=150)
            # feeding the training data to the model
            mlp.fit(X_train, y_train)
            pred_mlp = mlp.predict(X_test)
            self.report_result_of_classification(model_names[7], mlp, pred_mlp, X_train,
                                                 X_test, y_train, y_test)

            ''' The Part related with AdaBoost '''
            ada_boost = AdaBoostClassifier(random_state=1)
            ada_boost.fit(X_train, y_train)
            pred_ab = ada_boost.predict(X_test)
            self.report_result_of_classification(model_names[8], ada_boost, pred_ab, X_train,
                                                 X_test, y_train, y_test)

            ''' The Part related with Gradient Boosting '''
            gradient_boost = GradientBoostingClassifier(random_state=1)
            gradient_boost.fit(X_train, y_train)
            pred_gbc = gradient_boost.predict(X_test)
            self.report_result_of_classification(model_names[9], gradient_boost, pred_gbc, X_train,
                                                 X_test, y_train, y_test)

            ''' The Part related with XGBoost '''
            xg_boost = xgb.XGBClassifier(random_state=1)
            xg_boost.fit(X_train, y_train)
            pred_xgb = xg_boost.predict(X_test)
            self.report_result_of_classification(model_names[10], xg_boost, pred_xgb, X_train,
                                                 X_test, y_train, y_test)

            ''' Conclusion '''

            ''' Accuracy classification score '''
            accuracy_summary = pd.DataFrame({'models': model_names, 'accuracies': self.accuracies})
            print(accuracy_summary.sort_values(by='accuracies', ascending=False), '\n')

            ''' Evaluate a score by cross-validation '''
            # Checking accuracy of different classification models, K-fold cross validation
            model_classifiers = [svc, rfc, knn, logistic_regression, decision_tree, nbc, sgd, mlp, ada_boost,
                                 gradient_boost, xg_boost]
            models = pd.DataFrame({'modelNames': model_names, 'modelClassifiers': model_classifiers})
            counter, score = 0, []
            for each in models['modelClassifiers']:
                accuracy = cross_val_score(each, X_train, y_train, scoring='accuracy', cv=10)
                print('Cross-validation accuracy of %s Classification model is %.2f' % (
                models.iloc[counter, 0], accuracy.mean()))
                score.append(accuracy.mean())
                counter += 1

            ''' Plotting the accuracies of different models '''
            pd.DataFrame({'Model Name': model_names, 'Score': score}).sort_values(by='Score', ascending=True).plot(x=0,
                          y=1, kind='bar', figsize=(15, 5),
                          title='Comparison of accuracies of different classification models')
            plt.show()

            ''' Random Forest Feature Importance '''
            importance = rfc.feature_importances_
            rfc_feature_imp = pd.Series(importance, index=X_features.columns).sort_values(ascending=False)
            print('Random Forest Feature Importance', rfc_feature_imp, sep='\n')

            # Visualize the Importance Creating a bar plot
            sns.barplot(x=rfc_feature_imp, y=rfc_feature_imp.index)
            # Add labels to the graph
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Features')
            plt.title("Visualizing Important Features")
            plt.show()

        except Exception as exp:
            logging.error("%s | Something went wrong in start_ml_process, caused: %s" % (self.name, str(exp)))

    def report_result_of_classification(self, method, classifier, predict, x_train, x_test, y_train, y_test):
        """
        Calculate and print the classification report, actual and predicted head, accuracy, confusion matrix,
        f1 and precision score of requested classification result
        :return: self
        """
        try:
            print(method+' Classification Report', classification_report(y_test, predict), sep='\n')
            y_compare_rfc = pd.DataFrame({'Actual': y_test, 'Predicted': predict})
            print(y_compare_rfc.head(), '\n')
            print("Training accuracy :", classifier.score(x_train, y_train))
            print("Testing accuracy :", classifier.score(x_test, y_test))
            print('Confusion matrix:', confusion_matrix(y_test, predict), sep='\n')
            classifier_f1_score = f1_score(y_test, predict, average='micro')
            print('F1 Score: ', classifier_f1_score)
            print('Precision Score:', precision_score(y_test, predict, average="micro"))
            accuracy_classification_score = accuracy_score(y_test, predict)
            print("Accuracy: ", accuracy_classification_score, '\n')
            self.accuracies.append(accuracy_classification_score)
        except Exception as exp:
            logging.error(
                "%s | Something went wrong in report_result_of_classification, caused: %s" % (self.name, str(exp)))


if __name__ == '__main__':
    uci_file = "redwinequality.csv"
    rw_quality_predictor = WinePredictor(uci_file)
    rw_quality_predictor.run()
