import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import warnings

warnings.filterwarnings("ignore")




#load data
rawdata = pd.read_csv('../data/diabetes.csv',encoding="ISO-8859-1")
rawdata.head()



#Explore the data
print(rawdata.dtypes)
print(rawdata.head)
print(rawdata.dtypes)
print(rawdata.nunique())
rawdata.describe()
print(rawdata.shape)



#Analyzie the data
rawdata.describe()



#Diabetes Count
diabetes_count = rawdata['Outcome'].value_counts()
diabetes_count



#columns
print(rawdata.columns)



#check if there are missing values
missing_values = {}
variables = rawdata.columns.tolist()

for var in variables:
    missing_values[var] = rawdata[var].isnull().sum()
missing_values_df = pd.DataFrame.from_dict(missing_values, orient='index', columns=['Missing Values'])

print("Missing values by variable:")
print(missing_values_df)



#Histplots
variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

for var in variables:
    if var != 'Pregnancies':
        filtered_data = rawdata[rawdata[var] != 0]
    else:
        filtered_data = rawdata
        
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(x=filtered_data[var], kde=True, ax=ax)
    ax.set_title(f"Histogram of {var}")
    
    output_path = os.path.join("..", "output", f"Histplot of {var}.png")
    plt.savefig(output_path)
    plt.close(fig)



#scatterplot
filtered_data = rawdata[(rawdata['Glucose'] != 0) & (rawdata['BMI'] != 0)]
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=filtered_data)
plt.title('Glucose Levels vs. BMI (Colored by Diabetes Outcome)')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.tight_layout()
plt.savefig("../output/Scatterplot_glucoseBMI")
plt.show()

#########################################################################################################################################

#Classification

# Utility function to plot the diagonal line - complete, no input needed here.
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def get_confusion_matrix(y,y_pred):
    """
    compute the confusion matrix of a classifier yielding
    predictions y_pred for the true class labels y
    :param y: true class labels
    :type y: numpy array

    :param y_pred: predicted class labels
    :type y_pred: numpy array

    :return: comfusion matrix comprising the
             true positives (tp),
             true negatives  (tn),
             false positives (fp),
             and false negatives (fn)
    :rtype: four integers
    """
    
    # true/false pos/neg. - this is a block of code that's needed
    # HINT: consider using a for loop.
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1


    return tn, fp, fn, tp


def evaluation_metrics(clf, y, X, ax, legend_entry='my legendEntry'):
    """
    Compute multiple evaluation metrics for the provided classifier given the true labels
    and input features. Provides a plot of the ROC curve on the given axis with the legend
    entry for this plot specified.

    :param clf: Classifier object
    :param y: True class labels
    :param X: Feature matrix
    :param ax: Matplotlib axis to plot on
    :param legend_entry: The legend entry for the plot
    :return: List of evaluation metrics
    """
    y_test_pred = clf.predict(X)

    try:
        tn, fp, fn, tp = get_confusion_matrix(y, y_test_pred)
    except KeyError:
        y = pd.Series(y).values
        y_test_pred = pd.Series(y_test_pred).values
        tn, fp, fn, tp = get_confusion_matrix(y, y_test_pred)

    # Calculate the metrics
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    # Get the ROC curve 
    y_test_predict_proba = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_test_predict_proba)

    # Calculate the area under the ROC curve
    roc_auc = roc_auc_score(y, y_test_predict_proba)

    # Plot on the provided axis
    ax.plot(fpr, tpr, label=legend_entry)

    return [accuracy, precision, recall, specificity, f1, roc_auc]


# Import data
df = pd.read_csv('../data/diabetes.csv')



X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Perform stratified 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create a dataframe to store the performance metrics
df_performance = pd.DataFrame(columns=['Fold', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])




# Use this counter to save your performance metrics for each crossvalidation fold
# also plot the roc curve for each model and fold into a joint subplot
fold = 0
fig,axs = plt.subplots(1,2,figsize=(9, 4))



# Loop over all splits
for train_index, test_index in skf.split(X, y):
    
    print('Working on fold', fold)

    # Get the relevant subsets for training and testing
    X_test  = X.iloc[test_index]
    y_test  = y.iloc[test_index]
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]


    # Standardize the numerical features using training set statistics
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)


    # Creat prediction models and fit them to the training data

    # Logistic regression
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_sc, y_train)

    # Random forest
    clf2 = RandomForestClassifier(random_state=42)
    clf2.fit(X_train_sc, y_train)


  # Evaluate your classifiers - ensure to use the correct inputs
    eval_metrics = evaluation_metrics(clf, y_test, X_test_sc, axs[0],legend_entry=str(fold))
    df_performance.loc[len(df_performance), ['Fold', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'AUC']] = [fold, 'LR'] + eval_metrics
    eval_metrics_RF = evaluation_metrics(clf2, y_test, X_test_sc, axs[1],legend_entry=str(fold))
    df_performance.loc[len(df_performance), ['Fold', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'AUC']] = [fold, 'RF'] + eval_metrics_RF
    # increase counter for folds
    fold += 1

# Edit the plot and make it nice
model_names = ['LR','RF']
for i,ax in enumerate(axs):
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    add_identity(ax, color="r", ls="--",label = 'random\nclassifier')
    ax.legend(title='Model', loc='lower right')
    ax.set_title(model_names[i])


# Save the plot - no change needed - ensure to submit with this exact output path
plt.tight_layout()
plt.savefig('../output/roc_curves.png')


# Summarize the performance metrics over all folds
### this may be more than one line of code

print('df_performance:', df_performance)

df_mean_std_performance = df_performance.groupby('Classifier').agg(['mean', 'std'])
print('df_mean_std_performance:', df_mean_std_performance)
df_mean_std_performance.to_csv('../output/performance.csv')


# Average the feature importance across the five folds and sort them
# HINT: consider how to sort a pandas data frame
### this may be a short block of code

df_LR_normcoef['mean_coef'] = df_LR_normcoef.mean(axis=1)
df_LR_normcoef_sorted = df_LR_normcoef.sort_values('mean_coef', ascending=False)
print('df_LR_normcoef_sorted.head(15):', df_LR_normcoef_sorted.head(15))


# Visualize the normalized feat ure importance averaged across the five folds
# FOR THE TOP 15 features and add error bars to indicate the std
fig, ax = plt.subplots()
top_features = df_LR_normcoef_sorted[:15]

#print(top_features)
#print(top_features['mean_coef'].shape)

lab = top_features.index
df_plot = pd.DataFrame({'Genes':lab.tolist(), 'Mean Coefficient':top_features['mean_coef'].values.tolist()})

### add a short block of code to create a nice plot with all required labels etc.


ax = df_plot.plot.bar(x='Genes', y='Mean Coefficient', yerr=top_features.std(axis=1), rot=90, color='grey')
ax.set_xlabel('Genes')
ax.set_ylabel('Mean Coefficient')
ax.set_title('Top 15 Feature Importance')
plt.savefig('../output/importance.png')
plt.close()





#Plot with seaborn

sns.set_style('whitegrid')
ax = sns.barplot(x=top_features.index, y='mean_coef', data=top_features, color='grey')
ax.errorbar(x=top_features.index, y=top_features['mean_coef'], yerr=top_features.std(axis=1), fmt='none', color='black', capsize=3)
ax.set_xlabel('Variables')
ax.set_ylabel('Mean Coefficient')
ax.set_title('Top 15 Features Importance')
ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.savefig('../output/importance_sns.png')

# Get the two most important features
top_features_2 = top_features[:2]
print(top_features_2.index)
print(top_features_2)


##############################################################################################################################################3

#linear regression

rawdata = pd.read_csv('../data/diabetes.csv')
rawdata.head()

X = rawdata.drop(columns=["Outcome"], axis=1)
y = rawdata["Outcome"]


# Split the dataset into training and testing sets using the sklearn 'train_test_split' function into 80% for training and 20% for testing,
# set the random seed to 2023.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

print('Training set size: {}, test set size: {}'.format(len(X_train), len(X_test)))


X_train.head()



# Standardize the numerical features using training set statistics
# Note the difference between the StandardScaler 'fit_transform' and 'transform' methods as outlined in the documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#sc = StandardScaler()
#X_train[num_cols] = sc.fit_transform(X_train[num_cols])
#X_test[num_cols] = sc.transform(X_test[num_cols])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


# Convert pandas DataFrame to numpy array
X_train, X_test, y_train, y_test = (
    np.array(X_train),
    np.array(X_test),
    np.array(y_train),
    np.array(y_test),
)



# Initialize the model
LR = LinearRegression()



# Fit the model
LR.fit(X_train, y_train)

coefficients = LR.coef_
intercept = LR.intercept_


print("y_hat = ", end="")
for i, col in enumerate(X.columns):
    print(f"{coefficients[i]:.2f} * {col} + ", end="")
print(f"{intercept:.2f}")



# Make predictions using the fitted model
y_pred = LR.predict(X_test)



# Implement RMSE and R2 Score manually (i.e., do NOT use package implementations for this)
# Refer to the definition of RMSE and R2 from lecture 5.
def R2(test, pred):
    ssr = ((pred - test.mean()) ** 2).sum()
    sst = ((test - test.mean()) ** 2).sum()
    r2 = 1 - (ssr / sst)
    return r2

def RMSE(test, pred):
    mse = ((pred - test) ** 2).mean()
    rmse = np.sqrt(mse)
    return rmse



# Calculate the performance
r2score = R2(y_test, y_pred)
rmse = RMSE(y_test, y_pred)

# Report the performance
print("R2 score: {:.3f}, RMSE: {:.3f}".format(r2score, rmse))



# Utility function to plot the diagonal line for visualizing the optimal fit of the model
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes



# Make a scatter plot (prediction vs ground truth)
# Add a dashed line indicating the optimal fit (you can use the add_identity function or you can implement this by yourself)
# Ensure the label is annotated
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
ax = add_identity(ax, linestyle='--', color='red')
ax.set_xlabel("Ground Truth")
ax.set_ylabel("Prediction")
ax.set_title("Linear Regression Model")
ax.text(0.1, 0.9, f"R2 Score: {r2score:.2f}\nRMSE: {rmse:.2f}", transform=ax.transAxes)
# Save the figure in this specified path (upon submission this path has to be as written below!)
plt.savefig("../output/LinearRegression.png", dpi=100)
plt.close()

#############################################################################################################################################

#Clustering

def plot_cluster_vs_actual_class(X, cluster_labels, algorithm, unsupervised_score, supervised_score):
    """
    Plots the clusters assigned by the clustering algorithm.

    Parameters:
        X (numpy array): The data to be clustered.
        cluster_labels (numpy array): The labels assigned by the clustering algorithm
        algorithm (string): "KMeans" or "DBSCAN"
        unsupervised_score: score to measure clustering performances without using the class labels. These metrics evaluate
                            for example the comparison between the distances among the samples grouped in the same cluster
                            with the distances among samples that belong to different clusters
                            Have a look at scikit-learn docs:
                            https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
        supervised_score: it can be calculated only if you know the class labels as in this case. A famous score
                          is the adjusted rand index (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
                          which basically evaluates how well clusters are formed with respect to the true labels. If 1 you have a perfect match
    """
    n_clusters = len(np.unique(cluster_labels))

    fig, ax = plt.subplots(figsize=(8, 6))

    # check X shape. If it is already 2 dimensional it means that we already applied PCA or the dataset has only 2 features
    # Leave this part as it is
    if X.shape[1] != 2:
        # apply PCA for visualization purposes
        X_plot = PCA(n_components=2).fit_transform(X)
    else:
        X_plot = X.copy()

    # create legend labels using the cluster assigned
    cluster_labels_plot = [f'cluster {i}' for i in cluster_labels]

    # sort the labels from lowest to highest
    hue_order = np.sort(np.unique(cluster_labels_plot))

    # create a scatterplot of the 2 pca components created.
    # The hue should be assigned to the clusters found by the clustering algorithm
    # Use the hue order calculated before
    sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=cluster_labels_plot, hue_order=hue_order, ax=ax)

    # set title and labels
    ax.set_title(f"Clusters Assigned by {algorithm}")
    ax.set_xlabel("PCA comp 1")
    ax.set_ylabel("PCA comp 2")

    # set global title specifying the number of clusters, the supervised score (ARI) and the unsupervised score
    # The title should look like: 'n clusters: ... - WCSS: .... - ARI: ....
    plt.title(f'n clusters: {n_clusters} - WCSS: {unsupervised_score:.2f} - ARI: {supervised_score:.2f}',
              fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"../output/{algorithm}_Plot_clusters_for_{n_clusters}_clusters.png")

# Function to calculate the sum of squared distances as a performance measure for the clustering
def calculate_WCSS(X, cluster_centers):
    """
    Calculate sum of squared distances between samples and cluster centers.
    You need to calculate all the distances, then take the minimum distance
    to find to which cluster the sample belong, and sum these distances.

    Parameters:
        X (numpy array): The feature matrix. Dimension n_samples * n_features
        cluster_centers (numpy array): The centers of each cluster. The dimension is n_clusters * n_features

    Output:
        WCSS score
    """
    # the function of sklearn "cdist" calculate the distance of each sample from each cluster center
    # then we take the minimum of the distance to find to which center the sample is closer
    # Output of cdist will be n_samples * n_cluster
    # Calculate squared euclidean distance between samples
    # Have a look at scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    distance_matrix = cdist(X, cluster_centers, 'euclidean')
    # Find the minimum distance to find to which center the sample is closer.
    # Specify the correct axis from which you want to compute the minimum.
    # The dimension of min_distances is n_samples
    min_distances = np.min(distance_matrix, axis=1)
    # compute the square of the distances
    min_distances_square = min_distances ** 2
    # compute the sum
    wcss = np.sum(min_distances_square)
    return wcss

def apply_kmeans(X, n_clusters):
    """
    Plots the true classes vs the groups found by kmeans.

    Parameters:
        X (numpy array): The data to be clustered.
        y (numpy array): True classes
        n_clusters (numpy array or list): different number of clusters you want to test stored in the array
    Output:
        wcss: list that contains WCSS scores
        ari_scores (list): list that contains ARI scores
    """
    wcss = []
    ari_scores = []
    predicted_clusters_list = []

    for i in n_clusters:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        kmeans_score = kmeans.inertia_

        try:
            unsupervised_score = calculate_WCSS(X, kmeans.cluster_centers_)
            if abs(unsupervised_score - kmeans_score) < 1e-4:
                print('Your implementation is correct. Using your WCSS')
            else:
                print('Your implementation is NOT correct. Using KMeans attribute')
                unsupervised_score = kmeans_score
        except:
            print('Your implementation has an error. Using KMeans attribute')
            unsupervised_score = kmeans_score
        wcss.append(unsupervised_score)

        predicted_clusters = kmeans.labels_
        supervised_score = adjusted_rand_score(true_classes, predicted_clusters)

        ari_scores.append(supervised_score)
        predicted_clusters_list.append(predicted_clusters)

        # Call the plot function inside the loop with the final predicted clusters
        plot_cluster_vs_actual_class(X, predicted_clusters_list[-1], 'KMeans', unsupervised_score, supervised_score)
    return wcss, ari_scores

def apply_dbscan(X, eps_range, min_samples_range):
    """
    Plots the true classes vs the groups found by kmeans.

    Parameters:
        X (numpy array): The data to be clustered.
        y (numpy array): True classes
        eps_range (numpy array or list): list of values to iterate on for eps hyperparameter
        min_samples_range (numpy array or list): list of values to iterate on for min_samples hyperparameter
    Output:
        scores_silhouette (list): list that contains silouhette scores
        ari_scores (list): list that contains ARI scores
        summary (pandas DataFrame): dataframe that contains the hyperparameters and the scores
    """

    # initialize the lists to store the results of the scores. Define one supervised (you know the true classes)
    # and one unsupervised (you don't know the true classes) score
    scores_silhouette = []
    scores_ari = []
    # Define summary dataframe
    summary = pd.DataFrame(columns = ['eps', 'min_samples', 'n_clusters', 'silhouette', 'ari'])

    count = 0
    for eps in eps_range:
        for min_samples in min_samples_range:
            #instantiate DBSCAN object
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            #fit algorithm
            dbscan.fit(X)
            # compute silhouette score which is an unsupervised metric to evaluate.
            # The Silhouette Coefficient is calculated using the mean intra-cluster distance
            # and the mean nearest-cluster distance for each sample. To clarify,
            # the mean nearest-cluster distance is the distance between a sample and the nearest cluster
            # that the sample is not a part of.
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
            try:
                silhouette = silhouette_score(X, dbscan.labels_)
            except:
                # if the algorithm only find 1 cluster the function cannot calculate the score and we set this to 0
                silhouette = 0.0
            # compute ARI score
            ari = adjusted_rand_score(true_classes, dbscan.labels_)

            # Fill the summary dataframe for the specific row (iteration) and column
            summary.loc[count] = [eps, min_samples, n_clusters, silhouette, ari]
            count +=1
            #store eps, min_samples and the corresponding score values in the lists you initialized.
            scores_silhouette.append((eps, min_samples, silhouette))
            scores_ari.append((eps, min_samples, ari))

    return scores_silhouette, scores_ari, summary

# Read the dataset
raw_data = pd.read_csv('../data/diabetes.csv')
print('Info of missing data in raw_data:', raw_data.isna().sum())

# Remove NaN from y
data = raw_data.drop('Outcome', axis=1)
data[data == 0] = np.nan

# Standardize X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Define true_classes
true_classes = raw_data.iloc[:, -1].values.reshape(-1)

# Define the range for the number of clusters to test
max_n_cluster = 10
n_clusters = np.arange(2, max_n_cluster + 1)

# Apply K-Means using the function defined above
wcss, ari_scores = apply_kmeans(X, n_clusters)

# create a figure to show the elbow plot or the distance score (output from kmeans)
# and save it in the folder "output/" with the name Elbow_plot.png.
# Note that you should use multiple lines of code. Remember to put labels, title ...
plt.figure(figsize=(8, 6))
plt.plot(n_clusters, wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Plot for K-Means')
plt.savefig('../output/Elbow_plot.png')


# ##### DBSCAN #####
# scores manually defined to make a search.
# Define a list for eps and min_samples parameters of DBSCAN algorithm
# define 2 lists on which to iterate to find the best set of DBSCAN hyperparameters. Specify 5 values for each. Have a look at the
# documentation to have an idea of the ranges for the parameters
# Define a list for eps and min_samples parameters of DBSCAN algorithm
eps_range = [1.0, 1.5, 2.0, 2.5, 3.0]
min_samples_range = [5, 10, 15, 20, 25]

# Use the function that you defined to apply DBSCAN to evaluate the results for all combinations of parameters.
scores_silhouette, scores_ari, summary = apply_dbscan(X, eps_range, min_samples_range)
# HINT: sort the dataframe and take the first 5 columns
#sort summary according to silouhette
summary_silhouette = summary.sort_values('silhouette', ascending=False).head(5)
summary_silhouette.to_csv('../output/summary_DBSCAN_silhouette.csv')

# Extract the 5 best sets of hyperparameters according to ARI score.
summary_ari = summary.sort_values('ari', ascending=False).head(5)
summary_ari.to_csv('../output/summary_DBSCAN_ARI.csv')

# Extract best values for the hyperparameters according to silhouette score and use them to perform the final clustering (DBSCAN).
best_eps_silhouette, best_min_samples_silhouette, silhouette_score_value = max(summary_silhouette[['eps', 'min_samples', 'silhouette']].values, key=lambda x: x[2])
dbscan_silhouette_labels = DBSCAN(eps=best_eps_silhouette, min_samples=int(best_min_samples_silhouette)).fit(X).labels_

# Use the 'plot_cluster_vs_actual_class' function to plot the groups formed by the DBSCAN algorithm with silhouette score.
plot_cluster_vs_actual_class(X, dbscan_silhouette_labels, 'DBSCAN_silhouette', silhouette_score_value, summary_silhouette.iloc[0]['ari'])

# Extract best values for the hyperparameters according to ARI score and use them to perform the final clustering (DBSCAN).
best_eps_ari, best_min_samples_ari, ari = max(summary_ari[['eps', 'min_samples', 'ari']].values, key=lambda x: x[2])
dbscan_ari_labels = DBSCAN(eps=best_eps_ari, min_samples=int(best_min_samples_ari)).fit(X).labels_

# Compute the performance metrics (both unsupervised and supervised) for DBSCAN with ARI score.
try:
    silhouette_ari = silhouette_score(X, dbscan_ari_labels)
except:
    silhouette_ari = 0.0
ari_ari = ari

# using the 'plot_cluster_vs_actual_class' function that you implemented, plot the groups formed by the clustering algorithm and the true classes.
plot_cluster_vs_actual_class(X, dbscan_ari_labels, 'DBSCAN_ARI', silhouette_ari, ari_ari)
