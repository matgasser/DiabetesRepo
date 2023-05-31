import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import os
from sklearn.cluster import DBSCAN
import pandas as pd


def plot_cluster_vs_actual_class(X, y_true, cluster_labels, algorithm, unsupervised_score, supervised_score):
    """
    Plots the true classes vs the groups found by kmeans.

    Parameters:
        X (numpy array): The data to be clustered.
        y (numpy array): True classes
        cluster_labels (numpy array): The labels assigned by the clustering algorithm
        score (float): clustering score
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
    # compute number of clusters
    n_clusters = len(np.unique(cluster_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # 2nd Plot showing the actual clusters formed.
    # check X shape. If it is already 2 dimensional it means that we already applied PCA or the dataset has only 2 features
    if X.shape[1] != 2:
        # apply PCA for visualization purposes
        X_plot = PCA().fit_transform(X)
    else:
        X_plot = X.copy()

    # create legend labels using the cluster assigned
    cluster_labels_plot = [f'cluster {i}' for i in cluster_labels]

    # sort the labels from lowest to highest
    hue_order = np.sort(np.unique(cluster_labels_plot))

    sns.scatterplot(x=X_plot[:, 0],
                    y=X_plot[:, 1],
                    hue=cluster_labels_plot,
                    hue_order=hue_order,
                    ax=ax1,
                    palette=sns.color_palette(n_colors=len(np.unique(cluster_labels)))
                    )

    # set title and labels
    ax1.set_title(f"Groups assigned by clustering algorithm")
    ax1.set_xlabel("PCA comp 1")
    ax1.set_ylabel("PCA comp 2")

    y_labels_plot = [MAPPING[i] for i in y_true]

    # 2nd plot.
    sns.scatterplot(x=X_plot[:, 0],
                    y=X_plot[:, 1],
                    hue=y_labels_plot,
                    ax=ax2,
                    palette=sns.color_palette(n_colors=len(np.unique(y_true)))
                    )

    # set title and labels
    ax2.set_title(f'True labels')
    ax2.set_xlabel("PCA comp 1")
    ax2.set_ylabel("PCA comp 2")

    if algorithm == 'KMeans':
        score = 'WCSS'
    else:
        score = 'silouhette'
        n_clusters = n_clusters - (1 if -1 in cluster_labels else 0)

    # set global title
    plt.suptitle(
        f"n_clusters = {n_clusters} - {score} score: {unsupervised_score:.2f} - ARI score: {supervised_score:.2f}",
        fontsize=14,
        fontweight="bold",
        )
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../output/{algorithm} Plot_true_vs_cluster_for_{n_clusters}_clusters.png")


# function to manually calculate the sum of squared distances to evaluate clustering performances.
# It is one of the attribute of K-Means
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
    # X is n_samples * n_features
    # cluster centers n_cluster * n_features
    # cdist calculate the distance of each sample from each cluster center
    # then we take the minimum of the distance to find to which center the sample is closer
    # Output of cdist will be n_samples * n_cluster
    # Calculate squared euclidean distance between samples
    # Have a look at scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    distance_matrix = cdist(X, cluster_centers, 'euclidean')
    # square euclidean distance
    distance_matrix_squared = distance_matrix ** 2
    # Find the minimum distance to find to which center the sample is closer. Specify the correct axis
    # from which you want to compute the minimum
    min_distances = np.min(distance_matrix_squared, axis=1)
    # compute the sum
    wcss = np.sum(min_distances)

    return wcss


def apply_kmeans(X, y, n_clusters):
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
    # Within cluster Squared Sum
    wcss = []
    ari_scores = []

    # iterate through the array to test different n_clusters for K-Means
    for i in n_clusters:
        # instantiate K means object
        kmeans = KMeans(n_clusters=i,
                        init='k-means++',
                        random_state=0)
        # fir algorithm
        kmeans.fit(X)
        # compute score to evaluate how well data are clustered around the cluster centers
        # Have a look at Elbow method
        try:
            unsupervised_score = calculate_WCSS(X, kmeans.cluster_centers_)
            if abs(unsupervised_score - kmeans.inertia_) < 1e-4:
                print('Your implementation is correct. Using your WCSS')
            else:
                print('Your implementation is NOT correct. Using KMeans attribute')
                unsupervised_score = kmeans.inertia_
        except:
            print('Your implementation has an error. Using KMeans attribute')
            unsupervised_score = kmeans.inertia_

        wcss.append(unsupervised_score)

        # Compute the Adjusted Rand Index to evaluate the clustering performance
        # if you have the true classes
        supervised_score = adjusted_rand_score(y, kmeans.labels_)
        ari_scores.append(supervised_score)

        plot_cluster_vs_actual_class(X, y, kmeans.labels_, 'KMeans', unsupervised_score, supervised_score)

    return wcss, ari_scores


def apply_dbscan(X, y, eps_range, min_samples_range):
    """
    Plots the true classes vs the groups found by kmeans.

    Parameters:
        X (numpy array): The data to be clustered.
        y (numpy array): True classes
        eps_range (numpy array or list): list of values to iterate on for eps hyperparameter
        min_samples_range (numpy array or list): list of values to iterate on for min_samples hyperparameter
    """
    scores_silouhette = []
    scores_ari = []
    # find the best eps and min_samples values
    # Define for loop to test all the combinations of eps and min_samples
    summary = pd.DataFrame(columns=['eps', 'min_samples', 'n_clusters', 'silouhette', 'ari'])
    count = 0
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X)
            labels = dbscan.labels_
            ari = adjusted_rand_score(y, dbscan.labels_)
            try:
                silouhette = silhouette_score(X, labels)
            except:
                silouhette = 0.0

            summary.loc['iteration_' + str(count + 1), 'eps'] = eps
            summary.loc['iteration_' + str(count + 1), 'min_samples'] = min_samples
            summary.loc['iteration_' + str(count + 1), 'n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
            summary.loc['iteration_' + str(count + 1), 'silouhette'] = silouhette
            summary.loc['iteration_' + str(count + 1), 'ari'] = adjusted_rand_score(y, labels)
            count += 1
            # stores values in a list
            scores_silouhette.append((eps, min_samples, silouhette))
            scores_ari.append((eps, min_samples, ari))

    return scores_silouhette, scores_ari, summary




cleandat = pd.read_csv(r'../data/clean_Data.csv')
cd = cleandat.dropna()
scaler = StandardScaler()
# Read data
X = pd.DataFrame(scaler.fit_transform(cd.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = cd.Outcome

""""#X = X.loc[y.index.to_list()]
# Standardize the dataset
#scaler = preprocessing.StandardScaler()
#X = scaler.fit_transform(X)"""

y = y.values.reshape(-1)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
MAPPING = {i: le.classes_[i] for i in np.arange(len(le.classes_))}

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

# next, we sort and plot the results
distances = np.sort(distances, axis=0)
distances = distances[:,1]

fig = px.scatter(
    distances,
    title='Distance Curve')
fig.update_xaxes(title_text='Distances')
fig.update_yaxes(title_text='Distance threashold (espsilon)')
fig.update_layout(showlegend=False)
fig.show()


max_n_clusters = 12
# create an array that contain the different number of clusters from 2 to max_n_cluster
n_clusters = np.arange(2, max_n_clusters)

wcss, ari_scores = apply_kmeans(X, y, n_clusters)

plt.figure(figsize=(12, 8))
plt.plot(n_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.tight_layout()
plt.savefig('../Output/plots')
plt.show()

# DBSCAN
# scores manually defined to make a search.
# Define a list for eps and min_samples parameters of DBSCAN algorithm
eps_range = np.arange(2, 3, 0.1).tolist()
#eps_range = [x in range(2,3,+0.1)]
min_samples_range = np.arange(2, 10, 1).tolist()
# lists to store the results of the scores. Define one supervised (you know the true classes)
# and one unsupervised (you don't know the true classes) score

scores_silouhette, scores_ari, summary = apply_dbscan(X, y, eps_range, min_samples_range)

# sort summary according to silouhette and ARI
summary_silouhette = summary.sort_values(by='silouhette', ascending=False)[:5]

summary_silouhette.to_csv('../Output/summary_DBSCAN_silouhette_5_best_al.csv')

summary_ari = summary.sort_values(by='ari', ascending=False)[:5]

summary_ari.to_csv('../Output/summary_DBSCAN_ari_5_best_al.csv')

# extract best values for the hyperparameters according to ARI and fit again the DBSCAN with
# the extracted best parameters
best_eps, best_min_samples, silouhette = max(scores_silouhette, key=lambda x: x[2])

print(f"Optimal value of eps: {best_eps}")
print(f"Optimal value of min_samples: {best_min_samples}")
print(f"Silouhette score is: {silouhette}")

dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan.fit(X)
# Calculate scores
try:
    silouhette = silhouette_score(X, dbscan.labels_)
except:
    silouhette = 0.0

ari = adjusted_rand_score(y, dbscan.labels_)
# call function to plot
plot_cluster_vs_actual_class(X, y, dbscan.labels_, "DBSCAN_silouhette", silouhette, ari)

# extract best values for the hyperparameters according to ARI and fit again the DBSCAN with
# the extracted best parameters
best_eps, best_min_samples, ari = max(scores_ari, key=lambda x: x[2])

print(f"Optimal value of eps: {best_eps}")
print(f"Optimal value of min_samples: {best_min_samples}")
print(f"ARI score is: {ari}")

dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan.fit(X)
# Calculate scores
try:
    silouhette = silhouette_score(X, dbscan.labels_)
except:
    silouhette = 0.0

ari = adjusted_rand_score(y, dbscan.labels_)
# call function to plot
plot_cluster_vs_actual_class(X, y, dbscan.labels_, "DBSCAN_ARI", silouhette, ari)