
# You-Are-What-You-Eat-Customer-Segmentation (KMeans_Clustering)
![image](https://user-images.githubusercontent.com/100878908/190882867-f6c96482-e2e6-4c9e-841e-afc5be92c351.png)


In this project we used k-means clustering to segment up the customer base in order to increase business understanding, and to enhance the relevancy of targeted messaging & customer communications.




## Overview
### 1. Context
The Senior Management team from our client, a supermarket chain, are disagreeing about how customers are shopping, and how lifestyle choices may affect which food areas customers are shopping into, or more interestingly, not shopping into.

They have asked us to use data, and Machine Learning to help segment up their customers based upon their engagement with each of the major food categories - aiding business understanding of the customer base, and to enhance the relevancy of targeted messaging & customer communications.

### 2. Actions
We firstly needed to compile the necessary data from sevaral tables in the database, namely the transactions table and the product_areas table. We joined together the relevant information using Pandas, and then aggregated the transactional data across product areas, from the most recent six month to a customer level. The final data for clustering is, for each customer, the percentage of sales allocated to each product area.

As a starting point, we test & apply k-means clustering for this task. We need to apply some data pre-processing, most importantly feature scaling to ensure all variables exist on the same scale - a very important consideration for distance based algorithms such as k-means.

As k-means is an unsupervised learning approach, in other words there are no labels - we use a process known as Within Cluster Sum of Squares (WCSS) to understand what a “good” number of clusters or segments is.

Based upon this, we apply the k-means algorithm onto the product area data, append the clusters to our customer base, and then profile the resulting customer segments to understand what the differentiating factors were!

### 3. Results
Based upon iterative testing using WCSS we settled on a customer segmentation with 3 clusters. These clusters ranged in size, with Cluster 0 accounting for 73.6% of the customer base, Cluster 2 accounting for 14.6%, and Cluster 1 accounting for 11.8%.

There were some extremely interesting findings from profiling the clusters.

For Cluster 0 we saw a significant portion of spend being allocated to each of the product areas - showing customers without any particular dietary preference.

For Cluster 1 we saw quite high proportions of spend being allocated to Fruit & Vegetables, but very little to the Dairy & Meat product areas. It could be hypothesised that these customers are following a vegan diet.

Finally customers in Cluster 2 spent significant portions within Dairy, Fruit & Vegetables, but very little in the Meat product area - so similarly, we would make an early hypothesis that these customers are more along the lines of those following a vegetarian diet.

To help embed this segmentation into the business, we have proposed to call this the “You Are What You Eat” segmentation.

### 4. Growth/Next Steps
It would be interesting to run this clustering/segmentation at a lower level of product areas, so rather than just the four areas of Meat, Dairy, Fruit, Vegetables - clustering spend across the sub-categories below those categories. This would mean we could create more specific clusters, and get an even more granular understanding of dietary preferences within the customer base.

Here we’ve just focused on variables that are linked directly to sales - it could be interesting to also include customer metrics such as distance to store, gender etc to give a even more well-rounded customer segmentation.

It would be useful to test other clustering approaches such as hierarchical clustering or DBSCAN to compare the results.


## Data Overview
We are primarily looking to discover segments of customers based upon their transactions within food based product areas so we will need to only select those.

In the code below, we:

* Import the required python packages & libraries
* Import the tables from the database
* Merge the tables to tag on product_area_name which only exists in the product_areas table
* Drop the non-food categories
* Aggregate the sales data for each product area, at customer level
* Pivot the data to get it into the right format for clustering
* Change the values from raw dollars, into a percentage of spend for each customer (to ensure each customer is comparable)

![sss](https://user-images.githubusercontent.com/100878908/190884254-52f5f83e-467f-44a2-8766-7877faf61ed5.png)
![ss](https://user-images.githubusercontent.com/100878908/190884287-e4a16174-f3a2-4c8f-bde4-d5e03737d94d.png)

The dataset sample below is at customer level, and we have a column for each of the highest level food product areas. Within each of those we have the percentage of sales that each customer allocated to that product area over the past six months.

![s3](https://user-images.githubusercontent.com/100878908/190884420-2f7b13bd-b4d0-4b69-8ec6-2b196b957c7f.png)
## KMeans Clustering
### Concept Overview
K-Means is an unsupervised learning algorithm, meaning that it does not look to predict known labels or values, but instead looks to isolate patterns within unlabelled data.

The algorithm works in a way where it partitions data-points into distinct groups (clusters) based upon their similarity to each other.

This similarity is most often the eucliedean (straight-line) distance between data-points in n-dimensional space. Each variable that is included lies on one of the dimensions in space.

The number of distinct groups (clusters) is determined by the value that is set for “k”.

The algorithm does this by iterating over four key steps, namely:

* It selects “k” random points in space (these points are known as centroids)
* It then assigns each of the data points to the nearest centroid (based upon euclidean distance)
*  It then repositions the centroids to the mean dimension values of it’s cluster
* It then reassigns each data-point to the nearest centroid
* Steps 3 & 4 continue to iterate until no data-points are reassigned to a closer centroid.

### Data Preprocessing
There are three vital preprocessing steps for k-means, namely:

* Missing values in the data
* The effect of outliers
* Feature Scaling

#### Missing Values
Missing values can cause issues for k-means, as the algorithm won’t know where to plot those data-points along the dimension where the value is not present. If we have observations with missing values, the most common options are to either remove the observations, or to use an imputer to fill-in or to estimate what those value might be.

As we aggregated our data for each customer, we actually don’t suffer from missing values so we don’t need to deal with that here.

#### Outliers
As k-means is a distance based algorithm, outliers can cause problems. The main issue we face is when we come to scale our input variables, a very important step for a distance based algorithm.

We don’t want any variables to be “bunched up” due to a single outlier value, as this will make it hard to compare their values to the other input variables. We should always investigate outliers rigorously - however in our case where we’re dealing with percentages, we thankfully don’t face this issue!

#### Feature Scaling
Again, as k-means is a distance based algorithm, in other words it is reliant on an understanding of how similar or different data points are across different dimensions in n-dimensional space, the application of Feature Scaling is extremely important.


The below code uses the in-built MinMaxScaler functionality from scikit-learn to apply Normalisation to all of our variables. The reason we create a new object (here called data_for_clustering_scaled) is that we want to use the scaled data for clustering, but when profiling the clusters later on, we may want to use the actual percentages as this may make more intuitive business sense, so it’s good to have both options available!

We pickled the scaler file to use it later for the deployment.

![s4](https://user-images.githubusercontent.com/100878908/190884780-229aa4cc-7f4a-4b1b-aa88-e940a5a42045.png)

## Finding A Good Value For k
At this point here, our data is ready to be fed into the k-means clustering algorithm. Before that however, we want to understand what number of clusters we want the data split into.
Finding the “right” value for k, can feel more like art than science, but there are some data driven approaches that can help us!

The approaches we will utilise here is known as Within Cluster Sum of Squares (WCSS) and
KElbowVisualizer. 

In the code below we will test multiple values for k, and plot how WCSS and KElbowVisualizer metric changes. As we increase the value for k (in other words, as we increase the number or centroids or clusters) the WCSS value will always decrease. However, these decreases will get smaller and smaller each time we add another centroid and we are looking for a point where this decrease is quite prominent before this point of diminishing returns.


![s5](https://user-images.githubusercontent.com/100878908/190884989-45b8e164-b8f7-45ba-850e-93e29bdfbb16.png)
![s6](https://user-images.githubusercontent.com/100878908/190884992-b7811f84-b9a2-46bc-806b-deaf824c3be4.png)


That code gives us the below plots - which visualises our results and k=3 seems a good option based on both WCCS and KElbowVisualizer.


![s7](https://user-images.githubusercontent.com/100878908/190885088-ee908b4c-265c-45bc-aa4c-f67bff5fb5d9.png)

![s8](https://user-images.githubusercontent.com/100878908/190885090-fd0da7e3-7369-478f-9427-854b31940718.png)


## Model fitting 
The below code will instantiate our k-means object using a value for k equal to 3. We then fit this object to our scaled dataset to separate our data into three distinct segments or clusters.

![s9](https://user-images.githubusercontent.com/100878908/190885215-23f60cf1-8364-4c53-8fa2-3563625d9597.png)
## Appending Clusters To Customers and Cluster Profiling
With the k-means algorithm fitted to our data, we can now append those clusters to our original dataset, meaning that each customer will be tagged with the cluster number that they most closely fit into based upon their sales data over each product area. 

Once we have our data separated into distinct clusters, our client needs to understand what is is that is driving the separation. This means the business can understand the customers within each, and the behaviours that make them unique. 

![s10](https://user-images.githubusercontent.com/100878908/190885389-3b3a5631-585f-43c1-a9ff-860c90e6756e.png)



### Cluster Sizes
The three clusters are different in size, with the following proportions:

* Cluster 0: 73.6% of customers
* Cluster 2: 14.6% of customers
* Cluster 1: 11.8% of customers
Based on these results, it does appear we do have a skew toward Cluster 0 with Cluster 1 & Cluster 2 being proportionally smaller. This isn’t right or wrong, it is simply showing up pockets of the customer base that are exhibiting different behaviours - and this is exactly what we want.

### Cluster Attributes
To understand what these different behaviours or characteristics are, we can look to analyse the attributes of each cluster, in terms of the variables we fed into the k-means algorithm.

![s11](https://user-images.githubusercontent.com/100878908/190885401-0de3de36-39e0-48d2-affb-3009f6574dff.png)


For Cluster 0 we see a reasonably significant portion of spend being allocated to each of the product areas. For Cluster 1 we see quite high proportions of spend being allocated to Fruit & Vegetables, but very little to the Dairy & Meat product areas. It could be hypothesised that these customers are following a vegan diet. Finally customers in Cluster 2 spend, on average, significant portions within Dairy, Fruit & Vegetables, but very little in the Meat product area - so similarly, we would make an early hypothesis that these customers are more along the lines of those following a vegetarian diet - very interesting!


## Pickling the Centriods
To use the model for deployment, we freezed the centriods and saved them to pickle file. 

![s12](https://user-images.githubusercontent.com/100878908/190885947-bbf12abe-e5ef-41d3-8722-bd9a78d2f82c.png)
