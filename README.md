
# You-Are-What-You-Eat-Customer-Segmentation (KMeans_Clustering)
![image](https://user-images.githubusercontent.com/100878908/190882867-f6c96482-e2e6-4c9e-841e-afc5be92c351.png)


In this project we use k-means clustering to segment up the customer base in order to increase business understanding, and to enhance the relevancy of targeted messaging & customer communications.




## Overview
### Context
The Senior Management team from our client, a supermarket chain, are disagreeing about how customers are shopping, and how lifestyle choices may affect which food areas customers are shopping into, or more interestingly, not shopping into.

They have asked us to use data, and Machine Learning to help segment up their customers based upon their engagement with each of the major food categories - aiding business understanding of the customer base, and to enhance the relevancy of targeted messaging & customer communications.

### Actions
We firstly needed to compile the necessary data from sevaral tables in the database, namely the transactions table and the product_areas table. We joined together the relevant information using Pandas, and then aggregated the transactional data across product areas, from the most recent six month to a customer level. The final data for clustering is, for each customer, the percentage of sales allocated to each product area.

As a starting point, we test & apply k-means clustering for this task. We need to apply some data pre-processing, most importantly feature scaling to ensure all variables exist on the same scale - a very important consideration for distance based algorithms such as k-means.

As k-means is an unsupervised learning approach, in other words there are no labels - we use a process known as Within Cluster Sum of Squares (WCSS) to understand what a “good” number of clusters or segments is.

Based upon this, we apply the k-means algorithm onto the product area data, append the clusters to our customer base, and then profile the resulting customer segments to understand what the differentiating factors were!

### Results
Based upon iterative testing using WCSS we settled on a customer segmentation with 3 clusters. These clusters ranged in size, with Cluster 0 accounting for 73.6% of the customer base, Cluster 2 accounting for 14.6%, and Cluster 1 accounting for 11.8%.

There were some extremely interesting findings from profiling the clusters.

For Cluster 0 we saw a significant portion of spend being allocated to each of the product areas - showing customers without any particular dietary preference.

For Cluster 1 we saw quite high proportions of spend being allocated to Fruit & Vegetables, but very little to the Dairy & Meat product areas. It could be hypothesised that these customers are following a vegan diet.

Finally customers in Cluster 2 spent significant portions within Dairy, Fruit & Vegetables, but very little in the Meat product area - so similarly, we would make an early hypothesis that these customers are more along the lines of those following a vegetarian diet.

To help embed this segmentation into the business, we have proposed to call this the “You Are What You Eat” segmentation.

### Growth/Next Steps
It would be interesting to run this clustering/segmentation at a lower level of product areas, so rather than just the four areas of Meat, Dairy, Fruit, Vegetables - clustering spend across the sub-categories below those categories. This would mean we could create more specific clusters, and get an even more granular understanding of dietary preferences within the customer base.

Here we’ve just focused on variables that are linked directly to sales - it could be interesting to also include customer metrics such as distance to store, gender etc to give a even more well-rounded customer segmentation.

It would be useful to test other clustering approaches such as hierarchical clustering or DBSCAN to compare the results.


## Data Overview
We are primarily looking to discover segments of customers based upon their transactions within food based product areas so we will need to only select those.

In the code below, we:

Import the required python packages & libraries
Import the tables from the database
Merge the tables to tag on product_area_name which only exists in the product_areas table
Drop the non-food categories
Aggregate the sales data for each product area, at customer level
Pivot the data to get it into the right format for clustering
Change the values from raw dollars, into a percentage of spend for each customer (to ensure each customer is comparable)
