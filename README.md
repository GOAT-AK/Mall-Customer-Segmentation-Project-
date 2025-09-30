# Mall-Customer-Segmentation-Project
This project focuses on clustering mall customers into meaningful groups based on their Annual Income and Spending Score using K-Means clustering. The goal is to identify distinct customer segments that can guide targeted marketing strategies.

# Table of Content

* [Brief](#Brief)  
* [DataSet](#DataSet)  
* [How_It_Works](#How_It_Works)  
* [Tools](#Tools)
* [Cluster_Insights](#Cluster_Insights)
* [Strategic_Recommendations](#Strategic_Recommendations)  
* [Remarks](#Remarks)  
* [Usage](#Usage)  
* [Sample_Run](#Sample_Run)


# Brief

Customer segmentation is a crucial step in personalized marketing and business decision-making.
This project applies K-Means clustering to group customers based on spending behavior and income levels, helping businesses identify premium customers, cautious wealthy clients, impulsive buyers, and more.



# DataSet
The dataset used in this project is the [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) from Kaggle. It includes demographic and spending behavior attributes of mall visitors.


### Column Descriptions

| Attribute                  | Description                                                                |
|----------------------------|----------------------------------------------------------------------------|
| CustomerID                 | Unique customer ID.                                                        |
| Gender                     | Percentage of classes attended.                                            |
| Age                        | Age of the customer.                                                       |
| Annual Income (k$)         | Annual income of the customer (in $1000s).                                 |
| Spending Score (1-100)     | Score assigned by the mall based on spending behavior and loyalty.         |


# How_It_Works

- Load and preprocess the dataset (handle scaling for income & spending score).
- Perform **Exploratory Data Analysis (EDA)** to explore distributions, gender differences, and natural groupings. 
- Apply **Elbow Method and Silhouette Analysis** to determine the optimal number of clusters.  
- Run **K-Means clustering** and assign each customer to a cluster.
- Visualize results with scatter plots, bar charts, and cluster centroids.
- Build a **Streamlit dashboard** for interactive analysis and business insights.


# Tools & Libraries

I. Jupyter Notebook & VS Code  
II. Python 3.x  
III. pandas, numpy  
IV. matplotlib, seaborn  
V. scikit-learn  
VI. Streamlit   


# Cluster_Insights

- ***Cluster 0 (Average Spenders):*** Moderate income, average spending – steady but not exceptional.
- ***Cluster 1 (Premium Customers):*** High income, high spending – profitable and loyal, top priority. 
- ***Cluster 2 (Standard Customers):*** Average income, high spending – strong potential for upselling. 
- ***Cluster 3 (Cautious Wealthy):*** High income, low spending – need trust-building & personalized offers.
- ***Cluster 4 (Impulsive Customers):*** Low income, high spending – respond well to discounts & deals.


# Strategic_Recommendations 

* **Focus heavily on Clusters 1 & 2** → Provide loyalty programs, VIP services, and personalized experiences to retain them.
  
* **Nurture Cluster 0** → Encourage higher spending through targeted campaigns.
  
* **Investigate Cluster 3** → Understand why wealthy customers aren’t spending; improve value proposition.

* **Engage Cluster 4 carefully** → Offer discounts and deals but avoid over-investment, as their income limits long-term profitability.



# Remarks
* This Python program was run and tested in Jupyter Notebook.
* Ensure the required libraries are installed by running:

  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn streamlit

# Usage

To begin utilizing this application, follow these steps:

1. Clone this repository:
   
   ```bash
   git clone https://github.com/GOAT-AK/Mall-Customer-Segmentation-Project

2. Navigate to the cloned repository:

   ```bash
   cd Mall-Customer-Segmentation-Project

3. Run the Jupyter Notebook:

   ```bash
   Customer Segmentation.ipynb

4. Launch the Streamlit app:
   
   ```bash
   streamlit run Script1.py


# Sample_Run


* Pic 1

![Image](https://github.com/user-attachments/assets/3f6be520-abf3-462b-aba6-3c6f47d82ceb)

   
