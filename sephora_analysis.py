# -*- coding: utf-8 -*-
"""
Project: Data Analysis of Sephora Products

Summary: Analyzes Sephora product data to identify trends, recommend similar products, and predict ratings using ML models.
Dataset: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import layers

# Load and clean data
product_overview = pd.read_csv('product_info.csv')
product_overview_filter = product_overview[product_overview['price_usd'] < 50].drop_duplicates(subset=['product_name'])
product_overview['highlights'] = product_overview['highlights'].fillna('')
product_overview['loves_count'] = product_overview['loves_count'].astype(float)

# --- Exploratory Visualizations ---
sns.barplot(data=product_overview_filter, x='rating', y='primary_category')
sns.scatterplot(data=product_overview_filter, y='price_usd', x='rating')
sns.jointplot(data=product_overview_filter, y='price_usd', x='rating', kind='hex', color='blue')
sns.scatterplot(data=product_overview_filter, y='price_usd', x='loves_count')

# --- Content-based Recommendation System ---
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
tfidf_matrix = tfidf.fit_transform(product_overview['highlights'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(product_overview.index, index=product_overview['product_name']).drop_duplicates()

def get_recommendations(product_name, cosine_sim=cosine_sim):
    idx = indices[product_name]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
    product_indices = [i[0] for i in sim_scores]
    return product_overview[['product_name', 'price_usd']].iloc[product_indices]

# Example
print(get_recommendations('Superstars Anti-Aging Serum and Eye Kit'))

# --- Linear Regression ---
product_overview_filter.dropna(subset=['loves_count', 'rating', 'new', 'price_usd', 'primary_category', 'online_only', 'limited_edition', 'sephora_exclusive', 'reviews'], inplace=True)
df_feature = product_overview_filter[['loves_count', 'price_usd', 'new', 'limited_edition', 'primary_category', 'online_only', 'sephora_exclusive', 'reviews']]
df_target = product_overview_filter[['rating']]

X_train, X_test, y_train, y_test = train_test_split(df_feature, df_target, test_size=0.2, random_state=42)
X_train = sm.add_constant(pd.get_dummies(X_train, columns=['primary_category'], dtype=int))
X_test = sm.add_constant(pd.get_dummies(X_test, columns=['primary_category'], dtype=int))

ols_model = sm.OLS(y_train, X_train).fit()
print(ols_model.summary())

# --- Random Forest Regressor ---
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(product_overview_filter[['price_usd']], product_overview_filter[['rating']], test_size=0.2, random_state=42)
rf_model = RandomForestRegressor()
rf_model.fit(X_train_rf, y_train_rf)
pred_rf = rf_model.predict(X_test_rf)
print("R2 score (Random Forest):", r2_score(y_test_rf, pred_rf))

# --- Neural Network Model ---
column_names = ['loves_count', 'rating', 'price_usd']
dataset = product_overview[column_names].dropna()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe().drop(columns='rating').transpose()
train_labels = train_dataset.pop('rating')
test_labels = test_dataset.pop('rating')

norm = lambda x: (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mae', 'mse'])
model.fit(normed_train_data, train_labels, epochs=100, validation_split=0.2, verbose=0)

test_predictions = model.predict(normed_test_data).flatten()

plt.figure(figsize=(6,6))
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values Rating')
plt.ylabel('Predictions Rating')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [rating 1-5]")
plt.ylabel("Count")
plt.show()
