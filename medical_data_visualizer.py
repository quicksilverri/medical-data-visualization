import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
data = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
data['BMI'] = data['weight'] * 10000 / data['height'] ** 2
data['overweight'] = data['BMI'] > 25

data['overweight'] = data['BMI'].apply(lambda x: 1 if x > 25 else 0)
data = data.drop('BMI', axis=1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
normalize = lambda x: 0 if x == 1 else 1
data['cholesterol'] = data['cholesterol'].apply(normalize)
data['gluc'] = data['gluc'].apply(normalize)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    long_data = pd.melt(data, id_vars=['id', 'cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    long_data['total'] = 1
    long_data = long_data.groupby(['cardio', 'variable', 'value'], as_index=False).count()
    
    # Draw the catplot with 'sns.catplot()'
    
    fig = sns.catplot(kind='bar', x='variable', y='total', hue='value', data=long_data, col='cardio').fig
    
# Do not modify the next two lines
    fig.savefig('catplot.png')

    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    mask_pressure = data['ap_lo'] <= data['ap_hi']
    mask_height_b = data['height'] >= data['height'].quantile(q=0.025)
    mask_height_up = data['height'] <= data['height'].quantile(q=0.975)
    mask_weight_b = data['weight'] >= data['weight'].quantile(q=0.025)
    mask_weight_up = data['weight'] <= data['weight'].quantile(q=0.975)

    clean_data = data[mask_pressure & mask_height_b & mask_height_up & mask_weight_b & mask_weight_up]

    # Calculate the correlation matrix
    # Generate a mask for the upper triangle
    plt.figure(figsize=(12, 10))
    corr = clean_data.corr()
    matrix = np.triu(corr)
    

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(clean_data.corr(method="pearson"), annot=True, mask=matrix, ax=ax, fmt='.1f', square=True)


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
