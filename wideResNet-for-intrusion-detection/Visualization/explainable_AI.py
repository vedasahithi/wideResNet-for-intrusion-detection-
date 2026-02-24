import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import train_test_split
import shap
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

data = pd.read_csv("sel-fea.csv",nrows=5000)
data_=np.array(data)
att_name=["serror_rate","rerror_rate","same_srv_rate","diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_srv_serror_rate","dst_host_srv_rerror_rate"]
#np.savetxt("pro_att.csv",[producttt],delimiter=',',fmt='%s')
data.head()
X = data[att_name]

Y=np.random.randint(2,size=len(X))

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,train_size=0.8, test_size=0.2,stratify=Y)
l_model = LogisticRegression()
xtrain=np.nan_to_num(xtrain)
l_model.fit(xtrain,ytrain)
background_c = shap.maskers.Independent(X, max_samples=1000)
explainer = shap.Explainer(l_model, background_c,
feature_names=list(X.columns))
shap_values_c = explainer(X)

shap.plots.scatter(shap_values_c)
shap.plots.bar(shap_values_c)
shap.plots.bar(shap_values_c.abs.max(0))
shap.plots.beeswarm(shap_values_c)
#shap.plots.heatmap(shap_values_c[:1000])


#The bar plot ranks features based on their average SHAP values, showing their importance.
shap.summary_plot(shap_values_c, X, plot_type="bar")

# Visualize a single prediction's breakdown
#.waterfall_plot(shap.Explanation(values=shap_values_c[1], base_values=explainer.expected_value, data=X.iloc[:,1]))
shap.plots.waterfall(shap_values_c[1], max_display=13)

# Decision plot
#shap.decision_plot(explainer.expected_value, shap_values_c.values, X,ignore_warnings=True)

#Beeswarm Plot
my_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#7FFFD4", "#FFCBA4"])

shap.summary_plot(shap_values_c, X, plot_type="dot",cmap=my_cmap)

num_samples = 4  # matches your example (a, b, c, d)
features = X.columns

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i in range(num_samples):
    vals = shap_values_c.values[i]  # SHAP values for sample i
    colors = ['#7FFFD4' if v < 0 else '#FFCBA4' for v in vals]

    axes[i].bar(features, vals, color=colors)
    axes[i].set_title(f"Group {i + 1}")
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()

