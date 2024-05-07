from sklearn.linear_model import LinearRegression
import numpy as np


print('-'*50 + "Fitting extend" + '-'*50)
# Fit a linear regression model to attention part of prompt calculation
# Example dataset
query_tokens = [
1,
16,
64,
128,
256,
512,
1024,
2048,
4096,
8192,
14166,
28232,

256,
256,
512,
1024,
1024,
512,
512,
512,
4096,
]

ctx_lens = [
8192,
8192,
8192,
8192,
8192,
8192,
8192,
8192,
8192,
8192,
14166,
28232,

4096,
8192,
4096,
4096,
8192,
8192,
16384,
32768,
4096,
]

ops = [q * c for q, c in zip(query_tokens, ctx_lens)]

X = [[q, c, o] for q, c, o in zip(query_tokens, ctx_lens, ops)]
y = [
56.57,
56.6,
66.4,
63.72,
67.42,
80.15,
122.98,
191.95,
357.6,
771.88,
1338.21,
2942.76,

52.29,
65.74,
66.37,
106.91,
121.27,
81.9,
106.89,
158.81,
368.25,
]

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Print the coefficients and intercept
print("multi query attention Coefficients:", model.coef_)
print("multi query attention Intercept:", model.intercept_)

# Calculate and print the R2 value
r2_score = model.score(X, y)
print("multi query attention R2 value:", r2_score)

new_query = 8192
new_ctx = 8192

p = model.predict([[new_query, new_ctx, new_query * new_ctx]])
# p = model.predict(X)
print(p)
