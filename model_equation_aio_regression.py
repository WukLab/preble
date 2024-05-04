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
72,
90,
100.01,
95.3,
126.48,
191.11,
271.66,
432.36,
780.74,
1709.52,
3280.41,
7302.69,

114.77,
125.73,
155.44,
247.47,
271.66,
191.11,
211.13,
277.95,
757.33,
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
