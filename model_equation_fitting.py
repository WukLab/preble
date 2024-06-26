from sklearn.linear_model import LinearRegression
import numpy as np

print('-'*50 + "Fitting Linear" + '-'*50)
# Fit a linear regression model to the linear part 
X = np.array([
256,
512,
1024,
2048,
4096,
8192,
14166,
28232,
]).reshape(-1, 1)
y = [
15.318,
30.128,
59,
110.4,
208.842,
417,
725,
1423,
]
linear = LinearRegression()
# Fit the model
linear.fit(X, y)
print("Linear Coefficients:", linear.coef_)
print("Linear Intercept:", linear.intercept_)

# Calculate and print the R2 value
r2_score = linear.score(X, y)
print("Linear R2 value:", r2_score)

query = 512
p = linear.predict([[query]])
print(p)

print('-'*50 + "Fitting extend attention" + '-'*50)
# Fit a linear regression model to attention part of prompt calculation
# Example dataset
query_tokens = [
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
12.939,
13.122,
19.001,
29.326,
48.529,
75.11,
101,
287.931,
1113,

6.661,
13.122,
9.54,
14.3,
29.326,
19.001,
37.893,
75.709,
1.283,
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

new_query = 1024
new_ctx = 1024

p = model.predict([[new_query, new_ctx, new_query * new_ctx]])
# p = model.predict(X)
print(p)

print('-'*50 + "Fitting other percentage " + '-'*50)
query_tokens = [
64, 128, 192, 224, 256, 384, 512, 1024, 2048, 4096, 8192, 14166, 28232,
512,
256, 256, 512, 1024, 512, 512, 512,
]

seq_lens = [
2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 4096, 8192, 14166, 28232,
512,
4096, 8192, 4096, 4096, 8192, 16384, 32768,
]

y = [
0.209375, 0.2256690998, 0.197572314, 0.1590909091, 0.1521709786, 0.2258379434, 0.1184210526, 0.1028571429, 0.08667635118, 0.08542552624, 0.07769639776, 0.06940854917, 0.05582986066,
0.1716473001, 
0.1785189597, 0.1328608924, 0.1193159449, 0.1021236646, 0.1325932778, 0.09036196005, 0.0660668071,
]

X = [[q, s, 1/q, 1/s] for q, s in zip(query_tokens, seq_lens)]

model = LinearRegression()

# Fit the model
model.fit(X, y)

# Print the coefficients and intercept
print("multi query other percent Coefficients:", model.coef_)
print("multi query other percent Intercept:", model.intercept_)

# Calculate and print the R2 value
r2_score = model.score(X, y)
print("multi query other percent R2 value:", r2_score)

new_query = 512
new_ctx = 8291

p = model.predict([[new_query, new_ctx, 1/new_query, 1/new_ctx]])
print(p)