from sklearn.linear_model import LinearRegression

# Example dataset
query_tokens = [
1024, 2048, 4096, 8192, 14166, 28232,
256, 512, 1024, 512, 512, 512, 256,
]

ctx_lens = [
1024,2048,4096,8192,14166,28232,
4096, 4096, 4096, 8192, 16384, 32768, 8192,
]

ops = [q * c for q, c in zip(query_tokens, ctx_lens)]

X = [[q, c, o] for q, c, o in zip(query_tokens, ctx_lens, ops)]
y = [
9.6, 13.2, 45, 169, 482, 1827,
7.02, 11.9, 21.769, 24.1, 47.255, 95.273, 14.829
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

new_query = 512
new_ctx = 8192

p = model.predict([[new_query, new_ctx, new_query * new_ctx]])
# p = model.predict(X)
print(p)