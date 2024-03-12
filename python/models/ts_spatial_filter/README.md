# Spatial filter for time-series data
- 참조 : [Github](https://github.com/statefb/ts-spatial-filter)

- Example
```python
# hyper-parameter
win_size = len(y) // 50
if win_size % 2 == 0:
    win_size+=1

sigma_d = y.std()
sigma_i = y.std() * 5

# filter
filt = BilateralFilter(win_size=win_size,sigma_d=sigma_d,sigma_i=sigma_i)
y_filtered = filt.fit_transform(np.array(y))  # x: 1d array
```