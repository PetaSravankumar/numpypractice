 About This Repository
This notebook contains all my Day 1 NumPy practice exercises as part of my 45-day Machine Learning and AI study plan.
Every exercise was written from scratch, tested, reviewed, and corrected with detailed explanations.
DetailInfo📅 Completed OnDay 1 of 45-Day ML Journey📓 Notebooknumpy_notes.ipynb🐍 LanguagePython 3.13📦 LibraryNumPy🏆 Score85 / 100✅ Exercises10 Cells / 12 Concepts🛠️ ToolJupyter Notebook (Anaconda)

📚 What I Learned
NumPy Arrays → Reshape → Slicing → Boolean Indexing →
Stacking → Math Operations → Statistics → nditer →
Student Grades Problem → Normalization → House Dataset (ML Simulation) 

🗂️ Exercise Breakdown

🔵 Exercise 1 — Array Reshape & Flatten
File: Cell 1 | Difficulty: 🟢 Beginner
What I practiced:

np.arange() to create sequential arrays
.reshape(3, -1) — using the -1 trick to auto-calculate dimensions
.reshape(2, 6) and .reshape(2, 2, 3) — 2D and 3D reshaping
.flatten() — converting multi-dimensional array back to 1D

Code:
pythonimport numpy as np
a = np.arange(1, 13)
print(a)
print("reshaped to 3x4:", a.reshape(3, -1))
print("reshaped to 2x6:", a.reshape(2, 6))
print("reshaped to 2x2x3:", a.reshape(2, 2, 3))
print("flattened:", a.flatten())
Output:
[ 1  2  3  4  5  6  7  8  9 10 11 12]
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
[[[ 1  2  3]
  [ 4  5  6]]
 [[ 7  8  9]
  [10 11 12]]]
[ 1  2  3  4  5  6  7  8  9 10 11 12]
Key Concept Learned:

-1 in reshape means "NumPy, calculate this dimension automatically".
reshape(3, -1) on 12 elements → NumPy calculates 12 ÷ 3 = 4 columns.
This is used in every deep learning model when flattening images before feeding into a neural network.


🔵 Exercise 2 — Array Slicing & Indexing
File: Cell 2 | Difficulty: 🟡 Intermediate
What I practiced:

arr[row][col] vs arr[row, col] — two ways to access elements
arr[:, col] — extracting entire column
arr[r1:r2, c1:c2] — extracting 2D blocks
Negative indexing for bottom rows

Code:
pythonimport numpy as np
arr = np.array([[10, 20, 30, 40],
                [50, 60, 70, 80],
                [90,100,110,120]])

print("array:", arr)
print("element 70:", arr[1][2])   # or arr[1, 2]
print("3rd column:", arr[:, 2])
print("top-right 2x2:", arr[:2, 2:])
print("bottom-left 2x2:", arr[1:, :2])
Output:
element 70: 70
3rd column: [ 30  70 110]
top-right 2x2:
[[ 30  40]
 [ 70  80]]
bottom-left 2x2:
[[ 50  60]
 [ 90 100]]
Key Concept Learned:

arr[row, col] is preferred NumPy style over arr[row][col].
arr[:, 2] = entire column 2.
arr[-2:, :2] = last 2 rows, first 2 columns.


🔵 Exercise 3 — Boolean Indexing
File: Cell 3 | Difficulty: 🟡 Intermediate
What I practiced:

Filtering elements with conditions: arr[arr > 30]
Finding even numbers: arr[arr % 2 == 0]
Multiple conditions: (arr > 20) & (arr < 60)
Replacing values: arr[arr < 20] = 0
Counting with conditions: np.sum(arr > 50)

Code:
pythonimport numpy as np
arr = np.array([15, 42, 8, 73, 29, 55, 6, 88, 34, 61])

print(">30:", arr[arr > 30])
print("even:", arr[arr % 2 == 0])
print("between 20-60:", arr[(arr > 20) & (arr < 60)])
arr[arr < 20] = 0
print("after replacing <20 with 0:", arr)
print("count >50:", np.sum(arr > 50))
Output:
>30: [42 73 55 88 34 61]
even: [42  8  6 88 34]
between 20-60: [42 29 55 34]
after replacing: [ 0 42  0 73 29 55  0 88 34 61]
count >50: 4
Key Concept Learned:

Always use arr_copy = arr.copy() before modifying with boolean indexing
so the original array stays unchanged.
np.sum(condition) counts how many elements satisfy the condition.


🔵 Exercise 4 — Array Stacking
File: Cell 4 | Difficulty: 🟡 Intermediate
What I practiced:

np.vstack() — vertical stacking (adds more rows)
np.hstack() — horizontal stacking (adds more columns)
Always printing .shape after stacking to verify

Code:
pythonimport numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])

c = np.vstack((a, b))
d = np.hstack((a, b))

print("vertical stack:\n", c)
print("horizontal stack:\n", d)
print("vertical shape:", c.shape)
print("horizontal shape:", d.shape)
Output:
vertical stack:
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
shape: (4, 3)

horizontal stack:
[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]
shape: (2, 6)
Key Concept Learned:

vstack = more rows (stack on top).
hstack = more columns (stack side by side).
Always verify shape after stacking — shape errors are the #1 bug in ML!


🔵 Exercise 5 — Math Operations
File: Cell 5 | Difficulty: 🟢 Beginner
What I practiced:

Element-wise: +, -, *, /
np.dot() — dot product (matrix multiplication)
a ** 2 — squaring every element
np.sqrt() — square root of every element

Code:
pythonimport numpy as np
a = np.array([10, 20, 30, 40, 50])
b = np.array([ 2,  4,  6,  8, 10])

print("add:", a + b)
print("subtract:", a - b)
print("multiply:", a * b)
print("divide:", a / b)
print("dot product:", np.dot(a, b))
print("square:", a ** 2)
print("sqrt:", np.sqrt(b))
Output:
add:        [12 24 36 48 60]
subtract:   [ 8 16 24 32 40]
multiply:   [ 20  80 180 320 500]
divide:     [5. 5. 5. 5. 5.]
dot product: 1100
square:     [ 100  400  900 1600 2500]
sqrt:       [1.414 2.0 2.449 2.828 3.162]
Key Concept Learned:

np.dot(a, b) = (10×2)+(20×4)+(30×6)+(40×8)+(50×10) = 1100.
The dot product is the core operation of every neural network layer —
output = np.dot(inputs, weights) + bias


🔵 Exercise 6 — Statistics Functions
File: Cell 6 | Difficulty: 🟢 Beginner
What I practiced:

np.sum(), np.mean(), np.max(), np.min()
np.std(), np.median()
np.argmax(), np.argmin() — index of max/min values

Code:
pythonimport numpy as np
sales = np.array([45, 78, 62, 91, 55, 83, 70, 48, 95, 67, 74, 88])

print("total:", np.sum(sales))       # 856
print("average:", np.mean(sales))    # 71.33
print("max:", np.max(sales))         # 95
print("min:", np.min(sales))         # 45
print("std dev:", np.std(sales))     # 15.87
print("median:", np.median(sales))   # 72.0
print("best month index:", np.argmax(sales))   # 8
print("worst month index:", np.argmin(sales))  # 0
Output:
total:            856
average:          71.33
max:              95
min:              45
std deviation:    15.87
median:           72.0
best month index: 8   → September
worst month index: 0  → January
Key Concept Learned:

argmax() returns the INDEX of the maximum — not the value itself.
In neural networks: np.argmax(predictions) finds which class the model predicted.
Example: [0.1, 0.7, 0.2] → argmax = 1 → class "dog" 🐕


🔵 Exercise 7 — nditer (Array Iteration)
File: Cell 7 | Difficulty: 🟡 Intermediate
What I practiced:

np.nditer() — iterating every element in a 2D array
Adding conditions inside the loop
Math operations inside the loop
Using end="" to print on same line

Code:
pythonimport numpy as np
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# All elements
for x in np.nditer(arr2d):
    print(x, end=" ")
print("\n-------------")

# Elements > 5
for x in np.nditer(arr2d):
    if x > 5:
        print(x, end=" ")
print("\n-------------")

# Squares
for x in np.nditer(arr2d):
    print(x**2, end=" ")
Output:
1 2 3 4 5 6 7 8 9
-------------
6 7 8 9
-------------
1 4 9 16 25 36 49 64 81
Key Concept Learned:

Regular for row in arr2d loops row by row.
np.nditer() goes element by element — very useful for custom operations on every value.


🔵 Exercise 8 — Student Grades (Real-World Problem)
File: Cell 8 | Difficulty: 🔴 Advanced
What I practiced:

axis=1 — mean across columns (per student average)
axis=0 — mean down rows (per subject average)
np.argmax() on computed averages
np.argmin() to find lowest subject
Min-Max normalization on 2D array

Code:
pythonimport numpy as np
grades = np.array([[85, 92, 78, 88],
                   [72, 68, 90, 75],
                   [95, 88, 92, 96],
                   [60, 74, 65, 70],
                   [88, 79, 83, 91]])

# Average per student (row-wise)
print(np.mean(grades, axis=1))

# Average per subject (column-wise)
print(np.mean(grades, axis=0))

# Best student index
print(np.argmax(np.mean(grades, axis=1)))

# Lowest subject index
print(np.argmin(np.mean(grades, axis=0)))

# Normalize grades to 0-1
min1 = np.min(grades)
max1 = np.max(grades)
print((grades - min1) / (max1 - min1))
Output:
Student averages: [85.75  76.25  92.75  67.25  85.25]
Subject averages: [80.  80.2  81.6  84. ]
Best student index: 2   → Student 3 (avg 92.75)
Lowest subject: 0       → Math (avg 80.0)
Normalized grades:
[[0.69 0.89 0.50 0.78]
 [0.33 0.22 0.83 0.42]
 [0.97 0.78 0.89 1.00]
 [0.00 0.39 0.14 0.28]
 [0.78 0.53 0.64 0.86]]
Key Concept Learned:

axis=0 → goes DOWN rows → result per column (subject average).
axis=1 → goes ACROSS columns → result per row (student average).
This is the most confusing concept in NumPy — draw it to remember!


🔵 Exercise 9 — Normalization (Critical for ML!)
File: Cell 9 | Difficulty: 🔴 Advanced
What I practiced:

Min-Max Normalization — scales data to 0–1 range
Z-score Standardization — scales data to mean=0, std=1
Understanding when to use each method

Code:
pythonimport numpy as np
data = np.array([200, 450, 1200, 800, 350, 950, 100, 600])

# Method 1: Min-Max Normalization (0 to 1)
min1 = np.min(data)
max1 = np.max(data)
norm = (data - min1) / (max1 - min1)
print("Min-Max:", norm)

# Method 2: Z-score Standardization (mean=0, std=1)
z = (data - np.mean(data)) / np.std(data)
print("Z-score:", z)
Output:
Min-Max: [0.09 0.32 1.0  0.64 0.23 0.77 0.0  0.45]
Z-score: [-1.07 -0.37  1.74  0.61 -0.65  1.04 -1.35  0.05]
Key Concept Learned:

This is exactly what StandardScaler and MinMaxScaler do in scikit-learn!
You manually coded what sklearn does automatically — now you understand it from the inside.
MethodFormulaRangeUse WhenMin-Max(x-min)/(max-min)0 to 1Neural networks, imagesZ-score(x-mean)/stdmean=0 std=1Most ML algorithms


🔵 Exercise 10 — House Price Dataset (ML Simulation)
File: Cell 10 | Difficulty: 🔴 Advanced
What I practiced:

Separating features (X) and target (y) — the first step in every ML project
np.argmax() to find most expensive house
Boolean indexing to filter houses by bedrooms
Normalizing a specific column
np.corrcoef() to find correlation between features

Code:
pythonimport numpy as np
houses = np.array([
    [1200, 3, 10,  45],
    [1800, 4,  5,  72],
    [ 900, 2, 20,  32],
    [2200, 5,  2,  95],
    [1500, 3,  8,  58],
    [ 800, 2, 25,  28],
    [2500, 5,  1, 110],
    [1100, 2, 15,  40],
    [1700, 4,  6,  68],
    [2000, 4,  3,  85]
])
# columns: [size_sqft, bedrooms, age_years, price_lakhs]

# Separate features and target
X = houses[:, :3]    # first 3 columns → features
y = houses[:, 3]     # last column → price

print("X shape:", X.shape)    # (10, 3)
print("y shape:", y.shape)    # (10,)
print("avg price:", np.mean(y))
print("most expensive:", np.argmax(y))
print("4+ bedroom houses:\n", houses[houses[:, 1] > 3])

# Normalize size column
size = X[:, 0]
norm = (size - np.min(size)) / (np.max(size) - np.min(size))
print("normalized size:", np.round(norm, 2))

# Correlation between size and price
corr = np.corrcoef(X[:, 0], y)[0, 1]
print("correlation:", round(corr, 3))
Output:
X shape: (10, 3)
y shape: (10,)
avg price: 63.3
most expensive index: 6  → [2500, 5, 1, 110]
4+ bedroom houses:
[[1800   4   5  72]
 [2200   5   2  95]
 [2500   5   1 110]
 [1700   4   6  68]
 [2000   4   3  85]]
normalized size: [0.24 0.59 0.06 0.82 0.41 0.0 1.0 0.18 0.53 0.71]
correlation: 0.998
Key Concept Learned:

Correlation of 0.998 means size and price move together almost perfectly!
This is exactly how a Linear Regression model selects which features to use.
This exercise simulates the first 10 minutes of every real ML project.
