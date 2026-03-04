# MinHash and LSH – Machine Learning with Big Data (Assignment 2)

## Introduction

This assignment implements techniques used for large-scale similarity detection including **k-gram shingling, Jaccard similarity, Min-Hashing, and Locality Sensitive Hashing (LSH)**.

These techniques are commonly used in large data systems for applications such as:

* Document similarity detection
* Plagiarism detection
* Recommendation systems
* Information retrieval

The assignment is divided into five main parts.

1. Construction of **k-grams** and computation of Jaccard similarity between documents.
2. Implementation of **Min-Hashing** to approximate Jaccard similarity.
3. Use of **Locality Sensitive Hashing (LSH)** to identify similar document pairs efficiently.
4. Application of **Min-Hashing on the MovieLens dataset** to compute similarity between users.
5. Application of **LSH on MovieLens dataset** to detect candidate pairs with high similarity.

# Dataset Structure

Repository contents:

mlbd_a2
│
├── minhash
│   ├── D1.txt
│   ├── D2.txt
│   ├── D3.txt
│   └── D4.txt
│
├── assign.py
└── README.md

Additional dataset used:

MovieLens 100K dataset

```
u.data
```

The MovieLens dataset contains:

* 943 users
* 1682 movies
* user–movie rating interactions

# Methodology

## 1. k-Gram Construction

Three types of shingles were constructed:

* Character 2-grams
* Character 3-grams
* Word 2-grams

Shingles are stored as **sets**, therefore duplicate shingles are automatically removed.

# 1(A) Min-Hash Signature using 3-grams

The Jaccard similarity between **D1 and D2** is estimated using Min-Hash signatures with different numbers of hash functions.

Exact Jaccard similarity:

```
0.977979274611399
```

Approximate similarities:

| t (hash functions) | Approximate Jaccard |
| ------------------ | ------------------- |
| 20                 | 1.0                 |
| 60                 | 0.9833333333333333  |
| 150                | 0.9933333333333333  |
| 300                | 0.9866666666666667  |
| 600                | 0.9716666666666667  |

---

# 1(B) Jaccard Similarity for All Document Pairs

### Character 2-grams

| Pair  | Similarity         |
| ----- | ------------------ |
| D1 D2 | 0.9811320754716981 |
| D1 D3 | 0.8156996587030717 |
| D1 D4 | 0.6444444444444445 |
| D2 D3 | 0.8                |
| D2 D4 | 0.6412698412698413 |
| D3 D4 | 0.6529968454258676 |

### Character 3-grams

| Pair  | Similarity          |
| ----- | ------------------- |
| D1 D2 | 0.977979274611399   |
| D1 D3 | 0.5803571428571429  |
| D1 D4 | 0.3050847457627119  |
| D2 D3 | 0.5680473372781065  |
| D2 D4 | 0.30590339892665475 |
| D3 D4 | 0.31212381771281167 |

### Word 2-grams

| Pair  | Similarity           |
| ----- | -------------------- |
| D1 D2 | 0.9407665505226481   |
| D1 D3 | 0.18234165067178504  |
| D1 D4 | 0.03024193548387097  |
| D2 D3 | 0.1736641221374046   |
| D2 D4 | 0.030303030303030304 |
| D3 D4 | 0.01607142857142857  |

Total values reported: **18**

# 2(A) Min-Hash Approximation

Approximation using different numbers of hash functions:

| t   | Approximate Jaccard |
| --- | ------------------- |
| 20  | 1.0                 |
| 60  | 0.9333333333333333  |
| 150 | 0.9733333333333334  |
| 300 | 0.9666666666666667  |
| 600 | 0.98                |

# 2(B) Selecting a Good Value of t

To determine an appropriate number of hash functions (*t*), additional experiments were performed by varying *t* from **10 to 600**.
The approximate Jaccard similarity for **D1 and D2** was computed for each value.

### Experimental Results

| t   | Approximate Jaccard |
| --- | ------------------- |
| 10  | 1.0                 |
| 20  | 0.95                |
| 30  | 1.0                 |
| 40  | 0.975               |
| 50  | 0.98                |
| 60  | 0.9833333333333333  |
| 70  | 0.9714285714285714  |
| 80  | 0.975               |
| 90  | 0.9888888888888889  |
| 100 | 0.98                |
| 110 | 0.990909090909091   |
| 120 | 0.9833333333333333  |
| 130 | 0.9692307692307692  |
| 140 | 0.9857142857142858  |
| 150 | 0.98                |
| 160 | 0.9875              |
| 170 | 0.9823529411764705  |
| 180 | 0.9888888888888889  |
| 190 | 0.9789473684210527  |
| 200 | 0.965               |
| 210 | 0.9904761904761905  |
| 220 | 0.9681818181818181  |
| 230 | 0.9869565217391304  |
| 240 | 0.9708333333333333  |
| 250 | 0.996               |
| 260 | 0.9884615384615385  |
| 270 | 0.9777777777777777  |
| 280 | 0.9642857142857143  |
| 290 | 0.9827586206896551  |
| 300 | 0.9766666666666667  |
| 450 | 0.9822222222222222  |
| 600 | 0.9666666666666667  |

### Observation

Lower values of *t* show larger fluctuations in the estimated similarity. As *t* increases, the approximation becomes more stable and closer to the exact Jaccard similarity value.

From the experimental results, values around **150–200 hash functions** provide a stable approximation with reasonable computational cost.


# 3(A) LSH Parameter Selection

Total MinHash signature size: t = 160
Chosen parameters:
r = 4
b = 40

f(0.7) ≈ 0.99998

# 3(B) LSH Probability for Each Pair

| Pair  | Probability         |
| ----- | ------------------- |
| D1 D2 | 1.0                 |
| D1 D3 | 0.9919044206534928  |
| D1 D4 | 0.29392984078239415 |
| D2 D3 | 0.9876980128156715  |
| D2 D4 | 0.2965848032379953  |
| D3 D4 | 0.3171289518403788  |

# 4(A) Min-Hashing on MovieLens Dataset

Users with Jaccard similarity ≥ 0.5 were computed using exact similarity.
Exact pairs count = 10

# 4(B) MinHash Approximation on MovieLens

Average results over **5 runs**:

| Hash Functions | Avg False Positives | Avg False Negatives |
| -------------- | ------------------- | ------------------- |
| 50             | 172.0               | 2.6                 |
| 100            | 39.0                | 2.4                 |
| 200            | 6.8                 | 2.4                 |

Increasing the number of hash functions significantly reduces false positives.

# 5. LSH on MovieLens Dataset

Average results over **5 runs**:

| t   | r  | b  | Avg FP | Avg FN |
| --- | -- | -- | ------ | ------ |
| 50  | 5  | 10 | 2.2    | 6.4    |
| 100 | 5  | 20 | 0.6    | 7.2    |
| 200 | 5  | 40 | 0.0    | 6.8    |
| 200 | 10 | 20 | 0.0    | 8.2    |

Increasing rows per band reduces false positives, while increasing bands increases false negatives.
