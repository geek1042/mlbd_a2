import os
import itertools
import random
from collections import defaultdict

MINHASH_PATH = r"D:\BDML_Assign2\minhash"
MOVIELENS_PATH = r"D:\BDML_Assign2\u.data"

def read_doc(name):
    with open(os.path.join(MINHASH_PATH, name), "r", encoding="utf-8") as f:
        return f.read().strip()

docs = {
    "D1": read_doc("D1.txt"),
    "D2": read_doc("D2.txt"),
    "D3": read_doc("D3.txt"),
    "D4": read_doc("D4.txt")
}

def char_kgrams(text, k):
    return set(text[i:i+k] for i in range(len(text)-k+1))

def word_kgrams(text, k):
    words = text.split()
    return set(tuple(words[i:i+k]) for i in range(len(words)-k+1))

def jaccard(a,b):
    return len(a & b) / len(a | b)

char2 = {k: char_kgrams(v,2) for k,v in docs.items()}
char3 = {k: char_kgrams(v,3) for k,v in docs.items()}
word2 = {k: word_kgrams(v,2) for k,v in docs.items()}

# Solution 1(A)

print("\n")
print("1(A) Min-Hash using 3-grams (D1 vs D2)")
print("\n")

m = 20011

def generate_hash(t):
    funcs = []
    for _ in range(t):
        a = random.randint(1,m-1)
        b = random.randint(0,m-1)
        funcs.append((a,b))
    return funcs

def hash_value(x,a,b):
    return (a*x + b) % m

universe = set()
for s in char3.values():
    universe |= s
shingle_map = {g:i for i,g in enumerate(universe)}

def minhash(shingles, hash_funcs):
    indices = [shingle_map[s] for s in shingles]
    sig = []
    for a,b in hash_funcs:
        sig.append(min(hash_value(i,a,b) for i in indices))
    return sig

def approx(sig1,sig2):
    return sum(1 for a,b in zip(sig1,sig2) if a==b)/len(sig1)

exact_val = jaccard(char3["D1"], char3["D2"])
print("Exact Jaccard =", exact_val)

for t in [20,60,150,300,600]:
    funcs = generate_hash(t)
    s1 = minhash(char3["D1"], funcs)
    s2 = minhash(char3["D2"], funcs)
    print("t =",t,"Approximate Jaccard =",approx(s1,s2))


# Solution 1(B)

print("\n")
print("1(B) Exact Jaccard for All Pairs")
print("\n")

print("\nCharacter 2-grams")
for a,b in itertools.combinations(docs.keys(),2):
    print(a,b,jaccard(char2[a],char2[b]))

print("\nCharacter 3-grams")
for a,b in itertools.combinations(docs.keys(),2):
    print(a,b,jaccard(char3[a],char3[b]))

print("\nWord 2-grams")
for a,b in itertools.combinations(docs.keys(),2):
    print(a,b,jaccard(word2[a],word2[b]))

# Solution 2(A)

print("\n")
print("2(A) Min-Hash Approximation")
print("\n")

for t in [20,60,150,300,600]:
    funcs = generate_hash(t)
    s1 = minhash(char3["D1"], funcs)
    s2 = minhash(char3["D2"], funcs)
    print("t =",t,"Approximate Jaccard =",approx(s1,s2))

# Solution 2(B)

print("\n")
print("2(B) Good value of t")

for t in [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,450,600]:
    funcs = generate_hash(t)
    s1 = minhash(char3["D1"], funcs)
    s2 = minhash(char3["D2"], funcs)
    print("t =",t,"Approximate Jaccard =",approx(s1,s2))

print("\n")

# Solution 3(B)

print("\n")
print("3(B) LSH Probability for Each Pair")
print("\n")
r=4
b=40

def lsh_prob(s,r,b):
    return 1 - (1 - s**r)**b

for a,b_doc in itertools.combinations(docs.keys(),2):
    sim = jaccard(char3[a],char3[b_doc])
    print(a,b_doc,"Probability =",lsh_prob(sim,r,b))

# Solution 4(A)

print("\n")
print("4(A) Exact Jaccard >= 0.5 (MovieLens)")
print("\n")

def load_movielens():
    users = defaultdict(set)
    with open(MOVIELENS_PATH,"r") as f:
        for line in f:
            user,movie,_,_ = line.strip().split("\t")
            users[int(user)].add(int(movie))
    return users

users = load_movielens()

keys = list(users.keys())
exact_pairs = set()

for i in range(len(keys)):
    for j in range(i+1,len(keys)):
        if jaccard(users[keys[i]],users[keys[j]]) >= 0.5:
            exact_pairs.add((keys[i],keys[j]))

print("Exact pairs count =",len(exact_pairs))

# Solution 4(B)

print("\n")
print("4(B) Min-Hash Results (Average over 5 runs)")
print("\n")

def user_minhash(users,t):
    universe = set()
    for s in users.values():
        universe |= s
    mapping = {g:i for i,g in enumerate(universe)}
    funcs = generate_hash(t)
    signatures = {}
    for u in users:
        indices = [mapping[i] for i in users[u]]
        sig = []
        for a,b in funcs:
            sig.append(min(hash_value(i,a,b) for i in indices))
        signatures[u] = sig
    return signatures

def approx_user_pairs(signatures,threshold):
    keys = list(signatures.keys())
    pairs = set()
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            if approx(signatures[keys[i]],signatures[keys[j]]) >= threshold:
                pairs.add((keys[i],keys[j]))
    return pairs

for t in [50,100,200]:
    fp_total = 0
    fn_total = 0
    for _ in range(5):
        sigs = user_minhash(users,t)
        approx_pairs = approx_user_pairs(sigs,0.5)
        fp_total += len(approx_pairs - exact_pairs)
        fn_total += len(exact_pairs - approx_pairs)
    print("t =",t,"avg FP =",fp_total/5,"avg FN =",fn_total/5)

# Solution 5

print("\n")
print("5) LSH on MovieLens (Average over 5 runs)")
print("\n")

def lsh_users(signatures,r,b):
    bands = []
    for i in range(b):
        bands.append(defaultdict(list))
    for u,sig in signatures.items():
        for i in range(b):
            start = i*r
            end = start+r
            key = tuple(sig[start:end])
            bands[i][key].append(u)
    candidates = set()
    for band in bands:
        for bucket in band.values():
            if len(bucket)>1:
                for pair in itertools.combinations(bucket,2):
                    candidates.add(tuple(sorted(pair)))
    return candidates

configs = [
    (50,5,10),
    (100,5,20),
    (200,5,40),
    (200,10,20)
]

for t,r,b in configs:
    fp_total = 0
    fn_total = 0
    for _ in range(5):
        sigs = user_minhash(users,t)
        candidates = lsh_users(sigs,r,b)
        predicted = set()
        for u,v in candidates:
            if approx(sigs[u],sigs[v]) >= 0.6:
                predicted.add((u,v))
        fp_total += len(predicted - exact_pairs)
        fn_total += len(exact_pairs - predicted)
    print("t =",t,"r =",r,"b =",b,"avg FP =",fp_total/5,"avg FN =",fn_total/5)
