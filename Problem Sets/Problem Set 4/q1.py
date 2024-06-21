import numpy as np
import matplotlib.pyplot as plt

#Update this to ur path
TXT_FILE_PATH = r'/Volumes/T7 Shield APFS/Data-319-Model-Data-Based-Methods/Problem Sets/Problem Set 4/ratings.train.txt'

#params
k = 20
lambd = 1/10
eta = 1/100
iterations = 40

#init P and Q
np.random.seed(4)
P = np.random.rand(1000, k) * np.sqrt(5/k)
Q = np.random.rand(90000, k) * np.sqrt(5/k)

def read_ratings(path):
    '''this function reads the numbers from disk and yeilds them one by one
    so that you dont need to use a with open statement every time
    @param The path to the txt file
    @return a generator that yields the user, item and rating'''
    with open(path, 'r') as f: #open file
        for line in f: #iterate thru lines
            u, i, r = map(int, line.split('\t'))
            yield u, i, r #yeild values
            
def calc_error(R, Q, P, lambd):
    '''This function calculates the errror of the model.
    @param R the path to the ratings file
    @param Q a vector 
    @param P a Vector
    @param lambd the regularization param
    @return the calculated error'''
    error = 0
    for u, i, r in read_ratings(R): #read u, i, and r from the file
        error += (r - np.dot(Q[i], P[u].T)) ** 2
    error += lambd * (np.sum(np.linalg.norm(P, axis = 1) ** 2) + np.sum(np.linalg.norm(Q, axis = 1) ** 2))
    return error

errors = [] #holds error values for graph
PrevQ = Q.copy()

#SGD
for iteration in range(iterations):
    prevQ = Q.copy()
    for u, i, r in read_ratings(TXT_FILE_PATH):
        e = 2 * (r - np.dot(Q[i], P[u].T))
        Q[i] += eta * (e * P[u] - 2 * lambd * Q[i])
        P[u] += eta * (e * prevQ[i] - 2 * lambd * P[u])
        
    error = calc_error(TXT_FILE_PATH, Q, P, lambd)
    errors.append(error)
    print(f"Iteration {iteration + 1}, Error: {error}")
    if error < 70000:
        print('Error is Below 70,0000 :) ')
        
#Plots
plt.plot(range(1, iterations + 1), errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()        
    
        