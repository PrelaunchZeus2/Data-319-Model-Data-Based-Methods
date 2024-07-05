import numpy as np

def inv_sqrt_diag(matrix):
    '''this function takes a matrx and computes the inverse square root of the diagonal elements'''
    return np.diag(1/np.sqrt(np.diag(matrix)))

R = np.loadtxt('Problem Sets/Problem Set 4/user-shows.txt')
shows = np.genfromtxt('Problem Sets/Problem Set 4/shows.txt', dtype=str, delimiter="\n")

#Compute Matricies P and Q
P = np.diag(np.sum(R, axis = 1))
Q = np.diag(np.sum(R, axis = 0))

#Compute invsqrts
P_inv_sq = inv_sqrt_diag(P)
Q_inv_sq = inv_sqrt_diag(Q)

#compute user-user similarity
S_U = P_inv_sq @ R @ R.T @ P_inv_sq

#item tem
S_I = Q_inv_sq @ R.T @ R @ Q_inv_sq

#Compute Gammas
Gamma_U = P_inv_sq @ R @ R.T @ P_inv_sq @ R
Gamma_I = R @ Q_inv_sq @ R.T @ R @ Q_inv_sq

#User 500 item and user
U500_user = Gamma_U[499]
U500_item = Gamma_I[499]

#Zero Entries
U500_user[:100] = 0
U500_item[:100] = 0

#top 5 shows
top_5_user = np.argsort(U500_user)[-5:]
top_5_item = np.argsort(U500_item)[-5:]

print('top 5 shows for user-user:', top_5_user)
print('top 5 shows item-item:', top_5_item)



