import numpy as np

# following is based off from algorithm 26.1
def hessenberg(matrix):
    A=matrix.copy().astype(float)
    m = A.shape[0]

    for k in range(m-2):
        x=A[k+1:m,k]
        v = _getSign(x[0])*np.linalg.norm(x,2)*_getE1(x.shape[0]) + x
        v = v/np.linalg.norm(v,2)

        A[k+1:m,k:m]=A[k+1:m,k:m]-2*np.outer(v,np.matmul(v,A[k+1:m,k:m]))
        A[0:m,k+1:m]=A[0:m,k+1:m]-2*np.outer(np.matmul(A[0:m,k+1:m],v.T),v)
        
    return(A.round(8))

def fullHessenberg(matrix):
    A=matrix.copy().astype(float)
    m = A.shape[0]
    Q=np.identity(A.shape[0])

    for k in range(m-2):
        x=A[k+1:m,k]
        v = _getSign(x[0])*np.linalg.norm(x,2)*_getE1(x.shape[0]) + x
        v = v/np.linalg.norm(v,2)

        q_k=_getHV(v)
        P=np.identity(A.shape[0])
        P[k+1:,k+1:] = q_k
        Q=np.matmul(Q,P)

        A[k+1:m,k:m]=A[k+1:m,k:m]-2*np.outer(v,np.matmul(v,A[k+1:m,k:m]))
        A[0:m,k+1:m]=A[0:m,k+1:m]-2*np.outer(np.matmul(A[0:m,k+1:m],v.T),v)   
    return(A.round(8),Q)

def householder(matrix):
    A=matrix.copy().astype(float)
    n=matrix.shape[1]
    v=np.zeros(matrix.shape, float)
    q_star=np.identity(A.shape[0])
    for k in range(n):
        x=A[k:,k]
        v[k:,k]=_getSign(x[0])*np.linalg.norm(x,2)*_getE1(x.shape[0])+x
        v[k:,k]=v[k:,k]/np.linalg.norm(v[k:,k],2)
        q_k=np.identity(A.shape[0])
        q_k[k:,k:]=_getHV(v[k:,k])
        q_star=np.matmul(q_k,q_star)
        # print(v[k:,k][np.newaxis].T.shape)
        # print(v[k:,k][np.newaxis].shape)
        A[k:,k:]=A[k:,k:]-2*np.matmul(v[k:,k][np.newaxis].T,np.matmul(v[k:,k][np.newaxis],A[k:,k:]))
    
    Q=q_star.T
    R=A.round(8)
    return(Q,R)

def solveSystem(A,b):
    Q,R = householder(A)
    # Solve for y in Qy=b, Q^*Qy=y=Q^*b
    y = np.matmul(Q.T,b)
    # Solve for x in Rx=y using back sub
    x = _backwardSub(R,y)
    return(x)

def _getHV(v):
    I=np.identity(v.shape[0])
    return(I-2*np.outer(v,v))

# function that returns an e1 vector of length m
def _getE1(m):
    e1=np.zeros((m))
    e1[0]=1
    return(e1)

# a sign function that checks and corrects for x=0 according to notes in book
def _getSign(x):
    sign=np.sign(x)
    if sign==0: sign=1
    return(sign)

# FOLLOWING FUNCTION IS BORROWED FROM MY SUBMISSION FOR HW09
# Backward Substitution Algorithm
def _backwardSub(U,y):
    n = y.shape[0]
    x = np.ones(n) # construct basic y
    for i in range(n-1,-1,-1):
        # print('i',i)
        c=0
        for j in range(n-1,i,-1):
            # print(' j',j)
            c+=U[i,j]*x[j]
        x[i] = (y[i]-c)/U[i,i]
        # print(i)
    
    return(x)