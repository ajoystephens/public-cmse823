import numpy as np
# shell is loosly modeled after sklearn.decomposition.PCA so that the two can be easily compared


class NumericPCA:
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self,X):
        self.n_ = X.shape[0]
        self.mean_ = np.mean(X,axis=0)
        self.covariance_ = self._getCovariance(X)
        self.eigVals = _practical_qr_algorithm(self.covariance_,5,1e-5)

        
        self._calcComponents()
        
        
        # i = np.argsort(np.abs(self.eigVals))[::-1][:self.n_components]
        # self.explained_variance_ = np.take(self.eigVals,i)
        # self.components_ = np.zeros((self.n_components,self.covariance_.shape[0]))
        # for i in range(self.explained_variance_.shape[0]):
        #     # print(largest_eigvals[i])
        #     self.components_[i,:] = _getEigenvector(self.covariance_, self.explained_variance_[i])
        #     # print(largest_eigvect[:,i])
        
        # return(S)
    def transform(self,X):
        ones = np.ones(X.shape[0])
        X_diff = X - np.outer(ones,self.mean_)
        transformation = np.matmul(X_diff,self.components_.T)
        return(transformation)
    
    def inverse_transform(self,X):
        ones = np.ones(X.shape[0])
        X_diff = np.matmul(X,self.components_)
        X_trans = X_diff + np.outer(ones,self.mean_)
        return(X_trans)
    
    def change_n_cnt(self,n_components):
        self.n_components = n_components
        self._calcComponents()
        
    def _calcComponents(self):
        i = np.argsort(np.abs(self.eigVals))[::-1][:self.n_components]
        self.explained_variance_ = np.take(self.eigVals,i)
        self.components_ = np.zeros((self.n_components,self.covariance_.shape[0]))
        for i in range(self.explained_variance_.shape[0]):
            # print(largest_eigvals[i])
            self.components_[i,:] = _getEigenvector(self.covariance_, self.explained_variance_[i])
            
    def _getCovariance(self,X):
        n= X.shape[0]
        ones = np.ones(n)
        X_mean = np.outer(ones,self.mean_)
        X_diff = X-X_mean # subtract the mean to center the matrix
        S = np.matmul(X_diff.T,X_diff)/n
        return(S)
    

    

# ALGORITHMS FROM CLASS

# following is based off from algorithm 26.1
def _hessenberg(matrix):
    A=matrix.copy().astype(float)
    m = A.shape[0]

    for k in range(m-2):
        x=A[k+1:m,k]
        v = _getSign(x[0])*np.linalg.norm(x,2)*_getE1(x.shape[0]) + x
        if np.linalg.norm(v,2)!=0:v = v/np.linalg.norm(v,2)

        A[k+1:m,k:m]=A[k+1:m,k:m]-2*np.outer(v,np.matmul(v,A[k+1:m,k:m]))
        A[0:m,k+1:m]=A[0:m,k+1:m]-2*np.outer(np.matmul(A[0:m,k+1:m],v.T),v)

    return(A.round(8))

def _fullHessenberg(matrix):
    A=matrix.copy().astype(float)
    m = A.shape[0]
    Q=np.identity(A.shape[0])

    for k in range(m-2):
        x=A[k+1:m,k]
        v = _getSign(x[0])*np.linalg.norm(x,2)*_getE1(x.shape[0]) + x
        if np.linalg.norm(v,2)!=0:v = v/np.linalg.norm(v,2)

        q_k=_getHV(v)
        P=np.identity(A.shape[0])
        P[k+1:,k+1:] = q_k
        Q=np.matmul(Q,P)

        A[k+1:m,k:m]=A[k+1:m,k:m]-2*np.outer(v,np.matmul(v,A[k+1:m,k:m]))
        A[0:m,k+1:m]=A[0:m,k+1:m]-2*np.outer(np.matmul(A[0:m,k+1:m],v.T),v)   
    return(A.round(8),Q)

def _householder(matrix):
    A=matrix.copy().astype(float)
    n=matrix.shape[1]
    v=np.zeros(matrix.shape, float)
    q_star=np.identity(A.shape[0])
    for k in range(n):
        x=A[k:,k]
        v[k:,k]=_getSign(x[0])*np.linalg.norm(x,2)*_getE1(x.shape[0])+x
        # if np.linalg.norm(v[k:,k],2) !=0:v[k:,k]=v[k:,k]/np.linalg.norm(v[k:,k],2) # <--- line throws errors
        # v[k:,k]=v[k:,k]/np.linalg.norm(v[k:,k],2)
        if np.linalg.norm(v[k:,k],2)!=0.0: v[k:,k]=v[k:,k]/np.linalg.norm(v[k:,k],2)
        q_k=np.identity(A.shape[0])
        q_k[k:,k:]=_getHV(v[k:,k])
        q_star=np.matmul(q_k,q_star)
        # print(v[k:,k][np.newaxis].T.shape)
        # print(v[k:,k][np.newaxis].shape)
        A[k:,k:]=A[k:,k:]-2*np.matmul(v[k:,k][np.newaxis].T,np.matmul(v[k:,k][np.newaxis],A[k:,k:]))

    Q=q_star.T.round(8)
    R=A.round(8)
    return(Q,R)

def _practical_qr_algorithm(matrix,k_max,zero_cutoff):
    m = matrix.shape[0]
    A = np.ones((k_max,m,m),dtype=float)
    A[0] = _hessenberg(matrix)
    shift = np.ones(k_max)
    eig=[]
    should_return=False
    # print(matrix.round(5))
    for k in range(1,k_max):
        # print('k:',k)
        shift[k] = A[k-1,m-1,m-1]
        if m==1: 
            # print('last:',shift[k])
            # eig=eig+[shift[k].round(6)]
            eig=_combineEigenvalues(eig,[shift[k].round(8)])
            return(eig)
        Q,R = _householder(A[k-1,:,:]-shift[k]*np.identity(m))
        A[k,:,:] = np.matmul(R,Q)+shift[k]*np.identity(m)
        # print(A[k,:,:].round(5))
        for j in range(m-1):
            # print('  j:',j,'val:',A[k,j,j+1])
            if np.abs(A[k,j,j+1]) < zero_cutoff:
                should_return=True
                # print('shift:',shift[k],',',A[k,j,j+1])
                A[k,j,j+1]=0
                A[k,j+1,j]=0
                # print('shift:',shift[k],',',A[k,j,j+1])
                # break
                A_1 = A[k,:j+1,:j+1]
                A_2 = A[k,j+1:,j+1:]
                # eig=eig+[shift[k].round(6)]
                eig=_combineEigenvalues(eig,[shift[k].round(8)])
                if A_1.shape[0]>0: 
                    e_temp = _practical_qr_algorithm(A_1,k_max,zero_cutoff)
                    # print('eig:',eig,'etemp:',e_temp)
                    # print(e_temp)
                    # if abs(e_temp[0])>0:eig=eig+e_temp
                    # if e_temp!=None:eig=eig+e_temp
                    if e_temp!=None:eig=_combineEigenvalues(eig,e_temp)
                if A_2.shape[0]>0: 
                    e_temp = _practical_qr_algorithm(A_2,k_max,zero_cutoff)
                    # print('eig:',eig,'etemp:',e_temp)
                    # print(e_temp)
                    # if abs(e_temp[0])>0:eig=eig+e_temp
                    # if e_temp!=None:eig=eig+e_temp
                    if e_temp!=None:eig=_combineEigenvalues(eig,e_temp)
        # if should_return: return((list(set(eig))))

    # print('not found')  
    # return(shift[k])
    # print(eig)
    # eig = [ round(e, 6) % e for e in eig ] # round each value in eig list
    # print('hi')
    # return((list(set(eig))))
    return(eig)

# gets the eigenvector corresponding to some eigenvalue
#  uses one itteration of Alg 27.2 from book, pdf pg 220, book pg 206
def _getEigenvector(matrix, eigenvalue):
    A = matrix.copy().astype(np.double)
    n = matrix.shape[0]
    v = _getSomeUnitVector(n)
    w = _solveSystem(A-eigenvalue*np.identity(n),v)
    eigenvector = w/np.linalg.norm(w,2)
    return(eigenvector)

def _solveSystem(A,b):
    Q,R = _householder(A)
    # Solve for y in Qy=b, Q^*Qy=y=Q^*b
    y = np.matmul(Q.T,b)
    # Solve for x in Rx=y using back sub
    x = _backwardSub(R,y)
    return(x)

def _getHV(v):
    # print('v:',v)
    I=np.identity(v.shape[0])
    Hv = I-2*np.outer(v,v)
    # print('Hv:',Hv)
    return(Hv)

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



def _combineEigenvalues(old,new):
    # assuming new values will be more accurate
    # print('combining')
    # print(old)
    # print(new)
    # if old != []:
    for o in old:
        should_add = True
        for n in new:
            p = abs(n*.1)
            if o>=n-p and o<=n+p: should_add = False
            # print('o:',o,'n:',n,'p:',p,'should_add:',should_add)
        if should_add: new = [o] + new

    # print('returning:',new)
    return(new)




def _getSomeUnitVector(n):
    v = np.random.randint(0,10,n)
    v = v/np.linalg.norm(v,2)
    return(v)