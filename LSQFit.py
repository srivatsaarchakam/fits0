import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from math import log
from random import gauss

# config
xmin=1.0
xmax=20.0
npoints=12
sigma=0.2

lx=np.zeros(npoints)
ly=np.zeros(npoints)
ley=np.zeros(npoints)

pars=[0.5,1.3,0.5]

# Eq: y = a + b * log(x) + c * (log(x))^2
def f(x,par):
    return par[0] + par[1] * log(x) + par[2] * log(x) * log(x) 

# x = array-like
def getX(x):
    step=(xmax-xmin)/npoints
    for i in range(npoints):
        x[i]=xmin+i*step
        
# x,y,ey = array-like
def getY(x,y,ey):
    for i in range(npoints):
        y[i]=f(x[i],pars)+gauss(0,sigma)
        ey[i]=sigma

# get a random sampling of the (x,y) data points, rerun to generate different data sets for the plot below

getX(lx)
getY(lx,ly,ley)

fig, ax = plt.subplots()
ax.errorbar(lx, ly, yerr=ley)
ax.set_title("Pseudoexperiment")
fig.show()

# *** modify and add your code here ***

# matrix for model; for each i: [1, ln(x_i), (ln(x_i)^2)]
def matrix(x):
    z = np.log(x)
    return np.column_stack([np.ones_like(z), z, z**2])

# solve weighted normal equations
def weighted_fit(X, y, ey):
    w = 1 / (ey ** 2)
    sqrt_w = np.sqrt(w)

    Xw = X * sqrt_w[: , None]
    yw = y * sqrt_w

    XtX = Xw.T @ Xw
    XtY = Xw.T @ yw

    param = np.linalg.solve(XtX, XtY)
    covar = inv(XtX)

    res = y - X @ param
    chi2 = float(np.sum((res / ey) ** 2))
    dof = y.size - X.shape[1]

    return param, covar, chi2, dof

# matrix
X = matrix(lx)

nexperiments = 1000  # for example

par_a = np.random.rand(1000)   # simple placeholders for making the plot example
par_b = np.random.rand(1000)   # these need to be filled using results from your fits
par_c = np.random.rand(1000)

chi2_vals = np.empty(nexperiments)
chi2_reduced = np.empty(nexperiments)

for k in range(nexperiments):
    getY(lx, ly, ley)
    p, covar, chi2, dof = weighted_fit(X, ly, ley)

    par_a[k], par_b[k], par_c[k] = p
    chi2_vals[k] = chi2
    chi2_reduced[k] = chi2 / dof


# perform many least squares fits on different pseudo experiments here
# fill histograms w/ required data
pdf = "LSQFit.pdf"

# careful, the automated binning may not be optimal for displaying your results!
with PdfPages(pdf) as pdf:
    title = 8
    axis = 6

    # plot set 1
    fig1, axis1 = plt.subplots(2, 2, figsize = (8,6))
    plt.tight_layout(pad = 2.0)

    # par_a
    axis1[0, 0].hist(par_a, bins = 80)
    axis1[0, 0].set_title('Parameter a', fontsize = title)
    axis1[0,0].set_xlabel('a', fontsize = axis)
    axis1[0,0].set_ylabel('counts', fontsize = axis)

    # par_b
    axis1[0, 1].hist(par_b, bins = 80)
    axis1[0, 1].set_title('Parameter b', fontsize = title)
    axis1[0,1].set_xlabel('b', fontsize = axis)
    axis1[0,1].set_ylabel('counts', fontsize = axis)

    # par_c
    axis1[1, 0].hist(par_c, bins = 80)
    axis1[1, 0].set_title('Parameter c', fontsize = title)
    axis1[1,0].set_xlabel('c', fontsize = axis)
    axis1[1,0].set_ylabel('counts', fontsize = axis)

    # chi2
    axis1[1, 1].hist(chi2_vals, bins = 80)
    axis1[1, 1].set_title('chi2 Distribution', fontsize = title)
    axis1[1,1].set_xlabel('chi2', fontsize = axis)
    axis1[1,1].set_ylabel('counts', fontsize = axis)

    pdf.savefig(fig1)

    # plot set 2
    fig2, axis2 = plt.subplots(2, 2, figsize = (8,6))
    plt.tight_layout(pad = 2.0)

    # par_a & par_b
    axis2[0, 0].hist2d(par_a, par_b, bins = 50)
    axis2[0, 0].set_title('Parameter a vs b', fontsize = title)
    axis2[0,0].set_xlabel('a', fontsize = axis)
    axis2[0,0].set_ylabel('b', fontsize = axis)

    # par_a & par_c
    axis2[0, 1].hist2d(par_a, par_c, bins = 50)
    axis2[0, 1].set_title('Parameter a vs c', fontsize = title)
    axis2[0,1].set_xlabel('a', fontsize = axis)
    axis2[0,1].set_ylabel('c', fontsize = axis)

    # par_b & par_c
    axis2[1, 0].hist2d(par_b, par_c, bins = 50)
    axis2[1, 0].set_title('Parameter b vs c', fontsize = title)
    axis2[1, 0].set_xlabel('b', fontsize = axis)
    axis2[1, 0].set_ylabel('c', fontsize = axis)

    # chi2
    axis2[1, 1].hist(chi2_reduced, bins = 80)
    axis2[1, 1].set_title('Reduced chi2 Distribution', fontsize = title)
    axis2[1, 1].set_xlabel('chi2', fontsize = axis)
    axis2[1, 1].set_ylabel('counts', fontsize = axis)

    pdf.savefig(fig2)

fig.show()

# **************************************
  
input("hit Enter to exit")
