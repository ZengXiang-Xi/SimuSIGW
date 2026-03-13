#############################################################
# this is the code for HPC
############################################################

### import the lib
import numpy as np
import sys
import os
import time
from matplotlib import pyplot as plt
import torch as tc
tc.set_default_dtype(tc.float64)

# print(tc.cuda.is_available())
###########################################################
# define the function
###########################################################
time_start = time.time()
### the derivative matrix
def Fourier(number,xmin,period):
    """
    Function
    ----------
    Fourier pseudospectral method
    #period=period*pi

    Parameters
    ----------
    number : int
        the number of Fourier points.
        !!! Note : the number must be even
    xmin : int
        the minimum value of the domain.
    period : int
        the period of the domain, in the other word, the length of the domain.
        !!! Note : Actual domain period is period*pi, so the unit of period is pi

    Returns
    -------
    result1 : numpy.array
        $j \times period \times \pi/number,\quad j=0,……,number-1$, mapping to the domain.
        !!! Note : there are number (even) Fourier points, and include starting point of the domain, no end point of the domain
    result2 : numpy.array
        Fourier first-order differentiation matrix.
    result3 : numpy.array
        Fourier second-order differentiation matrix.
    result4 : numpy.array
        Fourier third-order differentiation matrix.

    """
    N=int(number)
    h=2*tc.pi/N

    D1=tc.zeros((N,N))
    D2=tc.zeros((N,N))
    D3=tc.zeros((N,N))

    x=xmin+tc.arange(0,N,dtype=int)*(period*np.pi)/N
    for i in tc.arange(1,N,dtype=int):
        for j in tc.arange(0,i,dtype=int):
            D1[i,j]=(-1)**(i-j)/(2*tc.tan((i-j)*h/2))
            D2[i,j]=-(-1)**(i-j)/(2*(tc.sin((i-j)*h/2))**2)
            D3[i,j]=(-1)**(i-j)*(3/(4*(tc.sin((i-j)*h/2))**2)-N**2/8)/tc.tan((i-j)*h/2)
    D1=D1-D1.T
    D2=D2+D2.T-((N**2+2)/12)*tc.eye(N)
    D3=D3-D3.T
    return x,(2/period)*D1,((2/period)**2)*D2,((2/period)**3)*D3

def gaussian_random_fields(N, power_spectrum, a):
    '''
    another method to generate Gaussian random fields (more quick)
    '''

    noise = np.fft.fftn(np.random.normal(size=(N,N,N)))
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    kz = np.fft.fftfreq(N) * N
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    k[k == 0] = 1e-12  # avoid zero divition

    amplitude = np.sqrt(power_spectrum(k)*(N/(2*np.pi))**3)
    field_k = noise * amplitude
    zetag = np.fft.ifftn(field_k).real
    field = zetag * (2/3)
    zeta = -a*np.log(np.abs(1 - zetag/a))
    field1 = zeta * (2/3)

    return field, field1

### the projection operator
def TT_project(N, a11, a12, a13, a22, a23, a33):

    """ 
    transfer the tensor field into Fourier space, then get the TT part by multiply a projection matrix
    N : the size of the lattice

    """

    kx = np.fft.fftfreq(N) * N 
    ky = np.fft.fftfreq(N) * N 
    kz = np.fft.fftfreq(N) * N 
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    k[k == 0] = 1e-10  

    # define the projection operator Pij

    P11 = np.ones((N,N,N)) - (kx/k)**2
    P12 = -(kx/k)*(ky/k)
    P21 = P12
    P13 = -(kx/k)*(kz/k)
    P31 = P13
    P22 = np.ones((N,N,N)) - (ky/k)**2
    P23 = -(ky/k)*(kz/k)
    P32 = P23
    P33 = np.ones((N,N,N)) - (kz/k)**2

    # do the Fourier transformation

    fa11 = np.fft.fftn(a11)
    fa12 = np.fft.fftn(a12)
    fa13 = np.fft.fftn(a13)
    fa22 = np.fft.fftn(a22)
    fa23 = np.fft.fftn(a23)
    fa33 = np.fft.fftn(a33)

    # define the new variables

    v11 = P11*fa11 + P12*fa12 + P13*fa13
    v12 = P11*fa12 + P12*fa22 + P13*fa23
    v13 = P11*fa13 + P12*fa23 + P13*fa33
    v21 = P21*fa11 + P22*fa12 + P23*fa13
    v22 = P21*fa12 + P22*fa22 + P23*fa23
    v23 = P21*fa13 + P22*fa23 + P23*fa33
    v31 = P31*fa11 + P32*fa12 + P33*fa13
    v32 = P31*fa12 + P32*fa22 + P33*fa23
    v33 = P31*fa13 + P32*fa23 + P33*fa33

    vt11 = P11*np.conj(fa11) + P12*np.conj(fa12) + P13*np.conj(fa13)
    vt12 = P11*np.conj(fa12) + P12*np.conj(fa22) + P13*np.conj(fa23)
    vt13 = P11*np.conj(fa13) + P12*np.conj(fa23) + P13*np.conj(fa33)
    vt21 = P21*np.conj(fa11) + P22*np.conj(fa12) + P23*np.conj(fa13)
    vt22 = P21*np.conj(fa12) + P22*np.conj(fa22) + P23*np.conj(fa23)
    vt23 = P21*np.conj(fa13) + P22*np.conj(fa23) + P23*np.conj(fa33)
    vt31 = P31*np.conj(fa11) + P32*np.conj(fa12) + P33*np.conj(fa13)
    vt32 = P31*np.conj(fa12) + P32*np.conj(fa22) + P33*np.conj(fa23)
    vt33 = P31*np.conj(fa13) + P32*np.conj(fa23) + P33*np.conj(fa33)

    # calculate the TT results

    trpupu = v11*vt11 + v22*vt22 + v33*vt33 + v12*vt21 + v21*vt12 + v13*vt31 + v31*vt13 + v23*vt32 + v32*vt23
    trpu = v11 + v22 + v33
    trput = vt11 + vt22 + vt33

    result = trpupu - (1/2)*trpu*trput
    result = result.real

    # get the real field results (do the inverse Fourier transformation)

    # c11 = np.fft.ifftn(b11).real
    # c12 = np.fft.ifftn(b12).real
    # c13 = np.fft.ifftn(b13).real
    # c22 = np.fft.ifftn(b22).real
    # c23 = np.fft.ifftn(b23).real
    # c33 = np.fft.ifftn(b33).real

    # discrete the momentum according to the amplitude of k

    dk = 1
    kmin = 1
    kmax = int((np.sqrt(3)/2)*N)+2
    
    # calculate the average of GW (also the energy spectrum)

    kbox = np.zeros(kmax)
    knumber = np.zeros(kmax)

    knumber[-1] = 1e-6

    for j in range(N):
        for l in range(N):
            for m in range(N):
                length = int(k[j,l,m] + 2/3)
                knumber[length] += 1
                kbox[length] += (np.pi/12)*((k[j,l,m])**3/(N**6))*result[j,l,m]

    kbox = kbox/knumber

    return knumber, kbox

### capture the energy spectrum of GWs
def load_GW(time_step, path, N, dt, name):

    """ 

    time_step: a list of time slices, like [200, 400, 600, 800, 1000]

    path: the path of your raw data

    """

    for i in range(len(time_step)):

        a11_i = np.load(path + '/a11_' + str(time_step[i]) + '.npy')
        a12_i = np.load(path + '/a12_' + str(time_step[i]) + '.npy')
        a13_i = np.load(path + '/a13_' + str(time_step[i]) + '.npy')
        a22_i = np.load(path + '/a22_' + str(time_step[i]) + '.npy')
        a23_i = np.load(path + '/a23_' + str(time_step[i]) + '.npy')
        a33_i = np.load(path + '/a33_' + str(time_step[i]) + '.npy')

        knumber, kbox = TT_project(N, a11_i, a12_i, a13_i, a22_i, a23_i, a33_i)
        kbox = kbox*((time_step[i]+5)*dt)**2

        np.save(gw + '/' + str(name) + '_' + str(time_step[i]) + '.npy', kbox)


### evolve the field equations, we will use fourth order Runge-Kutta method to evolve scalar fields, use leapfrog to evolve GWs

def Evolution_scalar(field, pi, t):

    # compute Dt_phi

    Dt_phi = pi

    # compute Dt_pi

    Dt_pi = (1/3) * (tc.einsum('Xx,xyz->Xyz', D2, field) + tc.einsum('Yy,xyz->xYz', D2, field) + tc.einsum('Zz,xyz->xyZ', D2, field)) - (4/t) * pi

    return Dt_phi, Dt_pi


def RK4_scalar_Evolution(dt, field, pi, t):

    dphi_1, dpi_1 = \
    Evolution_scalar(field, pi, t)
    dphi_2, dpi_2 = \
    Evolution_scalar(field + dphi_1*dt/2, pi + dpi_1*dt/2, t+dt/2)
    dphi_3, dpi_3 = \
    Evolution_scalar(field + dphi_2*dt/2, pi + dpi_2*dt/2, t+dt/2)
    dphi_4, dpi_4 = \
    Evolution_scalar(field + dphi_3*dt, pi + dpi_3*dt, t+dt)

    a1 = field + (dphi_1 + 2*dphi_2 + 2*dphi_3 + dphi_4)*(dt/6)
    a2 = pi + (dpi_1 + 2*dpi_2 + 2*dpi_3 + dpi_4)*(dt/6)

    return a1, a2

def Evolution_tensor(field, pi, h11, h12, h13, h22, h23, h33, t):

    #### To apply leapfrog method, we need to evolve t*h_{ij} rather than h_{ij}. Therefore, our h = t*h    
    # evolve the GWs

    Dt_a11 = (tc.einsum('Xx,xyz->Xyz', D2, h11) + tc.einsum('Yy,xyz->xYz', D2, h11) + tc.einsum('Zz,xyz->xyZ', D2, h11)) \
        - 4*t*(4*field*tc.einsum('Xx,xyz->Xyz', D2, field) + 2*(tc.einsum('Xx,xyz->Xyz', D1, field))**2 - (tc.einsum('Xx,xyz->Xyz', D1, t*pi + field))**2)
    Dt_a12 = (tc.einsum('Xx,xyz->Xyz', D2, h12) + tc.einsum('Yy,xyz->xYz', D2, h12) + tc.einsum('Zz,xyz->xyZ', D2, h12)) \
        - 4*t*(4*field*tc.einsum('Yy,xyz->xYz', D1, tc.einsum('Xx,xyz->Xyz', D1, field)) + 2*tc.einsum('Xx,xyz->Xyz', D1, field)*tc.einsum('Yy,xyz->xYz', D1, field) - (tc.einsum('Xx,xyz->Xyz', D1, t*pi + field))*(tc.einsum('Yy,xyz->xYz', D1, t*pi + field)))
    Dt_a13 = (tc.einsum('Xx,xyz->Xyz', D2, h13) + tc.einsum('Yy,xyz->xYz', D2, h13) + tc.einsum('Zz,xyz->xyZ', D2, h13)) \
        - 4*t*(4*field*tc.einsum('Zz,xyz->xyZ', D1, tc.einsum('Xx,xyz->Xyz', D1, field)) + 2*tc.einsum('Xx,xyz->Xyz', D1, field)*tc.einsum('Zz,xyz->xyZ', D1, field) - (tc.einsum('Xx,xyz->Xyz', D1, t*pi + field))*(tc.einsum('Zz,xyz->xyZ', D1, t*pi + field)))
    Dt_a22 = (tc.einsum('Xx,xyz->Xyz', D2, h22) + tc.einsum('Yy,xyz->xYz', D2, h22) + tc.einsum('Zz,xyz->xyZ', D2, h22)) \
        - 4*t*(4*field*tc.einsum('Yy,xyz->xYz', D2, field) + 2*(tc.einsum('Yy,xyz->xYz', D1, field))**2 - (tc.einsum('Yy,xyz->xYz', D1, t*pi + field))**2)
    Dt_a23 = (tc.einsum('Xx,xyz->Xyz', D2, h23) + tc.einsum('Yy,xyz->xYz', D2, h23) + tc.einsum('Zz,xyz->xyZ', D2, h23)) \
        - 4*t*(4*field*tc.einsum('Zz,xyz->xyZ', D1, tc.einsum('Yy,xyz->xYz', D1, field)) + 2*tc.einsum('Yy,xyz->xYz', D1, field)*tc.einsum('Zz,xyz->xyZ', D1, field) - (tc.einsum('Yy,xyz->xYz', D1, t*pi + field))*(tc.einsum('Zz,xyz->xyZ', D1, t*pi + field)))
    Dt_a33 = (tc.einsum('Xx,xyz->Xyz', D2, h33) + tc.einsum('Yy,xyz->xYz', D2, h33) + tc.einsum('Zz,xyz->xyZ', D2, h33)) \
        - 4*t*(4*field*tc.einsum('Zz,xyz->xyZ', D2, field) + 2*(tc.einsum('Zz,xyz->xyZ', D1, field))**2 - (tc.einsum('Zz,xyz->xyZ', D1, t*pi + field))**2)

    return Dt_a11, Dt_a12, Dt_a13, Dt_a22, Dt_a23, Dt_a33

def leapfrog_tensor_Evolution(dt, field, pi, h11, h12, h13, h22, h23, h33, t, v11_h, v12_h, v13_h, v22_h, v23_h, v33_h):

    ac11, ac12, ac13, ac22, ac23, ac33 = Evolution_tensor(field, pi, h11, h12, h13, h22, h23, h33, t)
    v11_new = v11_h + ac11 * dt
    v12_new = v12_h + ac12 * dt
    v13_new = v13_h + ac13 * dt
    v22_new = v22_h + ac22 * dt
    v23_new = v23_h + ac23 * dt
    v33_new = v33_h + ac33 * dt
    h11_new = h11 + v11_new * dt
    h12_new = h12 + v12_new * dt
    h13_new = h13 + v13_new * dt
    h22_new = h22 + v22_new * dt
    h23_new = h23 + v23_new * dt
    h33_new = h33 + v33_new * dt

    return h11_new, h12_new, h13_new, h22_new, h23_new, h33_new, \
         v11_new, v12_new, v13_new, v22_new, v23_new, v33_new

def step(field, h):

    zeta = (3/2)*field
    zeta1 = -(2/h)*(np.sqrt(np.abs(1 - h*zeta)) - 1)
    phi = (2/3)*zeta1

    return phi

#######################################################################
# generate the initial value
#######################################################################

### define the path
# Get the path
submit_dir = os.environ.get('SLURM_SUBMIT_DIR')
if submit_dir is None:
    # If there is no Slurm path，then use the current path
    submit_dir = os.getcwd()

# generate the output path
output_dir = os.path.join(submit_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)  
gw = os.path.join(output_dir, 'GW_data')
os.makedirs(gw, exist_ok=True)

path = os.path.join(output_dir, 'step')
os.makedirs(path, exist_ok=True)

### the power spectrum of the curvature perturbation
def ps(k, kstar, e, A):
    return A * (1/kstar)**3 / (np.sqrt(2*np.pi)*e) * np.exp(-(k/kstar - 1)**2 / (2*e**2))

# to avoid the zero divition, we have already divided with k**3
def power_spectrum(k):
    return ps(k, 20, 1/10, 1e-2)*(2*np.pi**2)

N = 256
a = 0.01
H = -20
field, field1 = gaussian_random_fields(N, power_spectrum, a)

field3 = step(field, H)

### set the initial spatial grid

space_length = 2  

x, D1, D2, D3 = Fourier(N, 0, space_length)

### initialize

max_time_step = 2000
dt = space_length / N / 5 
# define the initial time
t = space_length / N  # so the final t = (time_step + 5) * dt

phi = tc.empty([N, N, N])
pi = tc.empty([N, N, N])

h11 = tc.empty([N, N, N])
h12 = tc.empty([N, N, N])
h13 = tc.empty([N, N, N])
h22 = tc.empty([N, N, N])
h23 = tc.empty([N, N, N])
h33 = tc.empty([N, N, N])

v11 = tc.empty([N, N, N])
v12 = tc.empty([N, N, N])
v13 = tc.empty([N, N, N])
v22 = tc.empty([N, N, N])
v23 = tc.empty([N, N, N])
v33 = tc.empty([N, N, N])

phi = tc.tensor(field3)

### transfer the data to GPU

D1 = D1.to(tc.device('cuda'))
D2 = D2.to(tc.device('cuda'))
phi = phi.to(tc.device('cuda'))
pi = pi.to(tc.device('cuda'))
h11 = h11.to(tc.device('cuda'))
h12 = h12.to(tc.device('cuda'))
h13 = h13.to(tc.device('cuda'))
h22 = h22.to(tc.device('cuda'))
h23 = h23.to(tc.device('cuda'))
h33 = h33.to(tc.device('cuda'))
v11 = v11.to(tc.device('cuda'))
v12 = v12.to(tc.device('cuda'))
v13 = v13.to(tc.device('cuda'))
v22 = v22.to(tc.device('cuda'))
v23 = v23.to(tc.device('cuda'))
v33 = v33.to(tc.device('cuda'))

############################################################################
### Evolution
############################################################################

### start evolution
# time_start = time.time() # count the time

for i in range(0,max_time_step+1):
    
    phi_next, pi_next = RK4_scalar_Evolution(dt, phi, pi, t)
    h11_next, h12_next, h13_next, h22_next, h23_next, h33_next, v11_next, v12_next, v13_next, v22_next, v23_next, v33_next \
    = leapfrog_tensor_Evolution(dt, phi, pi, h11, h12, h13, h22, h23, h33, t, v11, v12, v13, v22, v23, v33)

    if(i%2000 == 0):

        h11_real = h11_next/t
        h12_real = h12_next/t
        h13_real = h13_next/t
        h22_real = h22_next/t
        h23_real = h23_next/t
        h33_real = h33_next/t
        a11_next = (v11_next - h11_real)/t
        a12_next = (v12_next - h12_real)/t
        a13_next = (v13_next - h13_real)/t
        a22_next = (v22_next - h22_real)/t
        a23_next = (v23_next - h23_real)/t
        a33_next = (v33_next - h33_real)/t

        phi_write = phi_next.to(tc.device('cpu'))
        pi_write = pi_next.to(tc.device('cpu'))
        h11_write = h11_real.to(tc.device('cpu'))
        h12_write = h12_real.to(tc.device('cpu'))
        h13_write = h13_real.to(tc.device('cpu'))
        h22_write = h22_real.to(tc.device('cpu'))
        h23_write = h23_real.to(tc.device('cpu'))
        h33_write = h33_real.to(tc.device('cpu'))
        a11_write = a11_next.to(tc.device('cpu'))
        a12_write = a12_next.to(tc.device('cpu'))
        a13_write = a13_next.to(tc.device('cpu'))
        a22_write = a22_next.to(tc.device('cpu'))
        a23_write = a23_next.to(tc.device('cpu'))
        a33_write = a33_next.to(tc.device('cpu'))

        np.save(path + '/phi_' + str(i) + '.npy', phi_write)
        np.save(path + '/pi_' + str(i) + '.npy', pi_write)
        np.save(path + '/h11_' + str(i) + '.npy', h11_write)
        np.save(path + '/h12_' + str(i) + '.npy', h12_write)
        np.save(path + '/h13_' + str(i) + '.npy', h13_write)
        np.save(path + '/h22_' + str(i) + '.npy', h22_write)
        np.save(path + '/h23_' + str(i) + '.npy', h23_write)
        np.save(path + '/h33_' + str(i) + '.npy', h33_write)
        np.save(path + '/a11_' + str(i) + '.npy', a11_write)
        np.save(path + '/a12_' + str(i) + '.npy', a12_write)
        np.save(path + '/a13_' + str(i) + '.npy', a13_write)
        np.save(path + '/a22_' + str(i) + '.npy', a22_write)
        np.save(path + '/a23_' + str(i) + '.npy', a23_write)
        np.save(path + '/a33_' + str(i) + '.npy', a33_write)


    t += dt

    phi = phi_next
    pi = pi_next
    h11 = h11_next
    h12 = h12_next
    h13 = h13_next
    h22 = h22_next
    h23 = h23_next
    h33 = h33_next
    v11 = v11_next
    v12 = v12_next  
    v13 = v13_next
    v22 = v22_next
    v23 = v23_next
    v33 = v33_next
    print(i)

### load GWs
name = 'step_h=20ks=20N=256'
time_step = [2000]
load_GW(time_step, path, N, dt, name)

### plot
#data = np.load(gw + '/' + str(name) + '_' + str(time_step[0]) + '.npy')

#plt.loglog(data)
#plt.ylim(1e-12, 1e-7)
#plt.savefig(gw + '/' + str(name) + '.pdf')

time_end = time.time()
time_tot = (time_end - time_start)/60
print(time_tot)