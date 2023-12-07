'''Exact solution of Richards' equation

Based on
[ogden2017soil] Ogden, Fred L., Myron B. Allen, Wencong Lai, Jianting Zhu, Mookwon Seo, Craig C.
                Douglas, and Cary A. Talbot.
                “The Soil Moisture Velocity Equation”.
                Journal of Advances in Modeling Earth Systems 9, nr. 2 (juni 2017): 1473–87.
                https://doi.org/10.1002/2017MS000931.
'''
# pylint: disable=invalid-name; We want to use variable names that match the paper
import numpy as np
import matplotlib.pyplot as plt

def relative_saturation(t, z, A, K1, D1, n):
    '''Equation (11) in [ogden2017soil]'''
    # pylint: disable=too-many-arguments
    Atz = np.clip(A*t-z, 0, None)
    S_e =  (A/K1 * (1 - np.exp((-n*K1*(Atz))/(D1))))**(1/n)
    t_p = ponding_time(A, K1, D1, n)
    if t >= t_p:
        saturated = z <= A*(t - t_p)
        S_e[saturated] = 1
    return S_e

def ponding_time(A, K1, D1, n):
    '''Equation (12) in [ogden2017soil]'''
    return D1/(n*K1*A) * np.log(A/(A-K1))

def soil_water_diffusivity(Se, D1, n):
    '''Equation (9) for D in [ogden2017soil]'''
    return D1*Se**n

def unsaturated_hydraulic_conductivity(Se, K1, n):
    '''Equation (9) for K in [ogden2017soil]'''
    return K1*Se**(n+1)

def power_law_soil_water_retention():
    '''Experiment described in Section 5.1 in [ogden2017soil]'''
    # pylint: disable=too-many-locals, duplicate-code
    h = 60*60
    A = 2/h
    K1 = 1/h
    D1 = 100/h
    ns = np.arange(3, 10)
    dz = 0.1
    max_z = 40
    z = np.linspace(0, max_z, int(max_z/dz))

    ts_end = {
        3 : 16.0*h,
        4 : 12.0*h,
        5 : 9.8*h,
        6 : 8.2*h,
        7 : 7.0*h,
        8 : 6.2*h,
        9 : 5.6*h
    }
    fig, axs = plt.subplots(2,4, sharex=True)
    fig2, axs2 = plt.subplots(2,4,sharex=True,sharey=True)
    ax_map = {
        3 : (0,0),
        4 : (0,1),
        5 : (0,2),
        6 : (0,3),
        7 : (1,0),
        8 : (1,1),
        9 : (1,2)
    }    
    for n in ns:
        ax = axs[ax_map[n]]
        ax2 = axs2[ax_map[n]]
        t_end = ts_end[n]
        t_p = ponding_time(A, K1, D1, n)
        for t in [0.5*t_p, t_p, t_end]:
            Se = relative_saturation(t, z, A, K1, D1, n)
            D = soil_water_diffusivity(Se, D1, n)
            K = unsaturated_hydraulic_conductivity(Se, K1, n)
            ax.plot(Se, z)
            ax2.plot(D, K)            
        ax.set_ylim((z[-1], z[0]))
        ax.set_xlim((0,1.05))
        ax.set_xlabel('$S_e$')
        ax.set_ylabel('z (cm)')
        ax.set_title(f'n = {n}')
        ax2.set_xlabel('D')
        ax2.set_ylabel('K')
        ax2.set_title(f'n = {n}')
    fig.suptitle('Power law water retention functions from [ogden2017soil]')
    fig2.suptitle('Power law water retention functions from [ogden2017soil]')
    plt.show()

if __name__ == '__main__':
    power_law_soil_water_retention()
