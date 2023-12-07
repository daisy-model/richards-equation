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

def relative_saturation(t, z, A, K1, alpha, gamma):
    '''Equation (16) in [ogden2017soil]'''
    # pylint: disable=too-many-arguments
    aK = (1 - alpha)*K1
    Atz = np.clip(A*t-z, 0, None)
    S_e = 1/alpha * (1 + (aK/(A - aK))*(1/(np.exp(-Atz/gamma) - A/(A - aK))))    
    t_p = ponding_time(A, K1, alpha, gamma)
    if t >= t_p:
        saturated = z <= A*(t - t_p)
        S_e[saturated] = 1
    return S_e

def ponding_time(A, K1, alpha, gamma):
    '''Equation (17) in [ogden2017soil]'''
    return gamma/A * np.log((A - (1 - alpha)*K1)/(A - K1))


def soil_water_diffusivity(Se, alpha, beta):
    '''Equation (13) for D in [ogden2017soil]'''
    return (beta*Se)/(1 - alpha*Se)**2

def unsaturated_hydraulic_conductivity(Se, alpha, beta, gamma):
    '''Equation (13) for K in [ogden2017soil]'''
    return (beta*Se)/((1 - alpha*Se)*alpha*gamma)


def get_beta(K1, alpha, gamma):
    '''Relation from Equation (14) in [ogden2017soil]'''
    return K1 * (1 - alpha)*alpha*gamma

def nonlinear_soil_water_constitutive_relations():
    '''Experiment described in Section 5.2 in [ogden2017soil]'''
    # pylint: disable=too-many-locals, duplicate-code
    h = 60*60
    A = 2/h
    K1 = 1/h
    gamma = 25
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
    ts_end = {
        0.1 : 1.4*h,
        0.2 : 2.7*h,
        0.3 : 4.0*h,
        0.4 : 5.2*h,
        0.5 : 6.4*h,
        0.6 : 7.5*h,
        0.7 : 8.6*h,
        0.8 : 9.7*h,
        0.95 : 11.7*h,
    }

    dz = 0.1
    max_zs = {
        0.1 : 4,
        0.2 : 10,
        0.3 : 10,
        0.4 : 15,
        0.5 : 20,
        0.6 : 20,
        0.7 : 20,
        0.8 : 20,
        0.95 : 30,
    }
    zs = { k : np.linspace(0, v, int(v/dz)) for k,v in max_zs.items() }

    fig, axs = plt.subplots(3,3, sharex=True)
    fig2, axs2 = plt.subplots(3,3,sharex=True,sharey=True)
    ax_map = {
        0.1 : (0,0),
        0.2 : (0,1),
        0.3 : (0,2),
        0.4 : (1,0),
        0.5 : (1,1),
        0.6 : (1,2),
        0.7 : (2,0),
        0.8 : (2,1),
        0.95 : (2,2),
    }    
    for alpha in alphas:
        beta = get_beta(K1, alpha, gamma)
        ax = axs[ax_map[alpha]]
        ax2 = axs2[ax_map[alpha]]
        t_end = ts_end[alpha]
        z = zs[alpha]
        t_p = ponding_time(A, K1, alpha, gamma)
        for t in [0.5*t_p, t_p, t_end]:
            Se = relative_saturation(t, z, A, K1, alpha, gamma)
            D = soil_water_diffusivity(Se, alpha, beta)
            K = unsaturated_hydraulic_conductivity(Se, alpha, beta, gamma)
            ax.plot(Se, z)
            ax2.plot(D, K)
        ax.set_ylim((z[-1], z[0]))
        ax.set_xlim((0,1.05))
        ax.set_xlabel('$S_e$')
        ax.set_ylabel('z')
        ax.set_title(f'$\\alpha = {alpha}$')
        ax2.set_xlabel('D')
        ax2.set_ylabel('K')
        ax2.set_title(f'$\\alpha = {alpha}, \\beta = {beta:.2e}$')
    fig.suptitle('Nonlinear water retention functions from [ogden2017soil]')
    fig2.suptitle('Nonlinear water retention functions from [ogden2017soil]')
    plt.show()

if __name__ == '__main__':
    nonlinear_soil_water_constitutive_relations()
