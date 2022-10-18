# -*- coding: utf-8 -*-
"""

LTE MODEL

Created on Tue Oct 12 12:25:02 2021

"""

import numpy as np
import sqlite3
import astropy.constants as const  # h,k_B,c # SI units
from astropy import units as u
import matplotlib.pyplot as plt
import time
import pickle as pkl
import pandas as pd
from file_path import FILEPATH
from scipy import integrate


class SimpleSpectrum:
    def __init__(self, xarray, yarray, xunit='mhz', yunit='K'):
        self.xval = xarray
        self.yval = yarray
        self.xunit = xunit
        self.yunit = yunit


class ModelSpectrum:
    def __init__(self, cpt_list, fmhz_min=115.0e3, fmhz_max=116.0e3, dfmhz=0.1, eup_min=0.0, eup_max=150.0, aij_min=0.0,
                 telescope='alma_400m', tcmb=2.73, tc=0):
        """
        #accessing the components list
        """
        self.cpt_list = cpt_list
        self.fmhz_min = fmhz_min
        self.fmhz_max = fmhz_max
        self.dfmhz = dfmhz
        self.eup_min = eup_min
        self.eup_max = eup_max
        self.aij_min = aij_min
        self.telescope = telescope
        self.tcmb = tcmb
        self.tc = tc
        self.frequencies = np.arange(self.fmhz_min, self.fmhz_max, self.dfmhz)
        self.intensities = np.zeros_like(self.frequencies)

    def get_transition_list(self, cpt, species=None):
        if species is None:
            self.species_list = cpt.species_list
        else:
            self.species_list = [species]
        self.db = cpt.db
        transition_list = []
        for sp in self.species_list:
            command = "SELECT * FROM transitions WHERE catdir_id = " + str(sp.tag) + " and fMhz < " + str(self.fmhz_max) + \
                      " and fMhz > " + str(self.fmhz_min) + " and eup < " + str(self.eup_max)
            self.db.execute(command)
            all_rows = self.db.fetchall()
            for row in all_rows:
                trans = Transition(self.db, sp, row[1], row[3], row[4], row[6])  # tag, freq, aij, elo_cm, gup
                #print('tr =',trans)
                transition_list.append(trans)
        transition_list.sort(key=lambda x: x.f_trans_mhz)
        #print('t =',transition_list)
        return transition_list

    def compute_model(self):
        cpt_list_interacting = [cpt for cpt in self.cpt_list if cpt.isInteracting]
        cpt_list_non_interacting = [cpt for cpt in self.cpt_list if not cpt.isInteracting]

        tbefore = self.tc + jnu(self.frequencies, self.tcmb)
        #print(tbefore)

        intensity_all_cpt = 0.

        for k, cpt in enumerate(cpt_list_non_interacting):
            # dilution_factor = tr.mol_size ** 2 / (tr.mol_size**2 + get_beam_size(model.telescope,freq)**2)
            # dilution_factor = (1. - np.cos(tr.mol_size/3600./180.*np.pi)) / ( (1. - np.cos(tr.mol_size/3600./180.*np.pi))
            #                    + (1. - np.cos(get_beam_size(model.telescope,freq)/3600./180.*np.pi)) )
            ff_g = dilution_factor(cpt.size, get_beam_size(model.telescope, self.frequencies))
            # ff_d = dilution_factor(cpt.size, get_beam_size(model.telescope, self.frequencies), geometry='disc')

            # compute the sum of the difference between outgoing temp and incoming temp
            intensity_all_cpt = intensity_all_cpt + cpt.compute_delta_t_comp(self.frequencies,
                                                                             model.get_transition_list(cpt),
                                                                             tbefore, ff_g)

        # add "background"
        intensity_all_cpt += tbefore
        #print(intensity_all_cpt)

        for k, cpt in enumerate(cpt_list_interacting):
            tbefore = intensity_all_cpt  # re-define the "background" intensity
            # ff_d = 1.
            # ff_g = 1.
            ff_g = dilution_factor(cpt.size, get_beam_size(model.telescope, self.frequencies))
            # ff_d = dilution_factor(cpt.size, get_beam_size(model.telescope, self.frequencies), geometry='disc')
            # sum_tau_cpt = cpt.compute_sum_tau(frequencies, model.get_transition_list(cpt))

            intensity_all_cpt = cpt.compute_delta_t_comp(self.frequencies, model.get_transition_list(cpt),
                                                         tbefore, ff_g, interacting=True) + tbefore

            cpt.assign_spectrum(SimpleSpectrum(self.frequencies, intensity_all_cpt))

        self.intensities = intensity_all_cpt - jnu(self.frequencies, self.tcmb)  # Ton - Toff
        #print(self.intensities)
        return self.intensities


class Component:
    def __init__(self, db, species_list, isInteracting=False, vlsr=0.0, size=3.0, tex=100.):
        # super().__init__()
        self.db = db
        self.species_list = species_list
        self.tag_list = [sp.tag for sp in self.species_list]
        self.isInteracting = isInteracting
        self.vlsr = vlsr  # km/s
        self.size = size  # arcsec
        self.tex = tex  # K

    def get_fwhm(self, transition):
        tag = transition.tag
        return next(sp for sp in self.species_list if sp.tag == tag).fwhm

    def get_tex(self, transition):
        tag = transition.tag
        return next(sp for sp in self.species_list if sp.tag == tag).tex

    def get_ntot(self, transition):
        tag = transition.tag
        return next(sp for sp in self.species_list if sp.tag == tag).ntot

    def compute_sum_tau(self, fmhz, transition_list):
        sum_tau = 0
        for tr in transition_list:
            num = fmhz - tr.f_trans_mhz + kms_to_mhz(self.vlsr, tr.f_trans_mhz)
            den = fwhm_to_sigma(kms_to_mhz(self.get_fwhm(tr), tr.f_trans_mhz))
            sum_tau += tr.tau0 * np.exp(-0.5 * (num / den) ** 2)
        return sum_tau

    def compute_delta_t_comp(self, fmhz, transition_list, intensity_before, filling_factor, old=False,
                             interacting=False, tcmb=2.73):
        """Computes the difference between outgoing and incoming intensity of a component.
        fmhz : float or numpy array
        component : object Component()
        intensity_before : incoming intensity
        filling_factor : beam filling factor
        """
        # old : deltaT = Tc*(exp(-tau_comp)-1) + sum_mol( ff*(Jtex - Jtcmb)*(1.-exp(-tau_mol))
        # new : deltaT = ff * ( sum_mol(Jtex*(1.-exp(-tau_mol))) + tbefore*(exp(-tau_comp)-1.) )
        sum_tau_cpt = self.compute_sum_tau(fmhz, transition_list)
        deltaT = jnu(fmhz, self.tex) * (1. - np.exp(-sum_tau_cpt)) - intensity_before * (1. - np.exp(-sum_tau_cpt))
        deltaT = deltaT * filling_factor

        return deltaT

    def assign_spectrum(self, spec: SimpleSpectrum):
        self.model_spec = spec
        
    def __iter__(self):
        pass
    
class Species:
    def __init__(self, tag, ntot=7.0e14, tex=100., fwhm=1.0):
        # super().__init__(self)
        self.tag = tag
        self.ntot = ntot  # total column density [cm-2]
        self.tex = tex  # excitation temperature [K]
        self.fwhm = fwhm  # line width [km/s]
        # self.mol_size = mol_size # size of the molecular region[arcsec]


class Transition:
    def __init__(self, db, species, f_trans_mhz, aij, elo_cm, gup):
        self.db = db
        self.f_trans_mhz = f_trans_mhz
        self.aij = aij
        self.elo_cm = elo_cm
        self.elo_J = self.elo_cm * const.h.value * const.c.value * 100
        self.eup_J = self.elo_J + self.f_trans_mhz * 1.e6 * const.h.value
        self.gup = gup
        self.eup = (elo_cm + self.f_trans_mhz * 1.e6 / (const.c.value * 100)) * 1.4389  # [K]
        # print(self.eup)
        self.tag = species.tag
        self.tex = species.tex
        self.ntot = species.ntot
        self.fwhm = species.fwhm

        # def calc_tau0(self):
        # opacity at the line center
        qtex = get_partition_function(self.db, self.tag, self.tex)
        # print("qtex = ", qtex)
        self.nup = self.ntot * self.gup / qtex / np.exp(self.eup_J / const.k_B.value / self.tex)  # [cm-2]
        self.tau0 = const.c.value ** 3 * self.aij * self.nup * 1.e4 \
                    * (np.exp(const.h.value * self.f_trans_mhz * 1.e6 / const.k_B.value / self.tex) - 1.) \
                    / (4. * np.pi * (self.f_trans_mhz * 1.e6) ** 3 * self.fwhm * 1.e3 * np.sqrt(np.pi / np.log(2.)))
        # return tau0


def frange(start, stop, step):
    # stop at i => stop
    i = start
    while i < stop:
        yield i
        i += step


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_id(array, value):
    return (np.abs(array - value)).argmin()


def find_nearest_trans(trans_list, value):
    f_trans_list = []
    for tr in trans_list:
        f_trans_list.append(tr.f_trans_mhz)
    idx = (np.abs(np.array(f_trans_list) - value)).argmin()
    return trans_list[idx]


def get_partition_function(db, tag, temp):
    tref = []
    qlog = []
    for row in db.execute("SELECT * FROM cassis_parti_funct WHERE catdir_id = " + str(tag)):
        tref.append(row[1])
        qlog.append(np.power(10., row[2]))
    tref.sort()
    qlog.sort()
    return np.interp(temp, tref, qlog)
    # return np.power(10., qlog[find_nearest_id(np.array(tref),temp)])


def fwhm_to_sigma(value, reverse=False):
    if reverse:
        return value * (2. * np.sqrt(2. * np.log(2.)))
    else:
        return value / (2. * np.sqrt(2. * np.log(2.)))


def kms_to_mhz(value, fref_mhz, reverse=False):
    if reverse:
        return (value / fref_mhz) * const.c.value * 1.e-3
    else:
        return value * 1.e3 * fref_mhz / const.c.value


def jnu(fmhz, temp: float):
    fmhz_arr = np.array(fmhz) if type(fmhz) == list else fmhz
    res = (const.h.value * fmhz_arr * 1.e6 / const.k_B.value) / \
          (np.exp(const.h.value * fmhz_arr * 1.e6 / (const.k_B.value * temp)) - 1.)
    return list(res) if type(fmhz) == list else res


def get_beam_size(tel, freq_mhz):
    #tel?
    tel_dic = {'iram': 30.,
               'apex': 12.,
               'jcmt': 15.,
               'alma_400m': 400.}
    return (1.22 * const.c.value / (freq_mhz * 1.e6)) / tel_dic[tel] * 3600. * 180. / np.pi


def dilution_factor(source_size, beam_size, geometry='gaussian'):
    if geometry == 'disc':
        return 1. - np.exp(-np.log(2.) * (source_size / beam_size) ** 2)
    else:
        return source_size ** 2 / (source_size ** 2 + beam_size ** 2)
    
    
if __name__ == '__main__':  
    
    #GET DATABASE
    conn = sqlite3.connect(FILEPATH)

    db = conn.cursor()


    # tc = 1.  # 0.16
    tc = 0.
    tcmb = 2.73
    
    #FOR GENERATING A RANGE OF MODELS
    nrows = 1
    
    np.random.seed(1234)
    
    #RANDOMLY GENERATED PARAMETERS IN RANGES

    colden = np.array([1e19,1e15])
    extemp = np.array([400.,200.])
    fullwidth = np.array([5.,5.])
    lsr_vel = np.array([0.,0.])
    sourcesize = np.array([5.0,0.05])
    isoratio = 60.

    
    inputs = {'Column density': colden, 'Excitation temperature':extemp, 'FWHM':fullwidth, 'Velocity':lsr_vel, 'Size':sourcesize}
#    input_data = pd.DataFrame(inputs)
    
    freqmin = 238850
    freqmax = 239180
    length = (freqmax - freqmin) * 10
    intensities = np.zeros(shape=(nrows,length))
    intensities_1 = np.zeros(shape=(nrows,length))
    intensities_2 = np.zeros(shape=(nrows,length))
    
    for i in range(0,nrows):
        
        N = colden #column density
        T = extemp #exictation temperature
        W = fullwidth #full width at half maximum
        V = lsr_vel #local standard of rest velocity
        S = sourcesize #angular size?
        
        #CH3CN has species ID 137 in the database (I think CO has ID 50)
        # CH3{13}CN  154
        # 13CH3CN    153

        # CH3CN
        species1 = 137
        species2 = 154
        species1_lab = 'CH3CN'
        species2_lab = 'CH3{13}CN'
        
        
        #CH3CN
        cpt1 = Component(db, [Species(species1, ntot=N[0], tex=T[0], fwhm=W[0])],
                              isInteracting=False,vlsr=V[0], size=S[0])
        cpt2 = Component(db, [Species(species1, ntot=N[1], tex=T[1], fwhm=W[1])],
                              isInteracting=False,vlsr=V[1], size=S[1])

        # CH3{13}CN
        # cpt13_1 = Component(db, [Species(species2, ntot=N[0]/isoratio, tex=T[0],
                         # fwhm=W[0])], isInteracting=False,vlsr=V[0], size=S[0])
        
        # cpt13_2 = Component(db, [Species(species2, ntot=N[1]/isoratio, tex=T[1],
        #                 fwhm=W[1])],isInteracting=False,vlsr=V[1], size=S[1])


        
        cpt_list = [cpt1, cpt2]#,cpt13_1,cpt13_2]
        cpt_list_1 = [cpt1]
        cpt_list_2 = [cpt2]
        
        #both components
        model = ModelSpectrum(cpt_list, fmhz_min=freqmin, fmhz_max=freqmax,
                              eup_max=1000., telescope='alma_400m',
                              tc=tc, tcmb=tcmb)
        
        model.compute_model()
        intens = model.intensities
        intensities[i] = intens
        
        #cpt 1
        model_1 = ModelSpectrum(cpt_list_1, fmhz_min=freqmin, fmhz_max=freqmax,
                              eup_max=1000., telescope='alma_400m',
                              tc=tc, tcmb=tcmb)
        
        model_1.compute_model()
        intens_1 = model.intensities
        intensities_1[i] = intens_1
        
        # cpt 2
        model_2 = ModelSpectrum(cpt_list_2, fmhz_min=freqmin, fmhz_max=freqmax,
                              eup_max=1000., telescope='alma_400m',
                              tc=tc, tcmb=tcmb)
        
        model_2.compute_model()
        intens_2 = model_2.intensities
        intensities_2[i] = intens_2
    
    freqs = model.frequencies
    
    
    #PLOT both components
    plt.figure()
    plt.plot(freqs, intensities[0], linewidth = 2)
    plt.xticks(fontsize= 10)
    plt.yticks(fontsize= 10)
    #plt.grid()
    plt.xlabel('Frequency / MHz', fontsize = 15)
    plt.ylabel('Intensity / K', fontsize = 15)
    plt.title('extemp 1: ' + f"{extemp[0] :.3g} K, " + 'and extemp 2: '  + f"{extemp[1] :.3g} K", fontsize=15)    
    fig_name = 'two comps extemp '  + f"{extemp[0] :.3g}" + ' extemp '  + f"{extemp[1] :.3g}" + '.png'
    # plt.savefig(fig_name)
    
    #PLOT 1st component
    plt.figure()
    plt.plot(freqs, intensities_1[0], linewidth = 2)
    plt.xticks(fontsize= 10)
    plt.yticks(fontsize= 10)
    #plt.grid()
    plt.xlabel('Frequency / MHz', fontsize = 15)
    plt.ylabel('Intensity / K', fontsize = 15)
    plt.title('extemp - ' + f"{extemp[0] :.3g}, ", fontsize=15)    
    fig_name = 'varying extemp '  + f"{extemp[0] :.3g}" + '.png'
    # plt.savefig(fig_name)
    
    #PLOT 2nd component
    plt.figure()
    plt.plot(freqs, intensities_2[0], linewidth = 2)
    plt.xticks(fontsize= 10)
    plt.yticks(fontsize= 10)
    #plt.grid()
    plt.xlabel('Frequency / MHz', fontsize = 15)
    plt.ylabel('Intensity / K', fontsize = 15)
    plt.title('extemp - ' + f"{extemp[1] :.3g}", fontsize=15)    
    fig_name = 'varying extemp '  + f"{extemp[1] :.3g}" + '.png'
    # plt.savefig(fig_name)


    # area = integrate.simps(intensities[0], freqs)
    # print(area)
    # print(np.max(intensities[0]))


    #STORE SPECTRUM
    sp_frame = {'Frequency':freqs/1e3, 'Intensity':intensities}        
    filename = '2comp_model' + '_two_var_'+ 'extemp_1_' + f"{extemp[0] :.3g}" + '_extemp_2_' + f"{extemp[1] :.3g}" + '.dat'
    outfile = filename
    
    
    sp_frame_1 = {'Frequency':freqs/1e3, 'Intensity':intensities_1}    
    filename_1 = '2comp_model' + '_two_var_'+ '_extemp_' + f"{extemp[0] :.3g}" + '.dat'
    outfile_1 = filename_1
    
    
    sp_frame_2 = {'Frequency':freqs/1e3, 'Intensity':intensities_2}    
    filename_2 = '2comp_model' + '_two_var_'+ '_extemp_' + f"{extemp[1] :.3g}" + '.dat'
    outfile_2 = filename_2
    
    
    
    # colden = np.array([0.5e17,1e19])
    # extemp = np.array([150,350])
    # fullwidth = np.array([4.5,4.5])
    # lsr_vel = np.array([0,0])
    # sourcesize = np.array([0.3,0.05])
    # isoratio = 60


    with open(outfile, 'w') as f:
        
#        f.write(f'#  Species 1: {species1} {species1_lab}'+'\n')
#        f.write(f'#  {N[0]:.3g} {T[0]:.1f} {W[0]:.1f} {V[0]:.1f} {S[0]:.1f}   '+'\n')

        f.write(f'#  Species 2: {species2} {species2_lab}'+'\n')
        f.write(f'#  {N[0]:.3g} {T[0]:.1f} {W[0]:.1f} {V[0]:.1f} {S[0]:.1f}   '+'\n')
#        f.write(f'#  {N[1]:.3g} {T[1]:.1f} {W[1]:.1f} {V[1]:.1f} {S[1]:.1f}   '+'\n')
        f.write(f'# N(cm^-2) T(K) dv(km/s) vlsr(km/s) size(\") \n')
#        f.write(f'# isorato:  {isoratio:.1f} '+'\n')
        for fval,tr in zip(freqs,intensities[0]):
            f.write(f'{fval}  {tr}'+'\n')
            
    
    with open(outfile_1, 'w') as f:
        
#        f.write(f'#  Species 1: {species1} {species1_lab}'+'\n')
#        f.write(f'#  {N[0]:.3g} {T[0]:.1f} {W[0]:.1f} {V[0]:.1f} {S[0]:.1f}   '+'\n')

        f.write(f'#  Species 2: {species2} {species2_lab}'+'\n')
        f.write(f'#  {N[0]:.3g} {T[0]:.1f} {W[0]:.1f} {V[0]:.1f} {S[0]:.1f}   '+'\n')
#        f.write(f'#  {N[1]:.3g} {T[1]:.1f} {W[1]:.1f} {V[1]:.1f} {S[1]:.1f}   '+'\n')
        f.write(f'# N(cm^-2) T(K) dv(km/s) vlsr(km/s) size(\") \n')
#        f.write(f'# isorato:  {isoratio:.1f} '+'\n')
        for fval,tr in zip(freqs,intensities_1[0]):
            f.write(f'{fval}  {tr}'+'\n')
            
    
    with open(outfile_2, 'w') as f:
        
#        f.write(f'#  Species 1: {species1} {species1_lab}'+'\n')
#        f.write(f'#  {N[0]:.3g} {T[0]:.1f} {W[0]:.1f} {V[0]:.1f} {S[0]:.1f}   '+'\n')

        f.write(f'#  Species 2: {species2} {species2_lab}'+'\n')
        f.write(f'#  {N[0]:.3g} {T[0]:.1f} {W[0]:.1f} {V[0]:.1f} {S[0]:.1f}   '+'\n')
#        f.write(f'#  {N[1]:.3g} {T[1]:.1f} {W[1]:.1f} {V[1]:.1f} {S[1]:.1f}   '+'\n')
        f.write(f'# N(cm^-2) T(K) dv(km/s) vlsr(km/s) size(\") \n')
#        f.write(f'# isorato:  {isoratio:.1f} '+'\n')
        for fval,tr in zip(freqs,intensities_2[0]):
            f.write(f'{fval}  {tr}'+'\n')


    print(f'{outfile} written'+'\n')
    print(f'{outfile_1} written'+'\n')
    print(f'{outfile_2} written'+'\n')
