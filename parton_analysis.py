import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import gzip

import itertools
from math import isclose
#import pylhe

import ROOT

###########
# classes #
###########
class Event:
    def __init__(self, lhe_evnt_data_str, lhe_particle_data_strs):
        self.num_particles = int(lhe_evnt_data_str[0])
        self.proc_id = int(lhe_evnt_data_str[1])
        self.evnt_weight = float(lhe_evnt_data_str[2])
        self.scale_GeV = float(lhe_evnt_data_str[3])
        self.alpha_QED = float(lhe_evnt_data_str[4])
        self.alpha_QCD = float(lhe_evnt_data_str[5])
        self.particles = []
        self.lp = None
        self.lm = None
        self.is_t_leptonic = False
        self.is_tbar_leptonic = False
        for i in range(self.num_particles):
            # create particle
            self.particles.append(Particle(lhe_particle_data_strs[13*i:13*(i+1)]))
            p = self.particles[-1]
            # assign important particles
            if p.type() == 't':
                self.t = p
            elif p.type() == 't~':
                self.tbar = p
            elif p.type() == 'w+':
                self.wp = p
            elif p.type() == 'w-':
                self.wm = p
            # add to children of parents
            p1 = p.parents[0]
            p2 = p.parents[1]
            if p1 >= 0:
                self.particles[p1].children.append(i)
            if p2 != p1 and p2 >= 0:
                self.particles[p2].children.append(i)
            # determine leptonic decays
            if abs(p.PDGcode) > 10 and abs(p.PDGcode) < 17 and p.PDGcode % 2:
                if p.PDGcode > 0:
                    self.lp = p
                    self.is_t_leptonic = True
                else:
                    self.lm = p
                    self.is_tbar_leptonic = True
        assert(self.is_t_leptonic != self.is_tbar_leptonic)

class Particle:
    # dictionary of PDG codes for fundamental particles 2020 rev:
    # https://pdg.lbl.gov/2020/reviews/rpp2020-rev-monte-carlo-numbering.pdf
    _PDGcodes = {1:'d', 2:'u', 3:'s', 4:'c', 5:'b', 6:'t', 9:'g (glueball)', 
                11:'e-', 12:'ve', 13:'mu-', 14:'vm', 15:'tau-', 16:'vt', 
                21:'g', 22:'a (gamma)', 23:'z', 24:'w+', 25:'h',
                -1:'d~', -2:'u~', -3:'s~', -4:'c~', -5:'b~', -6:'t~', 
                -11:'e+', -12:'ve~', -13:'mu+', -14:'vm~', -15:'tau+', -16:'vt~', 
                -24:'w-'}

    def __init__(self,lhe_particle_data_str):
        assert(len(lhe_particle_data_str) == 13)
        self.PDGcode = int(lhe_particle_data_str[0])
        self.status = int(lhe_particle_data_str[1])
        self.parents = [int(i)-1 for i in lhe_particle_data_str[2:4]]
        self.children = []
        self.colors = [int(i) for i in lhe_particle_data_str[4:6]]
        self.momentum_GeV = [float(i) for i in lhe_particle_data_str[6:9]]
        self.energy_GeV = float(lhe_particle_data_str[9])
        self.mass_GeV = float(lhe_particle_data_str[10])
        self.distance_mm = float(lhe_particle_data_str[11])
        self.helicity = float(lhe_particle_data_str[12])

    # why I love oop
    def __str__(self) -> str:
        # this might be bad code
        return self._PDGcodes[self.PDGcode]
    
    def __repr__(self) -> str:
        # this is probably bad code
        return self._PDGcodes[self.PDGcode]
    
    def type(self) -> str:
        return self._PDGcodes[self.PDGcode]

    def fourMomentum(self):
        return np.append(self.momentum_GeV,self.energy_GeV)
    
    def fourMomentumRoot_PxPyPzE(self):
        return ROOT.Math.LorentzVector['ROOT::Math::PxPyPzE4D<double>'](self.momentum_GeV[0],self.momentum_GeV[1],self.momentum_GeV[2],self.energy_GeV)
    
####################################
# event functions (event methods?) #
####################################     
def get_hadrons(w: Particle, ps):
    q1 = ps[w.children[0]]
    q2 = ps[w.children[1]]
    return q1,q2

def decays_leptonically(p: Particle, e: Event):
    if p.PDGcode > 0:
        return e.is_t_leptonic
    else:
        return e.is_tbar_leptonic
#####################
# physics functions #
#####################
def compute_helicity_basis(t: Particle, tbar: Particle):
    # fixed beam basis: ttbar center of mass frame, z points along beam
    # helicity basis: k points along t in ttbar center of mass frame (han p.12)
    t_4momentum = t.fourMomentumRoot_PxPyPzE()
    tbar_4momentum = tbar.fourMomentumRoot_PxPyPzE()
    k_hat = boost_to_CM(t_4momentum,t_4momentum,tbar_4momentum).Vect().Unit()
    z_hat = ROOT.Math.DisplacementVector3D['ROOT::Math::Cartesian3D<double>'](0,0,1)
    cos_theta = z_hat.Dot(k_hat)
    k_xz_proj = ROOT.Math.DisplacementVector3D['ROOT::Math::Cartesian3D<double>'](k_hat.X(),0,k_hat.Z())
    sin_theta = np.sign(z_hat.Cross(k_xz_proj).Y())*ROOT.TMath.Sqrt(1-cos_theta*cos_theta)
    r_hat = scalar_multiply(1/sin_theta, z_hat - scalar_multiply(cos_theta,k_hat))
    n_hat = r_hat.Cross(k_hat)
    return r_hat, k_hat, n_hat

def prob_dist_w_angle(cos_theta_w):
    # han p.37
    mt_GeV_2 = 172.6**2
    mw2_GeV_2 = 2*80.4**2
    total_m_2 = mt_GeV_2+mw2_GeV_2
    return 3*(1-cos_theta_w**2)*mt_GeV_2 / (4*total_m_2) + 3*mw2_GeV_2*(1-cos_theta_w)**2 / (8*total_m_2)

def prob_soft(cos_theta_w):
    # han 9.37
    theta = abs(cos_theta_w)
    P_soft = prob_dist_w_angle(-theta)/(prob_dist_w_angle(theta)+prob_dist_w_angle(-theta))
    return P_soft

def prob_hard(cos_theta_w):
    # han p.37
    theta = abs(cos_theta_w)
    P_hard = prob_dist_w_angle(theta)/(prob_dist_w_angle(theta)+prob_dist_w_angle(-theta))
    return P_hard

def print_3_vec(vec):
    print(vec.X(),vec.Y(),vec.Z())

def compute_optimal_hadronic_direction(t: Particle, b: Particle, w: Particle, q1: Particle, q2: Particle):
    # linear combination of hard and soft jets in t or tbar rest frame (han p.14,37)
    t_four_momentum = t.fourMomentumRoot_PxPyPzE()
    b_four_momentum = b.fourMomentumRoot_PxPyPzE()
    w_four_momentum = w.fourMomentumRoot_PxPyPzE()
    q1_four_momentum = q1.fourMomentumRoot_PxPyPzE()
    q2_four_momentum = q2.fourMomentumRoot_PxPyPzE()

    # compute helicity angle, angle between down type quark and W in w rest frame
    if abs(q1.PDGcode) % 2:
        q_down = q1
        q_down_four_momentum = q1_four_momentum
        q_up = q2
        q_up_four_momentum = q2_four_momentum
    else:
        q_down = q2
        q_down_four_momentum = q2_four_momentum
        q_up = q1
        q_up_four_momentum = q1_four_momentum
    # this should be doable without the b quark, check boosts
    b_four_w = boost_to_CM(b_four_momentum,w_four_momentum)
    w_dir = -b_four_w.Vect().Unit()
    q_down_dir_w_rest = boost_to_CM(q_down_four_momentum,w_four_momentum).Vect().Unit()
    cos_theta_w = w_dir.Dot(q_down_dir_w_rest)

    #print_3_vec(w_dir)
    #w_top = boost_to_CM(w_four_momentum,t_four_momentum)
    #print_3_vec(w_top.Vect().Unit())

    # forward emitted quark in W rest frame is the harder quark (https://arxiv.org/pdf/1401.3021 p.6)
    if cos_theta_w > 0:
        q_hard = q_down
        q_hard_four_momentum = q_down_four_momentum
        q_soft = q_up
        q_soft_four_momentum = q_up_four_momentum
    else:
        q_hard = q_up
        q_hard_four_momentum = q_up_four_momentum
        q_soft = q_down
        q_soft_four_momentum = q_down_four_momentum

    # boost hard and soft quarks to top rest frame
    q_soft_four_t = boost_to_CM(q_soft_four_momentum,t_four_momentum)
    q_hard_four_t = boost_to_CM(q_hard_four_momentum,t_four_momentum)
    assert(q_soft_four_t.E() < q_hard_four_t.E())
    q_soft_dir = q_soft_four_t.Vect().Unit()
    q_hard_dir = q_hard_four_t.Vect().Unit()
    opt_momentum = scalar_multiply(prob_soft(cos_theta_w),q_soft_dir) + scalar_multiply(prob_hard(cos_theta_w),q_hard_dir)
    return opt_momentum

def compute_Omega(p_rest: Particle, e: Event):
    # unit vector along decay product's momentum in parent's rest frame (han p.9,16)
    if decays_leptonically(p_rest,e):
        if p_rest.PDGcode > 0:
            l = e.lp
        else:
            l = e.lm
        decay_axis = boost_to_CM(l.fourMomentumRoot_PxPyPzE(),p_rest.fourMomentumRoot_PxPyPzE()).Vect().Unit()
    else:
        if p_rest.PDGcode > 0:
            w = e.wp
        else:
            w = e.wm
        q1,q2 = get_hadrons(w,e.particles)
        for i in p_rest.children:
            if abs(e.particles[i].PDGcode) < 10:
                b = e.particles[i]
        decay_axis = compute_optimal_hadronic_direction(p_rest,b,w,q1,q2).Unit()
    return decay_axis

def compute_p_opt(p_rest: Particle, e: Event):
    # unit vector along decay product's momentum in parent's rest frame (han p.9,16)
    if p_rest.PDGcode > 0:
        w = e.wp
    else:
        w = e.wm
    q1,q2 = get_hadrons(w,e.particles)
    for i in p_rest.children:
        if abs(e.particles[i].PDGcode) < 10:
            b = e.particles[i]
    p_opt = compute_optimal_hadronic_direction(p_rest,b,w,q1,q2)
    return p_opt


def compute_cos_theta_axis(parent: Particle, e: Event, axis):
    # cos of angle between decay product and an axis in parent's restframe (han p.9)
    omega = compute_Omega(parent,e)
    return omega.Dot(axis)

def compute_cos_theta_ttbar(t: Particle, tbar: Particle, e: Event):
    # cos of angle between t decay product and tbar in their respective rest frames (han p.16)
    t_Omega = compute_Omega(t,e)
    tbar_Omega = compute_Omega(tbar,e)
    return t_Omega.Dot(tbar_Omega)

def compute_asymmetry(x,x_min,x_max):
    # asymmetry of a distribution (han p.10)
    weights = np.ones(x.size)
    # does N bins matter?
    N_bins = 80
    dsigma = ROOT.TH1D("asym_hist", "dsigma", N_bins, x_min, x_max)
    dsigma.FillN(x.size,x,weights)
    dsigma.Scale(1.0/dsigma.Integral())
    bin_low = 0
    bin_mid = dsigma.FindFixBin(0)
    bin_high = N_bins + 1
    N_plus = dsigma.Integral(bin_mid,bin_high)
    N_minus = dsigma.Integral(bin_low,bin_mid-1)
    return (N_plus - N_minus) / (N_plus + N_minus)

def compute_Cij(cos_theta_i,cos_theta_j):
    # compute spin corelation matrix element with respect to ij axes
    # method 2 of han p.10
    kappa_A = 1
    kappa_B = .64
    cos_i_cos_j = cos_theta_i*cos_theta_j
    asymmetry = compute_asymmetry(cos_i_cos_j,-1,1)
    return -4*asymmetry/(kappa_A*kappa_B)

#################
# root wrappers #
#################
def scalar_multiply(scalar, displacement_vector_3D):
    # some root overloads break in python
    return ROOT.Math.DisplacementVector3D['ROOT::Math::Cartesian3D<double>'](scalar*displacement_vector_3D.x(),
                                                                        scalar*displacement_vector_3D.y(),
                                                                        scalar*displacement_vector_3D.z())

def boost_to_CM(p1_4momentum, p2_4momentum, p3_4momentum = None):
    # get four momentum of p1 in p2's rest frame
    # get boost matrix
    if p3_4momentum is None:
        p2_beta = p2_4momentum.BoostToCM()
    else:
        p2_beta = p2_4momentum.BoostToCM(p3_4momentum)
    p2_to_CM_mat = ROOT.Math.Boost(p2_beta)
    # boost p2
    p1_four_boosted = p2_to_CM_mat*p1_4momentum
    return p1_four_boosted    

#####################
# parsing functions #
#####################
def get_lhe_tags(f_name: str) -> set[str]:
    tags = set()
    with gzip.open(f_name, 'rb') as f:
        for event,element in ET.iterparse(f):
            if 'event' in tags:
                # are there tags after the events?
                break
            tags.add(element.tag)
    return tags

def get_simulation_meta_data(f_name: str):
    keys = ["beam1_PID",
            "beam2_PID",
            "beam1_energy_GeV",
            "beam1_energy_GeV",
            "PDF_author_group1",
            "PDF_author_group2",
            "PDFset1",
            "PDFset2",
            "weightingStrategy",
            "numProcesses",
            "sigma_pb",
            "sigma_error_pb",
            "max_weight",
            "tag"]
    with gzip.open(f_name, 'rb') as f:
        for event,element in ET.iterparse(f):
            if element.tag == 'MGGenerationInfo':
                event_info = element.text.split()
                N_events = int(event_info[5])
                cross_section = float(event_info[-1])
            elif element.tag == 'init':
                vals = element.text.split()
                meta_data = {pair[0]: (int(pair[1]) if pair[1].find('.') < 0 else float(pair[1])) for pair in zip(keys,vals)}
                break
        meta_data['integrated_weight_pb'] = cross_section
        meta_data['numEvents'] = N_events
        return meta_data
            

def get_events_from_lhe_gzip(f_name: str):
    with gzip.open(f_name, 'rb') as f:
        for event,element in ET.iterparse(f):
            if element.tag == 'event':
                evnt_raw = element.text.split()
                evnt_data = evnt_raw[:6]
                particle_data = evnt_raw[6:]
                yield Event(evnt_data,particle_data)

def get_awk_events_from_lhe_batched(f_name: str, num_events: int):
    # proper batched parsing requires low level xml parser, implement if parallel bathces 
    # https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLParser
    i = 0
    # create an empty awkward array of size num_events?
    events = []
    for e in get_events_from_lhe_gzip(f_name):
        events.append(e)
        if not i%num_events:
            print(i)
            print(len(events))
            yield ak.Array(events)
        i+=1

###########
# drivers #
###########
def run_analysis(f_name: str):
    events = get_events_from_lhe_gzip(f_name)
    meta_data = get_simulation_meta_data(f_name)
    sigma = meta_data['sigma_pb']
    cos_theta_t_n = []
    cos_theta_tbar_n = []
    cos_theta_ttbar = []
    #mtt = np.empty(N_evnts)
    i = 0
    for e in events:
        t = e.t
        tbar = e.tbar
        r_hat, k_hat, n_hat = compute_helicity_basis(t,tbar)
        cos_theta_t_n.append(compute_cos_theta_axis(t,e,n_hat))
        cos_theta_tbar_n.append(compute_cos_theta_axis(tbar,e,n_hat))
        cos_theta_ttbar.append(compute_cos_theta_ttbar(t,tbar,e))
        #mtt[i] = (t.fourMomentumRoot_PxPyPzE() + tbar.fourMomentumRoot_PxPyPzE()).M()
        if i%1000 == 0:
            print(i)
        i+=1
    cos_theta_t_n_arr = np.array(cos_theta_t_n)
    cos_theta_tbar_n_arr = np.array(cos_theta_tbar_n)
    compute_Cij(cos_theta_t_n_arr,cos_theta_tbar_n_arr)
    ctn_ctn_arr = cos_theta_t_n_arr*cos_theta_tbar_n_arr
    
    # Root histogram 
    c1 = ROOT.TCanvas()
    weights = np.ones(ctn_ctn_arr.size)
    h1 = ROOT.TH1D("h1", "cnn", 80, -1, 1)
    h1.FillN(ctn_ctn_arr.size,ctn_ctn_arr,weights)
    h1.Scale(sigma/h1.Integral())
    h1.Draw()
    c1.Print("cnn.pdf")

if __name__ == "__main__":
    f_name = 'ttbar-parton-madgraph.lhe.gz'
    #f_name = 'ttest.lhe.gz'
    events = get_events_from_lhe_gzip(f_name)
    meta_data = get_simulation_meta_data(f_name)
    sigma = meta_data['sigma_pb']
    #N_events = meta_data['numEvents']
    N_events = 10
    cos_theta_t_n = np.zeros(N_events)
    cos_theta_tbar_n = np.zeros(N_events)
    p_opts = np.zeros(N_events)
    for i,e in enumerate(itertools.islice(events,0,N_events)):
        t = e.t
        tbar = e.tbar
        r_hat, k_hat, n_hat = compute_helicity_basis(t,tbar)
        #cos_theta_t_n[i] = compute_cos_theta_axis(t,e,n_hat)
        #cos_theta_tbar_n[i] = compute_cos_theta_axis(tbar,e,n_hat)
        if e.is_t_leptonic:
            p_opt = compute_p_opt(tbar,e)
        else:
            p_opt = compute_p_opt(t,e)
        print(p_opt.R())
        p_opts[i]=p_opt.R()
    
    #run_analysis(f_name)
    #meta_data = get_simulation_meta_data(f_name)
    #events_batch = get_awk_events_from_lhe_batched(f_name,10)
    #for events in events_batch:
    #    print(events)
    #    break