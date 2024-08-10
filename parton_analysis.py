# python headers
#import xml.etree.ElementTree as ET
#from math import isclose
#import itertools
#import gzip

# python scientific headers
import matplotlib.pyplot as plt
import numpy as np

# scikit hep headers
#import awkward as ak
import vector
import uproot
import pylhe

# yuleigh compatibility
#import pandas as pd
#from pathlib import Path
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go

# root
#import ROOT

# local
#from root_wrapper import *

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
            elif abs(p.PDGcode) == 24:
                self.w = p
            # add to children of parents
            p1 = p.parents[0]
            p2 = p.parents[1]
            if p1 >= 0:
                self.particles[p1].children.append(i)
            if p2 != p1 and p2 >= 0:
                self.particles[p2].children.append(i)
            # determine leptonic decays
            if abs(p.PDGcode) > 10 and abs(p.PDGcode) < 17 and p.PDGcode % 2:
                self.l = p
                if p.PDGcode > 0:
                    self.is_t_leptonic = True
                else:
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
    
    def fourMomentumRoot_TLorentz(self):
        return ROOT.TLorentzVector(self.momentum_GeV[0],self.momentum_GeV[1],self.momentum_GeV[2],self.energy_GeV)
    
    def fourMomentumRoot_PxPyPzE(self):
        return ROOT.Math.LorentzVector['ROOT::Math::PxPyPzE4D<double>'](self.momentum_GeV[0],self.momentum_GeV[1],self.momentum_GeV[2],self.energy_GeV)
    
    def transverseMomentum(self):
        return self.fourMomentumRoot_PxPyPzE().Pt()
    
    def transverseEnergy(self):
        return self.fourMomentumRoot_PxPyPzE().Et()
    
    def pseudorapidity(self):
        return self.fourMomentumRoot_PxPyPzE().Eta()
    
###################
# Event functions #
###################
def decays_leptonically(p: Particle, e: Event):
    if p.PDGcode*e.l.PDGcode > 0:
        return True
    else:
        return False
    
#####################
# physics functions #
#####################
def compute_helicity_basis(t_4momentum, tbar_4momentum):
    k_hat = t_4momentum.boostCM_of_p4(t_4momentum+tbar_4momentum).to_Vector3D().unit()
    z_hat = vector.obj(x=0,y=0,z=1)
    #z_hat = (t_4momentum+tbar_4momentum).to_Vector3D().unit()
    cos_theta = z_hat.dot(k_hat)
    #k_xz_proj = vector.VectorNumpy3D({'x':k_hat.px,'y':0*k_hat.py,'z':k_hat.pz})
    #sin_theta = np.sign(z_hat.cross(k_xz_proj).y)*np.sqrt(1-cos_theta*cos_theta)
    sin_theta = np.sqrt(1-cos_theta*cos_theta)
    r_hat = (z_hat - cos_theta*k_hat)/sin_theta
    n_hat = r_hat.cross(k_hat)
    return np.abs(cos_theta), r_hat, k_hat, n_hat

def compute_Omega_leptonic(t_lep_4momentum, t_had_4momentum, l_4momentum):
    # unit vector along leptonic decay product momentum in parent's rest frame (han p.9,16)
    ttbar_CM_4momentum = t_lep_4momentum + t_had_4momentum
    
    t_lep_4momentum_CM = t_lep_4momentum.boostCM_of_p4(ttbar_CM_4momentum)

    l_4momentum_CM = l_4momentum.boostCM_of_p4(ttbar_CM_4momentum)
    l_4momentum_top = l_4momentum_CM.boostCM_of_p4(t_lep_4momentum_CM)
    return l_4momentum_top.to_Vector3D().unit()

def prob_dist_w_angle(cos_theta_w):
    # han p.37
    mt_GeV_2 = 172.6**2
    mw2_GeV_2 = 2*80.4**2
    total_m_2 = mt_GeV_2+mw2_GeV_2
    return 3*(1-cos_theta_w*cos_theta_w)*mt_GeV_2 / (4*total_m_2) + 3*mw2_GeV_2*(1-cos_theta_w)*(1-cos_theta_w) / (8*total_m_2)

def prob_soft(fp,fm):
    # han 9.37
    return fm/(fp+fm)

def prob_hard(fp,fm):
    # han p.37
    return fp/(fp+fm)

def compute_Omega_hadronic(t_lep_4momentum, t_had_4momentum, w_4momentum, qd_4momentum, qu_4momentum):
    # unit vector along optimal hadronic direction in parent's rest frame (han p.9,16)
    # linear combination of hard and soft jets in t or tbar rest frame (han p.14,37)
    # boost to relevant frames
    ttbar_CM_4momentum = t_lep_4momentum + t_had_4momentum
    
    t_had_four_momentum_CM = t_had_4momentum.boostCM_of_p4(ttbar_CM_4momentum)
    w_four_momentum_CM = w_4momentum.boostCM_of_p4(ttbar_CM_4momentum)
    qd_four_momentum_CM = qd_4momentum.boostCM_of_p4(ttbar_CM_4momentum)
    qu_four_momentum_CM = qu_4momentum.boostCM_of_p4(ttbar_CM_4momentum)

    w_four_momentum_top = w_four_momentum_CM.boostCM_of_p4(t_had_four_momentum_CM)
    qd_four_momentum_top = qd_four_momentum_CM.boostCM_of_p4(t_had_four_momentum_CM)
    qu_four_momentum_top = qu_four_momentum_CM.boostCM_of_p4(t_had_four_momentum_CM)

    qd_four_momentum_w = qd_four_momentum_top.boostCM_of_p4(w_four_momentum_top)
    #qu_four_momentum_w = qu_four_momentum_top.boostCM_of_p4(w_four_momentum_top)

    # compute helicity angle, angle between down type quark and W in w rest frame
    w_dir = w_four_momentum_top.to_Vector3D().unit()
    q_down_dir = qd_four_momentum_w.to_Vector3D().unit()
    cos_theta_w = w_dir.dot(q_down_dir)
    
    # forward emitted quark in W rest frame is the harder quark (https://arxiv.org/pdf/1401.3021 p.6)
    q_hard_four_momentum_top = vector.MomentumNumpy4D(np.where(cos_theta_w >= 0,qd_four_momentum_top,qu_four_momentum_top))
    q_soft_four_momentum_top = vector.MomentumNumpy4D(np.where(cos_theta_w < 0,qu_four_momentum_top,qd_four_momentum_top))
    assert(np.all(q_soft_four_momentum_top.E <= q_hard_four_momentum_top.E))
    q_soft_dir = q_soft_four_momentum_top.to_Vector3D().unit()
    q_hard_dir = q_hard_four_momentum_top.to_Vector3D().unit()

    # compute probability distributions
    c_theta = np.abs(cos_theta_w)
    fp = prob_dist_w_angle(c_theta)
    fm = prob_dist_w_angle(-c_theta)

    # form optimal hadronic direction
    opt_had_momentum = prob_soft(fp,fm)*q_soft_dir + prob_hard(fp,fm)*q_hard_dir
    return cos_theta_w, opt_had_momentum.to_Vector3D().unit()

def compute_helicity_basis_old(t: Particle, tbar: Particle):
    # fixed beam basis: ttbar center of mass frame, z points along beam
    # helicity basis: k points along t in ttbar center of mass frame (han p.12)
    t_4momentum = t.fourMomentumRoot_PxPyPzE()
    tbar_4momentum = tbar.fourMomentumRoot_PxPyPzE()
    k_hat = boost_to_CM(t_4momentum,t_4momentum,tbar_4momentum).Vect().Unit()
    z_hat = ROOT.Math.DisplacementVector3D['ROOT::Math::Cartesian3D<double>'](0,0,1)
    #z_hat = (t_4momentum+tbar_4momentum).Unit()
    cos_theta = z_hat.Dot(k_hat)
    #k_xz_proj = ROOT.Math.DisplacementVector3D['ROOT::Math::Cartesian3D<double>'](k_hat.X(),0,k_hat.Z())
    #sin_theta = np.sign(z_hat.Cross(k_xz_proj).Y())*ROOT.TMath.Sqrt(1-cos_theta*cos_theta)
    sin_theta = ROOT.TMath.Sqrt(1-cos_theta*cos_theta)
    r_hat = scalar_multiply(1/sin_theta, z_hat - scalar_multiply(cos_theta,k_hat))
    n_hat = r_hat.Cross(k_hat)
    return cos_theta, r_hat, k_hat, n_hat

def compute_optimal_hadronic_direction_old(t: Particle, tbar: Particle, b: Particle, w: Particle, q1: Particle, q2: Particle):
    # linear combination of hard and soft jets in t or tbar rest frame (han p.14,37)
    t_four_momentum = t.fourMomentumRoot_PxPyPzE()
    tbar_four_momentum = tbar.fourMomentumRoot_PxPyPzE()
    #b_four_momentum = b.fourMomentumRoot_PxPyPzE()
    w_four_momentum = w.fourMomentumRoot_PxPyPzE()
    q1_four_momentum = q1.fourMomentumRoot_PxPyPzE()
    q2_four_momentum = q2.fourMomentumRoot_PxPyPzE()

    t_four_momentum_CM = boost_to_CM(t_four_momentum,t_four_momentum,tbar_four_momentum)
    #tbar_four_momentum_CM = boost_to_CM(tbar_four_momentum,t_four_momentum,tbar_four_momentum)
    #b_four_momentum_CM = boost_to_CM(b_four_momentum,t_four_momentum,tbar_four_momentum)
    w_four_momentum_CM = boost_to_CM(w_four_momentum,t_four_momentum,tbar_four_momentum)
    q1_four_momentum_CM = boost_to_CM(q1_four_momentum,t_four_momentum,tbar_four_momentum)
    q2_four_momentum_CM = boost_to_CM(q2_four_momentum,t_four_momentum,tbar_four_momentum)

    #b_four_momentum_top = boost_to_CM(b_four_momentum_CM,t_four_momentum_CM)
    w_four_momentum_top = boost_to_CM(w_four_momentum_CM,t_four_momentum_CM)
    q1_four_momentum_top = boost_to_CM(q1_four_momentum_CM,t_four_momentum_CM)
    q2_four_momentum_top = boost_to_CM(q2_four_momentum_CM,t_four_momentum_CM)

    #b_four_momentum_w = boost_to_CM(b_four_momentum_top,w_four_momentum_top)
    q1_four_momentum_w = boost_to_CM(q1_four_momentum_top,w_four_momentum_top)
    q2_four_momentum_w = boost_to_CM(q2_four_momentum_top,w_four_momentum_top)

    # compute helicity angle, angle between down type quark and W in w rest frame
    if abs(q1.PDGcode) % 2:
        q_down = q1
        q_down_four_momentum_w = q1_four_momentum_w
        q_down_four_momentum_top = q1_four_momentum_top
        q_up = q2
        q_up_four_momentum_w = q2_four_momentum_w
        q_up_four_momentum_top = q2_four_momentum_top
    else:
        q_down = q2
        q_down_four_momentum_w = q2_four_momentum_w
        q_down_four_momentum_top = q2_four_momentum_top
        q_up = q1
        q_up_four_momentum_w = q1_four_momentum_w
        q_up_four_momentum_top = q1_four_momentum_top
    # this should be doable without the b quark, check boosts
    w_dir1 = w_four_momentum_top.Vect().Unit()
    #w_dir2 = -b_four_momentum_w.Vect().Unit()
    #print_3_vec(w_dir1)
    #print_3_vec(w_dir2)
    q_down_dir = q_down_four_momentum_w.Vect().Unit()
    cos_theta_w = w_dir1.Dot(q_down_dir)

    # forward emitted quark in W rest frame is the harder quark (https://arxiv.org/pdf/1401.3021 p.6)
    if cos_theta_w > 0:
        q_hard = q_down
        q_hard_four_momentum_top = q_down_four_momentum_top
        q_soft = q_up
        q_soft_four_momentum_top = q2_four_momentum_top
    else:
        q_hard = q_up
        q_hard_four_momentum_top = q2_four_momentum_top
        q_soft = q_down
        q_soft_four_momentum_top = q_down_four_momentum_top

    # boost hard and soft quarks to top rest frame
    assert(q_soft_four_momentum_top.E() <= q_hard_four_momentum_top.E())
    q_soft_dir = q_soft_four_momentum_top.Vect().Unit()
    q_hard_dir = q_hard_four_momentum_top.Vect().Unit()

    c_theta = abs(cos_theta_w)
    fp = prob_dist_w_angle(c_theta)
    fm = prob_dist_w_angle(-c_theta)

    opt_momentum = scalar_multiply(prob_soft(fp,fm),q_soft_dir) + scalar_multiply(prob_hard(fp,fm),q_hard_dir)
    return opt_momentum

def compute_Omega_old(t_parent: Particle, t_other: Particle, e: Event):
    # unit vector along decay product's momentum in parent's rest frame (han p.9,16)
    if decays_leptonically(t_parent,e):
        t_four_momentum = t_parent.fourMomentumRoot_PxPyPzE()
        tbar_four_momentum = t_other.fourMomentumRoot_PxPyPzE()
        l_four_momentum = e.l.fourMomentumRoot_PxPyPzE()

        t_four_momentum_CM = boost_to_CM(t_four_momentum,t_four_momentum,tbar_four_momentum)
        l_four_momentum_CM = boost_to_CM(l_four_momentum,t_four_momentum,tbar_four_momentum)

        l_four_momentum_top = boost_to_CM(l_four_momentum_CM,t_four_momentum_CM)

        decay_axis = l_four_momentum_top.Vect().Unit()
    else:
        for i in t_parent.children:
            if abs(e.particles[i].PDGcode) < 10:
                b = e.particles[i]
        qs = [e.particles[i] for i in e.w.children]
        decay_axis = compute_optimal_hadronic_direction_old(t_parent,t_other,b,e.w,qs[0],qs[1]).Unit()
    return decay_axis

def compute_cos_theta_axis_old(t_parent: Particle, t_other: Particle, e: Event, axis):
    # cos of angle between decay product and an axis in parent's restframe (han p.9)
    omega = compute_Omega_old(t_parent,t_other,e)
    return omega.Dot(axis)

def compute_cos_theta_ttbar_old(t: Particle, tbar: Particle, e: Event):
    # cos of angle between t decay product and tbar in their respective rest frames (han p.16)
    t_Omega = compute_Omega_old(t,tbar,e)
    tbar_Omega = compute_Omega_old(tbar,t,e)
    return t_Omega.Dot(tbar_Omega)

def compute_asymmetry(x,x_min,x_max,label: str=None):
    # asymmetry of a distribution (han p.10)
    weights = np.ones(x.size)
    # does N bins matter?
    N_bins = 80
    name = label
    title = label
    if name is None:
        name = "asym_hist"
        title = "dsigma"
    c = ROOT.TCanvas(name+"c", title)
    dsigma = ROOT.TH1D(name, title, N_bins, x_min, x_max)
    dsigma.FillN(x.size,x,weights)
    dsigma.Sumw2()
    #for i in range(21):
    #    print(dsigma.GetBinContent(i))
    dsigma.Scale(1.0/dsigma.Integral())
    if name is not None:
        dsigma.Draw()
        #c.Print("plots/"+title+".pdf")
    bin_low = 0
    bin_mid = dsigma.FindFixBin(0)
    bin_high = N_bins + 1
    N_plus = dsigma.Integral(bin_mid,bin_high)
    N_minus = dsigma.Integral(bin_low,bin_mid-1)
    print(N_plus,N_minus)
    return (N_plus - N_minus) / (N_plus + N_minus)

def compute_Cij(cos_i_cos_j,label: str=None):
    # compute spin corelation matrix element with respect to ij axes
    # method 2 of han p.10
    kappa_A = 1
    kappa_B = .64
    asymmetry = compute_asymmetry(cos_i_cos_j,-1,1,label)
    print(asymmetry)
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

def get_lhe_meta_data(f_name: str):
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
            

def get_lhe_events_gzip(f_name: str, resolved=False):
    with gzip.open(f_name, 'rb') as f:
        for event,element in ET.iterparse(f):
            if element.tag == 'event':
                evnt_raw = element.text.split()
                evnt_data = evnt_raw[:6]
                particle_data = evnt_raw[6:]
                if resolved:
                    e = Event(evnt_data,particle_data)
                    reject = False
                    for p in e.particles:
                        if p.status == 1:
                            if abs(p.PDGcode) < 10 or (abs(p.PDGcode) > 10 and p.PDGcode % 2):
                                if p.transverseMomentum() <= 25 or abs(p.pseudorapidity()) >= 2.5:
                                    reject = True
                            elif abs(p.PDGcode) > 10 and not p.PDGcode % 2:
                                if p.transverseEnergy() <= 30:
                                    reject = True
                    if not reject:
                        yield e
                else: 
                    yield Event(evnt_data,particle_data)

#def get_awk_events_from_lhe_batched(f_name: str, num_events: int):
    # proper batched parsing requires low level xml parser, implement if parallel bathces 
    # https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.XMLParser
#    i = 0
    # create an empty awkward array of size num_events?
#    events = []
#    for e in get_events_from_lhe_gzip(f_name):
#        events.append(e)
#        if not i%num_events:
#            print(i)
#            print(len(events))
#            yield ak.Array(events)
#        i+=1

###########
# drivers #
###########
def run_analysis(f_name: str):
    events = get_lhe_events_gzip(f_name)
    meta_data = get_lhe_meta_data(f_name)
    sigma = meta_data['sigma_pb']
    N_events = meta_data['numEvents']
    cos_theta_t_n = np.zeros(N_events)
    cos_theta_tbar_n = np.zeros(N_events)
    #cos_theta_ttbar = []
    #mtt = np.empty(N_evnts)
    i = 0
    for e in events:
        r_hat, k_hat, n_hat = compute_helicity_basis_old(e.t,e.tbar)
        cos_theta_t_n[i] = compute_cos_theta_axis_old(e.t,e.tbar,e,n_hat)
        cos_theta_tbar_n[i] = compute_cos_theta_axis_old(e.tbar,e.t,e,n_hat)
        #cos_theta_ttbar.append(compute_cos_theta_ttbar(t,tbar,e))
        #mtt[i] = (t.fourMomentumRoot_PxPyPzE() + tbar.fourMomentumRoot_PxPyPzE()).M()
        if i%1000 == 0:
            print(i)
        i+=1
    print(compute_Cij(cos_theta_t_n,cos_theta_tbar_n))
    
    # Root histogram 
    ctn_ctn_arr = cos_theta_t_n*cos_theta_tbar_n
    c1 = ROOT.TCanvas()
    weights = np.ones(ctn_ctn_arr.size)
    h1 = ROOT.TH1D("h1", "cnn", 80, -1, 1)
    h1.FillN(ctn_ctn_arr.size,ctn_ctn_arr,weights)
    h1.Scale(sigma/h1.Integral())
    h1.Draw()
    c1.Print("cnn.pdf")

def run_analysis_split(f1: str, f2: str):
    events1 = get_lhe_events_gzip(f1)
    events2 = get_lhe_events_gzip(f2)
    meta_data1 = get_lhe_meta_data(f1)
    meta_data2 = get_lhe_meta_data(f2)
    sigma = meta_data1['sigma_pb'] + meta_data2['sigma_pb']
    N_events = meta_data1['numEvents'] + meta_data2['numEvents']
    cos_theta_t_n = np.zeros(N_events)
    cos_theta_tbar_n = np.zeros(N_events)
    #cos_theta_ttbar = []
    #mtt = np.empty(N_evnts)
    i = 0
    for e in itertools.chain(events1,events2):
        _, r_hat, k_hat, n_hat = compute_helicity_basis_old(e.t,e.tbar)
        cos_theta_t_n[i] = compute_cos_theta_axis_old(e.t,e.tbar,e,n_hat)
        cos_theta_tbar_n[i] = compute_cos_theta_axis_old(e.tbar,e.t,e,n_hat)
        #cos_theta_ttbar.append(compute_cos_theta_ttbar(t,tbar,e))
        #mtt[i] = (t.fourMomentumRoot_PxPyPzE() + tbar.fourMomentumRoot_PxPyPzE()).M()
        if i%1000 == 0:
            print(i,end=' ')
        i+=1
    print(compute_Cij(cos_theta_t_n,cos_theta_tbar_n))
    
    # Root histogram 
    ctn_ctn_arr = cos_theta_t_n*cos_theta_tbar_n
    c1 = ROOT.TCanvas()
    weights = np.ones(ctn_ctn_arr.size)
    h1 = ROOT.TH1D("h1", "cnn", 80, -1, 1)
    h1.FillN(ctn_ctn_arr.size,ctn_ctn_arr,weights)
    h1.Scale(sigma/h1.Integral())
    h1.Draw()
    c1.Print("cnn_split.pdf")

def main(f_name: str):
    events = get_lhe_events_gzip(f_name)
    meta_data = get_lhe_meta_data(f_name)
    sigma = meta_data['sigma_pb']
    #N_events = meta_data['numEvents']
    N_events = 10
    cos_theta_t_n = np.zeros(N_events)
    cos_theta_tbar_n = np.zeros(N_events)
    for i,e in enumerate(itertools.islice(events,0,N_events)):
        _, r_hat, k_hat, n_hat = compute_helicity_basis_old(e.t,e.tbar)
        #r_hat, k_hat, n_hat = get_top_frame(e.t.fourMomentumRoot_TLorentz(),e.t.fourMomentumRoot_TLorentz()+e.tbar.fourMomentumRoot_TLorentz())
        cos_theta_t_n[i] = compute_cos_theta_axis_old(e.t,e.tbar,e,n_hat)
        cos_theta_tbar_n[i] = compute_cos_theta_axis_old(e.tbar,e.t,e,n_hat)

        qs = [e.particles[i] for i in e.w.children]
        if e.is_t_leptonic:
            cos_theta_leptonic = get_theta_lep(n_hat,e.t.fourMomentumRoot_TLorentz(),e.t.fourMomentumRoot_TLorentz()+e.tbar.fourMomentumRoot_TLorentz(),e.l.fourMomentumRoot_TLorentz())
            thetaW = CalculatePoptThetaW(e.w.fourMomentumRoot_TLorentz(),e.tbar.fourMomentumRoot_TLorentz(),e.t.fourMomentumRoot_TLorentz()+e.tbar.fourMomentumRoot_TLorentz(),qs[0].fourMomentumRoot_TLorentz(),qs[1].fourMomentumRoot_TLorentz())
            cos_theta_hadronic = get_theta_jet_opt(n_hat,e.tbar.fourMomentumRoot_TLorentz(),e.t.fourMomentumRoot_TLorentz()+e.tbar.fourMomentumRoot_TLorentz(),thetaW,qs[0].fourMomentumRoot_TLorentz(),qs[1].fourMomentumRoot_TLorentz())
            print(cos_theta_tbar_n[i]-cos_theta_hadronic)
        else:
            cos_theta_leptonic = get_theta_lep(n_hat,e.tbar.fourMomentumRoot_TLorentz(),e.t.fourMomentumRoot_TLorentz()+e.tbar.fourMomentumRoot_TLorentz(),e.l.fourMomentumRoot_TLorentz())
            thetaW = CalculatePoptThetaW(e.w.fourMomentumRoot_TLorentz(),e.t.fourMomentumRoot_TLorentz(),e.t.fourMomentumRoot_TLorentz()+e.tbar.fourMomentumRoot_TLorentz(),qs[0].fourMomentumRoot_TLorentz(),qs[1].fourMomentumRoot_TLorentz())
            cos_theta_hadronic = get_theta_jet_opt(n_hat,e.t.fourMomentumRoot_TLorentz(),e.t.fourMomentumRoot_TLorentz()+e.tbar.fourMomentumRoot_TLorentz(),thetaW,qs[0].fourMomentumRoot_TLorentz(),qs[1].fourMomentumRoot_TLorentz())
            print(cos_theta_t_n[i]-cos_theta_hadronic)

def differential_cross_section_func(cos_theta_ij, Cij):
    kappa_A = 1.0
    kappa_B = .64
    return -.5*(1-kappa_A*kappa_B*Cij[0]*cos_theta_ij[0])*ROOT.TMath.Log(abs(cos_theta_ij[0]))

if __name__ == "__main__":
    #f_name = 'ttbar-parton-madspin-positive.lhe.gz'
    #f_name2 = 'ttbar-parton-madspin-negative.lhe.gz'
    #run_analysis_split(f_name,f_name2)
    #f_name = 'ttbar-parton-madspin-low.lhe.gz'

    #df_ = uproot.concatenate(Path("/mnt/e/Root-Data/inclusive.root"), library="pd")
    f_name = '/mnt/e/Root-Data/events.inclusive.parton.root'
    with uproot.open(f_name) as f:
        t = f['LHEF']
    e_branch = t['Event']
    p_branch = t['Particle']
    print(e_branch.keys())
    print(p_branch.keys())
    weights = e_branch["Event.Weight"].array(library='np')
    # outer batch loop over vectorized functions
    idx = 0
    
    cnn = np.empty(1980000)
    cnr = np.empty(1980000)
    cnk = np.empty(1980000)

    crn = np.empty(1980000)
    crr = np.empty(1980000)
    crk = np.empty(1980000)

    ckn = np.empty(1980000)
    ckr = np.empty(1980000)
    ckk = np.empty(1980000)

 
    cnn_threshold = []
    cnr_threshold = []
    cnk_threshold = []

    crn_threshold = []
    crr_threshold = []
    crk_threshold = []

    ckn_threshold = []
    ckr_threshold = []
    ckk_threshold = []

    weights_threshold = []


    cnn_resolved = []
    cnr_resolved = []
    cnk_resolved = []

    crn_resolved = []
    crr_resolved = []
    crk_resolved = []

    ckn_resolved = []
    ckr_resolved = []
    ckk_resolved = []


    batch_size = 60000
    # entry_start=1980000/2-batch_size/2,entry_stop=1980000/2+batch_size/2,step_size=batch_size
    for ps in p_branch.iterate(p_branch.keys(),step_size=batch_size):
        # define filters, get particles, build momentum vectors
        t_filter = ps['Particle.PID']==6
        tbar_filter = ps['Particle.PID']==-6
        l_filter = (np.abs(ps['Particle.PID']) > 10) & (np.abs(ps['Particle.PID']) < 17) & (ps['Particle.PID'] % 2 == 1)
        
        ts = ps[t_filter][:,0]
        tbars = ps[tbar_filter][:,0]
        ls = ps[l_filter][:,0]
        
        t_leptonic_filter = (np.sign(ps['Particle.PID']) != np.sign(ls['Particle.PID'])) & (np.abs(ps['Particle.PID'])==6)
        t_hadronic_filter = (np.sign(ps['Particle.PID']) == np.sign(ls['Particle.PID'])) & (np.abs(ps['Particle.PID'])==6)
        w_hadronic_filter = (np.sign(ps['Particle.PID']) == np.sign(ls['Particle.PID'])) & (np.abs(ps['Particle.PID'])==24)
        qd_filter = (np.sign(ps['Particle.PID'][ps['Particle.Mother1']]) == np.sign(ls['Particle.PID'])) & (np.abs(ps['Particle.PID'][ps['Particle.Mother1']])==24) & (ps['Particle.PID'] % 2 == 1)
        qu_filter = (np.sign(ps['Particle.PID'][ps['Particle.Mother1']]) == np.sign(ls['Particle.PID'])) & (np.abs(ps['Particle.PID'][ps['Particle.Mother1']])==24) & (ps['Particle.PID'] % 2 != 1)

        ts_lep = ps[t_leptonic_filter][:,0]
        ts_had = ps[t_hadronic_filter][:,0]
        ws_had = ps[w_hadronic_filter][:,0]
        qds = ps[qd_filter][:,0]
        qus = ps[qu_filter][:,0]

        t_4momentum = vector.array({'px':ts['Particle.Px'],'py':ts['Particle.Py'],'pz':ts['Particle.Pz'],'E':ts['Particle.E']})
        tbar_4momentum = vector.array({'px':tbars['Particle.Px'],'py':tbars['Particle.Py'],'pz':tbars['Particle.Pz'],'E':tbars['Particle.E']})
        l_4momentum = vector.array({'px':ls['Particle.Px'],'py':ls['Particle.Py'],'pz':ls['Particle.Pz'],'E':ls['Particle.E']})
        t_lep_4momentum = vector.array({'px':ts_lep['Particle.Px'],'py':ts_lep['Particle.Py'],'pz':ts_lep['Particle.Pz'],'E':ts_lep['Particle.E']})
        t_had_4momentum = vector.array({'px':ts_had['Particle.Px'],'py':ts_had['Particle.Py'],'pz':ts_had['Particle.Pz'],'E':ts_had['Particle.E']})
        w_had_4momentum = vector.array({'px':ws_had['Particle.Px'],'py':ws_had['Particle.Py'],'pz':ws_had['Particle.Pz'],'E':ws_had['Particle.E']})
        qd_4momentum = vector.array({'px':qds['Particle.Px'],'py':qds['Particle.Py'],'pz':qds['Particle.Pz'],'E':qds['Particle.E']})
        qu_4momentum = vector.array({'px':qus['Particle.Px'],'py':qus['Particle.Py'],'pz':qus['Particle.Pz'],'E':qus['Particle.E']})
        
        # copmute helicity basis and omegas
        abs_cos_theta, r_hat, k_hat, n_hat = compute_helicity_basis(t_4momentum,tbar_4momentum)
        omega_leptonic = compute_Omega_leptonic(t_lep_4momentum,t_had_4momentum,l_4momentum)
        cos_theta_w, omega_hadronic = compute_Omega_hadronic(t_lep_4momentum,t_had_4momentum,w_had_4momentum,qd_4momentum,qu_4momentum)

        # separate leptonic/hadronic omegas into t/tbar
        event_filter_t_leptonic = ls['Particle.PID'] < 0
        omega_t = vector.VectorNumpy3D(np.where(event_filter_t_leptonic, omega_leptonic, omega_hadronic))
        omega_tbar = vector.VectorNumpy3D(np.where(~event_filter_t_leptonic, omega_leptonic, omega_hadronic))

        # dot with unit vectors
        cos_t_n = omega_t.dot(n_hat)
        cos_t_r = omega_t.dot(r_hat)
        cos_t_k = omega_t.dot(k_hat)

        cos_tbar_n = omega_tbar.dot(n_hat)
        cos_tbar_r = omega_tbar.dot(r_hat)
        cos_tbar_k = omega_tbar.dot(k_hat)

        cnn_tmp = cos_t_n*cos_tbar_n
        cnr_tmp = cos_t_n*cos_tbar_r
        cnk_tmp = cos_t_n*cos_tbar_k

        crn_tmp = cos_t_r*cos_tbar_n
        crr_tmp = cos_t_r*cos_tbar_r
        crk_tmp = cos_t_r*cos_tbar_k

        ckn_tmp = cos_t_k*cos_tbar_n
        ckr_tmp = cos_t_k*cos_tbar_r
        ckk_tmp = cos_t_k*cos_tbar_k


        # generate matrix of cosines
        cnn[batch_size*idx:(idx+1)*batch_size] = cnn_tmp
        cnr[batch_size*idx:(idx+1)*batch_size] = cnr_tmp
        cnk[batch_size*idx:(idx+1)*batch_size] = cnk_tmp

        crn[batch_size*idx:(idx+1)*batch_size] = crn_tmp
        crk[batch_size*idx:(idx+1)*batch_size] = crr_tmp
        crn[batch_size*idx:(idx+1)*batch_size] = crk_tmp

        ckn[batch_size*idx:(idx+1)*batch_size] = ckn_tmp
        ckr[batch_size*idx:(idx+1)*batch_size] = ckr_tmp
        ckk[batch_size*idx:(idx+1)*batch_size] = ckk_tmp

        # apply threshold selection
        ttbar_4momentum = t_4momentum + tbar_4momentum
        mtt = ttbar_4momentum.M
        abs_beta_tt = np.abs(ttbar_4momentum.beta)
        threshold_filter = ((mtt <= 400) | ((mtt <= 500) & (mtt > 400) & (abs_cos_theta >= np.cos(3*np.pi/20))) | ((mtt <= 600) & (mtt > 500) & (abs_cos_theta >= np.cos(np.pi/20)))) & (abs_beta_tt <= .9)

        cnn_threshold.append(cnn_tmp[threshold_filter])
        cnr_threshold.append(cnr_tmp[threshold_filter])
        cnk_threshold.append(cnk_tmp[threshold_filter])

        crn_threshold.append(crn_tmp[threshold_filter])
        crr_threshold.append(crr_tmp[threshold_filter])
        crk_threshold.append(crk_tmp[threshold_filter])

        ckn_threshold.append(ckn_tmp[threshold_filter])
        ckr_threshold.append(ckr_tmp[threshold_filter])
        ckk_threshold.append(ckk_tmp[threshold_filter])

        weights_threshold.append(weights[batch_size*idx:(idx+1)*batch_size][threshold_filter])

        
        # apply resolved preselction
        #leps_and_jets_filter = (ps['Particle.Status'] == 1) & ( (np.abs(ps['Particle.PID']) < 10) | ((np.abs(ps['Particle.PID']) > 10) & (np.abs(ps['Particle.PID']) < 17) & (ps['Particle.PID'] % 2 == 1)) )
        #neutrino_filter = (np.abs(ps['Particle.PID']) > 10) & (np.abs(ps['Particle.PID']) < 17) & (ps['Particle.PID'] % 2 == 0)

        #leps_and_jets = ps[leps_and_jets_filter]
        #nuts = ps[neutrino_filter][:,0]

        #leps_and_jets_4momentum = vector.array({'px':ak.flatten(leps_and_jets['Particle.Px']),
        #                                        'py':ak.flatten(leps_and_jets['Particle.Py']),
        #                                        'pz':ak.flatten(leps_and_jets['Particle.Pz']),
        #                                        'E':ak.flatten(leps_and_jets['Particle.E'])})
        #nuts_4momentum = vector.array({'px':nuts['Particle.Px'],'py':nuts['Particle.Py'],'pz':nuts['Particle.Pz'],'E':nuts['Particle.E']})

        #preselection_filter = np.all(leps_and_jets_4momentum.pt.reshape((batch_size,5)) > 25,axis=1) & (nuts_4momentum.Et > 30) & (np.all(np.abs(leps_and_jets_4momentum.eta.reshape((batch_size,5))) < 2.5,axis=1))

        #cnn_resolved.append(cnn_tmp[preselection_filter])
        #cnr_resolved.append(cnr_tmp[preselection_filter])
        #cnk_resolved.append(cnk_tmp[preselection_filter])

        #crn_resolved.append(crn_tmp[preselection_filter])
        #crr_resolved.append(crr_tmp[preselection_filter])
        #crk_resolved.append(crk_tmp[preselection_filter])

        #ckn_resolved.append(ckn_tmp[preselection_filter])
        #ckr_resolved.append(ckr_tmp[preselection_filter])
        #ckk_resolved.append(ckk_tmp[preselection_filter])

        print(idx)
        idx+=1

    # compute matrix elements
    Np = weights[cnn >= 0].sum()
    Nm = weights[cnn < 0].sum()
    print(Np,Nm)
    asym = (Np-Nm)/(Np+Nm)
    print(asym)
    Cnn = -4*asym/.64
    print(Cnn)

    print(compute_Cij(cnn,'cnn'))#, compute_Cij(cnr,'cnr'), compute_Cij(cnk,'cnk'))

    #print(compute_Cij(crn,'crn'), compute_Cij(crr,'crr'), compute_Cij(crk,'crk'))

    #print(compute_Cij(ckn,'ckn'), compute_Cij(ckr,'ckr'), compute_Cij(ckk,'ckk'))

    # threshold analysis
    cnn_threshold = np.concatenate(cnn_threshold)
    cnr_threshold = np.concatenate(cnr_threshold)
    cnk_threshold = np.concatenate(cnk_threshold)

    crn_threshold = np.concatenate(crn_threshold)
    crr_threshold = np.concatenate(crr_threshold)
    crk_threshold = np.concatenate(crk_threshold)

    ckn_threshold = np.concatenate(ckn_threshold)
    ckr_threshold = np.concatenate(ckr_threshold)
    ckk_threshold = np.concatenate(ckk_threshold)

    #print(compute_Cij(cnn_threshold,'cnn-threshold'), compute_Cij(cnr_threshold,'cnr-threshold'), compute_Cij(cnk_threshold,'cnk-threshold'))

    #print(compute_Cij(crn_threshold,'crn-threshold'), compute_Cij(crr_threshold,'crr-threshold'), compute_Cij(crk_threshold,'crk-threshold'))

    #print(compute_Cij(ckn_threshold,'ckn-threshold'), compute_Cij(ckr_threshold,'ckr-threshold'), compute_Cij(ckk_threshold,'ckk-threshold'))

    # cnn fitting
    #cross = ROOT.TF1("cross", differential_cross_section_func, -1,1,1)
    #cross.SetParameters(0)

    #c1 = ROOT.TCanvas()
    #weights = np.ones(cnn_threshold.size)
    #h1 = ROOT.TH1D("h1", "ctnn", 80, -1, 1)
    #h1.FillN(cnn_threshold.size,cnn_threshold,weights)
    #h1.Scale(74.88/h1.Integral())
    #h1.SetFillColor(ROOT.kRed)
    #h1.SetMarkerStyle(21)
    #h1.SetMarkerColor(ROOT.kRed)
    #h1.Draw()
    #c1.Print("ctnn_plot.pdf")
    #h1.Fit("cross","S")
    #c1.Print("ctnn_plot_fit.pdf")


    # resolved ananlysis
    #cnn_resolved = np.concatenate(cnn_resolved)
    #cnr_resolved = np.concatenate(cnr_resolved)
    #cnk_resolved = np.concatenate(cnk_resolved)

    #crn_resolved = np.concatenate(crn_resolved)
    #crr_resolved = np.concatenate(crr_resolved)
    #crk_resolved = np.concatenate(crk_resolved)

    #ckn_resolved = np.concatenate(ckn_resolved)
    #ckr_resolved = np.concatenate(ckr_resolved)
    #ckk_resolved = np.concatenate(ckk_resolved)

    #print(compute_Cij(cnn_resolved,'cnn-resolved'))
    #print(compute_Cij(cnr_resolved,'cnr-resolved'))
    #print(compute_Cij(cnk_resolved,'cnk-resolved'))

    #print(compute_Cij(crn_resolved,'crn-resolved'))
    #print(compute_Cij(crr_resolved,'crr-resolved'))
    #print(compute_Cij(crk_resolved,'crk-resolved'))

    #print(compute_Cij(ckn_resolved,'ckn-resolved'))
    #print(compute_Cij(ckr_resolved,'ckr-resolved'))
    #print(compute_Cij(ckk_resolved,'ckk-resolved'))


    #events = [event_tree[k].array() for k in event_tree.keys()]
    #particles = [particle_tree[k].array() for k in particle_tree.keys()]
    #print(events)
    #print(particles)