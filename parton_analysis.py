# python headers
import xml.etree.ElementTree as ET
#from math import isclose
#import itertools
import gzip

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

# dictionary of PDG codes for fundamental particles 2020 rev:
# https://pdg.lbl.gov/2020/reviews/rpp2020-rev-monte-carlo-numbering.pdf
_PDGcodes = {1:'d', 2:'u', 3:'s', 4:'c', 5:'b', 6:'t', 9:'g (glueball)', 
            11:'e-', 12:'ve', 13:'mu-', 14:'vm', 15:'tau-', 16:'vt', 
            21:'g', 22:'a (gamma)', 23:'z', 24:'w+', 25:'h',
            -1:'d~', -2:'u~', -3:'s~', -4:'c~', -5:'b~', -6:'t~', 
            -11:'e+', -12:'ve~', -13:'mu+', -14:'vm~', -15:'tau+', -16:'vt~', 
            -24:'w-'}

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

def compute_asymmetry(x,weights):
    # asymmetry of a distribution (han p.10)
    Np = weights[x >= 0].sum()
    Nm = weights[x < 0].sum()
    return (Np-Nm)/(Np+Nm)

def compute_Cij(cos_i_cos_j,weights):
    # compute spin corelation matrix element with respect to ij axes
    # method 2 of han p.10
    kappa_A = 1
    kappa_B = .64
    asymmetry = compute_asymmetry(cos_i_cos_j,weights)
    return -4*asymmetry/(kappa_A*kappa_B)

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
                yield evnt_data, particle_data

###########
# drivers #
###########
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
    print(compute_Cij(cnn,weights), compute_Cij(crn,weights), compute_Cij(cnk,weights))

    print(compute_Cij(cnr,weights), compute_Cij(crr,weights), compute_Cij(crk,weights))

    print(compute_Cij(cnk,weights), compute_Cij(crk,weights), compute_Cij(ckk,weights))

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

    weights_threshold = np.concatenate(weights_threshold)


    print(compute_Cij(cnn_threshold,weights_threshold), compute_Cij(crn_threshold,weights_threshold), compute_Cij(ckn_threshold,weights_threshold))

    print(compute_Cij(cnr_threshold,weights_threshold), compute_Cij(crr_threshold,weights_threshold), compute_Cij(ckr_threshold,weights_threshold))

    print(compute_Cij(cnk_threshold,weights_threshold), compute_Cij(crk_threshold,weights_threshold), compute_Cij(ckk_threshold,weights_threshold))

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