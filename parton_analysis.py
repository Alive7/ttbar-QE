# python headers
import time

# python scientific headers
#import matplotlib.pyplot as plt
import numpy as np

# scikit hep headers
import awkward as ak
import vector
import uproot
import pylhe

# root
#import ROOT

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
    return np.arccos(np.abs(cos_theta)), [n_hat, r_hat, k_hat]

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
    q_soft_four_momentum_top = vector.MomentumNumpy4D(np.where(cos_theta_w >= 0,qu_four_momentum_top,qd_four_momentum_top))
    #assert(np.all(q_soft_four_momentum_top.E <= q_hard_four_momentum_top.E))
    q_soft_dir = q_soft_four_momentum_top.to_Vector3D().unit()
    q_hard_dir = q_hard_four_momentum_top.to_Vector3D().unit()

    # compute probability distributions
    c_theta = np.abs(cos_theta_w)
    fp = prob_dist_w_angle(c_theta)
    fm = prob_dist_w_angle(-c_theta)

    # form optimal hadronic direction
    opt_had_momentum = prob_soft(fp,fm)*q_soft_dir + prob_hard(fp,fm)*q_hard_dir
    return opt_had_momentum.unit()

def compute_asymmetry(x,weights):
    # asymmetry of a distribution (han p.10)
    #Np = np.sum(weights[x >= 0])
    #Nm = np.sum(weights[x < 0])
    Np = np.sum(x >= 0)
    Nm = np.sum(x < 0)
    #print(Np, Nm)
    return (Np-Nm)/(Np+Nm)

def compute_Cij(cos_i_cos_j,weights):
    # compute spin corelation matrix element with respect to ij axes
    # method 2 of han p.10
    kappa_A = 1
    kappa_B = .64#1
    asymmetry = compute_asymmetry(cos_i_cos_j,weights)
    return -4*asymmetry/(kappa_A*kappa_B)

####################
# helper functions #
####################

#####################
# parsing functions #
#####################
def process_lhe(f_name: str):
    init = pylhe.read_lhe_init(f_name)
    print(init.keys())
    print(init['initInfo'])
    print(init['procInfo'])
    events = pylhe.to_awkward(pylhe.read_lhe_with_attributes(f_name))
    #print(events.fields)
    return events.eventinfo, events.particles

def process_root(f_name: str):
    with uproot.open(f_name_root) as f:
        t = f['LHEF']
    e_branch = t['Event']
    p_branch = t['Particle']
    print(e_branch.keys())
    print(p_branch.keys())
    return e_branch, p_branch

###########
# drivers #
###########
if __name__ == "__main__":
    #t_total = time.time()
    #f_name_lhe = 'ttbar-parton-madspin-low.lhe.gz'
    f_name_root = '/mnt/e/Root-Data/events.inclusive.parton.root'
    #event_info, particles = process_lhe(f_name_lhe)
    #print(particles.fields)
    #print(event_info.fields)
    #print(event_info.nparticles, event_info.weight)
    e_branch, p_branch = process_root(f_name_root)
    N_events = e_branch.num_entries
    #print(type(particles), type(p_branch))
    weights = e_branch["Event.Weight"].array()[:,0]
    # outer batch loop over vectorized functions
    idx = 0
    
    cos_ij = np.empty((3,3,N_events))
    cos_ij_threshold = []
    weights_threshold = []

    # define blocks
    mtt_start = 300
    mtt_stop = 2000
    mtt_step = 100
    mtt_low = np.arange(mtt_start,mtt_stop,mtt_step)
    mtt_high = mtt_low + mtt_step
    theta_start = 0
    theta_stop = np.pi/2
    theta_step = np.pi/20
    theta_low = np.arange(theta_start,theta_stop,theta_step)
    theta_high = theta_low + theta_step
    reco_counts = np.zeros((mtt_low.shape[0],theta_low.shape[0]),dtype=int)

    batch_size = 60000
    # entry_start=1980000/2-batch_size/2,entry_stop=1980000/2+batch_size/2,step_size=batch_size
    keys = ['Particle.PID','Particle.Status','Particle.Mother1','Particle.Px', 'Particle.Py', 'Particle.Pz', 'Particle.E']
    for ps in p_branch.iterate(keys,step_size=batch_size):
        # define filters, get particles, build momentum vectors
        t_filter = ps['Particle.PID']==6
        tbar_filter = ps['Particle.PID']==-6
        l_filter = (np.abs(ps['Particle.PID']) > 10) & (np.abs(ps['Particle.PID']) < 17) & (ps['Particle.PID'] % 2 == 1)
        
        #t = time.time_ns()
        ts = ps[t_filter][:,0]
        #print(time.time_ns() - t)
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
        #t = time.time_ns()
        qds = ps[qd_filter][:,0]
        #print(time.time_ns() - t)
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
        #t = time.time_ns()
        theta, nrk_bais = compute_helicity_basis(t_4momentum,tbar_4momentum)
        
        #print(time.time_ns() - t)
        #t = time.time_ns()
        omega_leptonic = compute_Omega_leptonic(t_lep_4momentum,t_had_4momentum,l_4momentum)
        #print(time.time_ns() - t)
        #omega_hadronic_ideal = compute_Omega_leptonic(t_had_4momentum, t_lep_4momentum, qd_4momentum)
        #t = time.time_ns()
        omega_hadronic = compute_Omega_hadronic(t_lep_4momentum,t_had_4momentum,w_had_4momentum,qd_4momentum,qu_4momentum)
        #print(time.time_ns() - t)

        # separate leptonic/hadronic omegas into t/tbar
        event_filter_t_leptonic = ls['Particle.PID'] < 0
        omega_t = vector.VectorNumpy3D(np.where(event_filter_t_leptonic, omega_leptonic, omega_hadronic))
        omega_tbar = vector.VectorNumpy3D(np.where(event_filter_t_leptonic, omega_hadronic, omega_leptonic))

        # dot with unit vectors
        cos_t_nrk = np.array([omega_t.dot(vec) for vec in nrk_bais])
        cos_tbar_nrk = np.array([omega_tbar.dot(vec) for vec in nrk_bais])
        
        # generate matrix of cosines
        cos_ij_tmp = np.einsum('ik,jk->ijk',cos_t_nrk,cos_tbar_nrk)
        cos_ij[:,:,batch_size*idx:(idx+1)*batch_size] = cos_ij_tmp

        # apply threshold selection
        ttbar_4momentum = t_4momentum + tbar_4momentum
        mtt = ttbar_4momentum.M
        abs_beta_tt = np.abs(ttbar_4momentum.beta)
        threshold_filter = ((mtt <= 400) | ((mtt <= 500) & (mtt > 400) & (theta <= 3*np.pi/20)) | ((mtt <= 600) & (mtt > 500) & (theta <= np.pi/20))) & (abs_beta_tt <= .9)
        cos_ij_threshold.append(cos_ij_tmp[:,:,threshold_filter])
        weights_threshold.append(weights[batch_size*idx:(idx+1)*batch_size][threshold_filter])
        
        # accumulate reco counts
        filter_mtt_min = mtt > mtt_low.reshape((-1,1))
        filter_mtt_max = mtt < mtt_high.reshape((-1,1))
        filter_theta_min = theta > theta_low.reshape((-1,1))
        filter_theta_max = theta < theta_high.reshape((-1,1))

        filter_mtt = filter_mtt_min & filter_mtt_max
        filter_theta = filter_theta_min & filter_theta_max
        filter_block = np.einsum('ik,jk->ijk',filter_mtt,filter_theta)
        reco_counts += filter_block.sum(axis = 2)
        
        # apply resolved preselction
        leps_and_jets_filter = (ps['Particle.Status'] == 1) & ( (np.abs(ps['Particle.PID']) < 10) | ((np.abs(ps['Particle.PID']) > 10) & (ps['Particle.PID'] % 2 == 1)) )
        neutrino_filter = (np.abs(ps['Particle.PID']) > 10) & (np.abs(ps['Particle.PID']) < 17) & (ps['Particle.PID'] % 2 == 0)

        #leps_and_jets = ps[leps_and_jets_filter]
        #nuts = ps[neutrino_filter][:,0]

        #leps_and_jets_4momentum = vector.array({'px':ak.flatten(leps_and_jets['Particle.Px']),
        #                                        'py':ak.flatten(leps_and_jets['Particle.Py']),
        #                                        'pz':ak.flatten(leps_and_jets['Particle.Pz']),
        #                                        'E':ak.flatten(leps_and_jets['Particle.E'])})
        #nuts_4momentum = vector.array({'px':nuts['Particle.Px'],'py':nuts['Particle.Py'],'pz':nuts['Particle.Pz'],'E':nuts['Particle.E']})

        #preselection_filter = np.all(leps_and_jets_4momentum.pt.reshape((batch_size,5)) > 25,axis=1) & (nuts_4momentum.Et > 30) & (np.all(np.abs(leps_and_jets_4momentum.eta.reshape((batch_size,5))) < 2.5,axis=1))

        print(idx)
        idx+=1

    # threshold analysis
    print(100*reco_counts/N_events)
    cos_ij_threshold = np.concatenate(cos_ij_threshold,axis=2)
    weights_threshold = ak.concatenate(weights_threshold).to_numpy()
    #t = time.time_ns()
    Cij = np.apply_along_axis(compute_Cij,-1,cos_ij_threshold,weights_threshold)
    #print(time.time_ns() - t)
    #print(time.time()-t_total)
    print(Cij)
    
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