#!/usr/bin/env python
# coding: utf-8

# In[1]:


#selecionar prótons
#selecionar eventons com 2 múons
# '' com 2 elétrons
# evento misto: 1 eletron e 1 múon com maior PT <- 

#import sys,os
#os.system("cat /eos/cms/store/group/phys_pps/Phase2/LHE/EFT/AAWW/FPMC_WW_14TeV_a0w_0E0_aCw_0E0_noHADR_pt0/split/FPMC_WW_14TeV_a0w_0E0_aCw_0E0_noHADR_pt0_0.lhe.xz")


# In[2]:


ls /eos/cms/store/group/phys_pps/Phase2/Delphes/PU200/FPMC_WW_14TeV_a0w_0E0_aCw_0E0_noHADR_pt0_ZLepDecays_Delphes_PU200


# In[3]:


import uproot4
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[4]:


path='/eos/cms/store/group/phys_pps/Phase2/Delphes/PU200/FPMC_WW_14TeV_a0w_0E0_aCw_0E0_noHADR_pt0_ZLepDecays_Delphes_PU200/'
name='FPMC_WW_14TeV_a0w_0E0_aCw_0E0_noHADR_pt0_0_ZLepDecays_Delphes_PU200.root'

file=path+name
root=uproot4.open(file)


# In[5]:


root.keys()


# In[6]:


root.classnames()


# In[7]:


root['Delphes']


# In[8]:


tree = root['Delphes']
tree.keys()


# In[9]:


ElectronTight_Charge=tree['ElectronTight.Charge'].array()
print(ElectronTight_Charge)


# In[10]:


Vertex_Size=tree['Vertex_size'].array()
n_events=len(Vertex_Size)
print(Vertex_Size)
print(n_events)
print(np.sum(Vertex_Size))


# In[11]:


Muon=tree['Muon_size'].array()
MuonLoose=tree['MuonLoose_size'].array()
MuonTight=tree['MuonTight_size'].array()
plt.hist(Muon, bins=10, range = (0,10))
plt.hist(MuonLoose, bins=10, range = (0,10))
plt.hist(MuonTight, bins=10, range = (0,10))


# In[12]:


Muon_PT=tree['Muon.PT'].array() # ver o "só múon"
plt.hist(ak.flatten(Muon_PT), bins=10, range=(0, 150))


# In[13]:


MuonLoose_PT=tree['MuonLoose.PT'].array()
plt.hist(ak.flatten(MuonLoose_PT), bins=20, range=(0,150)) #bin=5 
#plt n_Muons
#loop sobre n_Muons e printar o PT


# In[14]:


MuonTight_PT=tree['MuonTight.PT'].array()
plt.hist(ak.flatten(MuonTight_PT), bins=20, range=(0,150))


# In[15]:


n_protons=tree['GenProton_size'].array()
print(n_protons)
plt.hist(n_protons, bins=20, range = (400,800))


# In[16]:


#n_protons=tree['GenProton']
#plt.hist(ak.flatten(n_protons))
#print('O número total de prótons antes da seleção é',np.sum(n_protons))


# In[17]:


#Utilidades...
#protons_Pz = tree['GenProton.Pz'].array() #nota: array 'sem S' coloca um vetor
#branches = tree['GenProton.Pz'].arrays() #nota: arrays 'com S' coloca um nome na frente
#n_protons=tree['GenProton'].array()

#n_total=0;

#for i in range (len(protons_Pz)):
 #   n_total=n_total+len(protons_Pz[i])

#print(n_total)
#print(len(ak.flatten(branches['GenProton.Pz'])))
#print(protons_Pz)
#print(branches)
###


# In[18]:


# single beam energy:
ebeam=7000.
# collision energy:
ecms=2*ebeam
# xi acceptance from Table 4 on page 40 of https://cds.cern.ch/record/2750358/files/NOTE2020_008.pdf:
_mode='vertical'
xi_min = 0.0147 # using the 234m station
xi_max = 0.196 # using the 196m station
pz_min = (1-xi_max)*ebeam
pz_max = (1-xi_min)*ebeam
# print limiting xi and pz:
pz_min1 =(1-1.08*xi_max)*ebeam
pz_max1 =(1-0.92*xi_min)*ebeam
print('For xi range of (%2.3f,%2.3f) proton pz is in range from %2.2f to %2.2f GeV'%(xi_min,xi_max,pz_min,pz_max))
print(pz_min1)
print(pz_max1)


# In[19]:


Protons_Pz = tree['GenProton.Pz'].array()
Pz_mask = (abs(Protons_Pz) > (pz_min)) & (abs(Protons_Pz) < (pz_max))

print(Protons_Pz)
print(Pz_mask)
np.sum(Pz_mask)


# In[20]:


protons = tree.arrays(['GenProton.Pz','GenProton.Z','GenProton.IsPU','GenProton.T'],  cut="(abs(GenProton.Pz)>%g) & (abs(GenProton.Pz)<%g)"%(pz_min,pz_max))
print(len(protons))


# In[21]:


zpos=23400

# time resolution for main vertex:
tvertex=40e-12
# time resolution for proton in PPS:
tpps=30e-12

convert_nanosec=1e9
trP=(convert_nanosec)*tpps
trM=(convert_nanosec)*tvertex


# setting initial vars:
N_Protons=np.zeros(len(protons)).astype(int) # number of protons in each event that pass the PPS criteria
GenProton_pz=protons['GenProton.Pz']
GenProton_vz=protons['GenProton.Z']
GenProton_ispu=protons['GenProton.IsPU']
GenProton_t=(convert_nanosec)*protons['GenProton.T'] # convert timing to nanosec

# setting vars for elements passing criteria of pz_min and pz_max of protons:
PassPz_Proton_pz=[]
PassPz_Proton_vz=[]
PassPz_Proton_ispu=[]
PassPz_Proton_t=[]
PassPz_Proton_tsmeared=[]
PassPz_Proton_xi=[]
PassPz_Proton_PU=[]

# store sign of proton:
sig=[]


# In[22]:


# loop over events in GenProton_pz branch:
for i in tqdm(range(len(GenProton_pz))):
    _pz=GenProton_pz[i]
    _xi=1-np.abs(_pz)/ebeam
    # smear with 2% uncertainty:
    _uncert=0.02
    _xi_smear=_xi*(1+np.random.normal(0,_uncert,len(_xi)))
    _pz_smear=ebeam*(1-_xi_smear)
    _t=np.zeros(len(_pz))
    
    # converting light speed from m/s to cm/ns:
    convert_m_to_cm=1e2
    c=3e8 # why not 299 792 458 m/s ?
    lightspeed=c*convert_m_to_cm/convert_nanosec
    
    # loop over proton pz and store timing:
    for k in range(len(_pz)):
        if _pz[k]>0:
            _t[k]=(GenProton_t[i][k]+(zpos-GenProton_vz[i][k])/lightspeed)
        else: 
            _t[k]=(GenProton_t[i][k]+(zpos+GenProton_vz[i][k])/lightspeed)

    # smear timing around PPS resolution:
    _tsmear=_t+np.random.normal(0,trP,len(_t))
    
    # a counter for protons:
    _Npr=0

    # PassPz :: loop over proton pz and store passing events within smearing:
    # [i] == event
    # [k] == proton in event
    for k in range(len(_pz)):
        if (abs(_pz_smear[k])>pz_min) and (abs(_pz_smear[k])<pz_max):
            sig.append(np.sign(_pz[k])) #Storing the Pz sign of each proton
            PassPz_Proton_pz.append(_pz_smear[k])
            PassPz_Proton_xi.append(_xi_smear[k])
            PassPz_Proton_vz.append(GenProton_vz[i][k])
            PassPz_Proton_PU.append(GenProton_ispu[i][k])
            PassPz_Proton_t.append(_t[k])
            PassPz_Proton_tsmeared.append(_tsmear[k])
            _Npr=_Npr+1
    
    # count number of protons passing PPS criteria after the smearing
    N_Protons[i]=_Npr


# In[23]:


print('O número de eventos é:', len(N_Protons))
print('The total number of protons that passed the criteria is', np.sum(N_Protons))
plt.hist(N_Protons, bins=10, range=(1,18))
#for i in range(len(N_Protons)):
 #   print(N_Protons[i])


# At this point the vector N_Protons has stored the number of protons on each event that passed the PPS criteria after the smearing. This vetor is important because it's order contains how many (and which) protons are present in each event. In the next steps, when analysing the leptons, we must keep track of which event those leptons came from. This is done by keeping the order of each of those 250 events in the vector. 

# In[24]:


print("The total number of protons lost after applying the smearing was",  np.sum(Pz_mask)-np.sum(N_Protons))


# # Sorting Protons by Pz Direction

# In[25]:


print(sig)


# In[26]:


# prep store for proton sign and pairs:
ProtonsNeg = np.zeros(n_events)
ProtonsPos = np.zeros(n_events)
N_Pairs = np.zeros(n_events) # possible combinations for 2 protons on both sectors
###nprot = np.zeros(n_events) # can be removed?

# simple counter:
_count=0

# loop over protons that passed the PassPz criteria:
# ev == event (first index)
# nprotons == secondary index to loop over protons in event (ev)
for ev,_nprotons in tqdm(enumerate(N_Protons)):
    # check sign:
    for i in range(_nprotons):
        if (sig[_count+i]>0):
            ProtonsPos[ev] = ProtonsPos[ev]+1
        else:
            ProtonsNeg[ev] = ProtonsNeg[ev]+1
        N_Pairs = ProtonsPos*ProtonsNeg
    _count = _count + _nprotons


# In[27]:


print(np.sum(ProtonsPos)+np.sum(ProtonsNeg))


# # Analyzing Electrons and Muons separetely 

# Gathering variables for Electrons and Muons and defining usefull vectors.

# In[28]:


#ELECTRON TIGHT

N_Electrons=tree['ElectronTight_size'].array()
Electron_pt=tree['ElectronTight.PT'].array()
Electron_eta=tree['ElectronTight.Eta'].array()
Electron_phi=tree['ElectronTight.Phi'].array()
Electron_T=tree['ElectronTight.T'].array()


VertexT_CMS=(convert_nanosec)*tree['Vertex.T'].array() # converted to nanosec

Electron_VT=VertexT_CMS+np.random.normal(0,trM,len(VertexT_CMS)) 
Electron_Vz=tree['Vertex.Z'].array() #This is the same for all particles


AllElectron_eta=[]
AllElectron_pt=[]
AllElectron_phi=[]
noElectrons=np.zeros((len(N_Electrons)))


# Setting all variables for Electrons Tight:
Electron_yll=[]
Electron_yll=[]
Electron_SumET=[]
Electron_SumPz=[]
Electron_mll=[]
Electron_mll=[]
Electron_Lead_pt=[] # leading Electron
Electron_SubL_pt=[] # subleading Electron
Electron_T=[]
Electron_z=[]

#MUONS

N_Muons=tree['Muon_size'].array()
Muon_pt=tree['Muon.PT'].array()
Muon_eta=tree['Muon.Eta'].array()
Muon_phi=tree['Muon.Phi'].array()
Muon_T=tree['Muon.T'].array()
#Muon_z=tree_['Vertex.Z'].array()

#MUONS TIGHT
N_TMuons=tree['MuonTight_size'].array()
TMuon_pt=tree['MuonTight.PT'].array()
TMuon_eta=tree['MuonTight.Eta'].array()
TMuon_phi=tree['MuonTight.Phi'].array()
VertexT_CMS=(convert_nanosec)*tree['Vertex.T'].array() # converted to nanosec
TMuon_VT=VertexT_CMS+np.random.normal(0,trM,len(VertexT_CMS))
TMuon_Vz=tree['Vertex.Z'].array()

# setting initial vars for muons:
Muon_yll=[]
TMuon_yll=[]
TMuon_SumET=[]
TMuon_SumPz=[]
Muon_mll=[]
TMuon_mll=[]
TMuon_Lead_pt=[] # leading muon
TMuon_SubL_pt=[] # subleading muon
TMuon_T=[]
TMuon_z=[]

# Definindo coisas pro Muon:
Muon_Lead_pt=[]
Muon_Sub_pt=[]

# for muons before selection
AllTMuon_eta=[]
AllTMuon_pt=[]
AllTMuon_phi=[]

# save events without tight muon plus 2 protons:
noMuons=np.zeros((len(N_TMuons)))


# For event we must find the Muons and Electrons with greater 'pt' (leading) and second greater 'pt' (subleading).

# In[29]:


#loop over electrons tight:

for _ev,_nelec in tqdm(enumerate(N_Electrons)):
    # reset muon kinematics:
    _e1_pt=0; _e1_eta=0; _e1_phi=0; _e1_t=0; _e1_z=0
    _e2_pt=0; _e2_eta=0; _e2_phi=0; _e2_t=0; _e2_z=0

    for i in range(_nelec):
        # search for leading electron:
        if Electron_pt[_ev][i]>_e1_pt:
            # store subleading electron:
            _e2_pt = _e1_pt
            _e2_eta = _e1_eta
            _e2_phi = _e1_phi     
            # store leading electron:
            _e1_pt = Electron_pt[_ev][i]
            _e1_eta = Electron_eta[_ev][i]
            _e1_phi = Electron_phi[_ev][i]
        elif Electron_pt[_ev][i]>_e2_pt:
            # if last, store subleading muon:
            _e2_pt = Electron_pt[_ev][i]
            _e2_eta = Electron_eta[_ev][i]
            _e2_phi = Electron_phi[_ev][i]

    # Eta for all Electrons
    AllElectron_eta.append(_e1_eta)
    AllElectron_eta.append(_e2_eta)
    
    #Pt for all Electrons
    AllElectron_pt.append(_e1_pt)
    AllElectron_pt.append(_e2_pt)
    
    #Phi for all Electrons
    AllElectron_phi.append(_e1_phi)
    AllElectron_phi.append(_e2_phi)
    
    # if valid event with 2 ELECTRONS and 2 PROTONS in PPS:
    if (_e1_pt!=0) and (_e2_pt!=0) and (ProtonsPos[_ev]!=0) and (ProtonsNeg[_ev]!=0):
        # save the leading and subleading electrons:
        Electron_Lead_pt.append(_e1_pt)
        Electron_SubL_pt.append(_e2_pt)
        ########### compute invariant mass of lepton pair:
        Electron_SumET.append(_e1_pt*np.cosh(_e1_eta)+_e2_pt*np.cosh(_e2_eta)) # sumET = pt1*cosh(eta1) + pt2*cosh(eta2)
        _sumTPx = _e1_pt*np.cos(_e1_phi)+_e2_pt*np.cos(_e2_phi) # sumTPx = pt1*cos(phi1) + pt2*cos(phi2)
        _sumTPy = _e1_pt*np.sin(_e1_phi)+_e2_pt*np.sin(_e2_phi) # sumTPy = pt1*sin(phi1) + pt2*sin(phi2)
        Electron_SumPz.append(_e1_pt*np.sinh(_e1_eta)+_e2_pt*np.sinh(_e2_eta))  # sumTPz = pt1*sinh(eta1) + pt2*sinh(eta2)
        # building the dilepton:
        #_di_TMuonP = np.sqrt(_sumTPx**2 + _sumTPy**2 + ( _mu1_pt*np.sinh(_mu1_eta)+_mu2_pt*np.sinh(_mu2_eta) )**2)
        _di_ElectronP = np.sqrt( _sumTPx**2 + _sumTPy**2 + Electron_SumPz[-1]**2 )
        #_mllt2 = (_mu1_pt*np.cosh(_mu1_eta)+_mu2_pt*np.cosh(_mu2_eta))**2 - _di_TMuonP**2  # - sumTPx**2 - sumTPy**2 - sumTPz**2
        _mllt2 = Electron_SumET[-1]**2 - _di_ElectronP**2
        # physics check:
        if(_mllt2<0): _mllt2=0
        Electron_mll.append(np.sqrt(_mllt2))
        #TMuon_yll[_ev]=_mu1_eta+_mu2_eta
        Electron_T.append(Electron_VT[_ev])
        Electron_z.append(Electron_Vz[_ev])
        # define rapidity of dilepton:
        #TMuon_yll = (1/2) log (sumET+sumTPz/sumET-sumTPz)
        #TMuon_yll.append(0.5*np.log(((_mu1_pt*np.cosh(_mu1_eta)+_mu2_pt*np.cosh(_mu2_eta))+(_mu1_pt*np.sinh(_mu1_eta)+_mu2_pt*np.sinh(_mu2_eta)))/((_mu1_pt*np.cosh(_mu1_eta)+_mu2_pt*np.cosh(_mu2_eta))-(_mu1_pt*np.sinh(_mu1_eta)+_mu2_pt*np.sinh(_mu2_eta)))))
        # compute rapidity from last object appended ([-1])
        TMuon_yll.append( 0.5*np.log( (TMuon_SumET[-1] + TMuon_SumPz[-1])/(TMuon_SumET[-1] - TMuon_SumPz[-1]) ) )
    else: #if not 2 Electrons plus 2 protons:
        noElectrons[_ev] = 1
    


# In[30]:


# loop over tight muons:
for _ev,_nmuons in tqdm(enumerate(N_TMuons)):
    # reset muon kinematics:
    _mu1_pt=0; _mu1_eta=0; _mu1_phi=0; _mu1_t=0; _mu1_z=0
    _mu2_pt=0; _mu2_eta=0; _mu2_phi=0; _mu2_t=0; _mu2_z=0

    for i in range(_nmuons):
        # search for leading muon:
        if TMuon_pt[_ev][i]>_mu1_pt:
            # store subleading muon:
            _mu2_pt = _mu1_pt
            _mu2_eta = _mu1_eta
            _mu2_phi = _mu1_phi     
            # store leading muon:
            _mu1_pt = TMuon_pt[_ev][i]
            _mu1_eta = TMuon_eta[_ev][i]
            _mu1_phi = TMuon_phi[_ev][i]
        elif TMuon_pt[_ev][i]>_mu2_pt:
            # if last, store subleading muon:
            _mu2_pt = TMuon_pt[_ev][i]
            _mu2_eta = TMuon_eta[_ev][i]
            _mu2_phi = TMuon_phi[_ev][i]

    # Eta for all muons:
    AllTMuon_eta.append(_mu1_eta)
    AllTMuon_eta.append(_mu2_eta)  
        
    # Pt for all muons:
    AllTMuon_pt.append(_mu1_pt)
    AllTMuon_pt.append(_mu2_pt)
    
    # Phi for all muons:
    AllTMuon_phi.append(_mu1_phi)    
    AllTMuon_phi.append(_mu2_phi)
    
    
    
    # if valid event with 2 muons and 2 protons in PPS:
    if (_mu1_pt!=0) and (_mu2_pt!=0) and (ProtonsPos[_ev]!=0) and (ProtonsNeg[_ev]!=0):
        # save the muons for leading and subleading:
        TMuon_Lead_pt.append(_mu1_pt)
        TMuon_SubL_pt.append(_mu2_pt)
        # compute invariant mass of lepton pair:
        TMuon_SumET.append(_mu1_pt*np.cosh(_mu1_eta)+_mu2_pt*np.cosh(_mu2_eta)) # sumET = pt1*cosh(eta1) + pt2*cosh(eta2)
        _sumTPx = _mu1_pt*np.cos(_mu1_phi)+_mu2_pt*np.cos(_mu2_phi) # sumTPx = pt1*cos(phi1) + pt2*cos(phi2)
        _sumTPy = _mu1_pt*np.sin(_mu1_phi)+_mu2_pt*np.sin(_mu2_phi) # sumTPy = pt1*sin(phi1) + pt2*sin(phi2)
        TMuon_SumPz.append(_mu1_pt*np.sinh(_mu1_eta)+_mu2_pt*np.sinh(_mu2_eta))  # sumTPz = pt1*sinh(eta1) + pt2*sinh(eta2)
        # building the dilepton:
        #_di_TMuonP = np.sqrt(_sumTPx**2 + _sumTPy**2 + ( _mu1_pt*np.sinh(_mu1_eta)+_mu2_pt*np.sinh(_mu2_eta) )**2)
        _di_TMuonP = np.sqrt( _sumTPx**2 + _sumTPy**2 + TMuon_SumPz[-1]**2 )
        #_mllt2 = (_mu1_pt*np.cosh(_mu1_eta)+_mu2_pt*np.cosh(_mu2_eta))**2 - _di_TMuonP**2  # - sumTPx**2 - sumTPy**2 - sumTPz**2
        _mllt2 = TMuon_SumET[-1]**2 - _di_TMuonP**2
        # physics check:
        if(_mllt2<0): _mllt2=0
        TMuon_mll.append(np.sqrt(_mllt2))
        #TMuon_yll[_ev]=_mu1_eta+_mu2_eta
        TMuon_T.append(TMuon_VT[_ev])
        TMuon_z.append(TMuon_Vz[_ev])
        # define rapidity of dilepton:
        #TMuon_yll = (1/2) log (sumET+sumTPz/sumET-sumTPz)
        #TMuon_yll.append(0.5*np.log(((_mu1_pt*np.cosh(_mu1_eta)+_mu2_pt*np.cosh(_mu2_eta))+(_mu1_pt*np.sinh(_mu1_eta)+_mu2_pt*np.sinh(_mu2_eta)))/((_mu1_pt*np.cosh(_mu1_eta)+_mu2_pt*np.cosh(_mu2_eta))-(_mu1_pt*np.sinh(_mu1_eta)+_mu2_pt*np.sinh(_mu2_eta)))))
        # compute rapidity from last object appended ([-1])
        TMuon_yll.append( 0.5*np.log((TMuon_SumET[-1] + TMuon_SumPz[-1])/(TMuon_SumET[-1] - TMuon_SumPz[-1])))
    else:
        # if not 2 muons plus 2 protons:
        noMuons[_ev] = 1


# At this point we have two vectors that contain information for each lepton after the selection. 
#   
#   For Muons:
#     
#     AllTMuon_eta;
#     noMuons;
#     
#   For Electrons:
#   
#     AllElectrons_eta;
#     noElectrons;
#     
# Note that in the vectors noMuons and noElectrons the components with 1 represent "NO" and those with 0 represent "YES" to the question: "Are there at least 2 leptons and 2 protons in the event?"

# In[31]:


print("Vector with all Muons: \n",AllTMuon_eta)

print("Vector lengh is:",len(AllTMuon_eta))

print("\n Vector with all Electrons: \n", AllElectron_eta)

print("Vector lengh is:",len(AllElectron_eta))


# Now we must rearrange those vectores into arrays with 250 lines (one for each event) and 2 columns (for the leading and subleading).

# In[32]:


print(AllTMuon_eta)


# !Warning!
# 
# For some reason the lengh of the vectors 'AllTMuon_eta' and 'AllElectron_eta' is different (500 and 486). In order to compare them, we must add components in the the vector with least lengh untill they match.

# In[33]:


print(len(AllElectron_eta))
print(len(AllTMuon_eta))


# In[34]:


w, h = 2, (len(N_Protons))
Muon_eta = [[0 for i in range(w)] for k in range(h)] 
Electron_eta = [[0 for i in range(w)] for k in range(h)]


for i in range(len(AllTMuon_eta)-len(AllElectron_eta)):
    AllElectron_eta.append(0)
    
#Checking:
print(len(AllElectron_eta))


# An attempt to organize leptons information in arrays with two components, one for the Leading lepton and the other for Subleading. The final array must have len=250, which is the number of events. 

# In[35]:


j=0
for i in range(len(N_Protons)):
         
        Muon_eta[i][0]=AllTMuon_eta[j]
        Electron_eta[i][0]=AllElectron_eta[j]
        
        Muon_eta[i][1]=AllTMuon_eta[j+1]
        Electron_eta[i][1]=AllElectron_eta[j+1]

        j=2*(i+1)
#print(Muon_eta)
#print(len(Muon_eta))
print(Electron_eta)
#print(Electron_eta)


# In[36]:


#_valid_events_M (_valid_events_E) are the events with 2 Muons (Electrons) and 2 Protons

print("Vector containing the number of valid events Muons: \n",noMuons)
print("\nVector containing the number of valid events for Electrons: \n", noElectrons)

_valid_events_M=0
_valid_events_E=0

for i in range(len(noMuons)):
    if noMuons[i]==0:
        _valid_events_M=_valid_events_M+1

        
for i in range(len(noElectrons)):
    if noElectrons[i]==0:
        _valid_events_E=_valid_events_E+1
        

print("\n\nThe number of events with at least 2 Muons and 2 Protons is:", _valid_events_M)

print("The number of events with at least 2 Electrons and 2 Protons is:", _valid_events_E)


# In[37]:


#Counting the number of Muons and Electrons obtained:

_n_TMuons=0
_n_Electrons=0

for i in range(len(AllTMuon_eta)):
    if AllTMuon_eta[i]!=0:
        _n_TMuons=_n_TMuons+1
    if AllElectron_eta[i]!=0:
        _n_Electrons=_n_Electrons+1

print("The number of TMuons is:", _n_TMuons)
print("The number of Electrons is:", _n_Electrons)


# Now we must find events that contain at least 1 Muon and 1 Electron together with 2 protons in PPS. For this purpouse we can use the vector that contains Muons and Electrons individually and compare them. After that, we must check kinematics with 'crossed variables'.

# In[38]:


#Checking which events contain two protons:

Yes_2Protons=np.zeros(250)

for _ev in range(250): #250 is the number of events
    if ((ProtonsPos[_ev]!=0) and (ProtonsNeg[_ev]!=0)):
        Yes_2Protons[_ev]=1

print(Yes_2Protons)


# We now have an array named 'Electron_Muon' in which each component has an ordered pair with the 'pt' of 1 Electron and 1 Muon. We can now use the vectors containing the information about the presence of leptons and protons to check wich events are available with 1 electron + 1 muon and 2 protons. Then we must check the kinematics as we did before. 
