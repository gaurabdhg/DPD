import numpy as np
import matplotlib.pyplot as plt

#universal constants
L=15
sigma=1.
gam=4.5
rc=1
m=1
rho=4
pnum=rho*L*L

vwall=0
aij=np.array([[50,25,200],[25,25,200],[200,200,0]])
dt=0.01
tmax=1000
psx=np.zeros((tmax,pnum,2))

#problem specific constants
rs=0.3
Ks=100
nring=0
numel_rings=9
pnum_fluid=pnum- nring*numel_rings 
nA=nring*numel_rings

#interaction forces computation  
def force_calc(pos,vel,L,rc,gam,aij,nA,nF):
  
  forx=np.zeros_like(pos)
  for i in range(len(pos)-1):
    if i<nA:
      ii=0
    elif i>=nA+nF:
      ii=2
    else: 
      ii=1
    ftemp=np.array([0.,0.])
    for j in range(i+1,len(pos)):
      dists=pos[i]-pos[j]

      if dists[0]>L/2:
        dists[0]-=L
      elif dists[0]<-L/2:
        dists[0]+=L
      
      if dists[1]>L/2:
        dists[1]-=L
      elif dists[1]<-L/2:
        dists[1]+=L
      
      rij=np.linalg.norm(dists)
      dirn=dists/rij
      dv=vel[i]-vel[j]
      
      wR=1 - rij if rij<rc else 0
      wD=wR**2
     
      if j<nA:
        jj=0
      elif j>=nA+nF:
        jj=2
      else: jj=1
      aa=aij[ii,jj]
      zeta=np.random.randn()
      Fcons=aa*(1-rij/rc)*dirn if rij<rc else np.array([0., 0.])
      Fdiss=-gam*wD*np.dot(dirn,dv)*dirn*dt**-0.5
      Frand=np.sqrt(2*gam*kBT)*wR*zeta*dirn
      ftemp+=Fcons+Fdiss+Frand    
      forx[i,:]+=ftemp 
      forx[j,:]-=ftemp 
      ftemp=0

  return forx/m

#particles initialisation
posx=np.zeros((pnum_fluid, 2))
posx[:, 0] = (15*np.random.randn(pnum_fluid))%15
posx[:, 1] = (15*np.random.randn(pnum_fluid))%15

pos=np.zeros((pnum,2))
pos=posx
nF=pnum

#velocity initialisation
vel=np.zeros_like(pos)
accln=np.zeros_like(pos)
momentum=[]
temp=[]
FBODY=np.tile(np.array([0.,0.]),(nA+nF,1))
#ringp=np.reshape(rings,(90,2))

for te in range(tmax):  
  print(f"Running iteration::::{te}")
  pos += vel  * dt		#position update

  pos  = pos % L		#periodic position

  accln = force_calc(pos,vel,L,rc,gam,aij,nA,nF)
  accln[:nA+nF] = accln[:nA+nF]+FBODY

  #velocity update
  vel += accln  * dt
 
  #simulation tracking details
  p_inst=np.nansum(vel,axis=0)
  ke=(np.linalg.norm(p_inst*p_inst)**2)/(2*pnum)
  momentum.append(np.linalg.norm(p_inst))
  temp.append(ke)

#scatter plots
  print("plotting...")
  plt.scatter(pos[:nA,0],pos[:nA,1],color='purple',marker='.')
  plt.scatter(pos[nA+nF:,0],pos[nA+nF:,1],color='black',marker='.')
  plt.scatter(pos[nA:nA+nF,0], pos[nA:nA+nF,1],color='orange',marker='.')
  plt.xlim(0,15)
  plt.ylim(0,15)
  plt.tight_layout()
  plt.savefig(f'{te}.png')
  plt.close()

#momentum and temperature plots
plt.subplot(2,1,1)
plt.plot(range(len(momentum)),momentum)
plt.ylabel('momentum')
plt.title('evolution of momentum and temperature')  

plt.subplot(2,1,2)
plt.plot(range(len(temp)),temp)
plt.ylabel('temperature')
plt.savefig(f'details.png')
