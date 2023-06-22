import numpy as np
import matplotlib.pyplot as plt

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
kBT=1
tmax=5000
psx=np.zeros((tmax,pnum,2))
rs=0.3
Ks=100
nring=10
numel_rings=9
pnum_fluid=pnum- nring*numel_rings 
nA=nring*numel_rings

def ringforces(pos,rs,Ks,L,nring,numel_rings,connectivitymatrix):

  forces=np.zeros_like(pos[:nring*numel_rings,:])
  
  for cx in range(nring):
    elems=pos[cx*numel_rings:cx*numel_rings+nring-1,:]

    elex=elems[:,0]+1j*elems[:,1]
    elex=elex.reshape((9,1))
    distx=elex-elex.T
    distx.real = np.mod(distx.real + L / 2, L) - L / 2
    distx.imag = np.mod(distx.imag + L / 2, L) - L / 2

    mags=np.absolute(distx)
    mags += np.finfo(float).eps
    direx=distx/mags

    Fsi=Ks*(1-mags/rs)*direx

    Fsi=Fsi*connectivitymatrix
    Fsi=np.nansum(Fsi,axis=0)
    Fsi=Fsi.T
    forces[cx*numel_rings:cx*numel_rings+nring-1,0]=-Fsi.real
    forces[cx*numel_rings:cx*numel_rings+nring-1,1]=-Fsi.imag

  return forces
  
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

radius=.3
rings=np.zeros((nring,numel_rings,2))
centre=np.zeros((nring,2))
centre=2+(10*np.random.randn(nring,2))%12
theta = np.linspace(0, 2 * np.pi, numel_rings, endpoint=False)

ud = np.roll(np.eye(numel_rings), 1, axis=0)
ld = np.roll(np.eye(numel_rings), -1, axis=0)
connectivitymatrix = ud + ld

for i in range(nring):  
  rings[i,:,0] =  radius * np.cos(theta) +  centre[i,0]
  rings[i,:,1] = radius * np.sin(theta)  +  centre[i,1] 

posx=np.zeros((pnum_fluid, 2))
posx[:, 0] = (15*np.random.randn(pnum_fluid))%15
posx[:, 1] = (15*np.random.randn(pnum_fluid))%15

cond=np.logical_or(posx[:,0]<=1,posx[:,0]>=14)
pos_wall=posx[cond]
posf=posx[~cond]

nF=pnum_fluid- len(pos_wall)
posring=rings.reshape((nA,2))
pos=np.zeros((pnum,2))

pos[:nA]=posring
pos[nA:nA+nF]=posf
pos[nA+nF:]=pos_wall

vel=np.zeros_like(pos)
accln=np.zeros_like(pos)
momentum=[]
temp=[]
FBODY=np.tile(np.array([0,0.3]),(nA+nF,1))
#ringp=np.reshape(rings,(90,2))

for te in range(tmax):  
  print(f"Running iteration::::{te}")
  pos += vel  * dt
  pos[:nA+nF,0]=np.clip(pos[:nA+nF,0],1,14)
  pos  = pos % L

  accln = force_calc(pos,vel,L,rc,gam,aij,nA,nF)
  acclnring=ringforces(pos,rs,Ks,L,nring,numel_rings,connectivitymatrix)

  accln[:nA]+=acclnring
  accln[:nA+nF] = accln[:nA+nF]+FBODY

  vel += accln  * dt
  vel[nA+nF:]=0.

  p_inst=np.nansum(vel,axis=0)
  ke=(np.linalg.norm(p_inst*p_inst)**2)/(2*pnum)
  momentum.append(np.linalg.norm(p_inst))
  temp.append(ke)


  print("plotting...")
  plt.scatter(pos[:nA,0],pos[:nA,1],color='purple',marker='.')
  plt.scatter(pos[nA+nF:,0],pos[nA+nF:,1],color='black',marker='.')
  plt.scatter(pos[nA:nA+nF,0], pos[nA:nA+nF,1],color='orange',marker='.')
  plt.xlim(0,15)
  plt.ylim(0,15)
  plt.tight_layout()
  plt.savefig(f'{te}.png')
  plt.close()
  if te==1000:
    plt.subplot(2,1,1)
    plt.plot(range(len(momentum)),momentum)
    plt.ylabel('momentum')
    plt.title('evolution of momentum and temperature')  

    plt.subplot(2,1,2)
    plt.plot(range(len(temp)),temp)
    plt.ylabel('temperature')
    plt.savefig(f'details1000.png')    
    plt.close()

plt.subplot(2,1,1)
plt.plot(range(len(momentum)),momentum)
plt.ylabel('momentum')
plt.title('evolution of momentum and temperature')  

plt.subplot(2,1,2)
plt.plot(range(len(temp)),temp)
plt.ylabel('temperature')
plt.savefig(f'details.png')