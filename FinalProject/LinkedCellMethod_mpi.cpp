// ./LinkedCellMethod 1 1 3 3 10 14
// mpic++ -O3 -march=native -o LinkedCellMethod_mpi LinkedCellMethod_mpi.cpp
// mpirun -np 2 ./LinkedCellMethod_mpi 0.001 0.001 50000 5 100000 100000 1 2

#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

using std::string;

#define DIM 2
#define sigma 1
#define epsilon 1
#define m_range 1
#define v_range 10

typedef double real;

typedef struct{
  
  int id;
  real m; // mass
  real x[DIM]; // position
  real v[DIM]; // velocity
  real F[DIM]; // current force
  real F_old[DIM]; // previous step force, used to update the current velocity
  
} Particle;

typedef struct ParticleList{
  Particle p;
  struct ParticleList *next;
} ParticleList;

typedef struct {
  
  real l[DIM]; // size of simulation domain
  int nc[DIM]; // number of cells in simulation domain
  
  int myrank; // process number of the local process
  int numprocs; // number of processes started
  int ip[DIM]; // position of process in the process mesh
  int np[DIM]; // size of process mesh, also number of subdomains
  int ip_lower[DIM]; // process number of the neighbor processes
  int ip_upper[DIM];
  
  int ic_start[DIM]; // width of broader neighborhood, corresponds to the first local index in the interior of the subdomain
  int ic_stop[DIM]; // first local index in the upper border neighborhood
  int ic_number[DIM]; // number of cells in subdomain, including border neighborhood
  real cellh[DIM]; // dimension of a cell
  int ic_lower_global[DIM]; // global index of the first cell of the subdomain
  
} SubDomain;

typedef ParticleList* Cell;

#define sqr(x) ((x) * (x))
#define index(ic, nc) ((ic)[0] + (nc)[0] * (ic)[1])

#define iterate(ic, minnc, maxnc) \
for((ic)[0] = (minnc)[0]; (ic)[0] < (maxnc)[0]; (ic)[0]++) \
for((ic)[1] = (minnc)[1]; (ic)[1] < (maxnc)[1]; (ic)[1]++)

#define inverseIndex(i, nc, ic) \
((ic)[0] = (i) % (nc)[0], (ic)[1] = (i) / (nc)[0])

int mod(int a, int b){
  int c = a % b;
  return ((c < 0)? (c + b): c);
}

/* generate a random floating point number from min to max */
double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

// update the current position of a partical according to the timestep
void updateX(Particle *p, real delta_t, real *l){
  
  // according to the Velocity-Stormer-Verlet method,
  // (x_i)^(n + 1) = (x_i)^n + delta_t * (v_i)^n + ((F_i)^n * delta_t^2 / (2 * m_i))
  // printf("updateX: P(%d) from (%10f, %10f) to ", p->id, p->x[0], p->x[1]);
  real a = delta_t * 0.5 / p->m;
  for(int d = 0; d < DIM; d++){
    
    p->x[d] += delta_t * (p->v[d] + a * p->F[d]);
    // printf("p(%d).x[%d] move to %10f\n", p->id, d, p->x[d]);
    // Do not adjust x until moving the particle to another subdomain. Adjust x when the particle has been received by the other subdomain. Because we need the original x to calculate the cell.
    
    // store the force in the previous step in order to update the current velocity
    p->F_old[d] = p->F[d];
    
  }
  // printf("(%10f, %10f), m=%10f, v=(%10f, %10f), F=(%10f, %10f)\n", p->x[0], p->x[1], p->m, p->v[0], p->v[1], p->F[0], p->F[1] );
  
}

// update the current velocity of a partical according to the timestep
void updateV(Particle *p, real delta_t){
  
  // according to the Velocity-Stormer-Verlet method,
  // v_i^(n + 1) = v_i^n + ((F_i^n + F_i^(n + 1)) * delta_t) / (2 * m_i)
  real a = delta_t * 0.5 / p->m;
  for(int d = 0; d < DIM; d++){
    p->v[d] += a * (p->F[d] + p->F_old[d]);
  }
  
}

// Calculate Lennard-Jones force
void force(Particle *i, Particle *j, real *l){
  
  real r_sqr = 0;
  real dist_vec[DIM];
  for(int d = 0; d < DIM; d++){
    
    real nearUpperBound = std::max(i->x[d], j->x[d]);
    real nearLowerBound = std::min(i->x[d], j->x[d]);
    real dist_bound = (l[d] - nearUpperBound) + (nearLowerBound - 0);
    real dist_bound_sqr = sqr(dist_bound);
    real dist_normal = j->x[d] - i->x[d];
    real dist_normal_sqr = sqr(dist_normal);
    r_sqr += std::min(dist_bound_sqr, dist_normal_sqr);
    
    if(dist_bound < dist_normal){
      if(j->x[d] > i->x[d]){
        dist_vec[d] = (j->x[d] - l[d]) + (0 - i->x[d]);
      }else{
        dist_vec[d] = (j->x[d] - 0) + (l[d] - i->x[d]);
      }
    }else{
      dist_vec[d] = j->x[d] - i->x[d];
    }
    
    // printf("[d=%d] i->x[d]=%10f, j->x[d]=%10f, dist_bound=%10f, dist_normal=%10f, r_sqr=%10f, dist_vec[d]=%10f\n", d, i->x[d], j->x[d], dist_bound, dist_normal, r_sqr, dist_vec[d]);
    
  }
  
  real s = sqr(sigma) / r_sqr;
  s = sqr(s) * s;
  real f = 24 * epsilon * s / r_sqr * (1 - 2 * s);
  // printf("f=%10f\n", f);
  for(int d = 0; d < DIM; d++){
    i->F[d] += f * dist_vec[d];
    // printf("dimForce from P(%d) on P(%d): [%d] dist_vec=%10f, f=%10f\n", j->id, i->id, d, dist_vec[d], f * dist_vec[d]);
  }
  // printf("Calc Force from P(%d)(m=%10f, x=(%10f, %10f)) on P(%d)(m=%10f, x=(%10f, %10f)): F=(%10f, %10f)\n", j->id, j->m, j->x[0], j->x[1], i->id, i->m, i->x[0], i->x[1], i->F[0], i->F[1]);
  
}

// insert i into the particle list whose current root is root_list
void insertList(ParticleList **root_list, ParticleList *i){
  i->next = *root_list;
  *root_list = i;
}

void deleteList(ParticleList **q){
  *q = (*q)->next;
}

void deleteWholeCell(ParticleList **head_ref){
  
  /* deref head_ref to get the real head */
  ParticleList *current = *head_ref;
  ParticleList *next;
    
  while (current != NULL){
      next = current->next;
      free(current);
      current = next;
  }
        
  /* deref head_ref to affect the real head back
      in the caller. */
  *head_ref = NULL;
  
}

int lengthList(ParticleList *q){
  
  int count = 0;
  while(NULL != q){
    
    // printf("NULL != q\n");
    // printf("q->p.id=%d\n",q->p.id);
    
    count++;
    // printf("lengthList:count=%d\n", count);
    q = q->next;
    // printf("lengthList:count=%d: i = i->next\n", count);
  }
  return count;
  
}

void sendReceiveCell(Cell *grid, int *ic_number, int lowerProc, int *lowerIcStart, int *lowerIcStop, int *lowerIcStartReceive, int *lowerIcStopReceive, int upperProc, int *upperIcStart, int *upperIcStop, int *upperIcStartReceive, int *upperIcStopReceive, real *l, SubDomain *s){
  
  // printf("[rank=%d] enter sendReceiveCell\n", s->myrank);
  
  MPI_Status status;
  MPI_Request request;
  
  int sum_lengthSend = 0, sum_lengthReceive = 0;
  int k = 0, kReceive = 0, ncs = 1;
  int *ic_lengthSend = NULL, *ic_lengthReceive = NULL, ic[DIM];
  Particle *ip_particleSend = NULL, *ip_particleReceive = NULL;
  
  // Send to lowerProc, and receive from upperProc
  for(int d = 0; d < DIM; d++){
    ncs *= (lowerIcStop[d] - lowerIcStart[d]);
  }
  
  ic_lengthSend = (int*)malloc(ncs * sizeof(*ic_lengthSend));
  ic_lengthReceive = (int*)malloc(ncs * sizeof(*ic_lengthReceive));
  
  // printf("[rank=%d] start to iterate [d=0]=(%d, %d), [d=1]=(%d, %d)\n",  s->myrank, lowerIcStart[0], lowerIcStop[0], lowerIcStart[1], lowerIcStop[1]);
  iterate(ic, lowerIcStart, lowerIcStop){
    int length = lengthList(grid[index(ic, ic_number)]);
    // printf("[rank=%d] iterate k=%d end lengthList\n",  s->myrank, k);
    ic_lengthSend[k] = length;
    // printf("[rank=%d] iterate k=%d put lengthList into ic_lengthSend\n",  s->myrank, k);
    sum_lengthSend += ic_lengthSend[k++];
    // printf("[rank=%d] iterate ic_lengthSend k=%d\n",  s->myrank, k);
  }
  
  // printf("[rank=%d] MPI_Isend to lowerProc=%d\n", s->myrank, lowerProc);
  MPI_Isend(ic_lengthSend, ncs, MPI_INT, lowerProc, 1, MPI_COMM_WORLD, &request);
  MPI_Recv(ic_lengthReceive, ncs, MPI_INT, upperProc, 1, MPI_COMM_WORLD, &status);
  MPI_Wait(&request, &status);
  
  free(ic_lengthSend);
  
  for(k = 0; k < ncs; k++){
    sum_lengthReceive += ic_lengthReceive[k];
  }
  // printf("[rank=%d] sum_lengthReceive=%d\n", s->myrank, sum_lengthReceive);
  
  sum_lengthSend *= sizeof(*ip_particleSend);
  ip_particleSend = (Particle*)malloc(sum_lengthSend);
  
  sum_lengthReceive *= sizeof(*ip_particleReceive);
  ip_particleReceive = (Particle*)malloc(sum_lengthReceive);
  
  k = 0;
  iterate(ic, lowerIcStart, lowerIcStop){
    for(ParticleList *i = grid[index(ic, ic_number)]; NULL != i; i = i->next){
      ip_particleSend[k++] = i->p;
      // printf("[rank=%d] Sending particles(P(%d))(ic=(%d, %d), x=(%10f, %10f)) to lowerProc(%d).\n", s->myrank, i->p.id,  ic[0], ic[1], i->p.x[0], i->p.x[1], lowerProc);
    }
  }
  
  // printf("[rank=%d] Send to lowerProc=%d\n", s->myrank, lowerProc);
  MPI_Isend(ip_particleSend, sum_lengthSend, MPI_CHAR, lowerProc, 2, MPI_COMM_WORLD, &request);
  MPI_Recv(ip_particleReceive, sum_lengthReceive, MPI_CHAR, upperProc, 2, MPI_COMM_WORLD, &status);
  MPI_Wait(&request, &status);
  
  free(ip_particleSend);
  
  // printf("[rank=%d] Start to process recerived particles from upperProc.\n", s->myrank);
  kReceive = 0;
  iterate(ic, upperIcStartReceive, upperIcStopReceive){
    for(int icp = 0; icp < ic_lengthReceive[kReceive]; icp++){
      
      ParticleList *i = (ParticleList*)malloc(sizeof(*i));
      i->p = ip_particleReceive[k++];
      
      // Adjust x(only after moving particles could happen)
      // If in computing force stage, no paritcle's postion will < 0 or > l
      for(int d = 0; d < DIM; d++){
        if(i->p.x[d] >= l[d] || i->p.x[d] < 0){
          i->p.x[d] = i->p.x[d] - (floor(i->p.x[d] / l[d]) * l[d]);
        }
      }
      
      // printf("[rank=%d] recerived particles(P(%d))(ic=(%d, %d), x=(%10f, %10f)) from upperProc(%d).\n", s->myrank, i->p.id,  ic[0], ic[1], i->p.x[0], i->p.x[1], upperProc);
      insertList(&grid[index(ic, ic_number)], i);
      
    }
    kReceive++;
  }
  printf("[rank=%d] Recerived %d particles from upperProc(%d).\n", s->myrank, kReceive, upperProc);
  
  free(ic_lengthReceive);
  free(ip_particleReceive);
  
  // Send to upperProce, and receive from lowerProc
  ncs = 1;
  for(int d = 0; d < DIM; d++){
    ncs *= (upperIcStop[d] - upperIcStart[d]);
  }
  
  ic_lengthSend = (int*)malloc(ncs * sizeof(*ic_lengthSend));
  ic_lengthReceive = (int*)malloc(ncs * sizeof(*ic_lengthReceive));
  
  k = 0;
  iterate(ic, upperIcStart, upperIcStop){
    ic_lengthSend[k] = lengthList(grid[index(ic, ic_number)]);
    sum_lengthSend += ic_lengthSend[k++];
  }
  // printf("[rank=%d] sum_lengthSend=%d\n", s->myrank, sum_lengthSend);
  
  MPI_Isend(ic_lengthSend, ncs, MPI_INT, upperProc, 1, MPI_COMM_WORLD, &request);
  MPI_Recv(ic_lengthReceive, ncs, MPI_INT, lowerProc, 1, MPI_COMM_WORLD, &status);
  MPI_Wait(&request, &status);
  
  free(ic_lengthSend);
  
  for(k = 0; k < ncs; k++){
    sum_lengthReceive += ic_lengthReceive[k];
  }
  // printf("[rank=%d] sum_lengthReceive=%d\n", s->myrank, sum_lengthReceive);
  
  sum_lengthSend *= sizeof(*ip_particleSend);
  ip_particleSend = (Particle*)malloc(sum_lengthSend);
  
  sum_lengthReceive *= sizeof(*ip_particleReceive);
  ip_particleReceive = (Particle*)malloc(sum_lengthReceive);
  
  k = 0;
  
  // printf("[rank=%d] start to iterate to ip_particleSend from upperProc: [d=0]=(%d, %d), [d=1]=(%d, %d)\n",  s->myrank, upperIcStart[0], upperIcStop[0], upperIcStart[1], upperIcStop[1]);
  iterate(ic, upperIcStart, upperIcStop){
    
    // printf("[rank=%d] prepare ip_particleSend for ic=(%d, %d), grid[%d]: ", s->myrank, ic[0], ic[1], index(ic, ic_number));
    
    for(ParticleList *i = grid[index(ic, ic_number)]; NULL != i; i = i->next){
      // printf("P(%d) to ip_particleSend[%d]", i->p.id, k);
      ip_particleSend[k++] = i->p;
      // printf("[rank=%d] Sending particles(P(%d))(ic=(%d, %d), x=(%10f, %10f)) to upperProc(%d).\n", s->myrank, i->p.id,  ic[0], ic[1], i->p.x[0], i->p.x[1], upperProc);
      // printf(" -> ");
    }
    // printf("NULL\n");
    
  }
  
  // printf("[rank=%d] Send to upperProc=%d\n", s->myrank, upperProc);
  MPI_Isend(ip_particleSend, sum_lengthSend, MPI_CHAR, upperProc, 2, MPI_COMM_WORLD, &request);
  MPI_Recv(ip_particleReceive, sum_lengthReceive, MPI_CHAR, lowerProc, 2, MPI_COMM_WORLD, &status);
  MPI_Wait(&request, &status);
  
  free(ip_particleSend);
  
  // printf("[rank=%d] Start to process recerived particles from lowerProc.\n", s->myrank);
  kReceive = 0;
  iterate(ic, lowerIcStartReceive, lowerIcStopReceive){
    for(int icp = 0; icp < ic_lengthReceive[kReceive]; icp++){
      
      ParticleList *i = (ParticleList*)malloc(sizeof(*i));
      i->p = ip_particleReceive[k++];
      
      // Adjust x(only after moving particles could happen)
      // If in computing force stage, no paritcle's postion will < 0 or > l
      for(int d = 0; d < DIM; d++){
        if(i->p.x[d] >= l[d] || i->p.x[d] < 0){
          i->p.x[d] = i->p.x[d] - (floor(i->p.x[d] / l[d]) * l[d]);
        }
      }
      
      // printf("[rank=%d] recerived particles(P(%d))(ic=(%d, %d), x=(%10f, %10f)) from lowerProc(%d).\n", s->myrank, i->p.id,  ic[0], ic[1], i->p.x[0], i->p.x[1], lowerProc);
      
      insertList(&grid[index(ic, ic_number)], i);
      
    }
    kReceive++;
  }
  printf("[rank=%d] Recerived %d particles from upperProc(%d).\n", s->myrank, kReceive, upperProc);
  
  free(ic_lengthReceive);
  free(ip_particleReceive);
  
}

void setCommunication(SubDomain *s, int d, int *lowerIcStart, int *lowerIcStop, int *lowerIcStartReceive, int *lowerIcStopReceive, int *upperIcStart, int *upperIcStop, int *upperIcStartReceive, int *upperIcStopReceive){
  
  for(int dd = 0; dd < DIM; dd++){
    
    if(dd == d){
      
      lowerIcStart[dd] = s->ic_start[dd];
      lowerIcStop[dd] = lowerIcStart[dd] + s->ic_start[dd];
      lowerIcStartReceive[dd] = 0;
      lowerIcStopReceive[dd] = lowerIcStartReceive[dd] + s->ic_start[dd];
      
      upperIcStop[dd] = s->ic_stop[dd];
      upperIcStart[dd] = upperIcStop[dd] - s->ic_start[dd];
      upperIcStopReceive[dd] = s->ic_stop[dd] + s->ic_start[dd];
      upperIcStartReceive[dd] = upperIcStopReceive[dd] - s->ic_start[dd];
      
    }else if(dd > d){
      
      int stop = s->ic_stop[dd] + s->ic_start[dd];
      
      lowerIcStart[dd] = 0;
      lowerIcStop[dd] = stop;
      lowerIcStartReceive[dd] = 0;
      lowerIcStopReceive[dd] = stop;
      
      upperIcStart[dd] = 0;
      upperIcStop[dd] = stop;
      upperIcStartReceive[dd] = 0;
      upperIcStopReceive[dd] = stop;
      
    }else{
      
      lowerIcStart[dd] = s->ic_start[dd];
      lowerIcStop[dd] = s->ic_stop[dd];
      lowerIcStartReceive[dd] = s->ic_start[dd];
      lowerIcStopReceive[dd] = s->ic_stop[dd];
      
      upperIcStartReceive[dd] = s->ic_start[dd];
      upperIcStart[dd] = s->ic_start[dd];
      upperIcStopReceive[dd] = s->ic_stop[dd];
      upperIcStop[dd] = s->ic_stop[dd];
      
    }
    
  }
  
}

void compF_comm(Cell *grid, SubDomain *s){
  
  int lowerIcStart[DIM], lowerIcStop[DIM];
  int upperIcStart[DIM], upperIcStop[DIM];
  int lowerIcStartReceive[DIM], lowerIcStopReceive[DIM];
  int upperIcStartReceive[DIM], upperIcStopReceive[DIM];
  
  for(int d = DIM - 1; d < 0; d--){
    setCommunication(s, d, lowerIcStart, lowerIcStop, lowerIcStartReceive, lowerIcStopReceive, upperIcStart, upperIcStop, upperIcStartReceive, upperIcStopReceive);
    sendReceiveCell(grid, s->ic_number, s->ip_lower[d], lowerIcStart, lowerIcStop, lowerIcStartReceive, lowerIcStopReceive, s->ip_upper[d], upperIcStart, upperIcStop, upperIcStartReceive, upperIcStopReceive, s->l, s);
  }
  
}

void moveParticles_comm(Cell *grid, SubDomain *s){
  
  int lowerIcStart[DIM], lowerIcStop[DIM];
  int upperIcStart[DIM], upperIcStop[DIM];
  int lowerIcStartReceive[DIM], lowerIcStopReceive[DIM];
  int upperIcStartReceive[DIM], upperIcStopReceive[DIM];
  
  for(int d = 0; d < DIM; d++){
    setCommunication(s, d, lowerIcStartReceive, lowerIcStopReceive, lowerIcStart, lowerIcStop, upperIcStartReceive, upperIcStopReceive, upperIcStart, upperIcStop);
    // printf("[rank=%d] Start to sendReceiveCell(d=%d)\n", s->myrank, d);
    sendReceiveCell(grid, s->ic_number, s->ip_lower[d], lowerIcStart, lowerIcStop, lowerIcStartReceive, lowerIcStopReceive, s->ip_upper[d], upperIcStart, upperIcStop, upperIcStartReceive, upperIcStopReceive, s->l, s);
    // printf("[rank=%d] End of sendReceiveCell(d=%d)\n", s->myrank, d);
  }
  
}

void compF_LC(Cell *grid, SubDomain *s, real r_cut){
  
  int *nc = s->ic_number;
  real *l = s->l;
  
  int ic[DIM], kc[DIM], converted_kc[DIM], converted_ic_m1[DIM], converted_ic_p1[DIM];
  real cellUpperBound[DIM], cellLowerBound[DIM];
  
  // loop all cells in the grid
  for(ic[0] = s->ic_start[0]; ic[0] < s->ic_stop[0]; ic[0]++){
    for(ic[1] = s->ic_start[1]; ic[1] < s->ic_stop[1]; ic[1]++){
      
      // printf("[rank=%d] ic=(%d, %d)\n", s->myrank, ic[0], ic[1]);
      
      // loop every particle in the current cell
      for(ParticleList *i = grid[index(ic, nc)]; NULL != i; i = i->next){
        
        // reset the force in the current particel
        for(int d = 0; d < DIM; d++){
          i->p.F[d] = 0;
        }
        
        // printf("[rank=%d] particle i id=%d\n", s->myrank, i->p.id);
        
        // loop over all cells which is adjacent to the target cell
        for(kc[0] = ic[0] - 1; kc[0] <= ic[0] + 1; kc[0]++){
          for(kc[1] = ic[1] - 1; kc[1] <= ic[1] + 1; kc[1]++){
            
            // printf("[rank=%d] compF_LC: kc=(%d, %d)\n", s->myrank, kc[0], kc[1]);
            
            // treat kc[d] < 0 and kc[d] >= nc[d] according to boundary conditions
            for(int d = 0; d < DIM; d++){
              if(s->ip[d] == 0 && ic[d] == s->ic_start[0]){
                cellLowerBound[d] = 0;
              }else{
                cellLowerBound[d] = (s->ip[d] * (s->ic_stop[d] - s->ic_start[d]) + (ic[d] - s->ic_start[d])) * r_cut;
              }
            
              if(s->ip[d] == (s->np[d] - 1) && ic[d] == (s->ic_stop[d] - 1)){
                cellUpperBound[d] = l[d];
              }else{
                cellUpperBound[d] = (s->ip[d] * (s->ic_stop[d] - s->ic_start[d]) + (ic[d] - s->ic_start[d] + 1)) * r_cut;
              }
              
            }
            // printf("cellBound[0]=(%10f, %10f), cellBound[1]=(%10f, %10f)\n", cellLowerBound[0], cellUpperBound[0], cellLowerBound[1], cellUpperBound[1]);
            
            // Check the distance between the particle i and the cell kc. If the distance is over r_cut, we can skip the particles in the cell kc.
            real d_i_kc = 0;
            if(kc[0] == ic[0]){
              
              if(kc[1] == ic[1] - 1){
                d_i_kc = i->p.x[1] - cellLowerBound[1];
              }else if(kc[1] == ic[1] + 1){
                d_i_kc = cellUpperBound[1] - i->p.x[1];
              }
              
            }else if(kc[1] == ic[1]){
              
              if(kc[0] == ic[0] - 1){
                d_i_kc = i->p.x[0] - cellLowerBound[0];
              }else if(kc[0] == ic[0] + 1){
                d_i_kc = cellUpperBound[0] - i->p.x[0];
              }
              
            }else{
              
              real j_d_0 = 0;
              real j_d_1 = 0;
              
              if(kc[0] == ic[0] - 1){
                j_d_0 = cellLowerBound[0];
              }else if(kc[0] == ic[0] + 1){
                j_d_0 = cellUpperBound[0];
              }
              
              if(kc[1] == ic[1] - 1){
                j_d_1 = cellLowerBound[1];
              }else if(kc[1] == ic[1] + 1){
                j_d_1 = cellUpperBound[1];
              }
              
              d_i_kc = sqrt(sqr(i->p.x[0] - j_d_0) + sqr(i->p.x[1] - j_d_1));
              
            }
            // printf("[rank=%d] compF_LC-d_i_kc: kc=(%d, %d), d_i_kc=%10f\n", s->myrank, kc[0], kc[1], d_i_kc);
            
            if(d_i_kc <= r_cut){
              
              for(ParticleList *j = grid[index(kc, nc)]; NULL != j; j = j->next){
                
                if(NULL == j){
                  // printf("[rank=%d] compF_LC: no particle in kc=(%d, %d)\n", s->myrank, kc[0], kc[1]);
                  break;
                }
                
                if(i != j){
                  
                  // printf("[rank=%d] particle j id=%d\n", s->myrank, j->p.id);
                  
                  real r_sqr = 0;
                  for(int d = 0; d < DIM; d++){
                    
                    real nearUpperBound = std::max(i->p.x[d], j->p.x[d]);
                    real nearLowerBound = std::min(i->p.x[d], j->p.x[d]);
                    real dist_bound = sqr((l[d] - nearUpperBound) + (nearLowerBound - 0));
                    real dist_normal = sqr(j->p.x[d] - i->p.x[d]);
                    
                    r_sqr += std::min(dist_bound, dist_normal);
                    
                  }
                
                  
                  if(r_sqr <= sqr(r_cut)){
                    
                    force(&i->p, &j->p, l);
                    
                    // printf("[Particle: id=%d, f=(%10f, %10f)]\n", i->p.id, i->p.F[0], i->p.F[1]);
                    
                  }
                  
                }
              }
              
            }
            
          }
        }
        
      }
      
      // printf("[rank=%d] end of ic=(%d, %d)\n", s->myrank, ic[0], ic[1]);
      
    }
  }
  
}

void clearBoarderNeiborhood(Cell *grid, SubDomain *s){

  int ic[DIM];
                
                for(ic[0] = 0; ic[0] < s->ic_start[0]; ic[0]++){
                  for(ic[1] = s->ic_start[1]; ic[1] < s->ic_stop[1]; ic[1]++){
                    // printf("[rank=%d] start to deleteWholeCell: ic=(%d, %d) => grid[%d]\n", s->myrank, ic[0], ic[1], index(ic, s->ic_number));
                    deleteWholeCell(&grid[index(ic, s->ic_number)]);
                    // printf("[rank=%d] End deleteWholeCell: ic=(%d, %d) => grid[%d]\n", s->myrank, ic[0], ic[1], index(ic, s->ic_number));
                  }
                }
                
                for(ic[0] = s->ic_stop[0]; ic[0] < (s->ic_stop[0] + s->ic_start[0]); ic[0]++){
                  for(ic[1] = s->ic_start[1]; ic[1] < s->ic_stop[1]; ic[1]++){
                    // printf("[rank=%d] start to deleteWholeCell: ic=(%d, %d) => grid[%d]\n", s->myrank, ic[0], ic[1], index(ic, s->ic_number));
                    deleteWholeCell(&grid[index(ic, s->ic_number)]);
                    // printf("[rank=%d] End deleteWholeCell: ic=(%d, %d) => grid[%d]\n", s->myrank, ic[0], ic[1], index(ic, s->ic_number));
                  }
                }
                
                for(ic[1] = 0; ic[1] < s->ic_start[1]; ic[1]++){
                  for(ic[0] = 0; ic[0] < (s->ic_stop[0] + s->ic_start[0]); ic[0]++){
                    // printf("[rank=%d] start to deleteWholeCell: ic=(%d, %d) => grid[%d]\n", s->myrank, ic[0], ic[1], index(ic, s->ic_number));
                    deleteWholeCell(&grid[index(ic, s->ic_number)]);
                    // printf("[rank=%d] End deleteWholeCell: ic=(%d, %d) => grid[%d]\n", s->myrank, ic[0], ic[1], index(ic, s->ic_number));
                  }
                }
                
                for(ic[1] = s->ic_stop[1]; ic[1] < (s->ic_stop[1] + s->ic_start[1]); ic[1]++){
                  for(ic[0] = 0; ic[0] < (s->ic_stop[0] + s->ic_start[0]); ic[0]++){
                    // printf("[rank=%d] start to deleteWholeCell: ic=(%d, %d) => grid[%d]\n", s->myrank, ic[0], ic[1], index(ic, s->ic_number));
                    deleteWholeCell(&grid[index(ic, s->ic_number)]);
                    // printf("[rank=%d] End deleteWholeCell: ic=(%d, %d) => grid[%d]\n", s->myrank, ic[0], ic[1], index(ic, s->ic_number));
                  }
                }
                
}
                 
void compF_LC_mpi(Cell *grid, SubDomain *s, real r_cut){
  
  // printf("[rank=%d] inside start to compF_comm\n", s->myrank);
  compF_comm(grid, s);
  
  // printf("[rank=%d] inside start to compF_LC\n", s->myrank);
  compF_LC(grid, s, r_cut);
  
  // printf("[rank=%d] inside start to clearBoarderNeiborhood\n", s->myrank);
  clearBoarderNeiborhood(grid, s);
  
  // printf("[rank=%d] inside finished compF_LC_mpi\n", s->myrank);
    
}

void moveParticles_LC(Cell *grid, SubDomain *s, real r_cut){
  
  // printf("[rank=%d] inside start moveParticles_LC\n", s->myrank);
  
  int *nc = s->ic_number;
  real *l = s->l;
                
  int ic[DIM], kc[DIM];
  for(ic[0] = s->ic_start[0]; ic[0] < s->ic_stop[0]; ic[0]++){
    for(ic[1] = s->ic_start[1]; ic[1] < s->ic_stop[1]; ic[1]++){
      
      // printf("[rank=%d] moveParticles_LC: start to process ic=(%d, %d)\n", s->myrank, ic[0], ic[1]);
      
      ParticleList **q = &grid[index(ic, nc)];
      ParticleList *i = *q;
      while(NULL != i){
        
        // update the particle belongs to which cell based on the new position
        for(int d = 0; d < DIM; d++){
          
          real subdomainLowerBound = s->ic_lower_global[d] * r_cut;
          
          if(i->p.x[d] >= subdomainLowerBound){
            
            if(i->p.x[d] >= l[d]){
              kc[d] = s->ic_stop[d] + floor((i->p.x[d] - l[d]) / r_cut);
            }else{
              kc[d] = s->ic_start[d] + floor((i->p.x[d] - subdomainLowerBound) / r_cut);
            }
            
          }else{
            
            if(i->p.x[d] < 0){
              
              real lastCellh = l[d] - ((floor(l[d] - r_cut) - 1) * r_cut);
              if(i->p.x[d] >= (-1 * lastCellh)){
                kc[d] = s->ic_start[d] - 1;
              }else{
                kc[d] = s->ic_start[d] - 1 - ceil(((-1 * lastCellh) - i->p.x[d]) / r_cut);
              }
              
            }else{
              kc[d] = s->ic_start[d] - ceil((subdomainLowerBound - i->p.x[d]) / r_cut);
            }
            
          }
          // printf("[rank=%d] p(%d): d=%d\n, x=%10f, subdomainLowerBound=%10f, ic_start=%d, ic_stop=%d\n", s->myrank, i->p.id, d, i->p.x[d], subdomainLowerBound, s->ic_start[d], s->ic_stop[d]);
          
        }
        // printf("[rank=%d] p(%d): x=(%10f, %10f), kc=(%d, %d)\n", s->myrank, i->p.id, i->p.x[0], i->p.x[1], kc[0], kc[1]);
        
        if(ic[0] != kc[0] || ic[1] != kc[1]){
          // printf("[rank=%d] p(%d): x=(%10f, %10f), ic=(%d, %d), kc=(%d, %d)\n", s->myrank, i->p.id, i->p.x[0], i->p.x[1], ic[0], ic[1], kc[0], kc[1]);
          deleteList(q);
          insertList(&grid[index(kc, nc)], i);
        }else{
          q = &i->next;
        }
        i = *q;
        
      }
      
    }
  }
  
  // printf("[rank=%d] inside end moveParticles_LC\n", s->myrank);
  
}

void moveParticles_LC_mpi(Cell *grid, SubDomain *s, real r_cut){
  
  // printf("[rank=%d] Start to moveParticles_LC_mpi\n", s->myrank);
  
  moveParticles_LC(grid, s, r_cut);
  // printf("[rank=%d] End moveParticles_LC\n", s->myrank);
  
  int ic[DIM];
  for(ic[0] = 0; ic[0] < (s->ic_stop[0] + s->ic_start[0]); ic[0]++){
    for(ic[1] = 0; ic[1] < (s->ic_stop[1] + s->ic_start[1]); ic[1]++){
      // printf("[rank=%d] ic=(%d, %d):", s->myrank, ic[0], ic[1]);
      for(ParticleList *i = grid[index(ic, s->ic_number)]; NULL != i; i = i->next){
        // printf("P(%d)->", i->p.id);
      }
      // printf("NULL\n");
    }
  }
  
  // printf("[rank=%d] Start to moveParticles_comm\n", s->myrank);
  
  moveParticles_comm(grid, s);
  
  // printf("[rank=%d] Start to clearBoarderNeiborhood\n", s->myrank);
  
  clearBoarderNeiborhood(grid, s);
  
  // printf("[rank=%d] End of moveParticles_LC_mpi\n", s->myrank);
                
}

void compX_LC(Cell *grid, SubDomain *s, real delta_t, real r_cut){
  
  int ic[DIM];
  for(ic[0] = s->ic_start[0]; ic[0] < s->ic_stop[0]; ic[0]++){
    for(ic[1] = s->ic_start[1]; ic[1] < s->ic_stop[1]; ic[1]++){
      
      // printf("[rank=%d] start to updateX: ic=(%d, %d)\n", s->myrank, ic[0], ic[1]);
      
      for(ParticleList *i = grid[index(ic, s->ic_number)]; NULL != i; i = i->next){
        updateX(&i->p, delta_t, s->l);
      }
      
      // printf("[rank=%d] end updateX: ic=(%d, %d)\n", s->myrank, ic[0], ic[1]);
      
    }
  }
  // printf("[rank=%d] End of updateX, start to moveParticles_LC_mpi\n", s->myrank);
  
  moveParticles_LC_mpi(grid, s, r_cut);
  // printf("[rank=%d] End of moveParticles_LC_mpi\n", s->myrank);
  
}

void compV_LC(Cell *grid, SubDomain *s, real delta_t){
  
  int ic[DIM];
  for(ic[0] = s->ic_start[0]; ic[0] < s->ic_stop[0]; ic[0]++){
    for(ic[1] = s->ic_start[0]; ic[1] < s->ic_stop[1]; ic[1]++){
      
      for(ParticleList *i = grid[index(ic, s->ic_number)]; NULL != i; i = i->next){
        updateV(&i->p, delta_t);
      }
      
    }
  }
  
}

real compoutStatistic_LC(Cell *grid, SubDomain *s){
  
  real e = 0;
  
  int ic[DIM];
  for(ic[0] = s->ic_start[0]; ic[0] < s->ic_stop[0]; ic[0]++){
    for(ic[1] = s->ic_start[1]; ic[1] < s->ic_stop[1]; ic[1]++){
      
      for(ParticleList *i = grid[index(ic, s->ic_number)]; NULL != i; i = i->next){
        
        real v_sqr = 0;
        for(int d = 0; d < DIM; d++){
          v_sqr += sqr(i->p.v[d]);
        }
        e += 0.5 * i->p.m * v_sqr;
        
      }
      
    }
  }
  
  return e;
  
}

real timeIntegration_LC(real t, real delta_t, real t_end, Cell *grid, SubDomain *s, real r_cut){
  
  real e = 0;
  
  // printf("t=%10f\n", t);
  compF_LC_mpi(grid, s, r_cut);
  
  while(t < t_end){
    
    MPI_Barrier(MPI_COMM_WORLD);
    t += delta_t;
    if(s->myrank ==0){
      // printf("\n\n\n\n\nt=%10f\n", t);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // printf("[rank=%d][t=%10f] Start to compX_LC\n", s->myrank, t);
    
    compX_LC(grid, s, delta_t, r_cut);
    // printf("[rank=%d][t=%10f] Start to compF_LC_mpi\n", s->myrank, t);
    compF_LC_mpi(grid, s, r_cut);
    // printf("[rank=%d][t=%10f] Start to compV_LC\n", s->myrank, t);
    compV_LC(grid, s, delta_t);
    
    e = compoutStatistic_LC(grid, s);
    
  }
  
  return e;
  
}

void inputParameters_LC(int argc, char *argv[], real *delta_t, real *t_end, int *N, int *nc, real *l, real *r_cut){
  
  // delta_t, t_end, N, r_cut, l[0], l[1]
  // 0.00462 1 100 2.5 37.5 12.5
  if(argc >= 7){
    
    *delta_t = atof(argv[1]);
    
    *t_end = atof(argv[2]);
    
    // str = argv[3];
    *N = atoi(argv[3]);
    
    *r_cut = atof(argv[4]);
    
    int l_start = 5;
    for(int d = 0; d < DIM; d++){
      
      l[d] = atof(argv[l_start + d]);
      nc[d] = (int)floor(l[d] / *r_cut);
      
    }
    
  }
  
}

void inputParameters_LC_mpi(int argc, char *argv[], real *delta_t, real *t_end, int *N, SubDomain *s, real *r_cut){
  
  // set parameters as in the sequential case
  inputParameters_LC(argc, argv, delta_t, t_end, N, s->nc, s->l, r_cut);
  
  // set additional parameters for the parallelization
  MPI_Comm_size(MPI_COMM_WORLD, &s->numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &s->myrank);
  // printf("*** has rank=%d\n", s->myrank);
  
  int numprocs = 1;
  if(argc >= 9){
    int l_start = 7;
    for(int d = 0; d < DIM; d++){
      s->np[d] = atof(argv[l_start + d]);
      numprocs *= s->np[d];
    }
  }
  s->numprocs = numprocs;
  
  // determine position of myrank in the process mesh np[d]
  int ipTemp[DIM];
  inverseIndex(s->myrank, s->np, s->ip);
  for(int d = 0; d < DIM; d++){
    ipTemp[d] = s->ip[d];
  }
                
  // determine neighborhood processes
  for(int d = 0; d < DIM; d++){
    ipTemp[d] = (s->ip[d] - 1 + s->np[d]) % (s->np[d]);
    s->ip_lower[d] = index(ipTemp, s->np);
    ipTemp[d] = (s->ip[d] + 1 + s->np[d]) % (s->np[d]);
    s->ip_upper[d] = index(ipTemp, s->np);
    ipTemp[d] = s->ip[d];
  }
  // printf("[rank=%d] ip=(%d, %d)=%d, ip_lower=(%d, %d), ip_upper=(%d, %d)\n", s->myrank, s->ip[0], s->ip[1], index(s->ip, s->np), s->ip_lower[0], s->ip_lower[1], s->ip_upper[0], s->ip_upper[1]);
                
  // set local parameters
  for(int d = 0; d < DIM; d++){
    s->cellh[d] = s->l[d] / s->nc[d];
    s->ic_start[d] = (int) ceil(*r_cut / s->cellh[d]);
    s->ic_stop[d] = s->ic_start[d] + (s->nc[d] / s->np[d]);
    s->ic_number[d] = (s->ic_stop[d] - s->ic_start[d]) + 2 * (s->ic_start[d]);
    s->ic_lower_global[d] = s->ip[d] * (s->nc[d] / s->np[d]);
    // printf("[rank=%d] d=%d: ic_lower_global=%d, ip=%d, nc=%d, np=%d\n", s->myrank, d, s->ic_lower_global[d], s->ip[d], s->nc[d], s->np[d]);
  }
  // printf("[rank=%d] ic_number=(%d, %d), (ic_start, ic_stop): [d=%d](%d, %d); [d=%d](%d, %d)\n", s->myrank, s->ic_number[0], s->ic_number[1], 0, s->ic_start[0], s->ic_stop[0], 1, s->ic_start[1], s->ic_stop[1]);
  
}

void initData_LC(int N, Cell *grid, SubDomain *s, real r_cut){
  
  int rank = s->myrank;
  int numprocs = s->numprocs;
  int particleNumberPerSubDomain = (int)floor(N / numprocs);
  
  /*
  if(s->myrank == 0){
    printf("particleNumberPerSubDomain=%d\n", particleNumberPerSubDomain);
  }
  */
  
  int subNLowerBound = rank * particleNumberPerSubDomain;
  int subNUpperBound = (rank == numprocs - 1)? N: subNLowerBound + particleNumberPerSubDomain;
                
  real subDomainLowerBound[DIM], subDomainUpperBound[DIM];
  for(int d = 0; d < DIM; d++){
    subDomainLowerBound[d] = s->ic_lower_global[d] * r_cut;
    if(s->ip[d] == s->np[d] - 1){
      subDomainUpperBound[d] = s->l[d] - 0.001;
    }else{
      subDomainUpperBound[d] = subDomainLowerBound[d] + ((s->ic_stop[d] - s->ic_start[d]) * r_cut) - 0.001;
    }
  }
                  
  srand(time(NULL));
  
  int ic[DIM];
  // printf("[rank=%d] Start to generate particles: id from %d to %d\n", s->myrank, subNLowerBound, subNUpperBound);
  if(subNUpperBound > subNLowerBound){
    for(int i = subNLowerBound; i < subNUpperBound; i++){
    
      ParticleList *pl_p = (ParticleList*)malloc(sizeof(*pl_p));
    
      pl_p->p.id = i;
      pl_p->p.m = drand48() * m_range;
    
      for(int d = 0; d < DIM; d++){
      
        pl_p->p.x[d] = randfrom(subDomainLowerBound[d], subDomainUpperBound[d]);
        pl_p->p.v[d] = randfrom(-1.0 * v_range, 1.0 * v_range);
        real dist_p_lowerBound = pl_p->p.x[d] - subDomainLowerBound[d];
        ic[d] = s->ic_start[d] + ((((int)floor(dist_p_lowerBound / r_cut) >= (s->ic_stop[d] - s->ic_start[d] - 1)))? s->ic_stop[d] - s->ic_start[d] - 1: (int)floor(dist_p_lowerBound / r_cut));
      
      }
    
      // printf("[rank=%d][Particle: id=%d, m=%10f, x=(%10f, %10f), v=(%10f, %10f), ic=(%d, %d)]\n", s->myrank, pl_p->p.id, pl_p->p.m, pl_p->p.x[0], pl_p->p.x[1], pl_p->p.v[0], pl_p->p.v[1], ic[0], ic[1]);
    
      // printf("[rank=%d]Start to insert particle[id=%d] into grid[%d=index(%d, %d)]\n", s->myrank, pl_p->p.id, index(ic, s->ic_number), ic[0], ic[1]);
      insertList(&grid[index(ic, s->ic_number)], pl_p);
    
    }
  }else{
    // printf("[rank=%d] No particles need to be generated\n", s->myrank);
  }
  // printf("[rank=%d] End generate particles\n", s->myrank);
  
  /*
  iterate(ic, s->ic_start, s->ic_stop){
    printf("[rank=%d] ic=(%d, %d):", s->myrank, ic[0], ic[1]);
    for(ParticleList *i = grid[index(ic, s->ic_number)]; NULL != i; i = i->next){
      printf("P(%d)->", i->p.id);
    }
    printf("NULL\n");
  }
  */
  
}

void freeLists_LC(Cell *grid, int *nc){
  
  ParticleList *current, *prev;
  int ic[DIM];
  for(ic[0] = 0; ic[0] < nc[0]; ic[0]++){
    for(ic[1] = 0; ic[1] < nc[1]; ic[1]++){
      
      current = grid[index(ic, nc)];
      while(NULL != current){
        
        prev = current;
        current = current->next;
        free(prev);
        
      }
      
    }
  }
  
}

int main(int argc, char** argv){
  
  int N, pnc;
  real r_cut, delta_t, t_end;
  SubDomain s;
  
  MPI_Init(&argc, &argv);

  inputParameters_LC_mpi(argc, argv, &delta_t, &t_end, &N, &s, &r_cut);
  
  if(s.myrank == 0){
    printf("N=%d, r_cut=%10f, l=(%10f, %10f), nc=(%d, %d), s->numprocs=%d\n", N, r_cut, s.l[0], s.l[1], s.nc[0], s.nc[1], s.numprocs);
  }
  
  pnc = 1;
  for(int d = 0; d < DIM; d++){
    pnc *= s.ic_number[d];
  }
  Cell *grid = (Cell*)malloc(pnc * sizeof(*grid));
  
  bool isDirtyMemory = true;
  while(isDirtyMemory){
    
    isDirtyMemory = false;
    int ic[DIM];
    // printf("[rank=%d] Check memory data: \n", s.myrank);
    for(ic[0] = 0; ic[0] < (s.ic_stop[0] + s.ic_start[0]); ic[0]++){
      for(ic[1] = 0; ic[1] < (s.ic_stop[1] + s.ic_start[1]); ic[1]++){
        // printf("[rank=%d] ic=(%d, %d): ", s.myrank, ic[0], ic[1]);
        ParticleList *i = grid[index(ic, s.ic_number)];
        if(NULL != i){
          // printf("has dirty data\n");
          isDirtyMemory = isDirtyMemory || true;
          grid[index(ic, s.ic_number)] = NULL;
        }else{
          // printf("is clean\n");
          isDirtyMemory = isDirtyMemory || false;
        }
      }
    }
    if(isDirtyMemory == true){
      free(grid);
      // printf("[rank=%d] free memory\n", s.myrank);
      grid = (Cell*)malloc(pnc * sizeof(*grid));
      // printf("[rank=%d] allocate memory\n", s.myrank);
    }
    
  }
  
  // printf("[rank=%d] Check memory data again: \n", s.myrank);
  int ic[DIM];
  for(ic[0] = 0; ic[0] < (s.ic_stop[0] + s.ic_start[0]); ic[0]++){
    for(ic[1] = 0; ic[1] < (s.ic_stop[1] + s.ic_start[1]); ic[1]++){
    
      // printf("[rank=%d] ic=(%d, %d):", s.myrank, ic[0], ic[1]);
      for(ParticleList *i = grid[index(ic, s.ic_number)]; NULL != i; i = i->next){
        // printf("P(%d)->", i->p.id);
      }
      // printf("NULL\n");
      
    }
  }
  
  printf("[rank=%d] start to initData\n", s.myrank);
  initData_LC(N, grid, &s, r_cut);
  printf("[rank=%d] Finished initData\n", s.myrank);
  MPI_Barrier(MPI_COMM_WORLD);
  
  
  double start = MPI_Wtime();
  timeIntegration_LC(0, delta_t, t_end, grid, &s, r_cut);
  double end = MPI_Wtime();
  if(s.myrank == 0){
    printf("Finished simulation, eslapse time = %10f sec(s).\n", end - start);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  
  if(s.myrank == 0){
    printf("All finished, start to free memory\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  // printf("[rank=%d] Start to freeLists_LC\n", s.myrank);
  freeLists_LC(grid, s.ic_number);
  // printf("[rank=%d] Start to free(grid)\n", s.myrank);
  free(grid);
  printf("[rank=%d] Finished free\n", s.myrank);
  
  MPI_Finalize();
  
  return 0;
  
}
