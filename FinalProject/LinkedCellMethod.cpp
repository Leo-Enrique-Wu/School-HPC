// g++ -std=c++11 -O3 -march=native -o LinkedCellMethod  LinkedCellMethod.cpp

// ./LinkedCellMethod 0.001 0.001 50000 5 100000 100000

#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

using std::string;

#define DIM 2
#define sigma 1
#define epsilon 1
#define m_range 1
#define v_range 10

typedef double real;

typedef struct{
  
  long id;
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

typedef ParticleList* Cell;

#define sqr(x) ((x) * (x))
#define index(ic, nc) ((ic)[0] + (nc)[0] * (ic)[1])

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
  real a = delta_t * 0.5 / p->m;
  for(int d = 0; d < DIM; d++){
    
    p->x[d] += delta_t * (p->v[d] + a * p->F[d]);
    // printf("p(%d).x[%d] move to %10f\n", p->id, d, p->x[d]);
    if(p->x[d] >= l[d] || p->x[d] < 0){
      p->x[d] = p->x[d] - (floor(p->x[d] / l[d]) * l[d]);
      // printf("Adjust to %10f\n", p->x[d]);
    }
    
    // store the force in the previous step in order to update the current velocity
    p->F_old[d] = p->F[d];
    
  }
  
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
    
  }
  real s = sqr(sigma) / r_sqr;
  s = sqr(s) * s;
  real f = 24 * epsilon * s / r_sqr * (1 - 2 * s);
  for(int d = 0; d < DIM; d++){
    i->F[d] += f * dist_vec[d];
  }
  
}

// insert i into the particle list whose current root is root_list
void insertList(ParticleList **root_list, ParticleList *i){
  i->next = *root_list;
  *root_list = i;
}

void deleteList(ParticleList **q){
  *q = (*q)->next;
}

void compF_LC(Cell *grid, int *nc, real r_cut, real *l){
  
  int ic[DIM], kc[DIM], converted_kc[DIM], converted_ic_m1[DIM], converted_ic_p1[DIM];
  real cellUpperBound[DIM], cellLowerBound[DIM];
  
  // loop all cells in the grid
  for(ic[0] = 0; ic[0] < nc[0]; ic[0]++){
    for(ic[1] = 0; ic[1] < nc[1]; ic[1]++){
      
      ParticleList *temp_i = grid[index(ic, nc)];
      /*
      if(NULL != temp_i){
        printf("NULL != grid[%d=index(%d, %d)]\n", index(ic, nc), ic[0], ic[1]);
      }else{
        printf("NULL == grid[%d=index(%d, %d)]\n", index(ic, nc), ic[0], ic[1]);
      }
       */
      
      // loop every particle in the current cell
      for(ParticleList *i = grid[index(ic, nc)]; NULL != i; i = i->next){
        
        // reset the force in the current particel
        for(int d = 0; d < DIM; d++){
          i->p.F[d] = 0;
        }
        
        // loop over all cells which is adjacent to the target cell
        for(kc[0] = ic[0] - 1; kc[0] <= ic[0] + 1; kc[0]++){
          for(kc[1] = ic[1] - 1; kc[1] <= ic[1] + 1; kc[1]++){
            
            // treat kc[d] < 0 and kc[d] >= nc[d] according to boundary conditions
            for(int d = 0; d < DIM; d++){
              if(kc[d] < 0 || kc[d] >= nc[d]){
                converted_kc[d] = mod(kc[d], nc[d]);
              }else{
                converted_kc[d] = kc[d];
              }
            }
            
            for(int d = 0; d < DIM; d++){
              if(ic[d] - 1 < 0){
                converted_ic_m1[d] = mod(ic[d] - 1, nc[d]);
                cellLowerBound[d] = 0;
              }else{
                converted_ic_m1[d] = ic[d] - 1;
                cellLowerBound[d] = ic[d] * r_cut;
              }
            
              if(ic[d] + 1 >= nc[d]){
                converted_ic_p1[d] = mod(ic[d] + 1, nc[d]);
                cellUpperBound[d] = l[d];
              }else{
                converted_ic_p1[d] = ic[d] + 1;
                cellUpperBound[d] = (ic[d] + 1) * r_cut;
              }
              
            }
            // printf("cellBound[0]=(%10f, %10f), cellBound[1]=(%10f, %10f)\n", cellLowerBound[0], cellUpperBound[0], cellLowerBound[1], cellUpperBound[1]);
            
            // Check the distance between the particle i and the cell kc. If the distance is over r_cut, we can skip the particles in the cell kc.
            real d_i_kc = 0;
            if(converted_kc[0] == ic[0]){
              
              if(converted_kc[1] == converted_ic_m1[1]){
                d_i_kc = i->p.x[1] - cellLowerBound[1];
              }else if(converted_kc[1] == converted_ic_p1[1]){
                d_i_kc = cellUpperBound[1] - i->p.x[1];
              }
              
            }else if(converted_kc[1] == ic[1]){
              
              if(converted_kc[0] == converted_ic_m1[0]){
                d_i_kc = i->p.x[0] - cellLowerBound[0];
              }else if(converted_kc[0] == converted_ic_p1[0]){
                d_i_kc = cellUpperBound[0] - i->p.x[0];
              }
              
            }else{
              
              real j_d_0 = 0;
              real j_d_1 = 0;
              
              if(converted_kc[0] == converted_ic_m1[0]){
                j_d_0 = cellLowerBound[0];
              }else if(converted_kc[0] == converted_ic_p1[0]){
                j_d_0 = cellUpperBound[0];
              }
              
              if(converted_kc[1] == converted_ic_m1[1]){
                j_d_1 = cellLowerBound[1];
              }else if(converted_kc[1] == converted_ic_p1[1]){
                j_d_1 = cellUpperBound[1];
              }
              
              d_i_kc = sqrt(sqr(i->p.x[0] - j_d_0) + sqr(i->p.x[1] - j_d_1));
              
            }
            // printf("P[%d](ic=(%d, %d)) and kc(%d, %d): d_i_kc=%10f, p.x=(%10f, %10f)\n", i->p.id, ic[0], ic[1], converted_kc[0], converted_kc[1], d_i_kc, i->p.x[0], i->p.x[1]);
            
            if(d_i_kc <= r_cut){
              
              for(ParticleList *j = grid[index(converted_kc, nc)]; NULL != j; j = j->next){
                if(i != j){
                  
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
      
    }
  }
  
}

void moveParticles_LC(Cell *grid, int *nc, real *l, real r_cut){
  
  int ic[DIM], kc[DIM];
  for(ic[0] = 0; ic[0] < nc[0]; ic[0]++){
    for(ic[1] = 0; ic[1] < nc[1]; ic[1]++){
      
      ParticleList **q = &grid[index(ic, nc)];
      ParticleList *i = *q;
      while(NULL != i){
        
        // update the particle belongs to which cell based on the new position
        for(int d = 0; d < DIM; d++){
          
          // The particle might move over the upper bound. Based on the boundary condition, it should appear from the lower bound.
          real x = i->p.x[d];
          if(x >= l[d]){
            x = x - (floor(x / l[d]) * l[d]);
          }
          
          kc[d] = ((int)floor(i->p.x[d] / r_cut) >= nc[d])? nc[d] - 1: (int)floor(i->p.x[d] / r_cut);
          
        }
        
        if(ic[0] != kc[0] || ic[1] != kc[1]){
          // printf("p(%d): ic=(%d, %d), kc=(%d, %d)\n", i->p.id, ic[0], ic[1], kc[0], kc[1]);
          deleteList(q);
          insertList(&grid[index(kc, nc)], i);
        }else{
          q = &i->next;
        }
        i = *q;
        
      }
      
    }
  }
  
}

void compX_LC(Cell *grid, int *nc, real *l, real delta_t, real r_cut){
  
  int ic[DIM];
  for(ic[0] = 0; ic[0] < nc[0]; ic[0]++){
    for(ic[1] = 0; ic[1] < nc[1]; ic[1]++){
      
      for(ParticleList *i = grid[index(ic, nc)]; NULL != i; i = i->next){
        updateX(&i->p, delta_t, l);
      }
      
    }
  }
  
  moveParticles_LC(grid, nc, l, r_cut);
  
}

void compV_LC(Cell *grid, int *nc, real *l, real delta_t){
  
  int ic[DIM];
  for(ic[0] = 0; ic[0] < nc[0]; ic[0]++){
    for(ic[1] = 0; ic[1] < nc[1]; ic[1]++){
      
      for(ParticleList *i = grid[index(ic, nc)]; NULL != i; i = i->next){
        updateV(&i->p, delta_t);
      }
      
    }
  }
  
}

real compoutStatistic_LC(Cell *grid, int *nc){
  
  real e = 0;
  
  int ic[DIM];
  for(ic[0] = 0; ic[0] < nc[0]; ic[0]++){
    for(ic[1] = 0; ic[1] < nc[1]; ic[1]++){
      
      for(ParticleList *i = grid[index(ic, nc)]; NULL != i; i = i->next){
        
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

real timeIntegration_LC(real t, real delta_t, real t_end, Cell *grid, int *nc, real r_cut, real *l){
  
  real e = 0;
  
  // printf("t=%10f\n", t);
  compF_LC(grid, nc, r_cut, l);
  
  while(t < t_end){
    
    t += delta_t;
    // printf("t=%10f\n", t);
    
    compX_LC(grid, nc, l, delta_t, r_cut);
    compF_LC(grid, nc, r_cut, l);
    compV_LC(grid, nc, l, delta_t);
    
    e = compoutStatistic_LC(grid, nc);
    
  }
  
  return e;
  
}

void inputParameters_LC(int argc, char *argv[], real *delta_t, real *t_end, int *N, int *nc, real *l, real *r_cut){
  
  // delta_t, t_end, N, r_cut, l[0], l[1]
  // 0.001 0.001 50000 5 100000 100000
  if(argc >= 7){
    
    string str = argv[1];
    *delta_t = stod(str);
    
    str = argv[2];
    *t_end = stod(str);
    
    str = argv[3];
    *N = stoi(str);
    
    str = argv[4];
    *r_cut = stod(str);
    
    int l_start = 5;
    for(int d = 0; d < DIM; d++){
      
      str = argv[l_start + d];
      l[d] = stod(str);
      
      nc[d] = (int)floor(l[d] / *r_cut);
      
    }
    
  }
  
}

void initData_LC(int N, Cell *grid, int *nc, real *l, real r_cut){
  
  srand(time(NULL));
  
  int ic[DIM];
  for(int i = 0; i < N; i++){
    
    ParticleList *pl_p = (ParticleList*)malloc(sizeof(*pl_p));
    ParticleList p_pl = *pl_p;
    
    pl_p->p.id = i;
    pl_p->p.m = drand48() * m_range;
    
    for(int d = 0; d < DIM; d++){
      
      pl_p->p.x[d] = drand48() * l[d];
      if(pl_p->p.x[d] == l[d]){
        pl_p->p.x[d] = 0;
      }
      
      pl_p->p.v[d] = randfrom(-1.0 * v_range, 1.0 * v_range);
      
      if(pl_p->p.x[d] == l[d]){
        ic[d] = 0;
      }else{
        ic[d] = ((int)floor(pl_p->p.x[d] / r_cut) >= nc[d])? nc[d] - 1: (int)floor(pl_p->p.x[d] / r_cut);
      }
      
    }
    
    // printf("[Particle: id=%ld, m=%10f, x=(%10f, %10f), v=(%10f, %10f), ic=(%d, %d)]\n", pl_p->p.id, pl_p->p.m, pl_p->p.x[0], pl_p->p.x[1], pl_p->p.v[0], pl_p->p.v[1], ic[0], ic[1]);
    
    // printf("Start to insert particle[id=%d] into grid[%d=index(%d, %d)]\n", pl_p->p.id, index(ic, nc), ic[0], ic[1]);
    insertList(&grid[index(ic, nc)], pl_p);
    
  }
  
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
  
  int nc[DIM];
  int N, pnc;
  real l[DIM], r_cut;
  real delta_t, t_end;

  inputParameters_LC(argc, argv, &delta_t, &t_end, &N, nc, l, &r_cut);
  printf("N=%d, r_cut=%10f, l=(%10f, %10f), nc=(%d, %d)\n", N, r_cut, l[0], l[1], nc[0], nc[1]);
  
  pnc = 1;
  for(int d = 0; d < DIM; d++){
    pnc *= nc[d];
  }
  Cell *grid = (Cell*)malloc(pnc * sizeof(*grid));
  
  printf("Start to InitData\n");
  initData_LC(N, grid, nc, l, r_cut);
  printf("Finished to InitData\n");
  
  Timer t;
  t.tic();
  timeIntegration_LC(0, delta_t, t_end, grid, nc, r_cut, l);
  double time = t.toc();
  printf("Finished simulation, eslapse time = %10f sec(s).\n", time);
  
  freeLists_LC(grid, nc);
  free(grid);
  
  return 0;
  
}
