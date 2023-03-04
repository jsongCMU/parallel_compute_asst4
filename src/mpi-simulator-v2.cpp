#include "common.h"
#include "mpi.h"
#include "quad-tree.h"
#include "timing.h"

void simulateStep(const QuadTree &quadTree,
                  const std::vector<Particle> &particles,
                  std::vector<Particle> &newParticles, StepParameters params) {
  // TODO: paste your sequential implementation in Assignment 3 here.
  // (or you may also rewrite a new version)
}

// Given all particles, return only particles that matter to current bin
std::vector<Particle> boxFilter(Vec2 topLeft, float dimX, float dimY, std::vector<Particle> allParticles)
{
  Vec2 botRight = {topLeft.x+dimX, topLeft.y+dimY};
  std::vector<Particle> result;
  for(const Particle& particle : allParticles)
  {
    bool isXValid = (particle.position.x >= topLeft.x) && (particle.position.x < botRight.x);
    bool isYValid = (particle.position.y >= topLeft.y) && (particle.position.y < botRight.y);
    if(isXValid && isYValid)
      result.push_back(particle);
  }
  return result;
}

// Given all particles, compute top left and bottom right points
void getBounds(const std::vector<Particle> &particles, Vec2 &topLeft, Vec2 &botRight, float offset=0.01)
{
  Vec2 bmin(1e30f,1e30f);
  Vec2 bmax(-1e30f,-1e30f);
  for (int i = 0; i < particles.size(); i++)
  {
    bmin.x = (particles[i].position.x < bmin.x) ? particles[i].position.x : bmin.x;
    bmin.y = (particles[i].position.y < bmin.y) ? particles[i].position.y : bmin.y;
    bmax.x = (particles[i].position.x > bmax.x) ? particles[i].position.x : bmax.x;
    bmax.y = (particles[i].position.y > bmax.y) ? particles[i].position.y : bmax.y;
  }
  // Need padding for particles right on bounding box
  bmin.x+=-offset;
  bmin.y+=-offset;
  bmax.x+=offset;
  bmax.y+=offset;
  // Update
  topLeft = bmin;
  botRight = bmax;
}

int main(int argc, char *argv[]) {
  int pid;
  int nproc;
  // Hard-code since coding is hard TODO: Not this lol
  int thread0_neigh[] = {1,4,5};
  int thread1_neigh[] = {0,2,4,5,6};
  int thread2_neigh[] = {1,3,5,6,7};
  int thread3_neigh[] = {2,6,7};
  int thread4_neigh[] = {0,1,5,8,9};
  int thread5_neigh[] = {0,1,2,4,6,8,9,10};
  int thread6_neigh[] = {1,2,3,5,7,9,10,11};
  int thread7_neigh[] = {2,3,6,10,11};
  int thread8_neigh[] = {4,5,9,12,13};
  int thread9_neigh[] = {4,5,6,8,10,12,13,14};
  int thread10_neigh[] = {5,6,7,9,11,13,14,15};
  int thread11_neigh[] = {6,7,10,14,15};
  int thread12_neigh[] = {8,9,13};
  int thread13_neigh[] = {8,9,10,12,14};
  int thread14_neigh[] = {9,10,11,13,15};
  int thread15_neigh[] = {10,11,14};
  int *relevantThreads[] = {thread0_neigh,thread1_neigh,thread2_neigh,thread3_neigh,thread4_neigh,thread5_neigh,thread6_neigh,thread7_neigh,thread8_neigh,thread9_neigh,thread10_neigh,thread11_neigh,thread12_neigh,thread13_neigh,thread14_neigh,thread15_neigh};

  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // Get total number of processes specificed at start of run
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  StartupOptions options = parseOptions(argc, argv);

  std::vector<Particle> particles, newParticles;
  loadFromFile(options.inputFile, particles);

  // Get bounds based on particles
  Vec2 bmin, bmax;
  getBounds(particles, bmin, bmax);
  
  // Get bin dimensions for this thread
  int sqrtNproc = int(sqrt(nproc));
  float gridEdgeDimX = (bmax.x-bmin.x) / sqrtNproc;
  float gridEdgeDimY = (bmax.y-bmin.y) / sqrtNproc;
  int xCoord = pid % sqrtNproc;
  int yCoord = pid / sqrtNproc;
  Vec2 threadBinTL = {
    bmin.x+xCoord*gridEdgeDimX,
    bmin.y+yCoord*gridEdgeDimY};

  // Get particles for this thread
  particles = boxFilter(threadBinTL, gridEdgeDimX, gridEdgeDimY, particles);

  StepParameters stepParams = getBenchmarkStepParams(options.spaceSize);

  // Don't change the timeing for totalSimulationTime.
  MPI_Barrier(MPI_COMM_WORLD);
  Timer totalSimulationTimer;
  for (int i = 0; i < options.numIterations; i++) {
    // The following code is just a demonstration.
    QuadTree tree;
    QuadTree::buildQuadTree(particles, tree);
    simulateStep(tree, particles, newParticles, stepParams);
    particles.swap(newParticles);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double totalSimulationTime = totalSimulationTimer.elapsed();

  if (pid == 0) {
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    saveToFile(options.outputFile, particles);
  }

  MPI_Finalize();
}
