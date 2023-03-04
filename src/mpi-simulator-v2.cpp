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

  float minX, minY, maxX, maxY;
  minX = 1e30f;
  minY = 1e30f;
  maxX = -1e30f;
  maxY = -1e30f;

  for (int i = 0; i < particles.size(); i++)
  {
    minX = (particles[i].position.x < minX) ? particles[i].position.x : minX;
    minY = (particles[i].position.y < minY) ? particles[i].position.y : minY;
    maxX = (particles[i].position.x > maxX) ? particles[i].position.x : maxX;
    maxY = (particles[i].position.y > maxY) ? particles[i].position.y : maxY;
  }
  minX+=-0.1;
  minY+=-0.1;
  maxX+=0.1;
  maxY+=0.1;
  
  int sqrtNproc = int(sqrt(nproc));
  float gridEdgeDimX = (maxX-minX) / sqrtNproc;
  float gridEdgeDimY = (maxY-minY) / sqrtNproc;
  int xCoord = pid % sqrtNproc;
  int yCoord = pid / sqrtNproc;
  Vec2 threadBinTL = {
    minX+xCoord*gridEdgeDimX, 
    minY+yCoord*gridEdgeDimY};
  particles = boxFilter(threadBinTL, gridEdgeDimX, gridEdgeDimY, particles);
  printf("Thread %d has %ld particles; idx = (%d,%d); TL=(%f,%f)\n", pid, particles.size(), xCoord, yCoord, threadBinTL.x, threadBinTL.y);


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
