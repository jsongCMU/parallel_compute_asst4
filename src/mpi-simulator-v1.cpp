#include "common.h"
#include "mpi.h"
#include "quad-tree.h"
#include "timing.h"

void simulateStep(const QuadTree &quadTree,
                  std::vector<Particle> &particles, 
                  StepParameters params) {
  // Update particles for this thread
  for (int i = 0; i < particles.size(); i++)
  {
    Particle &curParticle = particles[i];

    Vec2 force = Vec2(0.0f, 0.0f);
    std::vector<Particle> nearbyParticles;
    quadTree.getParticles(nearbyParticles, curParticle.position, params.cullRadius);
    for (const Particle& nearbyP : nearbyParticles)
      force += computeForce(curParticle, nearbyP, params.cullRadius);
    curParticle = updateParticle(curParticle, force, params.deltaTime);
  }
}

int main(int argc, char *argv[]) {
  int pid;
  int nproc;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // Get total number of processes specificed at start of run
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  StartupOptions options = parseOptions(argc, argv);

  std::vector<Particle> particles, newParticles;
  loadFromFile(options.inputFile, particles);

  StepParameters stepParams = getBenchmarkStepParams(options.spaceSize);

  // Create struct for Particle (https://hpc-tutorials.llnl.gov/mpi/derived_data_types/struct_examples/)
  // Particle is 1 int (id) and 5 floats (mass, position, velocity)
  MPI_Aint offsets[2], lowerbound, extent;
  MPI_Datatype particleType, oldtypes[2];
  int blockcounts[2];
  // 1 Int
  offsets[0] = 0;
  oldtypes[0] = MPI_INT;
  blockcounts[0] = 1;
  // 5 floats
  MPI_Type_get_extent(MPI_INT, &lowerbound, &extent);
  offsets[1] = 1 * extent;
  oldtypes[1] = MPI_FLOAT;
  blockcounts[1] = 5;
  // Bind
  MPI_Type_create_struct(2, blockcounts, offsets, oldtypes, &particleType);
  MPI_Type_commit(&particleType);

  // Compute displacements and counts for allgatherv operation
  int minNumParticles = particles.size()/nproc;
  int particleDisplacements[nproc];
  int particleCounts[nproc];
  for(int i=0; i<nproc; i++)
  {
    particleDisplacements[i] = minNumParticles*i;
    particleCounts[i] = minNumParticles;
  }
  particleCounts[nproc-1] = particles.size()-particleDisplacements[nproc-1];
  // Set up new particles with range of interest
  newParticles.insert(newParticles.begin(), particles.begin()+particleDisplacements[pid], particles.begin()+particleDisplacements[pid]+particleCounts[pid]);
  
  // Don't change the timeing for totalSimulationTime.
  MPI_Barrier(MPI_COMM_WORLD);
  Timer totalSimulationTimer;
  for (int i = 0; i < options.numIterations; i++) {
    // Build quadtree of all particles
    QuadTree tree;
    QuadTree::buildQuadTree(particles, tree);
    // Update subset of particles
    simulateStep(tree, newParticles, stepParams);
    // Share and get updates
    MPI_Allgatherv(
      &newParticles[0], particleCounts[pid], particleType, // Send
      &particles[0], particleCounts, particleDisplacements, particleType, // Recieve
      MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double totalSimulationTime = totalSimulationTimer.elapsed();

  if (pid == 0) {
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    saveToFile(options.outputFile, particles);
  }

  MPI_Finalize();
}
