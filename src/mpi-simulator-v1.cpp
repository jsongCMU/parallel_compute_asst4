#include "common.h"
#include "mpi.h"
#include "quad-tree.h"
#include "timing.h"

void simulateStep(const QuadTree &quadTree,
                  const std::vector<Particle> &particles,
                  std::vector<Particle> &newParticles, StepParameters params) {
  // TODO: paste your sequential implementation in Assignment 3 here.
  // (or you may also rewrite a new version)
  for (int i = 0; i < particles.size(); i++)
  {
    Particle curParticle = particles[i];

    Vec2 force = Vec2(0.0f, 0.0f);
    std::vector<Particle> nearbyParticles;
    quadTree.getParticles(nearbyParticles, curParticle.position, params.cullRadius);

    for (const Particle& nearbyP : nearbyParticles)
    force += computeForce(curParticle, nearbyP, params.cullRadius);

    newParticles[i] = updateParticle(curParticle, force, params.deltaTime);
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
  if (pid == 0) {
    loadFromFile(options.inputFile, particles);
  }

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

  // Don't change the timeing for totalSimulationTime.
  MPI_Barrier(MPI_COMM_WORLD);
  Timer totalSimulationTimer;
  
  // Test particle sending: mpirun -n 4 nbody-release-v1  -n 50000 -i 5 -in src/benchmark-files/random-50000-init.txt -s 500.0 -o logs/random-50000.txt
  if(pid==0)
  {
    // Send particle as test
    MPI_Request request_status;
    for(int i = 0; i < 5; i++)
    {
      MPI_Isend(&particles[i], 1, particleType, 1, 0, MPI_COMM_WORLD, &request_status);
      printf("Sending particle %d with mass %f, p=(%f,%f), v=(%f,%f)\n",
        particles[i].id, particles[i].mass, particles[i].position.x, particles[i].position.y,
        particles[i].velocity.x, particles[i].velocity.y);
    }
  }
  if(pid==1)
  {
    MPI_Status comm_status;
    Particle p;
    for(int i = 0; i < 5; i++)
    {
      MPI_Recv(&p, 1, particleType, 0, 0, MPI_COMM_WORLD, &comm_status);
      printf("Received particle %d with mass %f, p=(%f,%f), v=(%f,%f)\n",
        p.id, p.mass, p.position.x, p.position.y, p.velocity.x, p.velocity.y);
    }
  }

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
