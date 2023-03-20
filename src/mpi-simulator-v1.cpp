#include "common.h"
#include "mpi.h"
#include "quad-tree.h"
#include "timing.h"

inline void simulateStep(const QuadTree<MasslessParticle> &quadTree,
                  std::vector<MasslessParticle> &particles,
                  const StepParameters& params) {
  // Update particles for this thread
  std::vector<MasslessParticle> result = particles;

  for (int i = 0; i < particles.size(); i++)
  {
    MasslessParticle& curParticle = result[i];

    Vec2 force = Vec2(0.0f, 0.0f);
    std::vector<MasslessParticle> nearbyParticles;
    quadTree.getParticles(nearbyParticles, curParticle.position, params.cullRadius);
    for (const MasslessParticle& nearbyP : nearbyParticles)
      force += computeForce(curParticle, nearbyP, params.cullRadius);
    curParticle = updateParticle(curParticle, force, params.deltaTime);
  }

  particles.swap(result);
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

  std::vector<Particle> allParticlesMass;
  loadFromFile(options.inputFile, allParticlesMass);

  // Create massless particles
  std::vector<float> massArray;
  std::vector<MasslessParticle> allParticles;
  for(int i=0; i<allParticlesMass.size(); i++)
  {
    massArray.push_back(allParticlesMass[i].mass);
    MasslessParticle particle;
    particle.position.x = allParticlesMass[i].position.x;
    particle.position.y = allParticlesMass[i].position.y;
    particle.velocity.x = allParticlesMass[i].velocity.x;
    particle.velocity.y = allParticlesMass[i].velocity.y;
    particle.id = allParticlesMass[i].id;
    allParticles.push_back(particle);
  }
  MasslessParticle::massArray = massArray.data();

  StepParameters stepParams = getBenchmarkStepParams(options.spaceSize);

  // Create struct for Particle (https://hpc-tutorials.llnl.gov/mpi/derived_data_types/struct_examples/)
  // Particle is 1 int (id) and 4 floats (position, velocity)
  MPI_Aint offsets[2], lowerbound, extent;
  MPI_Datatype masslessParticleType, oldtypes[2];
  int blockcounts[2];
  // 1 Int
  offsets[0] = 0;
  oldtypes[0] = MPI_INT;
  blockcounts[0] = 1;
  // 5 floats
  MPI_Type_get_extent(MPI_INT, &lowerbound, &extent);
  offsets[1] = 1 * extent;
  oldtypes[1] = MPI_FLOAT;
  blockcounts[1] = 4;
  // Bind
  MPI_Type_create_struct(2, blockcounts, offsets, oldtypes, &masslessParticleType);
  MPI_Type_commit(&masslessParticleType);

  // Compute displacements and counts for allgatherv operation
  int particleDisplacements[nproc];
  int particleCounts[nproc];
  {
    const int minNumParticles = allParticles.size()/nproc;
    const int numParticlesRem = allParticles.size() % minNumParticles;
    int particleToInc = numParticlesRem;
    for(int i=0; i<nproc; i++)
    {
        int curThreadParticles = particleToInc ? (minNumParticles+1) : minNumParticles;
        particleToInc = (particleToInc>0) ? particleToInc-1 : 0;
        particleDisplacements[i] = (i<numParticlesRem) ? (minNumParticles*i+i) : (minNumParticles*i+numParticlesRem);
        particleCounts[i] = curThreadParticles;
    }
  }
  // Set up new particles with range of interest
  std::vector<MasslessParticle> myParticles;
  myParticles.insert(myParticles.begin(), allParticles.begin()+particleDisplacements[pid], allParticles.begin()+particleDisplacements[pid]+particleCounts[pid]);
  
  // Set up tree
  QuadTree<MasslessParticle> tree;

  // Don't change the timeing for totalSimulationTime.
  MPI_Barrier(MPI_COMM_WORLD);
  Timer totalSimulationTimer;
  for (int i = 0; i < options.numIterations; i++) {
    // Build quadtree of all particles
    QuadTree<MasslessParticle>::buildQuadTree(allParticles, tree);

    // Update subset of particles
    simulateStep(tree, myParticles, stepParams);
    
    // Share and get updates
    MPI_Allgatherv(
      &myParticles[0], particleCounts[pid], masslessParticleType, // Send
      &allParticles[0], particleCounts, particleDisplacements, masslessParticleType, // Recieve
      MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double totalSimulationTime = totalSimulationTimer.elapsed();

  if (pid == 0) {
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    // Convert to particles with mass
    allParticlesMass.clear();
    for(int i=0; i<allParticles.size(); i++)
    {
      Particle particle;
      particle.id = allParticles[i].id;
      particle.mass = MasslessParticle::getMass(allParticles[i]);
      particle.position.x = allParticles[i].position.x;
      particle.position.y = allParticles[i].position.y;
      particle.velocity.x = allParticles[i].velocity.x;
      particle.velocity.y = allParticles[i].velocity.y;
      allParticlesMass.push_back(particle);
    }
    saveToFile(options.outputFile, allParticlesMass);
  }

  MPI_Finalize();
}
