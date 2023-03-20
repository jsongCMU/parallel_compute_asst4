#include "common.h"
#include "mpi.h"
#include "quad-tree.h"
#include "timing.h"
#include <string>
#include <bits/stdc++.h>

struct GridInfo{
    Vec2 gridMin;
    Vec2 gridMax;
    float binDimX;
    float binDimY;
    int numCols;
    int numRows;
};

struct BinInfo{
    int col;
    int row;
    Vec2 binMin;
    Vec2 binMax;
};

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

// Given all particles, return particles are owned and relevant to current bin
inline void binFilter(const BinInfo &binInfo, const std::vector<MasslessParticle> &allParticles, std::vector<MasslessParticle> &myParticles, std::vector<MasslessParticle> &relevParticles, const float radius)
{
  myParticles.clear();
  relevParticles.clear();
  bool isXValid, isYValid;
  for(const MasslessParticle& particle : allParticles)
  {
    isXValid = (particle.position.x >= binInfo.binMin.x) && (particle.position.x < binInfo.binMax.x);
    isYValid = (particle.position.y >= binInfo.binMin.y) && (particle.position.y < binInfo.binMax.y);
    if(isXValid && isYValid)
    {
      myParticles.push_back(particle);
      relevParticles.push_back(particle);
      continue;
    }
    isXValid = (particle.position.x >= binInfo.binMin.x-radius) && (particle.position.x < binInfo.binMax.x+radius);
    isYValid = (particle.position.y >= binInfo.binMin.y-radius) && (particle.position.y < binInfo.binMax.y+radius);
    if(isXValid && isYValid)
    {
      relevParticles.push_back(particle);
      continue;
    }
  }
}

// Given all particles, compute minimum and maximum bounds
inline void getBounds(const std::vector<MasslessParticle> &particles, Vec2 &bmin, Vec2 &bmax, float offset)
{
  Vec2 bmin_temp(1e30f,1e30f);
  Vec2 bmax_temp(-1e30f,-1e30f);
  for (int i = 0; i < particles.size(); i++)
  {
    bmin_temp.x = (particles[i].position.x < bmin_temp.x) ? particles[i].position.x : bmin_temp.x;
    bmin_temp.y = (particles[i].position.y < bmin_temp.y) ? particles[i].position.y : bmin_temp.y;
    bmax_temp.x = (particles[i].position.x > bmax_temp.x) ? particles[i].position.x : bmax_temp.x;
    bmax_temp.y = (particles[i].position.y > bmax_temp.y) ? particles[i].position.y : bmax_temp.y;
  }
  // Need padding for particles right on bounding box
  bmin_temp.x+=-offset;
  bmin_temp.y+=-offset;
  bmax_temp.x+=offset;
  bmax_temp.y+=offset;
  // Update
  bmin = bmin_temp;
  bmax = bmax_temp;
}

// Update grid info
inline void updateGridInfo(GridInfo &gridInfo, const std::vector<MasslessParticle> &particles, int nproc, float offset=0.01)
{
    // Update bounds
    getBounds(particles, gridInfo.gridMin, gridInfo.gridMax, offset);
    // Update number of columns and rows
    int sqrtNproc = int(sqrt(nproc));
    gridInfo.numRows = sqrtNproc;
    gridInfo.numCols = sqrtNproc;
    // Update per bin x and y dims
    gridInfo.binDimX = (gridInfo.gridMax.x-gridInfo.gridMin.x) / sqrtNproc;
    gridInfo.binDimY = (gridInfo.gridMax.y-gridInfo.gridMin.y) / sqrtNproc;
}

// Update bin info
inline void updateBinInfo(BinInfo &binInfo, const GridInfo &gridInfo, const int pid)
{
    binInfo.col = pid % gridInfo.numCols;
    binInfo.row = pid / gridInfo.numCols;
    binInfo.binMin = {gridInfo.gridMin.x, gridInfo.gridMin.y};
    for(int i=0; i<binInfo.col; i++)
        binInfo.binMin.x+=gridInfo.binDimX;
    for(int i=0; i<binInfo.row; i++)
        binInfo.binMin.y+=gridInfo.binDimY;
    binInfo.binMax = {binInfo.binMin.x+gridInfo.binDimX, binInfo.binMin.y+gridInfo.binDimY
    };
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

  // Starting configuration
  StartupOptions options = parseOptions(argc, argv);
  std::vector<Particle> allParticlesMass;
  loadFromFile(options.inputFile, allParticlesMass);
  StepParameters stepParams = getBenchmarkStepParams(options.spaceSize);

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

  // Setup variables
  std::vector<MasslessParticle> myParticles, relevParticles;
  GridInfo gridInfo;
  BinInfo binInfo;
  MasslessParticle *recv_buffer;
  MasslessParticle *const recv_buffer_start = new MasslessParticle[allParticles.size()];
  int recv_buffer_rem;
  std::vector<MPI_Request> requests(nproc);
  std::vector<MPI_Status> statuses(nproc);
  MPI_Status comm_status;
  int numel;
  QuadTree<MasslessParticle> tree;
  const int tag_id_relevant = 0;
  const int tag_id_all = 1;

  // Don't change the timeing for totalSimulationTime.
  MPI_Barrier(MPI_COMM_WORLD);
  Timer totalSimulationTimer;

  for (int timestep = 0; timestep < options.numIterations; timestep++) {
    // Update grid and bin using allParticles
    updateGridInfo(gridInfo, allParticles, nproc);
    updateBinInfo(binInfo, gridInfo, pid);
    // Compute myParticles and relevParticles
    binFilter(binInfo, allParticles, myParticles, relevParticles, stepParams.cullRadius);

    // Build quadtree and simulate
    QuadTree<MasslessParticle>::buildQuadTree(relevParticles, tree);
    simulateStep(tree, myParticles, stepParams);

    // Update allParticles
    for(int i=0; i<nproc; i++)
    {
      if(i==pid)
        continue;
      MPI_Isend(&myParticles[0], myParticles.size(), masslessParticleType, i, tag_id_all, MPI_COMM_WORLD, &requests[i]);
    }
    recv_buffer_rem = allParticles.size();
    recv_buffer = recv_buffer_start;
    for(int i=0; i < nproc; i++)
    {
      if(i==pid)
        continue;
      MPI_Recv(recv_buffer, recv_buffer_rem, masslessParticleType, i, tag_id_all, MPI_COMM_WORLD, &comm_status);
      MPI_Get_count(&comm_status, masslessParticleType, &numel);
      recv_buffer += numel;
      recv_buffer_rem -= numel;
    }
    MPI_Waitall(pid, &requests[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(nproc-pid-1, &requests[pid+1], MPI_STATUSES_IGNORE);

    // Copy from recv_buffer_start to allParticles
    std::move(recv_buffer_start, recv_buffer, allParticles.begin());
    
    // Add own particles
    std::move(myParticles.begin(), myParticles.end(), allParticles.end() - myParticles.size());
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double totalSimulationTime = totalSimulationTimer.elapsed();
  
  delete[] recv_buffer_start;

  if (pid == 0) {
    printf("total simulation time: %.6fs\n", totalSimulationTime);

    // allParticles is jumbled, so sort by id to fix
    std::sort(allParticles.begin(), allParticles.end(), [](const MasslessParticle& a, const MasslessParticle& b) {return a.id < b.id;});
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
