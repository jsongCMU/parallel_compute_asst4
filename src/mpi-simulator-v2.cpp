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

void simulateStep(const QuadTree &quadTree,
                  std::vector<Particle> &particles,
                  const StepParameters& params) {
  // Update particles for this thread
  std::vector<Particle> result = particles;

  for (int i = 0; i < particles.size(); i++)
  {
    Particle& curParticle = result[i];

    Vec2 force = Vec2(0.0f, 0.0f);
    std::vector<Particle> nearbyParticles;
    quadTree.getParticles(nearbyParticles, curParticle.position, params.cullRadius);
    for (const Particle& nearbyP : nearbyParticles)
      force += computeForce(curParticle, nearbyP, params.cullRadius);
    curParticle = updateParticle(curParticle, force, params.deltaTime);
  }

  particles.swap(result);
}

// Given all particles, return only particles that matter to current bin
std::vector<Particle> binFilter(const BinInfo &binInfo, const std::vector<Particle> &allParticles)
{
  std::vector<Particle> result;
  for(const Particle& particle : allParticles)
  {
    bool isXValid = (particle.position.x >= binInfo.binMin.x) && (particle.position.x < binInfo.binMax.x);
    bool isYValid = (particle.position.y >= binInfo.binMin.y) && (particle.position.y < binInfo.binMax.y);
    if(isXValid && isYValid)
      result.push_back(particle);
  }
  return result;
}

// Given all particles, compute minimum and maximum bounds
void getBounds(const std::vector<Particle> &particles, Vec2 &bmin, Vec2 &bmax, float offset=5)
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
void updateGridInfo(GridInfo &gridInfo, const std::vector<Particle> &particles, int nproc, float offset=0.1)
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
void updateBinInfo(BinInfo &binInfo, const GridInfo &gridInfo, const int pid)
{
    binInfo.col = pid % gridInfo.numCols;
    binInfo.row = pid / gridInfo.numCols;
    binInfo.binMin = {
        gridInfo.gridMin.x+binInfo.col*gridInfo.binDimX,
        gridInfo.gridMin.y+binInfo.row*gridInfo.binDimY};
    binInfo.binMax = {
        binInfo.binMin.x+gridInfo.binDimX,
        binInfo.binMin.y+gridInfo.binDimY
    };
}

std::vector<int> getRelevantNeighbors(const GridInfo &gridInfo, const BinInfo &binInfo, const float radius)
{
    // Expand boundaries by radius
    Vec2 bminExpand = {binInfo.binMin.x-radius, binInfo.binMin.y-radius};
    Vec2 bmaxExpand = {binInfo.binMax.x+radius, binInfo.binMax.y+radius};
    // Compute range of rows and columns that touch boundary
    int colStart = (bminExpand.x-gridInfo.gridMin.x)/gridInfo.binDimX;
    int colEnd = (bmaxExpand.x-gridInfo.gridMin.x)/gridInfo.binDimX;
    int rowStart = (bminExpand.y-gridInfo.gridMin.y)/gridInfo.binDimY;
    int rowEnd = (bmaxExpand.y-gridInfo.gridMin.y)/gridInfo.binDimY;
    // Cap
    colStart = (colStart < 0) ? 0 : colStart;
    colEnd = (colEnd > gridInfo.numCols-1) ? gridInfo.numCols-1 : colEnd;
    rowStart = (rowStart < 0) ? 0 : rowStart;
    rowEnd = (rowEnd > gridInfo.numRows-1) ? gridInfo.numRows-1 : rowEnd;
    // Accumulate
    std::vector<int> relevant_pids;
    for(int row = rowStart; row < rowEnd+1; row++)
    {
        for(int col = colStart; col < colEnd+1; col++)
        {
            if(col == binInfo.col && row == binInfo.row)
                continue;
            relevant_pids.push_back(row*gridInfo.numCols+col);
        }
    }
    return relevant_pids;

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

  // Starting configuration
  StartupOptions options = parseOptions(argc, argv);
  std::vector<Particle> allParticles;
  loadFromFile(options.inputFile, allParticles);
  StepParameters stepParams = getBenchmarkStepParams(options.spaceSize);

  // Setup variables
  std::vector<Particle> myParticles, relevParticles;
  GridInfo gridInfo;
  BinInfo binInfo;
  std::vector<int> relevantPIDs;
  Particle *recv_buffer;
  Particle *recv_buffer_start;
  int recv_buffer_rem;
  std::vector<MPI_Request> requests(nproc);
  std::vector<MPI_Status> statuses(nproc);

  // Don't change the timeing for totalSimulationTime.
  MPI_Barrier(MPI_COMM_WORLD);
  Timer totalSimulationTimer;

  for (int timestep = 0; timestep < options.numIterations; timestep++) {
    // Update grid and bin using allParticles
    updateGridInfo(gridInfo, allParticles, nproc, stepParams.cullRadius);
    updateBinInfo(binInfo, gridInfo, pid);
    // Compute myParticles
    myParticles = binFilter(binInfo, allParticles);
    
    // Get PIDs to send to / receive from
    relevantPIDs = getRelevantNeighbors(gridInfo, binInfo, stepParams.cullRadius);

    // Send info
    int tag_id = 0; // TODO: change later?
    for(int i = 0; i < relevantPIDs.size(); i++)
    {
        MPI_Isend(&myParticles[0], myParticles.size(), particleType, relevantPIDs[i], tag_id, MPI_COMM_WORLD, &requests[i]);
    }
    // Receive info
    MPI_Status comm_status;

    // Sync
    MPI_Barrier(MPI_COMM_WORLD);

    recv_buffer_start = new Particle[allParticles.size()];
    recv_buffer_rem = allParticles.size();
    recv_buffer = recv_buffer_start;
    
    for(int i = 0; i < relevantPIDs.size(); i++)
    {
      int numel;
      MPI_Recv(recv_buffer, recv_buffer_rem, particleType, relevantPIDs[i], tag_id, MPI_COMM_WORLD, &comm_status);
      MPI_Get_count(&comm_status, particleType, &numel);
      // Update recv buffer
      recv_buffer += numel;
      recv_buffer_rem -= numel;
    }
    MPI_Waitall(int(relevantPIDs.size()), &requests[0], &statuses[0]);

    // Copy recv buffer into relevParticles
    relevParticles.resize((allParticles.size() - recv_buffer_rem) + myParticles.size());
    std::copy(recv_buffer_start, recv_buffer, relevParticles.begin());
    
    // Add own particles
    std::copy(myParticles.begin(), myParticles.end(), relevParticles.end() - myParticles.size());

    delete[] recv_buffer_start;
    recv_buffer = nullptr;
    recv_buffer_start = nullptr;

    // Sync
    MPI_Barrier(MPI_COMM_WORLD);

    // Build quadtree and simulate
    QuadTree tree;
    QuadTree::buildQuadTree(relevParticles, tree);
    simulateStep(tree, myParticles, stepParams);

    // Update allParticles
    for(int i=0; i<nproc; i++)
    {
      if(i==pid)
        continue;
      MPI_Isend(&myParticles[0], myParticles.size(), particleType, i, tag_id, MPI_COMM_WORLD, &requests[i]);
    }

    // Sync
    MPI_Barrier(MPI_COMM_WORLD);
    
    recv_buffer_start = new Particle[allParticles.size()];
    recv_buffer_rem = allParticles.size();
    recv_buffer = recv_buffer_start;
    for(int i=0; i < nproc; i++)
    {
      if(i==pid)
        continue;
      int numel;
      MPI_Recv(recv_buffer, recv_buffer_rem, particleType, i, tag_id, MPI_COMM_WORLD, &comm_status);
      MPI_Get_count(&comm_status, particleType, &numel);
      recv_buffer += numel;
      recv_buffer_rem -= numel;
    }
    MPI_Waitall(pid, &requests[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(nproc-pid-1, &requests[pid+1], MPI_STATUSES_IGNORE);

    // Copy from recv_buffer_start to allParticles
    std::copy(recv_buffer_start, recv_buffer, allParticles.begin());
    
    // Add own particles
    std::copy(myParticles.begin(), myParticles.end(), allParticles.end() - myParticles.size());

    delete[] recv_buffer_start;
    recv_buffer = nullptr;
    recv_buffer_start = nullptr;

    // int myNum = (int) myParticles.size();
    // int allNum;
    // MPI_Allreduce(&myNum, &allNum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // printf("(%3d %d) \t| %4d %4d %d %d\n", pid, timestep, myParticles.size(), recv_buffer_rem, myParticles.size() == recv_buffer_rem, allNum);

    // Sync
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double totalSimulationTime = totalSimulationTimer.elapsed();

  if (pid == 0) {
    printf("total simulation time: %.6fs\n", totalSimulationTime);

    // allParticles is jumbled, so sort by id to fix
    std::sort(allParticles.begin(), allParticles.end(), [](const Particle& a, const Particle& b) {return a.id < b.id;});
    saveToFile(options.outputFile, allParticles);
  }

  MPI_Finalize();
}
