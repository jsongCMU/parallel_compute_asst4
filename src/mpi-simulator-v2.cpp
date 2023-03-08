#include "common.h"
#include "mpi.h"
#include "quad-tree.h"
#include "timing.h"
#include <string>

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
                  const std::vector<Particle> &particles,
                  std::vector<Particle> &newParticles, StepParameters params) {
  // TODO: paste your sequential implementation in Assignment 3 here.
  // (or you may also rewrite a new version)
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
void getBounds(const std::vector<Particle> &particles, Vec2 &bmin, Vec2 &bmax, float offset=0.01)
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
void updateGridInfo(GridInfo &gridInfo, const std::vector<Particle> &particles, int nproc)
{
    // Update bounds
    getBounds(particles, gridInfo.gridMin, gridInfo.gridMax);
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

std::vector<int> getRelevantNieghbors(const GridInfo &gridInfo, const BinInfo &binInfo, const float radius)
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

  
  nproc = 16;

  // Compute overall grid info
  GridInfo gridInfo;
  updateGridInfo(gridInfo, particles, nproc);

  StepParameters stepParams = getBenchmarkStepParams(options.spaceSize);

  for(pid = 0; pid < nproc; pid++)
  {
    // Compute bin specific info
    BinInfo binInfo;
    updateBinInfo(binInfo, gridInfo, pid);

    // Get particles for this thread
    particles = binFilter(binInfo, particles);

    // Get relevant, neighboring bins
    std::vector<int> relevantPIDs = getRelevantNieghbors(gridInfo, binInfo, stepParams.cullRadius);
    std::string toPrint;
    for(int i=0; i<relevantPIDs.size(); i++)
    {
        toPrint = toPrint + std::to_string(relevantPIDs[i]) + ", ";
    }
    printf("PID=%d: %s\n", pid, toPrint.c_str());
  }

  MPI_Finalize();
}
