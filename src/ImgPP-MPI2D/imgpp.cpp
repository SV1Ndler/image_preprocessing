/******************************************************************
 *  mpi_pipeline.cpp
 *  Сборка:   mpic++ -O3 mpi_pipeline.cpp -o mpi_pipeline
 *  Запуск:   mpirun -np 4 ./mpi_pipeline
 ******************************************************************/
#include <mpi.h> 
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "pp/mat/mat.hpp"
#include "pp/pixel/pixel.hpp"
#include "pp/transformation/transformation.hpp"

using pp::Mat;

constexpr int kPixelSize = 3; // r-g-b    

constexpr std::size_t kRows   = 150;
constexpr std::size_t kColls   = 150;
constexpr std::size_t kBorderSize   = 3;

enum { N, W, S, E };


inline std::size_t LocalLen (std::size_t global, int coord, int dims)
{
    const int base  = global / dims;
    const int extra = global % dims;
    return base + (coord < extra ? 1 : 0);
}
inline std::size_t LocalOff (std::size_t global, int coord, int dims)
{
    const int base  = global / dims;
    const int extra = global % dims;

    return (coord < extra)
           ? coord * (base + 1)
           : extra * (base + 1) + (coord - extra) * base;
}

void FillNeighboursCoords(MPI_Comm cart, int* nbr)
{
    MPI_Cart_shift(cart, 0, 1, &nbr[N], &nbr[S]);
    MPI_Cart_shift(cart, 1, 1, &nbr[W], &nbr[E]);
}

void ExchangeGhostCells(Mat& local,
                   const int* nbr,             // соседи или MPI_PROC_NULL
                   MPI_Datatype RowType,
                   MPI_Datatype ColType)
{
    MPI_Request req[4];
    std::size_t start;

    MPI_Isend(local.GetPtr(0, kBorderSize), 1, ColType,
              nbr[W],   10, MPI_COMM_WORLD, &req[0]);

    MPI_Irecv(local.GetPtr(0, 0), 1, ColType,
              nbr[W],   10, MPI_COMM_WORLD, &req[1]);

    start = local.cols - 2*kBorderSize;

    MPI_Isend(local.GetPtr(0, start), 1, ColType,
              nbr[E], 10, MPI_COMM_WORLD, &req[2]);

    MPI_Irecv(local.GetPtr(0, start + kBorderSize), 1, ColType,
              nbr[E], 10, MPI_COMM_WORLD, &req[3]);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);


    /* -------------------------- */
    MPI_Isend(local.GetPtr(kBorderSize, 0), kBorderSize, RowType,
              nbr[N],   20, MPI_COMM_WORLD, &req[0]);

    MPI_Irecv(local.GetPtr(0, 0), kBorderSize, RowType,
              nbr[N],   20, MPI_COMM_WORLD, &req[1]);

    start = local.rows - 2*kBorderSize;

    MPI_Isend(local.GetPtr(start,                     0), kBorderSize, RowType,
              nbr[S], 20, MPI_COMM_WORLD, &req[2]);

    MPI_Irecv(local.GetPtr(start + kBorderSize,       0), kBorderSize, RowType,
              nbr[S], 20, MPI_COMM_WORLD, &req[3]);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}

void InitImg(Mat& src, int rank, int size);


bool CmpImg(const Mat& img, const Mat& block, int coords[2], int dims  [2]);

void InitImg(Mat& local, int coords[2], int dims  [2]);


int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);     // заполняет dims[0] * dims[1] == size
    int periods[2] = {0, 0};            // без периодичности
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int cartRank;
    int coords[2];
    MPI_Comm_rank(cart, &cartRank);
    MPI_Cart_coords(cart, cartRank, 2, coords);   // coords[0] = row(proc), [1] = col(proc)

    int nbr[4];
    FillNeighboursCoords(cart, nbr);

    const std::size_t rowsLocal = LocalLen (kRows,  coords[0], dims[0]);
    const std::size_t colsLocal = LocalLen (kColls, coords[1], dims[1]);

    const std::size_t rowsWithGhostCells = rowsLocal + 2*kBorderSize;
    const std::size_t colsWithGhostCells = colsLocal + 2*kBorderSize;

    const int kRowStrideBytes = colsWithGhostCells * kPixelSize;


    MPI_Datatype RowType;
    MPI_Type_contiguous(
        colsWithGhostCells * kPixelSize,
        MPI_UINT8_T,
        &RowType);
    MPI_Type_commit(&RowType);

    MPI_Datatype ColType;
    MPI_Type_vector(
        rowsWithGhostCells,
        kPixelSize*kBorderSize,
        kRowStrideBytes,
        MPI_UINT8_T,
        &ColType);
    MPI_Type_commit(&ColType);

    double t;

    pp::Mat correctImg;
    {
        pp::Mat img1(kRows, kColls);
        pp::InitImg(img1);
        img1 = img1.CopyWithBorder(kBorderSize);
        pp::Mat img2(img1.rows, img1.cols, img1.borderSize);
        
        t = MPI_Wtime();
    
        
        pp::Mat* img = &img1;
        pp::Mat* imgNew = &img2;

        img->MakeMirrorBorder(kBorderSize);
        pp::DoFilter(*img, *imgNew, pp::MedianFilterProc(7));
        std::swap(img, imgNew);
        
        img->MakeMirrorBorder(kBorderSize);
        pp::DoFilter(*img, *imgNew, pp::MeanFilterProc(7));
        std::swap(img, imgNew);

        img->MakeMirrorBorder(kBorderSize);
        pp::DoFilter(*img, *imgNew, pp::SobelFilterProc());
        std::swap(img, imgNew);

        img->MakeMirrorBorder(kBorderSize);
        pp::DoFilter(*img, *imgNew, pp::ThresholdFilterProc(20));
        std::swap(img, imgNew);
       
        correctImg.swap(*img);

        t = MPI_Wtime() - t;
        if(rank == 0)
            std::printf("MPI(%d//%d ranks) (s)elapsed: %.3f s\n", rank, size, t);
    }

    Mat cur(rowsWithGhostCells, colsWithGhostCells, kBorderSize);
    Mat nxt(cur.rows, cur.cols, kBorderSize);
    
    ::InitImg(cur, coords, dims);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();                  

    auto runStage = [&](auto&& proc)
    {
        cur.MakeMirrorBorder(kBorderSize); 
        ExchangeGhostCells(cur, nbr, RowType, ColType);

        pp::DoFilter(cur, nxt, proc);
        std::swap(cur, nxt);
    };

    runStage(pp::MedianFilterProc(7));
    runStage(pp::MeanFilterProc(7));
    runStage(pp::SobelFilterProc());
    runStage(pp::ThresholdFilterProc(20));

    double dt = MPI_Wtime() - t0;
    double dtMax;
    MPI_Reduce(&dt, &dtMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    bool resultLocal = CmpImg(correctImg, cur, coords, dims);
    bool resultGlobal = false;

    MPI_Reduce(
        &resultLocal,
        &resultGlobal,
        1,
        MPI_CXX_BOOL,
        MPI_LAND,
        0,
        MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("\n");
        std::printf("MPI(%d//%d ranks) (p)elapsed: %.3f s\n", rank, size, dt);
        std::printf("MPI(%d//%d ranks) result: %d\n", rank, size, resultGlobal);
    }

    MPI_Type_free(&RowType);
    MPI_Type_free(&ColType);
    MPI_Finalize();
    return 0;
}

void InitImg(Mat& local, int coords[2], int dims  [2]) {
    const std::size_t rowOffset = LocalOff(kRows,  coords[0], dims[0]);
    const std::size_t colOffset = LocalOff(kColls, coords[1], dims[1]);

    const std::size_t b = local.borderSize;           // = kBorderSize

    for (std::size_t row = b; row < local.rows - b; ++row) {
        for (std::size_t col = b; col < local.cols - b; ++col) {

            const auto globalRow = (row - b) + rowOffset;
            const auto globalCol = (col - b) + colOffset;
            
            auto pixel = local.GetPixel(row, col);
            pixel.r() = (653 + globalRow * kColls + globalCol) % 256;
            pixel.g() = (1754 + globalRow * kColls + globalCol) % 256;
            pixel.b() = (1999 + globalRow * kColls + globalCol) % 256;
        }
    }
}

bool CmpImg(const Mat& img, const Mat& block, int coords[2], int dims  [2]) {
    const std::size_t rowOff = LocalOff(kRows,  coords[0], dims[0]);
    const std::size_t colOff = LocalOff(kColls, coords[1], dims[1]);

    const std::size_t b = block.borderSize;

    for (std::size_t row = b; row < block.rows - b; ++row) {
        for (std::size_t col = b; col < block.cols - b; ++col) {

            const std::size_t gr = rowOff + row - b;
            const std::size_t gc = colOff + col - b;

            const auto& pixelImg = img  .GetPixel(gr + b, gc + b);
            const auto& pixelBlock= block.GetPixel(row,  col);

            if (pixelImg.r() != pixelBlock.r() || pixelImg.g() != pixelBlock.g() || pixelImg.b() != pixelBlock.b())
                return false;
        }
    }
    return true;
}