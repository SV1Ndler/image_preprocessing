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


inline std::size_t GetRowsFor(int rank, int size)
{
    const int base = kRows / size;
    const int extra = kRows % size;
    return base + (rank < extra ? 1 : 0);
}

inline std::size_t GetRowOffsetFor(int rank, int size)
{
    const int base = kRows / size;
    const int extra = kRows % size;

    if(rank < extra) {
        return static_cast<std::size_t>(rank) * (base+1);
    } else {
        return static_cast<std::size_t>(extra) * (base+1) +
               static_cast<std::size_t>(rank - extra) * base;
    }
}

void ExchangeGhostCells(Mat& local, int rank, int size, const MPI_Datatype& RowType)
{
    MPI_Request req[4];

    const int top = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int bottom = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL; 


    MPI_Isend(local.GetPtr(kBorderSize,      0), kBorderSize, RowType,
              top,   0, MPI_COMM_WORLD, &req[0]);

    MPI_Irecv(local.GetPtr(0,                         0), kBorderSize, RowType,
              top,   0, MPI_COMM_WORLD, &req[1]);

    const std::size_t start = local.rows - 2*kBorderSize;

    MPI_Isend(local.GetPtr(start,                     0), kBorderSize, RowType,
              bottom, 0, MPI_COMM_WORLD, &req[2]);

    MPI_Irecv(local.GetPtr(start + kBorderSize,       0), kBorderSize, RowType,
              bottom, 0, MPI_COMM_WORLD, &req[3]);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}

void InitImg(Mat& src, int rank, int size);


bool CmpImg(Mat& img, Mat& stride, int rank, int size);


int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    const std::size_t rowsLocal = GetRowsFor(rank, size);
    const std::size_t rowsWithGhostCells = rowsLocal + 2*kBorderSize;
    const std::size_t colsWithGhostCells = kColls + 2*kBorderSize;

    Mat cur(rowsWithGhostCells, colsWithGhostCells, kBorderSize);
    Mat nxt(cur.rows, cur.cols, kBorderSize);
    
    ::InitImg(cur, rank, size);

    MPI_Datatype GhostRowType;
    MPI_Type_contiguous(
        (kColls + 2*kBorderSize) * kPixelSize,
        MPI_UINT8_T,
        &GhostRowType);
    MPI_Type_commit(&GhostRowType); 

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();                  

    auto runStage = [&](auto&& proc)
    {
        cur.MakeMirrorBorder(kBorderSize); 
        ExchangeGhostCells(cur, rank, size, GhostRowType);

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

    bool resultLocal = CmpImg(correctImg, cur, rank, size);
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

    MPI_Type_free(&GhostRowType);
    MPI_Finalize();
    return 0;
}

void InitImg(Mat& src, int rank, int size) {
    const auto rowOffset = GetRowOffsetFor(rank, size);

    for (std::size_t row = src.borderSize; row < src.rows - src.borderSize; ++row) {
        for (std::size_t col = src.borderSize; col < src.cols - src.borderSize; ++col) {
            const auto globalRow = row + rowOffset - src.borderSize;
            const auto globalCol = col - kBorderSize;
            
            auto pixel = src.GetPixel(row, col);
            pixel.r() = (653 + globalRow * kColls + globalCol) % 256;
            pixel.g() = (1754 + globalRow * kColls + globalCol) % 256;
            pixel.b() = (1999 + globalRow * kColls + globalCol) % 256;
        }
    }
}

bool CmpImg(Mat& img, Mat& stride, int rank, int size) {
    const auto rowOffset = GetRowOffsetFor(rank, size);

    for (std::size_t row = stride.borderSize; row < stride.rows - stride.borderSize; ++row) {
        for (std::size_t col = stride.borderSize; col < stride.cols - stride.borderSize; ++col) {
            auto pixelImg = img.GetPixel(row + rowOffset, col);
            auto pixelStride = stride.GetPixel(row, col);

            if(pixelImg.r() != pixelStride.r() ||
               pixelImg.g() != pixelStride.g() ||
               pixelImg.b() != pixelStride.b()) {
                return false;
            }
        }
    }

    return true;
}