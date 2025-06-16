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

constexpr std::size_t kRows   = 5;
constexpr std::size_t kColls   = 5;
constexpr std::size_t kBorderSize   = 3;


inline std::size_t GetRowsFor(int rank, int size)
{
    const int base = kRows / size;
    const int extra = kRows % size;
    return base + (rank < extra ? 1 : 0);
}

inline std::size_t GetRowOffsetFor(int rank, int size)
{
    std::size_t off = 0;
    for (int r = 0; r < rank; ++r) off += GetRowsFor(r, size);
    return off;
}

void ExchangeShadowFields(Mat& local, int rank, int size)
{
    const int bytesPerRow = local.cols * 3;
    MPI_Request req[4];
    int n = 0;

    /* вверх */
    if (rank != 0) {
        MPI_Isend(local.GetPtr(kBorderSize, 0),          kBorderSize*bytesPerRow, MPI_BYTE,
                  rank-1, 0, MPI_COMM_WORLD, &req[n++]);
        MPI_Irecv(local.GetPtr(0, 0),               kBorderSize*bytesPerRow, MPI_BYTE,
                  rank-1, 1, MPI_COMM_WORLD, &req[n++]);
    }
    /* вниз */
    if (rank != size-1) {
        std::size_t start = local.rows - kBorderSize;           // первая строка нижней зоны
        MPI_Isend(local.GetPtr(start, 0),        kBorderSize*bytesPerRow, MPI_BYTE,
                  rank+1, 1, MPI_COMM_WORLD, &req[n++]);
        MPI_Irecv(local.GetPtr(start+kBorderSize, 0), kBorderSize*bytesPerRow, MPI_BYTE,
                  rank+1, 0, MPI_COMM_WORLD, &req[n++]);
    }
    MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
}

/* -------------------------------------------------------------------------- */
int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t;

    pp::Mat correctImg;
    if (rank == 0) {
        pp::Mat img1(kRows, kColls);
        img1 = img1.CopyWithBorder(kBorderSize);
        pp::Mat img2(img1.rows, img1.cols, img1.borderSize);
        
        t = MPI_Wtime();
    
        pp::InitImg(img1);
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
        pp::DoFilter(*img, *imgNew, pp::ThresholdFilterProc(77));
        std::swap(img, imgNew);
       
        correctImg.swap(*img);

        t = MPI_Wtime() - t;
        std::printf("MPI (%d ranks) (s)elapsed: %.3f s\n", size, t);
    }

    /* размеры локального блока (с учётом рамки) */
    const std::size_t rowsLocal     = GetRowsFor(rank, size);
    const std::size_t rowsWithShadowFields = rowsLocal + 2*kBorderSize;
    const std::size_t colsWithShadowFields = kColls   + 2*kBorderSize;

    Mat cur(rowsWithShadowFields, colsWithShadowFields, kBorderSize);
    Mat nxt(cur.rows, cur.cols, kBorderSize);

 
    pp::InitImg(cur);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
                                

    auto runStage = [&](auto&& proc)
    {
        cur.MakeMirrorBorder(kBorderSize); 
        ExchangeShadowFields(cur, rank, size);

        pp::DoFilter(cur, nxt, proc);
        std::swap(cur, nxt);
    };

    runStage(pp::MedianFilterProc(7));
    runStage(pp::MeanFilterProc(7));
    runStage(pp::SobelFilterProc());
    runStage(pp::ThresholdFilterProc(77));
    /* ------------------------------------------------------------------ */
    MPI_Barrier(MPI_COMM_WORLD);
    double dt = MPI_Wtime() - t0;

    if (rank == 0)
        std::printf("MPI (%d ranks) (p)elapsed: %.3f s\n", size, dt);

        
/* ---------- сборка результата на rank 0 ---------- */
const int  bytesPerRow = kColls * 3;
const auto rowsSend      = rowsLocal;                    // сколько «полезных» строк у ранга
const unsigned char* sendPtr = cur.GetPtr(kBorderSize,   // начало полезной области
                                          kBorderSize);

/* Массивы размеров и смещений (только на rank 0) */
std::vector<int> recvcounts, displs;
if (rank == 0) {
    recvcounts.resize(size);
    displs.resize(size);
    for (int r = 0; r < size; ++r) {
        recvcounts[r] = static_cast<int>(GetRowsFor(r, size) * bytesPerRow);
        displs[r]     = static_cast<int>(GetRowOffsetFor(r, size) * bytesPerRow);
    }
}

/* Буфер-приёмник у rank 0: весь кадр + зеркальная рамка */
pp::Mat gathered;
if (rank == 0) {
    gathered = pp::Mat(kRows + 2 * kBorderSize,
                       kColls + 2 * kBorderSize,
                       kBorderSize);
}

/* Собираем полезные полосы из всех рангов */
MPI_Gatherv(
    /* send */  const_cast<unsigned char*>(sendPtr),
    rowsSend * bytesPerRow,  MPI_BYTE,
    /* recv */  rank == 0 ? gathered.GetPtr(kBorderSize, kBorderSize) : nullptr,
    rank == 0 ? recvcounts.data() : nullptr,
    rank == 0 ? displs.data()     : nullptr,
    MPI_BYTE, 0, MPI_COMM_WORLD);

/* После вызова gathered (у ранга 0) содержит всю картинку             */
/* ----------------------------------------------------------- */


    if(rank == 0) {
        for (std::size_t row = 0; row < correctImg.rows; ++row) {
            for (std::size_t col = 0; col < correctImg.cols; ++col) {
                std::printf("%d, ", correctImg.GetPixel(row, col).r());
            }
            std::printf("\n");
        }
        std::printf("\n");

        for (std::size_t row = 0; row < correctImg.rows; ++row) {
            for (std::size_t col = 0; col < correctImg.cols; ++col) {
                std::printf("%d, ", gathered.GetPixel(row, col).r());
            }
            std::printf("\n");
        }

        gathered.MakeMirrorBorder(kBorderSize);
        auto result = gathered == correctImg;
        std::printf("Result: %d\n", (result));
    }

    /* при необходимости можно собрать результат на rank 0                  */
    MPI_Finalize();
    return 0;
}
