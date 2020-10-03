// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// The algorithm below is a re-implementation of \src\LU\Inverse_SSE.h using NEON
// intrinsics. inv(M) = M#/|M|, where inv(M), M# and |M| denote the inverse of M,
// adjugate of M and determinant of M respectively. M# is computed block-wise
// using specific formulae. For proof, see:
// https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
// Variable names are adopted from \src\LU\Inverse_SSE.h.

// TODO: Unify implementations of different data types (i.e. float and double) and
// different sets of instrinsics (i.e. SSE and NEON)
#ifndef EIGEN_INVERSE_NEON_H
#define EIGEN_INVERSE_NEON_H

namespace Eigen
{
namespace internal
{
template <typename MatrixType, typename ResultType>
struct compute_inverse_size4<Architecture::NEON, float, MatrixType, ResultType>
{
  enum
  {
    MatrixAlignment = traits<MatrixType>::Alignment,
    ResultAlignment = traits<ResultType>::Alignment,
    StorageOrdersMatch = (MatrixType::Flags & RowMajorBit) == (ResultType::Flags & RowMajorBit)
  };
  typedef typename conditional<(MatrixType::Flags & LinearAccessBit), MatrixType const &, typename MatrixType::PlainObject>::type ActualMatrixType;

  // fuctionally equivalent to _mm_shuffle_ps in SSE when interleave
  // == false (i.e. shuffle(m, n, mask, false) equals _mm_shuffle_ps(m, n, mask)),
  // interleave m and n when interleave == true
  static Packet4f shuffle(const Packet4f &m, const Packet4f &n, int mask, bool interleave = false)
  {
    const float *a = reinterpret_cast<const float *>(&m);
    const float *b = reinterpret_cast<const float *>(&n);
    if (!interleave)
    {
      Packet4f res = {*(a + (mask & 3)), *(a + ((mask >> 2) & 3)), *(b + ((mask >> 4) & 3)), *(b + ((mask >> 6) & 3))};
      return res;
    }
    else
    {
      Packet4f res = {*(a + (mask & 3)), *(b + ((mask >> 2) & 3)), *(a + ((mask >> 4) & 3)), *(b + ((mask >> 6) & 3))};
      return res;
    }
  }

  static void run(const MatrixType &mat, ResultType &result)
  {
    ActualMatrixType matrix(mat);

    Packet4f _L1 = matrix.template packet<MatrixAlignment>(0);
    Packet4f _L2 = matrix.template packet<MatrixAlignment>(4);
    Packet4f _L3 = matrix.template packet<MatrixAlignment>(8);
    Packet4f _L4 = matrix.template packet<MatrixAlignment>(12);

    // Four 2x2 sub-matrices of the input matrix
    // input = [[A, B],
    //          [C, D]]
    Packet4f A, B, C, D;

    if (!StorageOrdersMatch)
    {
      A = shuffle(_L1, _L2, 0x50, true);
      B = shuffle(_L3, _L4, 0x50, true);
      C = shuffle(_L1, _L2, 0xFA, true);
      D = shuffle(_L3, _L4, 0xFA, true);
    }
    else
    {
      A = shuffle(_L1, _L2, 0x44);
      B = shuffle(_L1, _L2, 0xEE);
      C = shuffle(_L3, _L4, 0x44);
      D = shuffle(_L3, _L4, 0xEE);
    }

    Packet4f AB, DC, temp;

    // AB = A# * B, where A# denotes the adjugate of A, and * denotes matrix product.
    AB = shuffle(A, A, 0x0F);
    AB = pmul(AB, B);

    temp = shuffle(A, A, 0xA5);
    temp = pmul(temp, shuffle(B, B, 0x4E));
    AB = psub(AB, temp);

    // DC = D#*C
    DC = shuffle(D, D, 0x0F);
    DC = pmul(DC, C);
    temp = shuffle(D, D, 0xA5);
    temp = pmul(temp, shuffle(C, C, 0x4E));
    DC = psub(DC, temp);

    // determinants of the sub-matrices
    Packet4f dA, dB, dC, dD;

    dA = pmul(shuffle(A, A, 0x5F), A);
    dA = psub(dA, shuffle(dA, dA, 0xEE));

    dB = pmul(shuffle(B, B, 0x5F), B);
    dB = psub(dB, shuffle(dB, dB, 0xEE));

    dC = pmul(shuffle(C, C, 0x5F), C);
    dC = psub(dC, shuffle(dC, dC, 0xEE));

    dD = pmul(shuffle(D, D, 0x5F), D);
    dD = psub(dD, shuffle(dD, dD, 0xEE));

    Packet4f d, d1, d2;
    Packet2f sum;
    temp = shuffle(DC, DC, 0xD8);
    d = pmul(temp, AB);
    sum = vpadd_f32(vadd_f32(vget_low_f32(d), vget_high_f32(d)), vadd_f32(vget_low_f32(d), vget_high_f32(d)));
    d = vdupq_lane_f32(sum, 0);
    d1 = pmul(dA, dD);
    d2 = pmul(dB, dC);

    // determinant of the input matrix, det = |A||D| + |B||C| - trace(A#*B*D#*C)
    Packet4f det = psub(padd(d1, d2), d);

    // reciprocal of the determinant of the input matrix, rd = 1/det
    Packet4f rd = pdiv(vdupq_n_f32(float32_t(1.0)), det);

    // Four sub-matrices of the inverse
    Packet4f iA, iB, iC, iD;

    // iD = D*|A| - C*A#*B
    temp = shuffle(C, C, 0xA0);
    temp = pmul(temp, shuffle(AB, AB, 0x44));
    iD = shuffle(C, C, 0xF5);
    iD = pmul(iD, shuffle(AB, AB, 0xEE));
    iD = padd(iD, temp);
    iD = psub(vmulq_lane_f32(D, vget_low_f32(dA), 0), iD);

    // iA = A*|D| - B*D#*C
    temp = shuffle(B, B, 0xA0);
    temp = pmul(temp, shuffle(DC, DC, 0x44));
    iA = shuffle(B, B, 0xF5);
    iA = pmul(iA, shuffle(DC, DC, 0xEE));
    iA = padd(iA, temp);
    iA = psub(vmulq_lane_f32(A, vget_low_f32(dD), 0), iA);

    // iB = C*|B| - D * (A#B)# = C*|B| - D*B#*A
    iB = pmul(D, shuffle(AB, AB, 0x33));
    iB = psub(iB, pmul(shuffle(D, D, 0xB1), shuffle(AB, AB, 0x66)));
    iB = psub(vmulq_lane_f32(C, vget_low_f32(dB), 0), iB);

    // iC = B*|C| - A * (D#C)# = B*|C| - A*C#*D
    iC = pmul(A, shuffle(DC, DC, 0x33));
    iC = psub(iC, pmul(shuffle(A, A, 0xB1), shuffle(DC, DC, 0x66)));
    iC = psub(vmulq_lane_f32(B, vget_low_f32(dC), 0), iC);

    const Packet4f coeff = {1.0, -1.0, -1.0, 1.0};
    rd = pmul(vdupq_lane_f32(vget_low_f32(rd), 0), coeff);
    iA = pmul(iA, rd);
    iB = pmul(iB, rd);
    iC = pmul(iC, rd);
    iD = pmul(iD, rd);

    Index res_stride = result.outerStride();
    float *res = result.data();

    pstoret<float, Packet4f, ResultAlignment>(res + 0, shuffle(iA, iB, 0x77));
    pstoret<float, Packet4f, ResultAlignment>(res + res_stride, shuffle(iA, iB, 0x22));
    pstoret<float, Packet4f, ResultAlignment>(res + 2 * res_stride, shuffle(iC, iD, 0x77));
    pstoret<float, Packet4f, ResultAlignment>(res + 3 * res_stride, shuffle(iC, iD, 0x22));
  }
};

#if EIGEN_ARCH_ARM64 && !EIGEN_APPLE_DOUBLE_NEON_BUG

// same algorithm as above, except that each operand is split into
// halves for two registers to hold.
template <typename MatrixType, typename ResultType>
struct compute_inverse_size4<Architecture::NEON, double, MatrixType, ResultType>
{
  enum
  {
    MatrixAlignment = traits<MatrixType>::Alignment,
    ResultAlignment = traits<ResultType>::Alignment,
    StorageOrdersMatch = (MatrixType::Flags & RowMajorBit) == (ResultType::Flags & RowMajorBit)
  };
  typedef typename conditional<(MatrixType::Flags & LinearAccessBit),
                               MatrixType const &,
                               typename MatrixType::PlainObject>::type
      ActualMatrixType;

  // fuctionally equivalent to _mm_shuffle_pd in SSE (i.e. shuffle(m, n, mask) equals _mm_shuffle_pd(m,n,mask))
  static Packet2d shuffle(const Packet2d &m, const Packet2d &n, int mask)
  {
    const double *a = reinterpret_cast<const double *>(&m);
    const double *b = reinterpret_cast<const double *>(&n);
    Packet2d res = {*(a + (mask & 1)), *(b + ((mask >> 1) & 1))};
    return res;
  }

  static void run(const MatrixType &mat, ResultType &result)
  {
    ActualMatrixType matrix(mat);

    // Four 2x2 sub-matrices of the input matrix, each is further divided into upper and lower
    // row e.g. A1, upper row of A, A2, lower row of A
    // input = [[A, B],  =  [[[A1, [B1,
    //          [C, D]]        A2], B2]],
    //                       [[C1, [D1,
    //                         C2], D2]]]

    Packet2d A1, A2, B1, B2, C1, C2, D1, D2;

    if (StorageOrdersMatch)
    {
      A1 = matrix.template packet<MatrixAlignment>(0);
      B1 = matrix.template packet<MatrixAlignment>(2);
      A2 = matrix.template packet<MatrixAlignment>(4);
      B2 = matrix.template packet<MatrixAlignment>(6);
      C1 = matrix.template packet<MatrixAlignment>(8);
      D1 = matrix.template packet<MatrixAlignment>(10);
      C2 = matrix.template packet<MatrixAlignment>(12);
      D2 = matrix.template packet<MatrixAlignment>(14);
    }
    else
    {
      Packet2d temp;
      A1 = matrix.template packet<MatrixAlignment>(0);
      C1 = matrix.template packet<MatrixAlignment>(2);
      A2 = matrix.template packet<MatrixAlignment>(4);
      C2 = matrix.template packet<MatrixAlignment>(6);

      temp = A1;
      A1 = shuffle(A1, A2, 0);
      A2 = shuffle(temp, A2, 3);

      temp = C1;
      C1 = shuffle(C1, C2, 0);
      C2 = shuffle(temp, C2, 3);

      B1 = matrix.template packet<MatrixAlignment>(8);
      D1 = matrix.template packet<MatrixAlignment>(10);
      B2 = matrix.template packet<MatrixAlignment>(12);
      D2 = matrix.template packet<MatrixAlignment>(14);

      temp = B1;
      B1 = shuffle(B1, B2, 0);
      B2 = shuffle(temp, B2, 3);

      temp = D1;
      D1 = shuffle(D1, D2, 0);
      D2 = shuffle(temp, D2, 3);
    }

    // determinants of the sub-matrices
    Packet2d dA, dB, dC, dD;

    dA = shuffle(A2, A2, 1);
    dA = pmul(A1, dA);
    dA = psub(dA, vdupq_laneq_f64(dA, 1));

    dB = shuffle(B2, B2, 1);
    dB = pmul(B1, dB);
    dB = psub(dB, vdupq_laneq_f64(dB, 1));

    dC = shuffle(C2, C2, 1);
    dC = pmul(C1, dC);
    dC = psub(dC, vdupq_laneq_f64(dC, 1));

    dD = shuffle(D2, D2, 1);
    dD = pmul(D1, dD);
    dD = psub(dD, vdupq_laneq_f64(dD, 1));

    Packet2d DC1, DC2, AB1, AB2;

    // AB = A# * B, where A# denotes the adjugate of A, and * denotes matrix product.
    AB1 = pmul(B1, vdupq_laneq_f64(A2, 1));
    AB2 = pmul(B2, vdupq_laneq_f64(A1, 0));
    AB1 = psub(AB1, pmul(B2, vdupq_laneq_f64(A1, 1)));
    AB2 = psub(AB2, pmul(B1, vdupq_laneq_f64(A2, 0)));

    // DC = D#*C
    DC1 = pmul(C1, vdupq_laneq_f64(D2, 1));
    DC2 = pmul(C2, vdupq_laneq_f64(D1, 0));
    DC1 = psub(DC1, pmul(C2, vdupq_laneq_f64(D1, 1)));
    DC2 = psub(DC2, pmul(C1, vdupq_laneq_f64(D2, 0)));

    Packet2d d1, d2;

    // determinant of the input matrix, det = |A||D| + |B||C| - trace(A#*B*D#*C)
    Packet2d det;

    // reciprocal of the determinant of the input matrix, rd = 1/det
    Packet2d rd;

    d1 = pmul(AB1, shuffle(DC1, DC2, 0));
    d2 = pmul(AB2, shuffle(DC1, DC2, 3));
    rd = padd(d1, d2);
    rd = padd(rd, vdupq_laneq_f64(rd, 1));

    d1 = pmul(dA, dD);
    d2 = pmul(dB, dC);

    det = padd(d1, d2);
    det = psub(det, rd);
    det = vdupq_laneq_f64(det, 0);
    rd = pdiv(vdupq_n_f64(float64_t(1.0)), det);

    // rows of four sub-matrices of the inverse
    Packet2d iA1, iA2, iB1, iB2, iC1, iC2, iD1, iD2;

    // iD = D*|A| - C*A#*B
    iD1 = pmul(AB1, vdupq_laneq_f64(C1, 0));
    iD2 = pmul(AB1, vdupq_laneq_f64(C2, 0));
    iD1 = padd(iD1, pmul(AB2, vdupq_laneq_f64(C1, 1)));
    iD2 = padd(iD2, pmul(AB2, vdupq_laneq_f64(C2, 1)));
    dA = vdupq_laneq_f64(dA, 0);
    iD1 = psub(pmul(D1, dA), iD1);
    iD2 = psub(pmul(D2, dA), iD2);

    // iA = A*|D| - B*D#*C
    iA1 = pmul(DC1, vdupq_laneq_f64(B1, 0));
    iA2 = pmul(DC1, vdupq_laneq_f64(B2, 0));
    iA1 = padd(iA1, pmul(DC2, vdupq_laneq_f64(B1, 1)));
    iA2 = padd(iA2, pmul(DC2, vdupq_laneq_f64(B2, 1)));
    dD = vdupq_laneq_f64(dD, 0);
    iA1 = psub(pmul(A1, dD), iA1);
    iA2 = psub(pmul(A2, dD), iA2);

    // iB = C*|B| - D * (A#B)# = C*|B| - D*B#*A
    iB1 = pmul(D1, shuffle(AB2, AB1, 1));
    iB2 = pmul(D2, shuffle(AB2, AB1, 1));
    iB1 = psub(iB1, pmul(shuffle(D1, D1, 1), shuffle(AB2, AB1, 2)));
    iB2 = psub(iB2, pmul(shuffle(D2, D2, 1), shuffle(AB2, AB1, 2)));
    dB = vdupq_laneq_f64(dB, 0);
    iB1 = psub(pmul(C1, dB), iB1);
    iB2 = psub(pmul(C2, dB), iB2);

    // iC = B*|C| - A * (D#C)# = B*|C| - A*C#*D
    iC1 = pmul(A1, shuffle(DC2, DC1, 1));
    iC2 = pmul(A2, shuffle(DC2, DC1, 1));
    iC1 = psub(iC1, pmul(shuffle(A1, A1, 1), shuffle(DC2, DC1, 2)));
    iC2 = psub(iC2, pmul(shuffle(A2, A2, 1), shuffle(DC2, DC1, 2)));
    dC = vdupq_laneq_f64(dC, 0);
    iC1 = psub(pmul(B1, dC), iC1);
    iC2 = psub(pmul(B2, dC), iC2);

    const Packet2d PN = {1.0, -1.0};
    const Packet2d NP = {-1.0, 1.0};
    d1 = pmul(PN, rd);
    d2 = pmul(NP, rd);

    Index res_stride = result.outerStride();
    double *res = result.data();
    pstoret<double, Packet2d, ResultAlignment>(res + 0, pmul(shuffle(iA2, iA1, 3), d1));
    pstoret<double, Packet2d, ResultAlignment>(res + res_stride, pmul(shuffle(iA2, iA1, 0), d2));
    pstoret<double, Packet2d, ResultAlignment>(res + 2, pmul(shuffle(iB2, iB1, 3), d1));
    pstoret<double, Packet2d, ResultAlignment>(res + res_stride + 2, pmul(shuffle(iB2, iB1, 0), d2));
    pstoret<double, Packet2d, ResultAlignment>(res + 2 * res_stride, pmul(shuffle(iC2, iC1, 3), d1));
    pstoret<double, Packet2d, ResultAlignment>(res + 3 * res_stride, pmul(shuffle(iC2, iC1, 0), d2));
    pstoret<double, Packet2d, ResultAlignment>(res + 2 * res_stride + 2, pmul(shuffle(iD2, iD1, 3), d1));
    pstoret<double, Packet2d, ResultAlignment>(res + 3 * res_stride + 2, pmul(shuffle(iD2, iD1, 0), d2));
  }
};

#endif  // EIGEN_ARCH_ARM64 && !EIGEN_APPLE_DOUBLE_NEON_BUG

} // namespace internal
} // namespace Eigen
#endif
