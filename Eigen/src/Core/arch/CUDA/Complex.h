// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_CUDA_H
#define EIGEN_COMPLEX_CUDA_H

// clang-format off

#if defined(EIGEN_CUDACC) && defined(EIGEN_GPU_COMPILE_PHASE)

namespace Eigen {

namespace internal {

// Many std::complex methods such as operator+, operator-, operator* and
// operator/ are not constexpr. Due to this, clang does not treat them as device
// functions and thus Eigen functors making use of these operators fail to
// compile. Here, we manually specialize these functors for complex types when
// building for CUDA to avoid non-constexpr methods.

// Sum
template<typename T> struct scalar_sum_op<const std::complex<T>, const std::complex<T> > : binary_op_base<const std::complex<T>, const std::complex<T> > {
  typedef typename std::complex<T> result_type;

  EIGEN_EMPTY_STRUCT_CTOR(scalar_sum_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<T> operator() (const std::complex<T>& a, const std::complex<T>& b) const {
    return std::complex<T>(numext::real(a) + numext::real(b),
                           numext::imag(a) + numext::imag(b));
  }
};

template<typename T> struct scalar_sum_op<std::complex<T>, std::complex<T> > : scalar_sum_op<const std::complex<T>, const std::complex<T> > {};


// Difference
template<typename T> struct scalar_difference_op<const std::complex<T>, const std::complex<T> >  : binary_op_base<const std::complex<T>, const std::complex<T> > {
  typedef typename std::complex<T> result_type;

  EIGEN_EMPTY_STRUCT_CTOR(scalar_difference_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<T> operator() (const std::complex<T>& a, const std::complex<T>& b) const {
    return std::complex<T>(numext::real(a) - numext::real(b),
                           numext::imag(a) - numext::imag(b));
  }
};

template<typename T> struct scalar_difference_op<std::complex<T>, std::complex<T> > : scalar_difference_op<const std::complex<T>, const std::complex<T> > {};


// Product
template<typename T> struct scalar_product_op<const std::complex<T>, const std::complex<T> >  : binary_op_base<const std::complex<T>, const std::complex<T> > {
  enum {
    Vectorizable = packet_traits<std::complex<T> >::HasMul
  };
  typedef typename std::complex<T> result_type;

  EIGEN_EMPTY_STRUCT_CTOR(scalar_product_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<T> operator() (const std::complex<T>& a, const std::complex<T>& b) const {
    const T a_real = numext::real(a);
    const T a_imag = numext::imag(a);
    const T b_real = numext::real(b);
    const T b_imag = numext::imag(b);
    return std::complex<T>(a_real * b_real - a_imag * b_imag,
                           a_real * b_imag + a_imag * b_real);
  }
};

template<typename T> struct scalar_product_op<std::complex<T>, std::complex<T> > : scalar_product_op<const std::complex<T>, const std::complex<T> > {};


// Quotient
template<typename T> struct scalar_quotient_op<const std::complex<T>, const std::complex<T> > : binary_op_base<const std::complex<T>, const std::complex<T> > {
  enum {
    Vectorizable = packet_traits<std::complex<T> >::HasDiv
  };
  typedef typename std::complex<T> result_type;

  EIGEN_EMPTY_STRUCT_CTOR(scalar_quotient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::complex<T> operator() (const std::complex<T>& a, const std::complex<T>& b) const {
    const T a_real = numext::real(a);
    const T a_imag = numext::imag(a);
    const T b_real = numext::real(b);
    const T b_imag = numext::imag(b);
    const T norm = T(1) / (b_real * b_real + b_imag * b_imag);
    return std::complex<T>((a_real * b_real + a_imag * b_imag) * norm,
                           (a_imag * b_real - a_real * b_imag) * norm);
  }
};

template<typename T> struct scalar_quotient_op<std::complex<T>, std::complex<T> > : scalar_quotient_op<const std::complex<T>, const std::complex<T> > {};

template<typename T>
struct sqrt_impl<std::complex<T>> {
  static EIGEN_DEVICE_FUNC std::complex<T> run(const std::complex<T>& z) {
    // Computes the principal sqrt of the input.
    //
    // For a complex square root of the number x + i*y. We want to find real
    // numbers u and v such that
    //    (u + i*v)^2 = x + i*y  <=>
    //    u^2 - v^2 + i*2*u*v = x + i*v.
    // By equating the real and imaginary parts we get:
    //    u^2 - v^2 = x
    //    2*u*v = y.
    //
    // For x >= 0, this has the numerically stable solution
    //    u = sqrt(0.5 * (x + sqrt(x^2 + y^2)))
    //    v = y / (2 * u)
    // and for x < 0,
    //    v = sign(y) * sqrt(0.5 * (-x + sqrt(x^2 + y^2)))
    //    u = y / (2 * v)
    //
    // Letting w = sqrt(0.5 * (|x| + |z|)),
    //   if x == 0: u = w, v = sign(y) * w
    //   if x > 0:  u = w, v = y / (2 * w)
    //   if x < 0:  u = |y| / (2 * w), v = sign(y) * w

    const T x = numext::real(z);
    const T y = numext::imag(z);
    const T zero = T(0);
    const T cst_half = T(0.5);

    // Special case of isinf(y)
    if ((numext::isinf)(y)) {
      const T inf = std::numeric_limits<T>::infinity();
      return std::complex<T>(inf, y);
    }

    T w = numext::sqrt(cst_half * (numext::abs(x) + numext::abs(z)));
    return
      x == zero ? std::complex<T>(w, y < zero ? -w : w)
      : x > zero ? std::complex<T>(w, y / (2 * w))
        : std::complex<T>(numext::abs(y) / (2 * w), y < zero ? -w : w );
  }
};

}  // namespace internal
}  // namespace Eigen

#endif

#endif  // EIGEN_COMPLEX_CUDA_H
