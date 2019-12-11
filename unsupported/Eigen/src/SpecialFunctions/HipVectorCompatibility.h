#ifndef HIP_VECTOR_COMPATIBILITY_H
#define HIP_VECTOR_COMPATIBILITY_H

namespace hip_impl {
  template <typename, typename, unsigned int> struct Scalar_accessor;
}   // end namespace hip_impl

namespace Eigen {
namespace internal {

#define HIP_SCALAR_ACCESSOR_BUILDER(NAME) \
template <typename T, typename U, unsigned int n> \
struct NAME <hip_impl::Scalar_accessor<T, U, n>> : NAME <T> {};

#define HIP_SCALAR_ACCESSOR_BUILDER_IGAMMA(NAME) \
template <typename T, typename U, unsigned int n, IgammaComputationMode mode> \
struct NAME <hip_impl::Scalar_accessor<T, U, n>, mode> : NAME <T, mode> {};

#if EIGEN_HAS_C99_MATH
HIP_SCALAR_ACCESSOR_BUILDER(betainc_helper)
HIP_SCALAR_ACCESSOR_BUILDER(erf_impl)
HIP_SCALAR_ACCESSOR_BUILDER(erfc_impl)
HIP_SCALAR_ACCESSOR_BUILDER(igammac_impl)
HIP_SCALAR_ACCESSOR_BUILDER(incbeta_cfe)
HIP_SCALAR_ACCESSOR_BUILDER(lgamma_impl)
HIP_SCALAR_ACCESSOR_BUILDER(ndtri_impl)
HIP_SCALAR_ACCESSOR_BUILDER(polygamma_impl)
HIP_SCALAR_ACCESSOR_BUILDER_IGAMMA(igamma_generic_impl)
#endif

HIP_SCALAR_ACCESSOR_BUILDER(bessel_i0_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_i0e_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_i1_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_i1e_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_j0_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_j1_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_k0_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_k0e_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_k1_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_k1e_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_y0_impl)
HIP_SCALAR_ACCESSOR_BUILDER(bessel_y1_impl)
HIP_SCALAR_ACCESSOR_BUILDER(betainc_impl)
HIP_SCALAR_ACCESSOR_BUILDER(digamma_impl)
HIP_SCALAR_ACCESSOR_BUILDER(digamma_impl_maybe_poly)
HIP_SCALAR_ACCESSOR_BUILDER(gamma_sample_der_alpha_impl)
HIP_SCALAR_ACCESSOR_BUILDER(gamma_sample_der_alpha_retval)
HIP_SCALAR_ACCESSOR_BUILDER(igamma_der_a_impl)
HIP_SCALAR_ACCESSOR_BUILDER(igamma_der_a_retval)
HIP_SCALAR_ACCESSOR_BUILDER(igamma_impl)
HIP_SCALAR_ACCESSOR_BUILDER(zeta_impl)
HIP_SCALAR_ACCESSOR_BUILDER(zeta_impl_series)
HIP_SCALAR_ACCESSOR_BUILDER_IGAMMA(igamma_series_impl)
HIP_SCALAR_ACCESSOR_BUILDER_IGAMMA(igammac_cf_impl)

}  // end namespace internal
}  // end namespace Eigen

#endif  // HIP_VECTOR_COMPATIBILITY_H
