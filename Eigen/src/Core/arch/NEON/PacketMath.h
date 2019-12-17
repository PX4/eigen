// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Konstantinos Margaritis <markos@freevec.org>
// Heavily based on Gael's SSE version.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_NEON_H
#define EIGEN_PACKET_MATH_NEON_H

namespace Eigen {

namespace internal {

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
#endif

#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#if EIGEN_ARCH_ARM64
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 32
#else
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 16
#endif
#endif

#if EIGEN_COMP_MSVC

// In MSVC's arm_neon.h header file, all NEON vector types
// are aliases to the same underlying type __n128.
// We thus have to wrap them to make them different C++ types.
// (See also bug 1428)

template<typename T,int unique_id>
struct eigen_packet_wrapper
{
  operator T&() { return m_val; }
  operator const T&() const { return m_val; }
  eigen_packet_wrapper() {}
  eigen_packet_wrapper(const T &v) : m_val(v) {}
  eigen_packet_wrapper& operator=(const T &v)
  {
    m_val = v;
    return *this;
  }

  T m_val;
};
typedef eigen_packet_wrapper<float32x2_t,0> Packet2f;
typedef eigen_packet_wrapper<float32x4_t,1> Packet4f;
typedef eigen_packet_wrapper<int32x2_t  ,2> Packet2i;
typedef eigen_packet_wrapper<int32x4_t  ,3> Packet4i;
typedef eigen_packet_wrapper<uint32x2_t ,4> Packet2ui;
typedef eigen_packet_wrapper<uint32x4_t ,5> Packet4ui;

#else

typedef float32x2_t Packet2f;
typedef float32x4_t Packet4f;
typedef int32x2_t   Packet2i;
typedef int32x4_t   Packet4i;
typedef uint32x2_t  Packet2ui;
typedef uint32x4_t  Packet4ui;

#endif // EIGEN_COMP_MSVC

#define _EIGEN_DECLARE_CONST_Packet4f(NAME,X) \
  const Packet4f p4f_##NAME = pset1<Packet4f>(X)

#define _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(NAME,X) \
  const Packet4f p4f_##NAME = vreinterpretq_f32_u32(pset1<int32_t>(X))

#define _EIGEN_DECLARE_CONST_Packet4i(NAME,X) \
  const Packet4i p4i_##NAME = pset1<Packet4i>(X)

#if EIGEN_ARCH_ARM64
  // __builtin_prefetch tends to do nothing on ARM64 compilers because the
  // prefetch instructions there are too detailed for __builtin_prefetch to map
  // meaningfully to them.
  #define EIGEN_ARM_PREFETCH(ADDR)  __asm__ __volatile__("prfm pldl1keep, [%[addr]]\n" ::[addr] "r"(ADDR) : );
#elif EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  #define EIGEN_ARM_PREFETCH(ADDR) __builtin_prefetch(ADDR);
#elif defined __pld
  #define EIGEN_ARM_PREFETCH(ADDR) __pld(ADDR)
#elif EIGEN_ARCH_ARM32
  #define EIGEN_ARM_PREFETCH(ADDR) __asm__ __volatile__ ("pld [%[addr]]\n" :: [addr] "r" (ADDR) : );
#else
  // by default no explicit prefetching
  #define EIGEN_ARM_PREFETCH(ADDR)
#endif

template <>
struct packet_traits<float> : default_packet_traits
{
  typedef Packet4f type;
  typedef Packet2f half;
  enum
  {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasHalfPacket = 1,

    HasAdd       = 1,
    HasSub       = 1,
    HasMul       = 1,
    HasNegate    = 1,
    HasAbs       = 1,
    HasArg       = 0,
    HasAbs2      = 1,
    HasMin       = 1,
    HasMax       = 1,
    HasConj      = 1,
    HasSetLinear = 0,
    HasBlend     = 0,
    HasReduxp    = 1,

    HasDiv   = 1,
    HasFloor = 1,

    HasSin  = EIGEN_FAST_MATH,
    HasCos  = EIGEN_FAST_MATH,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 0,
    HasTanh = EIGEN_FAST_MATH,
    HasErf  = EIGEN_FAST_MATH
  };
};

template <>
struct packet_traits<int32_t> : default_packet_traits
{
  typedef Packet4i type;
  typedef Packet2i half;
  enum
  {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasHalfPacket = 1,

    HasAdd       = 1,
    HasSub       = 1,
    HasMul       = 1,
    HasNegate    = 1,
    HasAbs       = 1,
    HasArg       = 0,
    HasAbs2      = 1,
    HasMin       = 1,
    HasMax       = 1,
    HasConj      = 1,
    HasSetLinear = 0,
    HasBlend     = 0,
    HasReduxp    = 1
  };
};

template <>
struct packet_traits<uint32_t> : default_packet_traits
{
  typedef Packet4ui type;
  typedef Packet2ui half;
  enum
  {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasHalfPacket = 1,

    HasAdd       = 1,
    HasSub       = 1,
    HasMul       = 1,
    HasNegate    = 0,
    HasAbs       = 0,
    HasArg       = 0,
    HasAbs2      = 1,
    HasMin       = 1,
    HasMax       = 1,
    HasConj      = 1,
    HasSetLinear = 0,
    HasBlend     = 0,
    HasReduxp    = 1
  };
};

#if EIGEN_GNUC_AT_MOST(4, 4) && !EIGEN_COMP_LLVM
// workaround gcc 4.2, 4.3 and 4.4 compilatin issue
EIGEN_STRONG_INLINE float32x4_t vld1q_f32(const float* x) { return ::vld1q_f32((const float32_t*)x); }
EIGEN_STRONG_INLINE float32x2_t vld1_f32(const float* x) { return ::vld1_f32 ((const float32_t*)x); }
EIGEN_STRONG_INLINE float32x2_t vld1_dup_f32(const float* x) { return ::vld1_dup_f32 ((const float32_t*)x); }
EIGEN_STRONG_INLINE void vst1q_f32(float* to, float32x4_t from) { ::vst1q_f32((float32_t*)to,from); }
EIGEN_STRONG_INLINE void vst1_f32 (float* to, float32x2_t from) { ::vst1_f32 ((float32_t*)to,from); }
#endif

template<> struct unpacket_traits<Packet2f>
{
  typedef float type;
  typedef Packet2f half;
  typedef Packet2i integer_packet;
  enum
  {
    size = 2,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template<> struct unpacket_traits<Packet4f>
{
  typedef float type;
  typedef Packet2f half;
  typedef Packet4i integer_packet;
  enum
  {
    size = 4,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template<> struct unpacket_traits<Packet2i>
{
  typedef int32_t type;
  typedef Packet2i half;
  enum
  {
    size = 2,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template<> struct unpacket_traits<Packet4i>
{
  typedef int32_t type;
  typedef Packet2i half;
  enum
  {
    size = 4,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template<> struct unpacket_traits<Packet2ui>
{
  typedef uint32_t type;
  typedef Packet2ui half;
  enum
  {
    size = 2,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template<> struct unpacket_traits<Packet4ui>
{
  typedef uint32_t type;
  typedef Packet2ui half;
  enum
  {
    size = 4,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template<> EIGEN_STRONG_INLINE Packet2f pset1<Packet2f>(const float& from) { return vdup_n_f32(from); }
template<> EIGEN_STRONG_INLINE Packet4f pset1<Packet4f>(const float& from) { return vdupq_n_f32(from); }
template<> EIGEN_STRONG_INLINE Packet2i pset1<Packet2i>(const int32_t& from) { return vdup_n_s32(from); }
template<> EIGEN_STRONG_INLINE Packet4i pset1<Packet4i>(const int32_t& from) { return vdupq_n_s32(from); }
template<> EIGEN_STRONG_INLINE Packet2ui pset1<Packet2ui>(const uint32_t& from) { return vdup_n_u32(from); }
template<> EIGEN_STRONG_INLINE Packet4ui pset1<Packet4ui>(const uint32_t& from) { return vdupq_n_u32(from); }

template<> EIGEN_STRONG_INLINE Packet2f pset1frombits<Packet2f>(unsigned int from)
{ return vreinterpret_f32_u32(vdup_n_u32(from)); }
template<> EIGEN_STRONG_INLINE Packet4f pset1frombits<Packet4f>(unsigned int from)
{ return vreinterpretq_f32_u32(vdupq_n_u32(from)); }

template<> EIGEN_STRONG_INLINE Packet2f plset<Packet2f>(const float& a)
{
  const float c[] = {0.0f,1.0f};
  return vadd_f32(pset1<Packet2f>(a), vld1_f32(c));
}
template<> EIGEN_STRONG_INLINE Packet4f plset<Packet4f>(const float& a)
{
  const float c[] = {0.0f,1.0f,2.0f,3.0f};
  return vaddq_f32(pset1<Packet4f>(a), vld1q_f32(c));
}
template<> EIGEN_STRONG_INLINE Packet2i plset<Packet2i>(const int32_t& a)
{
  const int32_t c[] = {0,1};
  return vadd_s32(pset1<Packet2i>(a), vld1_s32(c));
}
template<> EIGEN_STRONG_INLINE Packet4i plset<Packet4i>(const int32_t& a)
{
  const int32_t c[] = {0,1,2,3};
  return vaddq_s32(pset1<Packet4i>(a), vld1q_s32(c));
}
template<> EIGEN_STRONG_INLINE Packet2ui plset<Packet2ui>(const uint32_t& a)
{
  const uint32_t c[] = {0,1};
  return vadd_u32(pset1<Packet2ui>(a), vld1_u32(c));
}
template<> EIGEN_STRONG_INLINE Packet4ui plset<Packet4ui>(const uint32_t& a)
{
  const uint32_t c[] = {0,1,2,3};
  return vaddq_u32(pset1<Packet4ui>(a), vld1q_u32(c));
}

template<> EIGEN_STRONG_INLINE Packet2f padd<Packet2f>(const Packet2f& a, const Packet2f& b) { return vadd_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4f padd<Packet4f>(const Packet4f& a, const Packet4f& b) { return vaddq_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2i padd<Packet2i>(const Packet2i& a, const Packet2i& b) { return vadd_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i padd<Packet4i>(const Packet4i& a, const Packet4i& b) { return vaddq_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui padd<Packet2ui>(const Packet2ui& a, const Packet2ui& b) { return vadd_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui padd<Packet4ui>(const Packet4ui& a, const Packet4ui& b) { return vaddq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f psub<Packet2f>(const Packet2f& a, const Packet2f& b) { return vsub_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4f psub<Packet4f>(const Packet4f& a, const Packet4f& b) { return vsubq_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2i psub<Packet2i>(const Packet2i& a, const Packet2i& b) { return vsub_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i psub<Packet4i>(const Packet4i& a, const Packet4i& b) { return vsubq_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui psub<Packet2ui>(const Packet2ui& a, const Packet2ui& b) { return vsub_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui psub<Packet4ui>(const Packet4ui& a, const Packet4ui& b) { return vsubq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pnegate(const Packet2f& a) { return vneg_f32(a); }
template<> EIGEN_STRONG_INLINE Packet4f pnegate(const Packet4f& a) { return vnegq_f32(a); }
template<> EIGEN_STRONG_INLINE Packet2i pnegate(const Packet2i& a) { return vneg_s32(a); }
template<> EIGEN_STRONG_INLINE Packet4i pnegate(const Packet4i& a) { return vnegq_s32(a); }

template<> EIGEN_STRONG_INLINE Packet2f pconj(const Packet2f& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet4f pconj(const Packet4f& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet2i pconj(const Packet2i& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet4i pconj(const Packet4i& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet2ui pconj(const Packet2ui& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet4ui pconj(const Packet4ui& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet2f pmul<Packet2f>(const Packet2f& a, const Packet2f& b) { return vmul_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4f pmul<Packet4f>(const Packet4f& a, const Packet4f& b) { return vmulq_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2i pmul<Packet2i>(const Packet2i& a, const Packet2i& b) { return vmul_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pmul<Packet4i>(const Packet4i& a, const Packet4i& b) { return vmulq_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui pmul<Packet2ui>(const Packet2ui& a, const Packet2ui& b) { return vmul_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pmul<Packet4ui>(const Packet4ui& a, const Packet4ui& b) { return vmulq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pdiv<Packet2f>(const Packet2f& a, const Packet2f& b)
{
#if EIGEN_ARCH_ARM64
  return vdiv_f32(a,b);
#else
  Packet2f inv, restep, div;

  // NEON does not offer a divide instruction, we have to do a reciprocal approximation
  // However NEON in contrast to other SIMD engines (AltiVec/SSE), offers
  // a reciprocal estimate AND a reciprocal step -which saves a few instructions
  // vrecpeq_f32() returns an estimate to 1/b, which we will finetune with
  // Newton-Raphson and vrecpsq_f32()
  inv = vrecpe_f32(b);

  // This returns a differential, by which we will have to multiply inv to get a better
  // approximation of 1/b.
  restep = vrecps_f32(b, inv);
  inv = vmul_f32(restep, inv);

  // Finally, multiply a by 1/b and get the wanted result of the division.
  div = vmul_f32(a, inv);

  return div;
#endif
}
template<> EIGEN_STRONG_INLINE Packet4f pdiv<Packet4f>(const Packet4f& a, const Packet4f& b)
{
#if EIGEN_ARCH_ARM64
  return vdivq_f32(a,b);
#else
  Packet4f inv, restep, div;

  // NEON does not offer a divide instruction, we have to do a reciprocal approximation
  // However NEON in contrast to other SIMD engines (AltiVec/SSE), offers
  // a reciprocal estimate AND a reciprocal step -which saves a few instructions
  // vrecpeq_f32() returns an estimate to 1/b, which we will finetune with
  // Newton-Raphson and vrecpsq_f32()
  inv = vrecpeq_f32(b);

  // This returns a differential, by which we will have to multiply inv to get a better
  // approximation of 1/b.
  restep = vrecpsq_f32(b, inv);
  inv = vmulq_f32(restep, inv);

  // Finally, multiply a by 1/b and get the wanted result of the division.
  div = vmulq_f32(a, inv);

  return div;
#endif
}

template<> EIGEN_STRONG_INLINE Packet2i pdiv<Packet2i>(const Packet2i& /*a*/, const Packet2i& /*b*/)
{
  eigen_assert(false && "packet integer division are not supported by NEON");
  return pset1<Packet2i>(0);
}
template<> EIGEN_STRONG_INLINE Packet4i pdiv<Packet4i>(const Packet4i& /*a*/, const Packet4i& /*b*/)
{
  eigen_assert(false && "packet integer division are not supported by NEON");
  return pset1<Packet4i>(0);
}
template<> EIGEN_STRONG_INLINE Packet2ui pdiv<Packet2ui>(const Packet2ui& /*a*/, const Packet2ui& /*b*/)
{
  eigen_assert(false && "packet integer division are not supported by NEON");
  return pset1<Packet2ui>(0);
}
template<> EIGEN_STRONG_INLINE Packet4ui pdiv<Packet4ui>(const Packet4ui& /*a*/, const Packet4ui& /*b*/)
{
  eigen_assert(false && "packet integer division are not supported by NEON");
  return pset1<Packet4ui>(0);
}

// Clang/ARM wrongly advertises __ARM_FEATURE_FMA even when it's not available,
// then implements a slow software scalar fallback calling fmaf()!
// Filed LLVM bug:
//     https://llvm.org/bugs/show_bug.cgi?id=27216
#if (defined __ARM_FEATURE_FMA) && !(EIGEN_COMP_CLANG && EIGEN_ARCH_ARM)
// See bug 936.
// FMA is available on VFPv4 i.e. when compiling with -mfpu=neon-vfpv4.
// FMA is a true fused multiply-add i.e. only 1 rounding at the end, no intermediate rounding.
// MLA is not fused i.e. does 2 roundings.
// In addition to giving better accuracy, FMA also gives better performance here on a Krait (Nexus 4):
// MLA: 10 GFlop/s ; FMA: 12 GFlops/s.
template<> EIGEN_STRONG_INLINE Packet4f pmadd(const Packet4f& a, const Packet4f& b, const Packet4f& c)
{ return vfmaq_f32(c,a,b); }
#else
template<> EIGEN_STRONG_INLINE Packet4f pmadd(const Packet4f& a, const Packet4f& b, const Packet4f& c)
{
#if EIGEN_COMP_CLANG && EIGEN_ARCH_ARM
  // Clang/ARM will replace VMLA by VMUL+VADD at least for some values of -mcpu,
  // at least -mcpu=cortex-a8 and -mcpu=cortex-a7. Since the former is the default on
  // -march=armv7-a, that is a very common case.
  // See e.g. this thread:
  //     http://lists.llvm.org/pipermail/llvm-dev/2013-December/068806.html
  // Filed LLVM bug:
  //     https://llvm.org/bugs/show_bug.cgi?id=27219
  Packet4f r = c;
  asm volatile(
    "vmla.f32 %q[r], %q[a], %q[b]"
    : [r] "+w" (r)
    : [a] "w" (a),
      [b] "w" (b)
    : );
  return r;
#else
  return vmlaq_f32(c,a,b);
#endif
}
#endif

// No FMA instruction for int, so use MLA unconditionally.
template<> EIGEN_STRONG_INLINE Packet2i pmadd(const Packet2i& a, const Packet2i& b, const Packet2i& c)
{ return vmla_s32(c,a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pmadd(const Packet4i& a, const Packet4i& b, const Packet4i& c)
{ return vmlaq_s32(c,a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui pmadd(const Packet2ui& a, const Packet2ui& b, const Packet2ui& c)
{ return vmla_u32(c,a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pmadd(const Packet4ui& a, const Packet4ui& b, const Packet4ui& c)
{ return vmlaq_u32(c,a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pmin<Packet2f>(const Packet2f& a, const Packet2f& b) { return vmin_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4f pmin<Packet4f>(const Packet4f& a, const Packet4f& b) { return vminq_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2i pmin<Packet2i>(const Packet2i& a, const Packet2i& b) { return vmin_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pmin<Packet4i>(const Packet4i& a, const Packet4i& b) { return vminq_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui pmin<Packet2ui>(const Packet2ui& a, const Packet2ui& b) { return vmin_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pmin<Packet4ui>(const Packet4ui& a, const Packet4ui& b) { return vminq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pmax<Packet2f>(const Packet2f& a, const Packet2f& b) { return vmax_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4f pmax<Packet4f>(const Packet4f& a, const Packet4f& b) { return vmaxq_f32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2i pmax<Packet2i>(const Packet2i& a, const Packet2i& b) { return vmax_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pmax<Packet4i>(const Packet4i& a, const Packet4i& b) { return vmaxq_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui pmax<Packet2ui>(const Packet2ui& a, const Packet2ui& b) { return vmax_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pmax<Packet4ui>(const Packet4ui& a, const Packet4ui& b) { return vmaxq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pcmp_le<Packet2f>(const Packet2f& a, const Packet2f& b)
{ return vreinterpret_f32_u32(vcle_f32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet4f pcmp_le<Packet4f>(const Packet4f& a, const Packet4f& b)
{ return vreinterpretq_f32_u32(vcleq_f32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet2i pcmp_le<Packet2i>(const Packet2i& a, const Packet2i& b)
{ return vreinterpret_s32_u32(vcle_s32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet4i pcmp_le<Packet4i>(const Packet4i& a, const Packet4i& b)
{ return vreinterpretq_s32_u32(vcleq_s32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet2ui pcmp_le<Packet2ui>(const Packet2ui& a, const Packet2ui& b)
{ return vcle_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pcmp_le<Packet4ui>(const Packet4ui& a, const Packet4ui& b)
{ return vcleq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pcmp_lt<Packet2f>(const Packet2f& a, const Packet2f& b)
{ return vreinterpret_f32_u32(vclt_f32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet4f pcmp_lt<Packet4f>(const Packet4f& a, const Packet4f& b)
{ return vreinterpretq_f32_u32(vcltq_f32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet2i pcmp_lt<Packet2i>(const Packet2i& a, const Packet2i& b)
{ return vreinterpret_s32_u32(vclt_s32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet4i pcmp_lt<Packet4i>(const Packet4i& a, const Packet4i& b)
{ return vreinterpretq_s32_u32(vcltq_s32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet2ui pcmp_lt<Packet2ui>(const Packet2ui& a, const Packet2ui& b)
{ return vclt_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pcmp_lt<Packet4ui>(const Packet4ui& a, const Packet4ui& b)
{ return vcltq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pcmp_eq<Packet2f>(const Packet2f& a, const Packet2f& b)
{ return vreinterpret_f32_u32(vceq_f32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet4f pcmp_eq<Packet4f>(const Packet4f& a, const Packet4f& b)
{ return vreinterpretq_f32_u32(vceqq_f32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet2i pcmp_eq<Packet2i>(const Packet2i& a, const Packet2i& b)
{ return vreinterpret_s32_u32(vceq_s32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet4i pcmp_eq<Packet4i>(const Packet4i& a, const Packet4i& b)
{ return vreinterpretq_s32_u32(vceqq_s32(a,b)); }
template<> EIGEN_STRONG_INLINE Packet2ui pcmp_eq<Packet2ui>(const Packet2ui& a, const Packet2ui& b)
{ return vceq_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pcmp_eq<Packet4ui>(const Packet4ui& a, const Packet4ui& b)
{ return vceqq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pcmp_lt_or_nan<Packet2f>(const Packet2f& a, const Packet2f& b)
{ return vreinterpret_f32_u32(vmvn_u32(vcge_f32(a,b))); }
template<> EIGEN_STRONG_INLINE Packet4f pcmp_lt_or_nan<Packet4f>(const Packet4f& a, const Packet4f& b)
{ return vreinterpretq_f32_u32(vmvnq_u32(vcgeq_f32(a,b))); }

template<> EIGEN_STRONG_INLINE Packet2f pfloor<Packet2f>(const Packet2f& a)
{
  const Packet2f cst_1 = pset1<Packet2f>(1.0f);
  /* perform a floorf */
  Packet2f tmp = vcvt_f32_s32(vcvt_s32_f32(a));

  /* if greater, substract 1 */
  Packet2ui mask = vcgt_f32(tmp, a);
  mask = vand_u32(mask, vreinterpret_u32_f32(cst_1));
  return vsub_f32(tmp, vreinterpret_f32_u32(mask));
}
template<> EIGEN_STRONG_INLINE Packet4f pfloor<Packet4f>(const Packet4f& a)
{
  const Packet4f cst_1 = pset1<Packet4f>(1.0f);
  /* perform a floorf */
  Packet4f tmp = vcvtq_f32_s32(vcvtq_s32_f32(a));

  /* if greater, substract 1 */
  Packet4ui mask = vcgtq_f32(tmp, a);
  mask = vandq_u32(mask, vreinterpretq_u32_f32(cst_1));
  return vsubq_f32(tmp, vreinterpretq_f32_u32(mask));
}

// Logical Operations are not supported for float, so we have to reinterpret casts using NEON intrinsics
template<> EIGEN_STRONG_INLINE Packet2f pand<Packet2f>(const Packet2f& a, const Packet2f& b)
{ return vreinterpret_f32_u32(vand_u32(vreinterpret_u32_f32(a),vreinterpret_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packet4f pand<Packet4f>(const Packet4f& a, const Packet4f& b)
{ return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a),vreinterpretq_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packet2i pand<Packet2i>(const Packet2i& a, const Packet2i& b) { return vand_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pand<Packet4i>(const Packet4i& a, const Packet4i& b) { return vandq_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui pand<Packet2ui>(const Packet2ui& a, const Packet2ui& b)
{ return vand_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pand<Packet4ui>(const Packet4ui& a, const Packet4ui& b)
{ return vandq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f por<Packet2f>(const Packet2f& a, const Packet2f& b)
{ return vreinterpret_f32_u32(vorr_u32(vreinterpret_u32_f32(a),vreinterpret_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packet4f por<Packet4f>(const Packet4f& a, const Packet4f& b)
{ return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a),vreinterpretq_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packet2i por<Packet2i>(const Packet2i& a, const Packet2i& b) { return vorr_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i por<Packet4i>(const Packet4i& a, const Packet4i& b) { return vorrq_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui por<Packet2ui>(const Packet2ui& a, const Packet2ui& b)
{ return vorr_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui por<Packet4ui>(const Packet4ui& a, const Packet4ui& b)
{ return vorrq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pxor<Packet2f>(const Packet2f& a, const Packet2f& b)
{ return vreinterpret_f32_u32(veor_u32(vreinterpret_u32_f32(a),vreinterpret_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packet4f pxor<Packet4f>(const Packet4f& a, const Packet4f& b)
{ return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a),vreinterpretq_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packet2i pxor<Packet2i>(const Packet2i& a, const Packet2i& b) { return veor_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pxor<Packet4i>(const Packet4i& a, const Packet4i& b) { return veorq_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui pxor<Packet2ui>(const Packet2ui& a, const Packet2ui& b)
{ return veor_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pxor<Packet4ui>(const Packet4ui& a, const Packet4ui& b)
{ return veorq_u32(a,b); }

template<> EIGEN_STRONG_INLINE Packet2f pandnot<Packet2f>(const Packet2f& a, const Packet2f& b)
{ return vreinterpret_f32_u32(vbic_u32(vreinterpret_u32_f32(a),vreinterpret_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packet4f pandnot<Packet4f>(const Packet4f& a, const Packet4f& b)
{ return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(a),vreinterpretq_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packet2i pandnot<Packet2i>(const Packet2i& a, const Packet2i& b)
{ return vbic_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i pandnot<Packet4i>(const Packet4i& a, const Packet4i& b)
{ return vbicq_s32(a,b); }
template<> EIGEN_STRONG_INLINE Packet2ui pandnot<Packet2ui>(const Packet2ui& a, const Packet2ui& b)
{ return vbic_u32(a,b); }
template<> EIGEN_STRONG_INLINE Packet4ui pandnot<Packet4ui>(const Packet4ui& a, const Packet4ui& b)
{ return vbicq_u32(a,b); }

template<int N> EIGEN_STRONG_INLINE Packet2i pshiftright(Packet2i a) { return vshr_n_s32(a,N); }
template<int N> EIGEN_STRONG_INLINE Packet4i pshiftright(Packet4i a) { return vshrq_n_s32(a,N); }
template<int N> EIGEN_STRONG_INLINE Packet2ui pshiftright(Packet2ui a) { return vshr_n_u32(a,N); }
template<int N> EIGEN_STRONG_INLINE Packet4ui pshiftright(Packet4ui a) { return vshrq_n_u32(a,N); }

template<int N> EIGEN_STRONG_INLINE Packet2i pshiftleft(Packet2i a) { return vshl_n_s32(a,N); }
template<int N> EIGEN_STRONG_INLINE Packet4i pshiftleft(Packet4i a) { return vshlq_n_s32(a,N); }
template<int N> EIGEN_STRONG_INLINE Packet2ui pshiftleft(Packet2ui a) { return vshl_n_u32(a,N); }
template<int N> EIGEN_STRONG_INLINE Packet4ui pshiftleft(Packet4ui a) { return vshlq_n_u32(a,N); }

template<> EIGEN_STRONG_INLINE Packet2f pload<Packet2f>(const float* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return vld1_f32(from); }
template<> EIGEN_STRONG_INLINE Packet4f pload<Packet4f>(const float* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return vld1q_f32(from); }
template<> EIGEN_STRONG_INLINE Packet2i pload<Packet2i>(const int32_t* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return vld1_s32(from); }
template<> EIGEN_STRONG_INLINE Packet4i pload<Packet4i>(const int32_t* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return vld1q_s32(from); }
template<> EIGEN_STRONG_INLINE Packet2ui pload<Packet2ui>(const uint32_t* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return vld1_u32(from); }
template<> EIGEN_STRONG_INLINE Packet4ui pload<Packet4ui>(const uint32_t* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return vld1q_u32(from); }

template<> EIGEN_STRONG_INLINE Packet2f ploadu<Packet2f>(const float* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return vld1_f32(from); }
template<> EIGEN_STRONG_INLINE Packet4f ploadu<Packet4f>(const float* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return vld1q_f32(from); }
template<> EIGEN_STRONG_INLINE Packet2i ploadu<Packet2i>(const int32_t* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return vld1_s32(from); }
template<> EIGEN_STRONG_INLINE Packet4i ploadu<Packet4i>(const int32_t* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return vld1q_s32(from); }
template<> EIGEN_STRONG_INLINE Packet2ui ploadu<Packet2ui>(const uint32_t* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return vld1_u32(from); }
template<> EIGEN_STRONG_INLINE Packet4ui ploadu<Packet4ui>(const uint32_t* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return vld1q_u32(from); }

template<> EIGEN_STRONG_INLINE Packet2f ploaddup<Packet2f>(const float* from)
{ return vld1_dup_f32(from); }
template<> EIGEN_STRONG_INLINE Packet4f ploaddup<Packet4f>(const float* from)
{ return vcombine_f32(vld1_dup_f32(from), vld1_dup_f32(from+1)); }
template<> EIGEN_STRONG_INLINE Packet2i ploaddup<Packet2i>(const int32_t* from)
{ return vld1_dup_s32(from); }
template<> EIGEN_STRONG_INLINE Packet4i ploaddup<Packet4i>(const int32_t* from)
{ return vcombine_s32(vld1_dup_s32(from), vld1_dup_s32(from+1)); }
template<> EIGEN_STRONG_INLINE Packet2ui ploaddup<Packet2ui>(const uint32_t* from)
{ return vld1_dup_u32(from); }
template<> EIGEN_STRONG_INLINE Packet4ui ploaddup<Packet4ui>(const uint32_t* from)
{ return vcombine_u32(vld1_dup_u32(from), vld1_dup_u32(from+1)); }

template<> EIGEN_STRONG_INLINE Packet4f ploadquad<Packet4f>(const float* from) { return vld1q_dup_f32(from); }
template<> EIGEN_STRONG_INLINE Packet4i ploadquad<Packet4i>(const int32_t* from) { return vld1q_dup_s32(from); }
template<> EIGEN_STRONG_INLINE Packet4ui ploadquad<Packet4ui>(const uint32_t* from) { return vld1q_dup_u32(from); }

template<> EIGEN_STRONG_INLINE void pstore<float>(float* to, const Packet2f& from)
{ EIGEN_DEBUG_ALIGNED_STORE vst1_f32(to,from); }
template<> EIGEN_STRONG_INLINE void pstore<float>(float* to, const Packet4f& from)
{ EIGEN_DEBUG_ALIGNED_STORE vst1q_f32(to,from); }
template<> EIGEN_STRONG_INLINE void pstore<int32_t>(int32_t* to, const Packet2i& from)
{ EIGEN_DEBUG_ALIGNED_STORE vst1_s32(to,from); }
template<> EIGEN_STRONG_INLINE void pstore<int32_t>(int32_t* to, const Packet4i& from)
{ EIGEN_DEBUG_ALIGNED_STORE vst1q_s32(to,from); }
template<> EIGEN_STRONG_INLINE void pstore<uint32_t>(uint32_t* to, const Packet2ui& from)
{ EIGEN_DEBUG_ALIGNED_STORE vst1_u32(to,from); }
template<> EIGEN_STRONG_INLINE void pstore<uint32_t>(uint32_t* to, const Packet4ui& from)
{ EIGEN_DEBUG_ALIGNED_STORE vst1q_u32(to,from); }

template<> EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet2f& from)
{ EIGEN_DEBUG_UNALIGNED_STORE vst1_f32(to,from); }
template<> EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet4f& from)
{ EIGEN_DEBUG_UNALIGNED_STORE vst1q_f32(to,from); }
template<> EIGEN_STRONG_INLINE void pstoreu<int32_t>(int32_t* to, const Packet2i& from)
{ EIGEN_DEBUG_UNALIGNED_STORE vst1_s32(to,from); }
template<> EIGEN_STRONG_INLINE void pstoreu<int32_t>(int32_t* to, const Packet4i& from)
{ EIGEN_DEBUG_UNALIGNED_STORE vst1q_s32(to,from); }
template<> EIGEN_STRONG_INLINE void pstoreu<uint32_t>(uint32_t* to, const Packet2ui& from)
{ EIGEN_DEBUG_UNALIGNED_STORE vst1_u32(to,from); }
template<> EIGEN_STRONG_INLINE void pstoreu<uint32_t>(uint32_t* to, const Packet4ui& from)
{ EIGEN_DEBUG_UNALIGNED_STORE vst1q_u32(to,from); }

template<> EIGEN_DEVICE_FUNC inline Packet2f pgather<float, Packet2f>(const float* from, Index stride)
{
  Packet2f res = vld1_dup_f32(from);
  res = vld1_lane_f32(from + 1*stride, res, 1);
  return res;
}
template<> EIGEN_DEVICE_FUNC inline Packet4f pgather<float, Packet4f>(const float* from, Index stride)
{
  Packet4f res = vld1q_dup_f32(from);
  res = vld1q_lane_f32(from + 1*stride, res, 1);
  res = vld1q_lane_f32(from + 2*stride, res, 2);
  res = vld1q_lane_f32(from + 3*stride, res, 3);
  return res;
}
template<> EIGEN_DEVICE_FUNC inline Packet2i pgather<int32_t, Packet2i>(const int32_t* from, Index stride)
{
  Packet2i res = vld1_dup_s32(from);
  res = vld1_lane_s32(from + 1*stride, res, 1);
  return res;
}
template<> EIGEN_DEVICE_FUNC inline Packet4i pgather<int32_t, Packet4i>(const int32_t* from, Index stride)
{
  Packet4i res = vld1q_dup_s32(from);
  res = vld1q_lane_s32(from + 1*stride, res, 1);
  res = vld1q_lane_s32(from + 2*stride, res, 2);
  res = vld1q_lane_s32(from + 3*stride, res, 3);
  return res;
}
template<> EIGEN_DEVICE_FUNC inline Packet2ui pgather<uint32_t, Packet2ui>(const uint32_t* from, Index stride)
{
  Packet2ui res = vld1_dup_u32(from);
  res = vld1_lane_u32(from + 1*stride, res, 1);
  return res;
}
template<> EIGEN_DEVICE_FUNC inline Packet4ui pgather<uint32_t, Packet4ui>(const uint32_t* from, Index stride)
{
  Packet4ui res = vld1q_dup_u32(from);
  res = vld1q_lane_u32(from + 1*stride, res, 1);
  res = vld1q_lane_u32(from + 2*stride, res, 2);
  res = vld1q_lane_u32(from + 3*stride, res, 3);
  return res;
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<float, Packet2f>(float* to, const Packet2f& from, Index stride)
{
  vst1_lane_f32(to + stride*0, from, 0);
  vst1_lane_f32(to + stride*1, from, 1);
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<float, Packet4f>(float* to, const Packet4f& from, Index stride)
{
  vst1q_lane_f32(to + stride*0, from, 0);
  vst1q_lane_f32(to + stride*1, from, 1);
  vst1q_lane_f32(to + stride*2, from, 2);
  vst1q_lane_f32(to + stride*3, from, 3);
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<int32_t, Packet2i>(int32_t* to, const Packet2i& from, Index stride)
{
  vst1_lane_s32(to + stride*0, from, 0);
  vst1_lane_s32(to + stride*1, from, 1);
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<int32_t, Packet4i>(int32_t* to, const Packet4i& from, Index stride)
{
  vst1q_lane_s32(to + stride*0, from, 0);
  vst1q_lane_s32(to + stride*1, from, 1);
  vst1q_lane_s32(to + stride*2, from, 2);
  vst1q_lane_s32(to + stride*3, from, 3);
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<uint32_t, Packet2ui>(uint32_t* to, const Packet2ui& from, Index stride)
{
  vst1_lane_u32(to + stride*0, from, 0);
  vst1_lane_u32(to + stride*1, from, 1);
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<uint32_t, Packet4ui>(uint32_t* to, const Packet4ui& from, Index stride)
{
  vst1q_lane_u32(to + stride*0, from, 0);
  vst1q_lane_u32(to + stride*1, from, 1);
  vst1q_lane_u32(to + stride*2, from, 2);
  vst1q_lane_u32(to + stride*3, from, 3);
}

template<> EIGEN_STRONG_INLINE void prefetch<float>(const float* addr) { EIGEN_ARM_PREFETCH(addr); }
template<> EIGEN_STRONG_INLINE void prefetch<int32_t>(const int32_t* addr) { EIGEN_ARM_PREFETCH(addr); }
template<> EIGEN_STRONG_INLINE void prefetch<uint32_t>(const uint32_t* addr) { EIGEN_ARM_PREFETCH(addr); }

template<> EIGEN_STRONG_INLINE float pfirst<Packet2f>(const Packet2f& a) { return vget_lane_f32(a,0); }
template<> EIGEN_STRONG_INLINE float pfirst<Packet4f>(const Packet4f& a) { return vgetq_lane_f32(a,0); }
template<> EIGEN_STRONG_INLINE int32_t pfirst<Packet2i>(const Packet2i& a) { return vget_lane_s32(a,0); }
template<> EIGEN_STRONG_INLINE int32_t pfirst<Packet4i>(const Packet4i& a) { return vgetq_lane_s32(a,0); }
template<> EIGEN_STRONG_INLINE uint32_t pfirst<Packet2ui>(const Packet2ui& a) { return vget_lane_u32(a,0); }
template<> EIGEN_STRONG_INLINE uint32_t pfirst<Packet4ui>(const Packet4ui& a) { return vgetq_lane_u32(a,0); }

template<> EIGEN_STRONG_INLINE Packet2f preverse(const Packet2f& a) { return vrev64_f32(a); }
template<> EIGEN_STRONG_INLINE Packet4f preverse(const Packet4f& a)
{
  const float32x4_t a_r64 = vrev64q_f32(a);
  return vcombine_f32(vget_high_f32(a_r64), vget_low_f32(a_r64));
}
template<> EIGEN_STRONG_INLINE Packet2i preverse(const Packet2i& a) { return vrev64_s32(a); }
template<> EIGEN_STRONG_INLINE Packet4i preverse(const Packet4i& a)
{
  const int32x4_t a_r64 = vrev64q_s32(a);
  return vcombine_s32(vget_high_s32(a_r64), vget_low_s32(a_r64));
}
template<> EIGEN_STRONG_INLINE Packet2ui preverse(const Packet2ui& a) { return vrev64_u32(a); }
template<> EIGEN_STRONG_INLINE Packet4ui preverse(const Packet4ui& a)
{
  const uint32x4_t a_r64 = vrev64q_u32(a);
  return vcombine_u32(vget_high_u32(a_r64), vget_low_u32(a_r64));
}

template<> EIGEN_STRONG_INLINE Packet2f pabs(const Packet2f& a) { return vabs_f32(a); }
template<> EIGEN_STRONG_INLINE Packet4f pabs(const Packet4f& a) { return vabsq_f32(a); }
template<> EIGEN_STRONG_INLINE Packet2i pabs(const Packet2i& a) { return vabs_s32(a); }
template<> EIGEN_STRONG_INLINE Packet4i pabs(const Packet4i& a) { return vabsq_s32(a); }
template<> EIGEN_STRONG_INLINE Packet2ui pabs(const Packet2ui& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet4ui pabs(const Packet4ui& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet2f pfrexp<Packet2f>(const Packet2f& a, Packet2f& exponent)
{ return pfrexp_float(a,exponent); }
template<> EIGEN_STRONG_INLINE Packet4f pfrexp<Packet4f>(const Packet4f& a, Packet4f& exponent)
{ return pfrexp_float(a,exponent); }

template<> EIGEN_STRONG_INLINE Packet2f pldexp<Packet2f>(const Packet2f& a, const Packet2f& exponent)
{ return pldexp_float(a,exponent); }
template<> EIGEN_STRONG_INLINE Packet4f pldexp<Packet4f>(const Packet4f& a, const Packet4f& exponent)
{ return pldexp_float(a,exponent); }

template<> EIGEN_STRONG_INLINE float predux<Packet2f>(const Packet2f& a) { return vget_lane_f32(vpadd_f32(a,a), 0); }
template<> EIGEN_STRONG_INLINE float predux<Packet4f>(const Packet4f& a)
{
  const float32x2_t sum = vadd_f32(vget_low_f32(a), vget_high_f32(a));
  return vget_lane_f32(vpadd_f32(sum, sum), 0);
}
template<> EIGEN_STRONG_INLINE int32_t predux<Packet2i>(const Packet2i& a) { return vget_lane_s32(vpadd_s32(a,a), 0); }
template<> EIGEN_STRONG_INLINE int32_t predux<Packet4i>(const Packet4i& a)
{
  const int32x2_t sum = vadd_s32(vget_low_s32(a), vget_high_s32(a));
  return vget_lane_s32(vpadd_s32(sum, sum), 0);
}
template<> EIGEN_STRONG_INLINE uint32_t predux<Packet2ui>(const Packet2ui& a) { return vget_lane_u32(vpadd_u32(a,a), 0); }
template<> EIGEN_STRONG_INLINE uint32_t predux<Packet4ui>(const Packet4ui& a)
{
  const uint32x2_t sum = vadd_u32(vget_low_u32(a), vget_high_u32(a));
  return vget_lane_u32(vpadd_u32(sum, sum), 0);
}

template<> EIGEN_STRONG_INLINE Packet2f preduxp<Packet2f>(const Packet2f* vecs)
{
  const float32x2x2_t vtrn = vzip_f32(vecs[0], vecs[1]);
  return vadd_f32(vtrn.val[0], vtrn.val[1]);
}
template<> EIGEN_STRONG_INLINE Packet4f preduxp<Packet4f>(const Packet4f* vecs)
{
  const float32x4x2_t vtrn1 = vzipq_f32(vecs[0], vecs[2]);
  const float32x4x2_t vtrn2 = vzipq_f32(vecs[1], vecs[3]);
  const float32x4x2_t res1 = vzipq_f32(vtrn1.val[0], vtrn2.val[0]);
  const float32x4x2_t res2 = vzipq_f32(vtrn1.val[1], vtrn2.val[1]);
  return vaddq_f32(vaddq_f32(res1.val[0], res1.val[1]), vaddq_f32(res2.val[0], res2.val[1]));
}
template<> EIGEN_STRONG_INLINE Packet2i preduxp<Packet2i>(const Packet2i* vecs)
{
  const int32x2x2_t vtrn = vzip_s32(vecs[0], vecs[1]);
  return vadd_s32(vtrn.val[0], vtrn.val[1]);
}
template<> EIGEN_STRONG_INLINE Packet4i preduxp<Packet4i>(const Packet4i* vecs)
{
  const int32x4x2_t vtrn1 = vzipq_s32(vecs[0], vecs[2]);
  const int32x4x2_t vtrn2 = vzipq_s32(vecs[1], vecs[3]);
  const int32x4x2_t res1 = vzipq_s32(vtrn1.val[0], vtrn2.val[0]);
  const int32x4x2_t res2 = vzipq_s32(vtrn1.val[1], vtrn2.val[1]);
  return vaddq_s32(vaddq_s32(res1.val[0], res1.val[1]), vaddq_s32(res2.val[0], res2.val[1]));
}
template<> EIGEN_STRONG_INLINE Packet2ui preduxp<Packet2ui>(const Packet2ui* vecs)
{
  const uint32x2x2_t vtrn = vzip_u32(vecs[0], vecs[1]);
  return vadd_u32(vtrn.val[0], vtrn.val[1]);
}
template<> EIGEN_STRONG_INLINE Packet4ui preduxp<Packet4ui>(const Packet4ui* vecs)
{
  uint32x4x2_t vtrn1, vtrn2, res1, res2;
  Packet4ui sum1, sum2, sum;

  // NEON zip performs interleaving of the supplied vectors.
  // We perform two interleaves in a row to acquire the transposed vector
  vtrn1 = vzipq_u32(vecs[0], vecs[2]);
  vtrn2 = vzipq_u32(vecs[1], vecs[3]);
  res1 = vzipq_u32(vtrn1.val[0], vtrn2.val[0]);
  res2 = vzipq_u32(vtrn1.val[1], vtrn2.val[1]);

  // Do the addition of the resulting vectors
  sum1 = vaddq_u32(res1.val[0], res1.val[1]);
  sum2 = vaddq_u32(res2.val[0], res2.val[1]);
  sum = vaddq_u32(sum1, sum2);

  return sum;
}

// Other reduction functions:
// mul
template<> EIGEN_STRONG_INLINE float predux_mul<Packet2f>(const Packet2f& a)
{ return vget_lane_f32(a, 0) * vget_lane_f32(a, 1); }
template<> EIGEN_STRONG_INLINE float predux_mul<Packet4f>(const Packet4f& a)
{ return predux_mul(vmul_f32(vget_low_f32(a), vget_high_f32(a))); }
template<> EIGEN_STRONG_INLINE int32_t predux_mul<Packet2i>(const Packet2i& a)
{ return vget_lane_s32(a, 0) * vget_lane_s32(a, 1); }
template<> EIGEN_STRONG_INLINE int32_t predux_mul<Packet4i>(const Packet4i& a)
{ return predux_mul(vmul_s32(vget_low_s32(a), vget_high_s32(a))); }
template<> EIGEN_STRONG_INLINE uint32_t predux_mul<Packet2ui>(const Packet2ui& a)
{ return vget_lane_u32(a, 0) * vget_lane_u32(a, 1); }
template<> EIGEN_STRONG_INLINE uint32_t predux_mul<Packet4ui>(const Packet4ui& a)
{ return predux_mul(vmul_u32(vget_low_u32(a), vget_high_u32(a))); }

// min
template<> EIGEN_STRONG_INLINE float predux_min<Packet2f>(const Packet2f& a)
{ return vget_lane_f32(vpmin_f32(a,a), 0); }
template<> EIGEN_STRONG_INLINE float predux_min<Packet4f>(const Packet4f& a)
{
  const float32x2_t min = vmin_f32(vget_low_f32(a), vget_high_f32(a));
  return vget_lane_f32(vpmin_f32(min, min), 0);
}
template<> EIGEN_STRONG_INLINE int32_t predux_min<Packet2i>(const Packet2i& a)
{ return vget_lane_s32(vpmin_s32(a,a), 0); }
template<> EIGEN_STRONG_INLINE int32_t predux_min<Packet4i>(const Packet4i& a)
{
  const int32x2_t min = vmin_s32(vget_low_s32(a), vget_high_s32(a));
  return vget_lane_s32(vpmin_s32(min, min), 0);
}
template<> EIGEN_STRONG_INLINE uint32_t predux_min<Packet2ui>(const Packet2ui& a)
{ return vget_lane_u32(vpmin_u32(a,a), 0); }
template<> EIGEN_STRONG_INLINE uint32_t predux_min<Packet4ui>(const Packet4ui& a)
{
  const uint32x2_t min = vmin_u32(vget_low_u32(a), vget_high_u32(a));
  return vget_lane_u32(vpmin_u32(min, min), 0);
}

// max
template<> EIGEN_STRONG_INLINE float predux_max<Packet2f>(const Packet2f& a)
{ return vget_lane_f32(vpmax_f32(a,a), 0); }
template<> EIGEN_STRONG_INLINE float predux_max<Packet4f>(const Packet4f& a)
{
  const float32x2_t max = vmax_f32(vget_low_f32(a), vget_high_f32(a));
  return vget_lane_f32(vpmax_f32(max, max), 0);
}
template<> EIGEN_STRONG_INLINE int32_t predux_max<Packet2i>(const Packet2i& a)
{ return vget_lane_s32(vpmax_s32(a,a), 0); }
template<> EIGEN_STRONG_INLINE int32_t predux_max<Packet4i>(const Packet4i& a)
{
  const int32x2_t max = vmax_s32(vget_low_s32(a), vget_high_s32(a));
  return vget_lane_s32(vpmax_s32(max, max), 0);
}
template<> EIGEN_STRONG_INLINE uint32_t predux_max<Packet2ui>(const Packet2ui& a)
{ return vget_lane_u32(vpmax_u32(a,a), 0); }
template<> EIGEN_STRONG_INLINE uint32_t predux_max<Packet4ui>(const Packet4ui& a)
{
  const uint32x2_t max = vmax_u32(vget_low_u32(a), vget_high_u32(a));
  return vget_lane_u32(vpmax_u32(max, max), 0);
}

template<> EIGEN_STRONG_INLINE bool predux_any(const Packet4f& x)
{
  uint32x2_t tmp = vorr_u32(vget_low_u32( vreinterpretq_u32_f32(x)),
                            vget_high_u32(vreinterpretq_u32_f32(x)));
  return vget_lane_u32(vpmax_u32(tmp, tmp), 0);
}

// this PALIGN_NEON business is to work around a bug in LLVM Clang 3.0 causing incorrect compilation errors,
// see bug 347 and this LLVM bug: http://llvm.org/bugs/show_bug.cgi?id=11074
#define PALIGN_NEON(Offset,Type,Command) \
template<>\
struct palign_impl<Offset,Type>\
{\
    EIGEN_STRONG_INLINE static void run(Type& first, const Type& second)\
    {\
        if (Offset!=0)\
            first = Command(first, second, Offset);\
    }\
};\

PALIGN_NEON(0, Packet2f, vext_f32)
PALIGN_NEON(1, Packet2f, vext_f32)

PALIGN_NEON(0, Packet4f, vextq_f32)
PALIGN_NEON(1, Packet4f, vextq_f32)
PALIGN_NEON(2, Packet4f, vextq_f32)
PALIGN_NEON(3, Packet4f, vextq_f32)

PALIGN_NEON(0, Packet2i, vext_s32)
PALIGN_NEON(1, Packet2i, vext_s32)

PALIGN_NEON(0, Packet4i, vextq_s32)
PALIGN_NEON(1, Packet4i, vextq_s32)
PALIGN_NEON(2, Packet4i, vextq_s32)
PALIGN_NEON(3, Packet4i, vextq_s32)

PALIGN_NEON(0, Packet2ui, vext_u32)
PALIGN_NEON(1, Packet2ui, vext_u32)

PALIGN_NEON(0, Packet4ui, vextq_u32)
PALIGN_NEON(1, Packet4ui, vextq_u32)
PALIGN_NEON(2, Packet4ui, vextq_u32)
PALIGN_NEON(3, Packet4ui, vextq_u32)

#undef PALIGN_NEON

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet2f, 2>& kernel)
{
  const float32x2x2_t z = vzip_f32(kernel.packet[0], kernel.packet[1]);
  kernel.packet[0] = z.val[0];
  kernel.packet[1] = z.val[1];
}
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet4f, 4>& kernel)
{
  const float32x4x2_t tmp1 = vzipq_f32(kernel.packet[0], kernel.packet[1]);
  const float32x4x2_t tmp2 = vzipq_f32(kernel.packet[2], kernel.packet[3]);

  kernel.packet[0] = vcombine_f32(vget_low_f32(tmp1.val[0]), vget_low_f32(tmp2.val[0]));
  kernel.packet[1] = vcombine_f32(vget_high_f32(tmp1.val[0]), vget_high_f32(tmp2.val[0]));
  kernel.packet[2] = vcombine_f32(vget_low_f32(tmp1.val[1]), vget_low_f32(tmp2.val[1]));
  kernel.packet[3] = vcombine_f32(vget_high_f32(tmp1.val[1]), vget_high_f32(tmp2.val[1]));
}
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet2i, 2>& kernel)
{
  const int32x2x2_t z = vzip_s32(kernel.packet[0], kernel.packet[1]);
  kernel.packet[0] = z.val[0];
  kernel.packet[1] = z.val[1];
}
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet4i, 4>& kernel)
{
  const int32x4x2_t tmp1 = vzipq_s32(kernel.packet[0], kernel.packet[1]);
  const int32x4x2_t tmp2 = vzipq_s32(kernel.packet[2], kernel.packet[3]);

  kernel.packet[0] = vcombine_s32(vget_low_s32(tmp1.val[0]), vget_low_s32(tmp2.val[0]));
  kernel.packet[1] = vcombine_s32(vget_high_s32(tmp1.val[0]), vget_high_s32(tmp2.val[0]));
  kernel.packet[2] = vcombine_s32(vget_low_s32(tmp1.val[1]), vget_low_s32(tmp2.val[1]));
  kernel.packet[3] = vcombine_s32(vget_high_s32(tmp1.val[1]), vget_high_s32(tmp2.val[1]));
}
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet2ui, 2>& kernel)
{
  const uint32x2x2_t z = vzip_u32(kernel.packet[0], kernel.packet[1]);
  kernel.packet[0] = z.val[0];
  kernel.packet[1] = z.val[1];
}
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet4ui, 4>& kernel)
{
  const uint32x4x2_t tmp1 = vzipq_u32(kernel.packet[0], kernel.packet[1]);
  const uint32x4x2_t tmp2 = vzipq_u32(kernel.packet[2], kernel.packet[3]);

  kernel.packet[0] = vcombine_u32(vget_low_u32(tmp1.val[0]), vget_low_u32(tmp2.val[0]));
  kernel.packet[1] = vcombine_u32(vget_high_u32(tmp1.val[0]), vget_high_u32(tmp2.val[0]));
  kernel.packet[2] = vcombine_u32(vget_low_u32(tmp1.val[1]), vget_low_u32(tmp2.val[1]));
  kernel.packet[3] = vcombine_u32(vget_high_u32(tmp1.val[1]), vget_high_u32(tmp2.val[1]));
}

//---------- double ----------

// Clang 3.5 in the iOS toolchain has an ICE triggered by NEON intrisics for double.
// Confirmed at least with __apple_build_version__ = 6000054.
#ifdef __apple_build_version__
// Let's hope that by the time __apple_build_version__ hits the 601* range, the bug will be fixed.
// https://gist.github.com/yamaya/2924292 suggests that the 3 first digits are only updated with
// major toolchain updates.
#define EIGEN_APPLE_DOUBLE_NEON_BUG (__apple_build_version__ < 6010000)
#else
#define EIGEN_APPLE_DOUBLE_NEON_BUG 0
#endif

#if EIGEN_ARCH_ARM64 && !EIGEN_APPLE_DOUBLE_NEON_BUG

// Bug 907: workaround missing declarations of the following two functions in the ADK
// Defining these functions as templates ensures that if these intrinsics are
// already defined in arm_neon.h, then our workaround doesn't cause a conflict
// and has lower priority in overload resolution.
template <typename T> uint64x2_t vreinterpretq_u64_f64(T a) { return (uint64x2_t) a; }

template <typename T> float64x2_t vreinterpretq_f64_u64(T a) { return (float64x2_t) a; }

typedef float64x2_t Packet2d;
typedef float64x1_t Packet1d;

template<> struct packet_traits<double>  : default_packet_traits
{
  typedef Packet2d type;
  typedef Packet2d half;
  enum
  {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,
    HasHalfPacket = 0,

    HasCast      = 1,
    HasCmp       = 1,
    HasAdd       = 1,
    HasSub       = 1,
    HasShift     = 1,
    HasMul       = 1,
    HasNegate    = 1,
    HasAbs       = 1,
    HasArg       = 0,
    HasAbs2      = 1,
    HasAbsDiff   = 1,
    HasMin       = 1,
    HasMax       = 1,
    HasConj      = 1,
    HasSetLinear = 0,
    HasBlend     = 0,
    HasInsert    = 1,
    HasReduxp    = 1,

    HasDiv   = 1,
    HasFloor = 0,

    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 0,
    HasExp  = 0,
    HasSqrt = 0,
    HasTanh = 0,
    HasErf  = 0
  };
};

template<> struct unpacket_traits<Packet2d>
{
  typedef double type;
  enum
  {
    size = 2,
    alignment = Aligned16,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
  typedef Packet2d half;
};

template<> EIGEN_STRONG_INLINE Packet2d pset1<Packet2d>(const double&  from) { return vdupq_n_f64(from); }

template<> EIGEN_STRONG_INLINE Packet2d plset<Packet2d>(const double& a)
{
  const double c[] = {0.0,1.0};
  return vaddq_f64(pset1<Packet2d>(a), vld1q_f64(c));
}

template<> EIGEN_STRONG_INLINE Packet2d padd<Packet2d>(const Packet2d& a, const Packet2d& b) { return vaddq_f64(a,b); }

template<> EIGEN_STRONG_INLINE Packet2d psub<Packet2d>(const Packet2d& a, const Packet2d& b) { return vsubq_f64(a,b); }

template<> EIGEN_STRONG_INLINE Packet2d pnegate(const Packet2d& a) { return vnegq_f64(a); }

template<> EIGEN_STRONG_INLINE Packet2d pconj(const Packet2d& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet2d pmul<Packet2d>(const Packet2d& a, const Packet2d& b) { return vmulq_f64(a,b); }

template<> EIGEN_STRONG_INLINE Packet2d pdiv<Packet2d>(const Packet2d& a, const Packet2d& b) { return vdivq_f64(a,b); }

#ifdef __ARM_FEATURE_FMA
// See bug 936. See above comment about FMA for float.
template<> EIGEN_STRONG_INLINE Packet2d pmadd(const Packet2d& a, const Packet2d& b, const Packet2d& c)
{ return vfmaq_f64(c,a,b); }
#else
template<> EIGEN_STRONG_INLINE Packet2d pmadd(const Packet2d& a, const Packet2d& b, const Packet2d& c)
{ return vmlaq_f64(c,a,b); }
#endif

template<> EIGEN_STRONG_INLINE Packet2d pmin<Packet2d>(const Packet2d& a, const Packet2d& b) { return vminq_f64(a,b); }

template<> EIGEN_STRONG_INLINE Packet2d pmax<Packet2d>(const Packet2d& a, const Packet2d& b) { return vmaxq_f64(a,b); }

// Logical Operations are not supported for float, so we have to reinterpret casts using NEON intrinsics
template<> EIGEN_STRONG_INLINE Packet2d pand<Packet2d>(const Packet2d& a, const Packet2d& b)
{ return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a),vreinterpretq_u64_f64(b))); }

template<> EIGEN_STRONG_INLINE Packet2d por<Packet2d>(const Packet2d& a, const Packet2d& b)
{ return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(a),vreinterpretq_u64_f64(b))); }

template<> EIGEN_STRONG_INLINE Packet2d pxor<Packet2d>(const Packet2d& a, const Packet2d& b)
{ return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a),vreinterpretq_u64_f64(b))); }

template<> EIGEN_STRONG_INLINE Packet2d pandnot<Packet2d>(const Packet2d& a, const Packet2d& b)
{ return vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(a),vreinterpretq_u64_f64(b))); }

template<> EIGEN_STRONG_INLINE Packet2d pcmp_le(const Packet2d& a, const Packet2d& b)
{ return vreinterpretq_f64_u64(vcleq_f64(a,b)); }

template<> EIGEN_STRONG_INLINE Packet2d pcmp_lt(const Packet2d& a, const Packet2d& b)
{ return vreinterpretq_f64_u64(vcltq_f64(a,b)); }

template<> EIGEN_STRONG_INLINE Packet2d pcmp_eq(const Packet2d& a, const Packet2d& b)
{ return vreinterpretq_f64_u64(vceqq_f64(a,b)); }

template<> EIGEN_STRONG_INLINE Packet2d pload<Packet2d>(const double* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return vld1q_f64(from); }

template<> EIGEN_STRONG_INLINE Packet2d ploadu<Packet2d>(const double* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return vld1q_f64(from); }

template<> EIGEN_STRONG_INLINE Packet2d ploaddup<Packet2d>(const double* from) { return vld1q_dup_f64(from); }
template<> EIGEN_STRONG_INLINE void pstore<double>(double* to, const Packet2d& from)
{ EIGEN_DEBUG_ALIGNED_STORE vst1q_f64(to,from); }

template<> EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const Packet2d& from)
{ EIGEN_DEBUG_UNALIGNED_STORE vst1q_f64(to,from); }

template<> EIGEN_DEVICE_FUNC inline Packet2d pgather<double, Packet2d>(const double* from, Index stride)
{
  Packet2d res = pset1<Packet2d>(0.0);
  res = vld1q_lane_f64(from + 0*stride, res, 0);
  res = vld1q_lane_f64(from + 1*stride, res, 1);
  return res;
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<double, Packet2d>(double* to, const Packet2d& from, Index stride)
{
  vst1q_lane_f64(to + stride*0, from, 0);
  vst1q_lane_f64(to + stride*1, from, 1);
}

template<> EIGEN_STRONG_INLINE void prefetch<double>(const double* addr) { EIGEN_ARM_PREFETCH(addr); }

// FIXME only store the 2 first elements ?
template<> EIGEN_STRONG_INLINE double pfirst<Packet2d>(const Packet2d& a) { return vgetq_lane_f64(a,0); }

template<> EIGEN_STRONG_INLINE Packet2d preverse(const Packet2d& a)
{ return vcombine_f64(vget_high_f64(a), vget_low_f64(a)); }

template<> EIGEN_STRONG_INLINE Packet2d pabs(const Packet2d& a) { return vabsq_f64(a); }

#if EIGEN_COMP_CLANG && defined(__apple_build_version__)
// workaround ICE, see bug 907
template<> EIGEN_STRONG_INLINE double predux<Packet2d>(const Packet2d& a)
{ return (vget_low_f64(a) + vget_high_f64(a))[0]; }
#else
template<> EIGEN_STRONG_INLINE double predux<Packet2d>(const Packet2d& a)
{ return vget_lane_f64(vget_low_f64(a) + vget_high_f64(a), 0); }
#endif

template<> EIGEN_STRONG_INLINE Packet2d preduxp<Packet2d>(const Packet2d* vecs)
{
  return vaddq_f64(vzip1q_f64(vecs[0], vecs[1]), vzip2q_f64(vecs[0], vecs[1]));
}
// Other reduction functions:
// mul
#if EIGEN_COMP_CLANG && defined(__apple_build_version__)
template<> EIGEN_STRONG_INLINE double predux_mul<Packet2d>(const Packet2d& a)
{ return (vget_low_f64(a) * vget_high_f64(a))[0]; }
#else
template<> EIGEN_STRONG_INLINE double predux_mul<Packet2d>(const Packet2d& a)
{ return vget_lane_f64(vget_low_f64(a) * vget_high_f64(a), 0); }
#endif

// min
template<> EIGEN_STRONG_INLINE double predux_min<Packet2d>(const Packet2d& a)
{ return vgetq_lane_f64(vpminq_f64(a,a), 0); }

// max
template<> EIGEN_STRONG_INLINE double predux_max<Packet2d>(const Packet2d& a)
{ return vgetq_lane_f64(vpmaxq_f64(a,a), 0); }

// this PALIGN_NEON business is to work around a bug in LLVM Clang 3.0 causing incorrect compilation errors,
// see bug 347 and this LLVM bug: http://llvm.org/bugs/show_bug.cgi?id=11074
#define PALIGN_NEON(Offset,Type,Command) \
template<>\
struct palign_impl<Offset,Type>\
{\
    EIGEN_STRONG_INLINE static void run(Type& first, const Type& second)\
    {\
        if (Offset!=0)\
            first = Command(first, second, Offset);\
    }\
};\

PALIGN_NEON(0, Packet2d, vextq_f64)
PALIGN_NEON(1, Packet2d, vextq_f64)
#undef PALIGN_NEON

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet2d, 2>& kernel)
{
  const float64x2_t tmp1 = vzip1q_f64(kernel.packet[0], kernel.packet[1]);
  const float64x2_t tmp2 = vzip2q_f64(kernel.packet[0], kernel.packet[1]);

  kernel.packet[0] = tmp1;
  kernel.packet[1] = tmp2;
}
#endif // EIGEN_ARCH_ARM64

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PACKET_MATH_NEON_H
