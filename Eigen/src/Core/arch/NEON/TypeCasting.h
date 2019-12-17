// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Rasmus Munk Larsen <rmlarsen@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_NEON_H
#define EIGEN_TYPE_CASTING_NEON_H

namespace Eigen {

namespace internal {

template<> struct type_casting_traits<float,numext::int32_t>
{ enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 }; };
template<> struct type_casting_traits<float,numext::uint32_t>
{ enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 }; };
template<> struct type_casting_traits<numext::int32_t,float>
{ enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 }; };
template<> struct type_casting_traits<numext::uint32_t,float>
{ enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 }; };

template<> EIGEN_STRONG_INLINE Packet2f pcast<Packet2i,Packet2f>(const Packet2i& a) { return vcvt_f32_s32(a); }
template<> EIGEN_STRONG_INLINE Packet2f pcast<Packet2ui,Packet2f>(const Packet2ui& a) { return vcvt_f32_u32(a); }
template<> EIGEN_STRONG_INLINE Packet4f pcast<Packet4i,Packet4f>(const Packet4i& a) { return vcvtq_f32_s32(a); }
template<> EIGEN_STRONG_INLINE Packet4f pcast<Packet4ui,Packet4f>(const Packet4ui& a) { return vcvtq_f32_u32(a); }
template<> EIGEN_STRONG_INLINE Packet2i pcast<Packet2f,Packet2i>(const Packet2f& a) { return vcvt_s32_f32(a); }
template<> EIGEN_STRONG_INLINE Packet4i pcast<Packet4f,Packet4i>(const Packet4f& a) { return vcvtq_s32_f32(a); }
template<> EIGEN_STRONG_INLINE Packet2ui pcast<Packet2f,Packet2ui>(const Packet2f& a) { return vcvt_u32_f32(a); }
template<> EIGEN_STRONG_INLINE Packet4ui pcast<Packet4f,Packet4ui>(const Packet4f& a) { return vcvtq_u32_f32(a); }

template<> EIGEN_STRONG_INLINE Packet2f preinterpret<Packet2f,Packet2i>(const Packet2i& a)
{ return vreinterpret_f32_s32(a); }
template<> EIGEN_STRONG_INLINE Packet2f preinterpret<Packet2f,Packet2ui>(const Packet2ui& a)
{ return vreinterpret_f32_u32(a); }
template<> EIGEN_STRONG_INLINE Packet4f preinterpret<Packet4f,Packet4i>(const Packet4i& a)
{ return vreinterpretq_f32_s32(a); }
template<> EIGEN_STRONG_INLINE Packet4f preinterpret<Packet4f,Packet4ui>(const Packet4ui& a)
{ return vreinterpretq_f32_u32(a); }
template<> EIGEN_STRONG_INLINE Packet2i preinterpret<Packet2i,Packet2f>(const Packet2f& a)
{ return vreinterpret_s32_f32(a); }
template<> EIGEN_STRONG_INLINE Packet4i preinterpret<Packet4i,Packet4f>(const Packet4f& a)
{ return vreinterpretq_s32_f32(a); }
template<> EIGEN_STRONG_INLINE Packet2ui preinterpret<Packet2ui,Packet2f>(const Packet2f& a)
{ return vreinterpret_u32_f32(a); }
template<> EIGEN_STRONG_INLINE Packet4ui preinterpret<Packet4ui,Packet4f>(const Packet4f& a)
{ return vreinterpretq_u32_f32(a); }

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TYPE_CASTING_NEON_H
