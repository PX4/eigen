// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_BLOCK_PANEL_H
#define EIGEN_GENERAL_BLOCK_PANEL_H

namespace Eigen { 
  
namespace internal {

template<typename _LhsScalar, typename _RhsScalar, bool _ConjLhs=false, bool _ConjRhs=false>
class gebp_traits;


/** \internal \returns b if a<=0, and returns a otherwise. */
inline std::ptrdiff_t manage_caching_sizes_helper(std::ptrdiff_t a, std::ptrdiff_t b)
{
  return a<=0 ? b : a;
}

#if EIGEN_ARCH_i386_OR_x86_64
const std::ptrdiff_t defaultL1CacheSize = 32*1024;
const std::ptrdiff_t defaultL2CacheSize = 256*1024;
const std::ptrdiff_t defaultL3CacheSize = 2*1024*1024;
#else
const std::ptrdiff_t defaultL1CacheSize = 16*1024;
const std::ptrdiff_t defaultL2CacheSize = 512*1024;
const std::ptrdiff_t defaultL3CacheSize = 512*1024;
#endif

/** \internal */
inline void manage_caching_sizes(Action action, std::ptrdiff_t* l1=0, std::ptrdiff_t* l2=0)
{
  static bool m_cache_sizes_initialized = false;
  static std::ptrdiff_t m_l1CacheSize = 0;
  static std::ptrdiff_t m_l2CacheSize = 0;
  static std::ptrdiff_t m_l3CacheSize = 0;

  if(!m_cache_sizes_initialized)
  {
    int l1CacheSize, l2CacheSize, l3CacheSize;
    queryCacheSizes(l1CacheSize, l2CacheSize, l3CacheSize);
    m_l1CacheSize = manage_caching_sizes_helper(l1CacheSize, defaultL1CacheSize);
    m_l2CacheSize = manage_caching_sizes_helper(l2CacheSize, defaultL2CacheSize);
    m_l3CacheSize = manage_caching_sizes_helper(l3CacheSize, defaultL3CacheSize);
    m_cache_sizes_initialized = true;
  }
  
  if(action==SetAction)
  {
    // set the cpu cache size and cache all block sizes from a global cache size in byte
    eigen_internal_assert(l1!=0 && l2!=0);
    m_l1CacheSize = *l1;
    m_l2CacheSize = *l2;
  }
  else if(action==GetAction)
  {
    eigen_internal_assert(l1!=0 && l2!=0);
    *l1 = m_l1CacheSize;
    *l2 = m_l2CacheSize;
  }
  else
  {
    eigen_internal_assert(false);
  }
}

/* Helper for computeProductBlockingSizes.
 *
 * Given a m x k times k x n matrix product of scalar types \c LhsScalar and \c RhsScalar,
 * this function computes the blocking size parameters along the respective dimensions
 * for matrix products and related algorithms. The blocking sizes depends on various
 * parameters:
 * - the L1 and L2 cache sizes,
 * - the register level blocking sizes defined by gebp_traits,
 * - the number of scalars that fit into a packet (when vectorization is enabled).
 *
 * \sa setCpuCacheSizes */

template<typename LhsScalar, typename RhsScalar, int KcFactor>
void evaluateProductBlockingSizesHeuristic(Index& k, Index& m, Index& n, Index num_threads = 1)
{
  typedef gebp_traits<LhsScalar,RhsScalar> Traits;

  // Explanations:
  // Let's recall that the product algorithms form mc x kc vertical panels A' on the lhs and
  // kc x nc blocks B' on the rhs. B' has to fit into L2/L3 cache. Moreover, A' is processed
  // per mr x kc horizontal small panels where mr is the blocking size along the m dimension
  // at the register level. This small horizontal panel has to stay within L1 cache.
  std::ptrdiff_t l1, l2, l3;
  manage_caching_sizes(GetAction, &l1, &l2, &l3);

  if (num_threads > 1) {
    typedef typename Traits::ResScalar ResScalar;
    enum {
      kdiv = KcFactor * (Traits::mr * sizeof(LhsScalar) + Traits::nr * sizeof(RhsScalar)),
      ksub = Traits::mr * Traits::nr * sizeof(ResScalar),
      k_mask = (0xffffffff/8)*8,

      mr = Traits::mr,
      mr_mask = (0xffffffff/mr)*mr,

      nr = Traits::nr,
      nr_mask = (0xffffffff/nr)*nr
    };
    Index k_cache = (l1-ksub)/kdiv;
    if (k_cache < k) {
      k = k_cache & k_mask;
      eigen_internal_assert(k > 0);
    }

    Index n_cache = (l2-l1) / (nr * sizeof(RhsScalar) * k);
    Index n_per_thread = numext::div_ceil(n, num_threads);
    if (n_cache <= n_per_thread) {
      // Don't exceed the capacity of the l2 cache.
      eigen_internal_assert(n_cache >= static_cast<Index>(nr));
      n = n_cache & nr_mask;
      eigen_internal_assert(n > 0);
    } else {
      n = (std::min<Index>)(n, (n_per_thread + nr - 1) & nr_mask);
    }

    if (l3 > l2) {
      // l3 is shared between all cores, so we'll give each thread its own chunk of l3.
      Index m_cache = (l3-l2) / (sizeof(LhsScalar) * k * num_threads);
      Index m_per_thread = numext::div_ceil(m, num_threads);
      if(m_cache < m_per_thread && m_cache >= static_cast<Index>(mr)) {
        m = m_cache & mr_mask;
        eigen_internal_assert(m > 0);
      } else {
        m = (std::min<Index>)(m, (m_per_thread + mr - 1) & mr_mask);
      }
    }
  }
  else {
    // In unit tests we do not want to use extra large matrices,
    // so we reduce the cache size to check the blocking strategy is not flawed
#ifdef EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS
    l1 = 9*1024;
    l2 = 32*1024;
    l3 = 512*1024;
#endif
    
    // Early return for small problems because the computation below are time consuming for small problems.
    // Perhaps it would make more sense to consider k*n*m??
    // Note that for very tiny problem, this function should be bypassed anyway
    // because we use the coefficient-based implementation for them.
    if((std::max)(k,(std::max)(m,n))<48)
      return;
    
    typedef typename Traits::ResScalar ResScalar;
    enum {
      k_peeling = 8,
      k_div = KcFactor * (Traits::mr * sizeof(LhsScalar) + Traits::nr * sizeof(RhsScalar)),
      k_sub = Traits::mr * Traits::nr * sizeof(ResScalar)
    };
    
    // ---- 1st level of blocking on L1, yields kc ----
    
    // Blocking on the third dimension (i.e., k) is chosen so that an horizontal panel
    // of size mr x kc of the lhs plus a vertical panel of kc x nr of the rhs both fits within L1 cache.
    // We also include a register-level block of the result (mx x nr).
    // (In an ideal world only the lhs panel would stay in L1)
    // Moreover, kc has to be a multiple of 8 to be compatible with loop peeling, leading to a maximum blocking size of:
    const Index max_kc = ((l1-k_sub)/k_div) & (~(k_peeling-1));
    const Index old_k = k;
    if(k>max_kc)
    {
      // We are really blocking on the third dimension:
      // -> reduce blocking size to make sure the last block is as large as possible
      //    while keeping the same number of sweeps over the result.
      k = (k%max_kc)==0 ? max_kc
                        : max_kc - k_peeling * ((max_kc-1-(k%max_kc))/(k_peeling*(k/max_kc+1)));
                        
      eigen_internal_assert(((old_k/k) == (old_k/max_kc)) && "the number of sweeps has to remain the same");
    }
    
    // ---- 2nd level of blocking on max(L2,L3), yields nc ----
    
    // TODO find a reliable way to get the actual amount of cache per core to use for 2nd level blocking, that is:
    //      actual_l2 = max(l2, l3/nb_core_sharing_l3)
    // The number below is quite conservative: it is better to underestimate the cache size rather than overestimating it)
    // For instance, it corresponds to 6MB of L3 shared among 4 cores.
    #ifdef EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS
    const Index actual_l2 = l3;
    #else
    const Index actual_l2 = 1572864; // == 1.5 MB
    #endif
    
    
    
    // Here, nc is chosen such that a block of kc x nc of the rhs fit within half of L2.
    // The second half is implicitly reserved to access the result and lhs coefficients.
    // When k<max_kc, then nc can arbitrarily growth. In practice, it seems to be fruitful
    // to limit this growth: we bound nc to growth by a factor x1.5.
    // However, if the entire lhs block fit within L1, then we are not going to block on the rows at all,
    // and it becomes fruitful to keep the packed rhs blocks in L1 if there is enough remaining space.
    Index max_nc;
    const Index lhs_bytes = m * k * sizeof(LhsScalar);
    const Index remaining_l1 = l1- k_sub - lhs_bytes;
    if(remaining_l1 >= Index(Traits::nr*sizeof(RhsScalar))*k)
    {
      // L1 blocking
      max_nc = remaining_l1 / (k*sizeof(RhsScalar));
    }
    else
    {
      // L2 blocking
      max_nc = (3*actual_l2)/(2*2*max_kc*sizeof(RhsScalar));
    }
    // WARNING Below, we assume that Traits::nr is a power of two.
    Index nc = std::min<Index>(actual_l2/(2*k*sizeof(RhsScalar)), max_nc) & (~(Traits::nr-1));
    if(n>nc)
    {
      // We are really blocking over the columns:
      // -> reduce blocking size to make sure the last block is as large as possible
      //    while keeping the same number of sweeps over the packed lhs.
      //    Here we allow one more sweep if this gives us a perfect match, thus the commented "-1"
      n = (n%nc)==0 ? nc
                    : (nc - Traits::nr * ((nc/*-1*/-(n%nc))/(Traits::nr*(n/nc+1))));
    }
    else if(old_k==k)
    {
      // So far, no blocking at all, i.e., kc==k, and nc==n.
      // In this case, let's perform a blocking over the rows such that the packed lhs data is kept in cache L1/L2
      // TODO: part of this blocking strategy is now implemented within the kernel itself, so the L1-based heuristic here should be obsolete.
      Index problem_size = k*n*sizeof(LhsScalar);
      Index actual_lm = actual_l2;
      Index max_mc = m;
      if(problem_size<=1024)
      {
        // problem is small enough to keep in L1
        // Let's choose m such that lhs's block fit in 1/3 of L1
        actual_lm = l1;
      }
      else if(l3!=0 && problem_size<=32768)
      {
        // we have both L2 and L3, and problem is small enough to be kept in L2
        // Let's choose m such that lhs's block fit in 1/3 of L2
        actual_lm = l2;
        max_mc = 576;
      }
      
      Index mc = (std::min<Index>)(actual_lm/(3*k*sizeof(LhsScalar)), max_mc);
      if (mc > Traits::mr) mc -= mc % Traits::mr;
      
      m = (m%mc)==0 ? mc
                    : (mc - Traits::mr * ((mc/*-1*/-(m%mc))/(Traits::mr*(m/mc+1))));
    }
  }
}

inline bool useSpecificBlockingSizes(Index& k, Index& m, Index& n)
{
#ifdef EIGEN_TEST_SPECIFIC_BLOCKING_SIZES
  if (EIGEN_TEST_SPECIFIC_BLOCKING_SIZES) {
    k = std::min<Index>(k, EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_K);
    m = std::min<Index>(m, EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_M);
    n = std::min<Index>(n, EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_N);
    return true;
  }
#else
  EIGEN_UNUSED_VARIABLE(k)
  EIGEN_UNUSED_VARIABLE(m)
  EIGEN_UNUSED_VARIABLE(n)
#endif
  return false;
}

/** \brief Computes the blocking parameters for a m x k times k x n matrix product
  *
  * \param[in,out] k Input: the third dimension of the product. Output: the blocking size along the same dimension.
  * \param[in,out] m Input: the number of rows of the left hand side. Output: the blocking size along the same dimension.
  * \param[in,out] n Input: the number of columns of the right hand side. Output: the blocking size along the same dimension.
  *
  * Given a m x k times k x n matrix product of scalar types \c LhsScalar and \c RhsScalar,
  * this function computes the blocking size parameters along the respective dimensions
  * for matrix products and related algorithms.
  *
  * The blocking size parameters may be evaluated:
  *   - either by a heuristic based on cache sizes;
  *   - or using a precomputed lookup table;
  *   - or using fixed prescribed values (for testing purposes).
  *
  * \sa setCpuCacheSizes */

template<typename LhsScalar, typename RhsScalar, int KcFactor>
void computeProductBlockingSizes(Index& k, Index& m, Index& n, Index num_threads = 1)
{
  if (!useSpecificBlockingSizes(k, m, n)) {
    if (!lookupBlockingSizesFromTable<LhsScalar, RhsScalar>(k, m, n, num_threads)) {
      evaluateProductBlockingSizesHeuristic<LhsScalar, RhsScalar, KcFactor>(k, m, n, num_threads);
    }
  }

  typedef gebp_traits<LhsScalar,RhsScalar> Traits;
  enum {
    kr = 8,
    mr = Traits::mr,
    nr = Traits::nr
  };
  if (k > kr) k -= k % kr;
  if (m > mr) m -= m % mr;
  if (n > nr) n -= n % nr;
}

template<typename LhsScalar, typename RhsScalar>
inline void computeProductBlockingSizes(Index& k, Index& m, Index& n, Index num_threads = 1)
{
  computeProductBlockingSizes<LhsScalar,RhsScalar,1>(k, m, n);
}

#ifdef EIGEN_HAS_FUSE_CJMADD
  #define MADD(CJ,A,B,C,T)  C = CJ.pmadd(A,B,C);
#else

  // FIXME (a bit overkill maybe ?)

  template<typename CJ, typename A, typename B, typename C, typename T> struct gebp_madd_selector {
    EIGEN_ALWAYS_INLINE static void run(const CJ& cj, A& a, B& b, C& c, T& /*t*/)
    {
      c = cj.pmadd(a,b,c);
    }
  };

  template<typename CJ, typename T> struct gebp_madd_selector<CJ,T,T,T,T> {
    EIGEN_ALWAYS_INLINE static void run(const CJ& cj, T& a, T& b, T& c, T& t)
    {
      t = b; t = cj.pmul(a,t); c = padd(c,t);
    }
  };

  template<typename CJ, typename A, typename B, typename C, typename T>
  EIGEN_STRONG_INLINE void gebp_madd(const CJ& cj, A& a, B& b, C& c, T& t)
  {
    gebp_madd_selector<CJ,A,B,C,T>::run(cj,a,b,c,t);
  }

  #define MADD(CJ,A,B,C,T)  gebp_madd(CJ,A,B,C,T);
//   #define MADD(CJ,A,B,C,T)  T = B; T = CJ.pmul(A,T); C = padd(C,T);
#endif

/* Vectorization logic
 *  real*real: unpack rhs to constant packets, ...
 * 
 *  cd*cd : unpack rhs to (b_r,b_r), (b_i,b_i), mul to get (a_r b_r,a_i b_r) (a_r b_i,a_i b_i),
 *          storing each res packet into two packets (2x2),
 *          at the end combine them: swap the second and addsub them 
 *  cf*cf : same but with 2x4 blocks
 *  cplx*real : unpack rhs to constant packets, ...
 *  real*cplx : load lhs as (a0,a0,a1,a1), and mul as usual
 */
template<typename _LhsScalar, typename _RhsScalar, bool _ConjLhs, bool _ConjRhs>
class gebp_traits
{
public:
  typedef _LhsScalar LhsScalar;
  typedef _RhsScalar RhsScalar;
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;

  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1,
    
    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,

    // register block size along the N direction (must be either 2 or 4)
    nr = NumberOfRegisters/4,

    // register block size along the M direction (currently, this one cannot be modified)
    mr = 2 * LhsPacketSize,
    
    WorkSpaceFactor = nr * RhsPacketSize,

    LhsProgress = LhsPacketSize,
    RhsProgress = RhsPacketSize
  };

  typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
  typedef typename packet_traits<ResScalar>::type  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;
  
  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }

  EIGEN_STRONG_INLINE void unpackRhs(DenseIndex n, const RhsScalar* rhs, RhsScalar* b)
  {
    for(DenseIndex k=0; k<n; k++)
      pstore1<RhsPacket>(&b[k*RhsPacketSize], rhs[k]);
  }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pload<RhsPacket>(b);
  }

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = pload<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, AccPacket& tmp) const
  {
    tmp = b; tmp = pmul(a,tmp); c = padd(c,tmp);
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = pmadd(c,alpha,r);
  }

protected:
//   conj_helper<LhsScalar,RhsScalar,ConjLhs,ConjRhs> cj;
//   conj_helper<LhsPacket,RhsPacket,ConjLhs,ConjRhs> pcj;
};

template<typename RealScalar, bool _ConjLhs>
class gebp_traits<std::complex<RealScalar>, RealScalar, _ConjLhs, false>
{
public:
  typedef std::complex<RealScalar> LhsScalar;
  typedef RealScalar RhsScalar;
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;

  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = false,
    Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1,
    
    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,
    nr = NumberOfRegisters/4,
    mr = 2 * LhsPacketSize,
    WorkSpaceFactor = nr*RhsPacketSize,

    LhsProgress = LhsPacketSize,
    RhsProgress = RhsPacketSize
  };

  typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
  typedef typename packet_traits<ResScalar>::type  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;

  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }

  EIGEN_STRONG_INLINE void unpackRhs(DenseIndex n, const RhsScalar* rhs, RhsScalar* b)
  {
    for(DenseIndex k=0; k<n; k++)
      pstore1<RhsPacket>(&b[k*RhsPacketSize], rhs[k]);
  }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pload<RhsPacket>(b);
  }

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = pload<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp) const
  {
    madd_impl(a, b, c, tmp, typename conditional<Vectorizable,true_type,false_type>::type());
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp, const true_type&) const
  {
    tmp = b; tmp = pmul(a.v,tmp); c.v = padd(c.v,tmp);
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsScalar& a, const RhsScalar& b, ResScalar& c, RhsScalar& /*tmp*/, const false_type&) const
  {
    c += a * b;
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = cj.pmadd(c,alpha,r);
  }

protected:
  conj_helper<ResPacket,ResPacket,ConjLhs,false> cj;
};

template<typename RealScalar, bool _ConjLhs, bool _ConjRhs>
class gebp_traits<std::complex<RealScalar>, std::complex<RealScalar>, _ConjLhs, _ConjRhs >
{
public:
  typedef std::complex<RealScalar>  Scalar;
  typedef std::complex<RealScalar>  LhsScalar;
  typedef std::complex<RealScalar>  RhsScalar;
  typedef std::complex<RealScalar>  ResScalar;
  
  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<RealScalar>::Vectorizable
                && packet_traits<Scalar>::Vectorizable,
    RealPacketSize  = Vectorizable ? packet_traits<RealScalar>::size : 1,
    ResPacketSize   = Vectorizable ? packet_traits<ResScalar>::size : 1,
    
    nr = 2,
    mr = 2 * ResPacketSize,
    WorkSpaceFactor = Vectorizable ? 2*nr*RealPacketSize : nr,

    LhsProgress = ResPacketSize,
    RhsProgress = Vectorizable ? 2*ResPacketSize : 1
  };
  
  typedef typename packet_traits<RealScalar>::type RealPacket;
  typedef typename packet_traits<Scalar>::type     ScalarPacket;
  struct DoublePacket
  {
    RealPacket first;
    RealPacket second;
  };

  typedef typename conditional<Vectorizable,RealPacket,  Scalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,DoublePacket,Scalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,ScalarPacket,Scalar>::type ResPacket;
  typedef typename conditional<Vectorizable,DoublePacket,Scalar>::type AccPacket;
  
  EIGEN_STRONG_INLINE void initAcc(Scalar& p) { p = Scalar(0); }

  EIGEN_STRONG_INLINE void initAcc(DoublePacket& p)
  {
    p.first   = pset1<RealPacket>(RealScalar(0));
    p.second  = pset1<RealPacket>(RealScalar(0));
  }

  /* Unpack the rhs coeff such that each complex coefficient is spread into
   * two packects containing respectively the real and imaginary coefficient
   * duplicated as many time as needed: (x+iy) => [x, ..., x] [y, ..., y]
   */
  EIGEN_STRONG_INLINE void unpackRhs(DenseIndex n, const Scalar* rhs, Scalar* b)
  {
    for(DenseIndex k=0; k<n; k++)
    {
      if(Vectorizable)
      {
        pstore1<RealPacket>((RealScalar*)&b[k*ResPacketSize*2+0],             real(rhs[k]));
        pstore1<RealPacket>((RealScalar*)&b[k*ResPacketSize*2+ResPacketSize], imag(rhs[k]));
      }
      else
        b[k] = rhs[k];
    }
  }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, ResPacket& dest) const { dest = *b; }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, DoublePacket& dest) const
  {
    dest.first  = pload<RealPacket>((const RealScalar*)b);
    dest.second = pload<RealPacket>((const RealScalar*)(b+ResPacketSize));
  }

  // nothing special here
  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = pload<LhsPacket>((const typename unpacket_traits<LhsPacket>::type*)(a));
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, DoublePacket& c, RhsPacket& /*tmp*/) const
  {
    c.first   = padd(pmul(a,b.first), c.first);
    c.second  = padd(pmul(a,b.second),c.second);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, ResPacket& c, RhsPacket& /*tmp*/) const
  {
    c = cj.pmadd(a,b,c);
  }
  
  EIGEN_STRONG_INLINE void acc(const Scalar& c, const Scalar& alpha, Scalar& r) const { r += alpha * c; }
  
  EIGEN_STRONG_INLINE void acc(const DoublePacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    // assemble c
    ResPacket tmp;
    if((!ConjLhs)&&(!ConjRhs))
    {
      tmp = pcplxflip(pconj(ResPacket(c.second)));
      tmp = padd(ResPacket(c.first),tmp);
    }
    else if((!ConjLhs)&&(ConjRhs))
    {
      tmp = pconj(pcplxflip(ResPacket(c.second)));
      tmp = padd(ResPacket(c.first),tmp);
    }
    else if((ConjLhs)&&(!ConjRhs))
    {
      tmp = pcplxflip(ResPacket(c.second));
      tmp = padd(pconj(ResPacket(c.first)),tmp);
    }
    else if((ConjLhs)&&(ConjRhs))
    {
      tmp = pcplxflip(ResPacket(c.second));
      tmp = psub(pconj(ResPacket(c.first)),tmp);
    }
    
    r = pmadd(tmp,alpha,r);
  }

protected:
  conj_helper<LhsScalar,RhsScalar,ConjLhs,ConjRhs> cj;
};

template<typename RealScalar, bool _ConjRhs>
class gebp_traits<RealScalar, std::complex<RealScalar>, false, _ConjRhs >
{
public:
  typedef std::complex<RealScalar>  Scalar;
  typedef RealScalar  LhsScalar;
  typedef Scalar      RhsScalar;
  typedef Scalar      ResScalar;

  enum {
    ConjLhs = false,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<RealScalar>::Vectorizable
                && packet_traits<Scalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1,
    
    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,
    nr = 4,
    mr = 2*ResPacketSize,
    WorkSpaceFactor = nr*RhsPacketSize,

    LhsProgress = ResPacketSize,
    RhsProgress = ResPacketSize
  };

  typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
  typedef typename packet_traits<ResScalar>::type  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;

  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }

  EIGEN_STRONG_INLINE void unpackRhs(DenseIndex n, const RhsScalar* rhs, RhsScalar* b)
  {
    for(DenseIndex k=0; k<n; k++)
      pstore1<RhsPacket>(&b[k*RhsPacketSize], rhs[k]);
  }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pload<RhsPacket>(b);
  }

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploaddup<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp) const
  {
    madd_impl(a, b, c, tmp, typename conditional<Vectorizable,true_type,false_type>::type());
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp, const true_type&) const
  {
    tmp = b; tmp.v = pmul(a,tmp.v); c = padd(c,tmp);
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsScalar& a, const RhsScalar& b, ResScalar& c, RhsScalar& /*tmp*/, const false_type&) const
  {
    c += a * b;
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = cj.pmadd(alpha,c,r);
  }

protected:
  conj_helper<ResPacket,ResPacket,false,ConjRhs> cj;
};

// helper for the rotating kernel below
template <typename GebpKernel, bool UseRotatingKernel = GebpKernel::UseRotatingKernel>
struct PossiblyRotatingKernelHelper
{
  // default implementation, not rotating

  typedef typename GebpKernel::Traits Traits;
  typedef typename Traits::RhsScalar RhsScalar;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::AccPacket AccPacket;

  const Traits& traits;
  PossiblyRotatingKernelHelper(const Traits& t) : traits(t) {}


  template <size_t K, size_t Index>
  void loadOrRotateRhs(RhsPacket& to, const RhsScalar* from) const
  {
    traits.loadRhs(from + (Index+4*K)*Traits::RhsProgress, to);
  }

  void unrotateResult(AccPacket&,
                      AccPacket&,
                      AccPacket&,
                      AccPacket&)
  {
  }
};

// rotating implementation
template <typename GebpKernel>
struct PossiblyRotatingKernelHelper<GebpKernel, true>
{
  typedef typename GebpKernel::Traits Traits;
  typedef typename Traits::RhsScalar RhsScalar;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::AccPacket AccPacket;

  const Traits& traits;
  PossiblyRotatingKernelHelper(const Traits& t) : traits(t) {}

  template <size_t K, size_t Index>
  void loadOrRotateRhs(RhsPacket& to, const RhsScalar* from) const
  {
    if (Index == 0) {
      to = pload<RhsPacket>(from + 4*K*Traits::RhsProgress);
    } else {
      EIGEN_ASM_COMMENT("Do not reorder code, we're very tight on registers");
      to = protate<1>(to);
    }
  }

  void unrotateResult(AccPacket& res0,
                      AccPacket& res1,
                      AccPacket& res2,
                      AccPacket& res3)
  {
    PacketBlock<AccPacket> resblock;
    resblock.packet[0] = res0;
    resblock.packet[1] = res1;
    resblock.packet[2] = res2;
    resblock.packet[3] = res3;
    ptranspose(resblock);
    resblock.packet[3] = protate<1>(resblock.packet[3]);
    resblock.packet[2] = protate<2>(resblock.packet[2]);
    resblock.packet[1] = protate<3>(resblock.packet[1]);
    ptranspose(resblock);
    res0 = resblock.packet[0];
    res1 = resblock.packet[1];
    res2 = resblock.packet[2];
    res3 = resblock.packet[3];
  }
};

/* optimized GEneral packed Block * packed Panel product kernel
 *
 * Mixing type logic: C += A * B
 *  |  A  |  B  | comments
 *  |real |cplx | no vectorization yet, would require to pack A with duplication
 *  |cplx |real | easy vectorization
 */
template<typename LhsScalar, typename RhsScalar, typename Index, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel
{
  typedef gebp_traits<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> Traits;
  typedef typename Traits::ResScalar ResScalar;
  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;
  typedef typename Traits::AccPacket AccPacket;

  enum {
    Vectorizable  = Traits::Vectorizable,
    LhsProgress   = Traits::LhsProgress,
    RhsProgress   = Traits::RhsProgress,
    ResPacketSize = Traits::ResPacketSize
  };


  static const bool UseRotatingKernel =
    EIGEN_ARCH_ARM &&
    internal::is_same<LhsScalar, float>::value &&
    internal::is_same<RhsScalar, float>::value &&
    internal::is_same<ResScalar, float>::value &&
    Traits::LhsPacketSize == 4 &&
    Traits::RhsPacketSize == 4 &&
    Traits::ResPacketSize == 4;

  EIGEN_DONT_INLINE
  void operator()(ResScalar* res, Index resStride, const LhsScalar* blockA, const RhsScalar* blockB, Index rows, Index depth, Index cols, ResScalar alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0, RhsScalar* unpackedB=0);
};

template<typename LhsScalar, typename RhsScalar, typename Index, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE
void gebp_kernel<LhsScalar,RhsScalar,Index,mr,nr,ConjugateLhs,ConjugateRhs>
  ::operator()(ResScalar* res, Index resStride, const LhsScalar* blockA, const RhsScalar* blockB, Index rows, Index depth, Index cols, ResScalar alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB, RhsScalar* unpackedB)
  {
    Traits traits;
    
    if(strideA==-1) strideA = depth;
    if(strideB==-1) strideB = depth;
    conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
    Index packet_cols4 = nr>=4 ? (cols/4) * 4 : 0;
    const Index peeled_mc3 = mr>=3*Traits::LhsProgress ? (rows/(3*LhsProgress))*(3*LhsProgress) : 0;
    const Index peeled_mc2 = mr>=2*Traits::LhsProgress ? peeled_mc3+((rows-peeled_mc3)/(2*LhsProgress))*(2*LhsProgress) : 0;
    const Index peeled_mc1 = mr>=1*Traits::LhsProgress ? (rows/(1*LhsProgress))*(1*LhsProgress) : 0;
    enum { pk = 8 }; // NOTE Such a large peeling factor is important for large matrices (~ +5% when >1000 on Haswell)
    const Index peeled_kc  = depth & ~(pk-1);
    const Index prefetch_res_offset = 32/sizeof(ResScalar);    
//     const Index depth2     = depth & ~1;

    //---------- Process 3 * LhsProgress rows at once ----------
    // This corresponds to 3*LhsProgress x nr register blocks.
    // Usually, make sense only with FMA
    if(mr>=3*Traits::LhsProgress)
    {      
      PossiblyRotatingKernelHelper<gebp_kernel> possiblyRotatingKernelHelper(traits);
      
      // Here, the general idea is to loop on each largest micro horizontal panel of the lhs (3*Traits::LhsProgress x depth)
      // and on each largest micro vertical panel of the rhs (depth * nr).
      // Blocking sizes, i.e., 'depth' has been computed so that the micro horizontal panel of the lhs fit in L1.
      // However, if depth is too small, we can extend the number of rows of these horizontal panels.
      // This actual number of rows is computed as follow:
      const Index l1 = defaultL1CacheSize; // in Bytes, TODO, l1 should be passed to this function.
      // The max(1, ...) here is needed because we may be using blocking params larger than what our known l1 cache size
      // suggests we should be using: either because our known l1 cache size is inaccurate (e.g. on Android, we can only guess),
      // or because we are testing specific blocking sizes.
      const Index actual_panel_rows = (3*LhsProgress) * std::max<Index>(1,( (l1 - sizeof(ResScalar)*mr*nr - depth*nr*sizeof(RhsScalar)) / (depth * sizeof(LhsScalar) * 3*LhsProgress) ));
      for(Index i1=0; i1<peeled_mc3; i1+=actual_panel_rows)
      {
        const Index actual_panel_end = (std::min)(i1+actual_panel_rows, peeled_mc3);
        for(Index j2=0; j2<packet_cols4; j2+=nr)
        {
          for(Index i=i1; i<actual_panel_end; i+=3*LhsProgress)
          {
          
          // We selected a 3*Traits::LhsProgress x nr micro block of res which is entirely
          // stored into 3 x nr registers.
          
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(3*LhsProgress)];
          prefetch(&blA[0]);

          // gets res block as register
          AccPacket C0, C1, C2,  C3,
                    C4, C5, C6,  C7,
                    C8, C9, C10, C11;
          traits.initAcc(C0);  traits.initAcc(C1);  traits.initAcc(C2);  traits.initAcc(C3);
          traits.initAcc(C4);  traits.initAcc(C5);  traits.initAcc(C6);  traits.initAcc(C7);
          traits.initAcc(C8);  traits.initAcc(C9);  traits.initAcc(C10); traits.initAcc(C11);

          LinearMapper r0 = res.getLinearMapper(i, j2 + 0);
          LinearMapper r1 = res.getLinearMapper(i, j2 + 1);
          LinearMapper r2 = res.getLinearMapper(i, j2 + 2);
          LinearMapper r3 = res.getLinearMapper(i, j2 + 3);

          r0.prefetch(0);
          r1.prefetch(0);
          r2.prefetch(0);
          r3.prefetch(0);

          // performs "inner" products
          const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
          prefetch(&blB[0]);
          LhsPacket A0, A1;

          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 3pX4");
            RhsPacket B_0, T0;
            LhsPacket A2;

#define EIGEN_GEBP_ONESTEP(K) \
            do { \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 3pX4"); \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              internal::prefetch(blA+(3*K+16)*LhsProgress); \
              if (EIGEN_ARCH_ARM) internal::prefetch(blB+(4*K+16)*RhsProgress); /* Bug 953 */ \
              traits.loadLhs(&blA[(0+3*K)*LhsProgress], A0);  \
              traits.loadLhs(&blA[(1+3*K)*LhsProgress], A1);  \
              traits.loadLhs(&blA[(2+3*K)*LhsProgress], A2);  \
              possiblyRotatingKernelHelper.template loadOrRotateRhs<K, 0>(B_0, blB); \
              traits.madd(A0, B_0, C0, T0); \
              traits.madd(A1, B_0, C4, T0); \
              traits.madd(A2, B_0, C8, B_0); \
              possiblyRotatingKernelHelper.template loadOrRotateRhs<K, 1>(B_0, blB); \
              traits.madd(A0, B_0, C1, T0); \
              traits.madd(A1, B_0, C5, T0); \
              traits.madd(A2, B_0, C9, B_0); \
              possiblyRotatingKernelHelper.template loadOrRotateRhs<K, 2>(B_0, blB); \
              traits.madd(A0, B_0, C2,  T0); \
              traits.madd(A1, B_0, C6,  T0); \
              traits.madd(A2, B_0, C10, B_0); \
              possiblyRotatingKernelHelper.template loadOrRotateRhs<K, 3>(B_0, blB); \
              traits.madd(A0, B_0, C3 , T0); \
              traits.madd(A1, B_0, C7,  T0); \
              traits.madd(A2, B_0, C11, B_0); \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 3pX4"); \
            } while(false)

            internal::prefetch(blB);
            EIGEN_GEBP_ONESTEP(0);
            EIGEN_GEBP_ONESTEP(1);
            EIGEN_GEBP_ONESTEP(2);
            EIGEN_GEBP_ONESTEP(3);
            EIGEN_GEBP_ONESTEP(4);
            EIGEN_GEBP_ONESTEP(5);
            EIGEN_GEBP_ONESTEP(6);
            EIGEN_GEBP_ONESTEP(7);

            blB += pk*4*RhsProgress;
            blA += pk*3*Traits::LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 3pX4");
          }
          else
          {
            RhsPacket B_0, T0;
            LhsPacket A2;
            EIGEN_GEBP_ONESTEP(0);
            blB += 4*RhsProgress;
            blA += 3*Traits::LhsProgress;
          }

#undef EIGEN_GEBP_ONESTEP

          possiblyRotatingKernelHelper.unrotateResult(C0, C1, C2, C3);
          possiblyRotatingKernelHelper.unrotateResult(C4, C5, C6, C7);
          possiblyRotatingKernelHelper.unrotateResult(C8, C9, C10, C11);

          ResPacket R0, R1, R2;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r0.loadPacket(1 * Traits::ResPacketSize);
          R2 = r0.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C8, alphav, R2);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
          r0.storePacket(2 * Traits::ResPacketSize, R2);

          R0 = r1.loadPacket(0 * Traits::ResPacketSize);
          R1 = r1.loadPacket(1 * Traits::ResPacketSize);
          R2 = r1.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C1, alphav, R0);
          traits.acc(C5, alphav, R1);
          traits.acc(C9, alphav, R2);
          r1.storePacket(0 * Traits::ResPacketSize, R0);
          r1.storePacket(1 * Traits::ResPacketSize, R1);
          r1.storePacket(2 * Traits::ResPacketSize, R2);

          R0 = r2.loadPacket(0 * Traits::ResPacketSize);
          R1 = r2.loadPacket(1 * Traits::ResPacketSize);
          R2 = r2.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C2, alphav, R0);
          traits.acc(C6, alphav, R1);
          traits.acc(C10, alphav, R2);
          r2.storePacket(0 * Traits::ResPacketSize, R0);
          r2.storePacket(1 * Traits::ResPacketSize, R1);
          r2.storePacket(2 * Traits::ResPacketSize, R2);

          R0 = r3.loadPacket(0 * Traits::ResPacketSize);
          R1 = r3.loadPacket(1 * Traits::ResPacketSize);
          R2 = r3.loadPacket(2 * Traits::ResPacketSize);
          traits.acc(C3, alphav, R0);
          traits.acc(C7, alphav, R1);
          traits.acc(C11, alphav, R2);
          r3.storePacket(0 * Traits::ResPacketSize, R0);
          r3.storePacket(1 * Traits::ResPacketSize, R1);
          r3.storePacket(2 * Traits::ResPacketSize, R2);          
          }
        }
        // process remaining peeled loop
        for(Index k=peeled_kc; k<depth; k++)
        {
          for(Index i=i1; i<actual_panel_end; i+=3*LhsProgress)
          {
          // One column at a time
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(3*Traits::LhsProgress)];
          prefetch(&blA[0]);

          // gets res block as register
          AccPacket C0, C4, C8;
          traits.initAcc(C0);
          traits.initAcc(C4);
          traits.initAcc(C8);

          LinearMapper r0 = res.getLinearMapper(i, j2);
          r0.prefetch(0);

          // performs "inner" products
          const RhsScalar* blB = &blockB[j2*strideB+offsetB];
          LhsPacket A0, A1, A2;
          
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            LhsPacket A0, A1;
            RhsPacket B_0;
            RhsPacket T0;

            traits.loadLhs(&blA[0*LhsProgress], A0);
            traits.loadLhs(&blA[1*LhsProgress], A1);
            traits.loadRhs(&blB[0*RhsProgress], B_0);
            traits.madd(A0,B_0,C0,T0);
            traits.madd(A1,B_0,C4,B_0);
            traits.loadRhs(&blB[1*RhsProgress], B_0);
            traits.madd(A0,B_0,C1,T0);
            traits.madd(A1,B_0,C5,B_0);
          }
          else
          {
            LhsPacket A0, A1;
            RhsPacket B_0, B1, B2, B3;
            RhsPacket T0;

            traits.loadLhs(&blA[0*LhsProgress], A0);
            traits.loadLhs(&blA[1*LhsProgress], A1);
            traits.loadRhs(&blB[0*RhsProgress], B_0);
            traits.loadRhs(&blB[1*RhsProgress], B1);

            traits.madd(A0,B_0,C0,T0);
            traits.loadRhs(&blB[2*RhsProgress], B2);
            traits.madd(A1,B_0,C4,B_0);
            traits.loadRhs(&blB[3*RhsProgress], B3);
            traits.madd(A0,B1,C1,T0);
            traits.madd(A1,B1,C5,B1);
            traits.madd(A0,B2,C2,T0);
            traits.madd(A1,B2,C6,B2);
            traits.madd(A0,B3,C3,T0);
            traits.madd(A1,B3,C7,B3);
          }

          blB += nr*RhsProgress;
          blA += mr;
        }

        if(nr==4)
        {
          ResPacket R0, R1, R2, R3, R4, R5, R6;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = ploadu<ResPacket>(r0);
          R1 = ploadu<ResPacket>(r1);
          R2 = ploadu<ResPacket>(r2);
          R3 = ploadu<ResPacket>(r3);
          R4 = ploadu<ResPacket>(r0 + ResPacketSize);
          R5 = ploadu<ResPacket>(r1 + ResPacketSize);
          R6 = ploadu<ResPacket>(r2 + ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C8, alphav, R2);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
          r0.storePacket(2 * Traits::ResPacketSize, R2);          
          }
        }
        else
        {
          ResPacket R0, R1, R4;
          ResPacket alphav = pset1<ResPacket>(alpha);

    //---------- Process 2 * LhsProgress rows at once ----------
    if(mr>=2*Traits::LhsProgress)
    {
      const Index l1 = defaultL1CacheSize; // in Bytes, TODO, l1 should be passed to this function.
      // The max(1, ...) here is needed because we may be using blocking params larger than what our known l1 cache size
      // suggests we should be using: either because our known l1 cache size is inaccurate (e.g. on Android, we can only guess),
      // or because we are testing specific blocking sizes.
      Index actual_panel_rows = (2*LhsProgress) * std::max<Index>(1,( (l1 - sizeof(ResScalar)*mr*nr - depth*nr*sizeof(RhsScalar)) / (depth * sizeof(LhsScalar) * 2*LhsProgress) ));

      for(Index i1=peeled_mc3; i1<peeled_mc2; i1+=actual_panel_rows)
      {
        Index actual_panel_end = (std::min)(i1+actual_panel_rows, peeled_mc2);
        for(Index j2=0; j2<packet_cols4; j2+=nr)
        {
          for(Index i=i1; i<actual_panel_end; i+=2*LhsProgress)
          {
          
          // We selected a 2*Traits::LhsProgress x nr micro block of res which is entirely
          // stored into 2 x nr registers.
          
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(2*Traits::LhsProgress)];
          prefetch(&blA[0]);

          // gets res block as register
          AccPacket C0, C1, C2, C3,
                    C4, C5, C6, C7;
          traits.initAcc(C0); traits.initAcc(C1); traits.initAcc(C2); traits.initAcc(C3);
          traits.initAcc(C4); traits.initAcc(C5); traits.initAcc(C6); traits.initAcc(C7);

          LinearMapper r0 = res.getLinearMapper(i, j2 + 0);
          LinearMapper r1 = res.getLinearMapper(i, j2 + 1);
          LinearMapper r2 = res.getLinearMapper(i, j2 + 2);
          LinearMapper r3 = res.getLinearMapper(i, j2 + 3);

          r0.prefetch(prefetch_res_offset);
          r1.prefetch(prefetch_res_offset);
          r2.prefetch(prefetch_res_offset);
          r3.prefetch(prefetch_res_offset);

          // performs "inner" products
          const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
          prefetch(&blB[0]);
          LhsPacket A0, A1;

            traits.loadLhs(&blA[0*LhsProgress], A0);
            traits.loadRhs(&blB[0*RhsProgress], B_0);
            traits.loadRhs(&blB[1*RhsProgress], B1);
            traits.madd(A0,B_0,C0,B_0);
            traits.loadRhs(&blB[2*RhsProgress], B_0);
            traits.madd(A0,B1,C1,B1);
            traits.loadLhs(&blA[1*LhsProgress], A0);
            traits.loadRhs(&blB[3*RhsProgress], B1);
            traits.madd(A0,B_0,C0,B_0);
            traits.loadRhs(&blB[4*RhsProgress], B_0);
            traits.madd(A0,B1,C1,B1);
            traits.loadLhs(&blA[2*LhsProgress], A0);
            traits.loadRhs(&blB[5*RhsProgress], B1);
            traits.madd(A0,B_0,C0,B_0);
            traits.loadRhs(&blB[6*RhsProgress], B_0);
            traits.madd(A0,B1,C1,B1);
            traits.loadLhs(&blA[3*LhsProgress], A0);
            traits.loadRhs(&blB[7*RhsProgress], B1);
            traits.madd(A0,B_0,C0,B_0);
            traits.madd(A0,B1,C1,B1);
          }
          else
          {
            LhsPacket A0;
            RhsPacket B_0, B1, B2, B3;

            traits.loadLhs(&blA[0*LhsProgress], A0);
            traits.loadRhs(&blB[0*RhsProgress], B_0);
            traits.loadRhs(&blB[1*RhsProgress], B1);

            traits.madd(A0,B_0,C0,B_0);
            traits.loadRhs(&blB[2*RhsProgress], B2);
            traits.loadRhs(&blB[3*RhsProgress], B3);
            traits.loadRhs(&blB[4*RhsProgress], B_0);
            traits.madd(A0,B1,C1,B1);
            traits.loadRhs(&blB[5*RhsProgress], B1);
            traits.madd(A0,B2,C2,B2);
            traits.loadRhs(&blB[6*RhsProgress], B2);
            traits.madd(A0,B3,C3,B3);
            traits.loadLhs(&blA[1*LhsProgress], A0);
            traits.loadRhs(&blB[7*RhsProgress], B3);
            traits.madd(A0,B_0,C0,B_0);
            traits.loadRhs(&blB[8*RhsProgress], B_0);
            traits.madd(A0,B1,C1,B1);
            traits.loadRhs(&blB[9*RhsProgress], B1);
            traits.madd(A0,B2,C2,B2);
            traits.loadRhs(&blB[10*RhsProgress], B2);
            traits.madd(A0,B3,C3,B3);
            traits.loadLhs(&blA[2*LhsProgress], A0);
            traits.loadRhs(&blB[11*RhsProgress], B3);

            traits.madd(A0,B_0,C0,B_0);
            traits.loadRhs(&blB[12*RhsProgress], B_0);
            traits.madd(A0,B1,C1,B1);
            traits.loadRhs(&blB[13*RhsProgress], B1);
            traits.madd(A0,B2,C2,B2);
            traits.loadRhs(&blB[14*RhsProgress], B2);
            traits.madd(A0,B3,C3,B3);

            traits.loadLhs(&blA[3*LhsProgress], A0);
            traits.loadRhs(&blB[15*RhsProgress], B3);
            traits.madd(A0,B_0,C0,B_0);
            traits.madd(A0,B1,C1,B1);
            traits.madd(A0,B2,C2,B2);
            traits.madd(A0,B3,C3,B3);
          }

          blB += nr*4*RhsProgress;
          blA += 4*LhsProgress;
        }
        // process remaining peeled loop
        for(Index k=peeled_kc; k<depth; k++)
        {
          if(nr==2)
          {
            LhsPacket A0;
            RhsPacket B_0, B1;

            traits.loadLhs(&blA[0*LhsProgress], A0);
            traits.loadRhs(&blB[0*RhsProgress], B_0);
            traits.loadRhs(&blB[1*RhsProgress], B1);
            traits.madd(A0,B_0,C0,B_0);
            traits.madd(A0,B1,C1,B1);
          }
          else
          {
            LhsPacket A0;
            RhsPacket B_0, B1, B2, B3;

            traits.loadLhs(&blA[0*LhsProgress], A0);
            traits.loadRhs(&blB[0*RhsProgress], B_0);
            traits.loadRhs(&blB[1*RhsProgress], B1);
            traits.loadRhs(&blB[2*RhsProgress], B2);
            traits.loadRhs(&blB[3*RhsProgress], B3);

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r0.loadPacket(1 * Traits::ResPacketSize);
          R2 = r1.loadPacket(0 * Traits::ResPacketSize);
          R3 = r1.loadPacket(1 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C1, alphav, R2);
          traits.acc(C5, alphav, R3);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
          r1.storePacket(0 * Traits::ResPacketSize, R2);
          r1.storePacket(1 * Traits::ResPacketSize, R3);

          R0 = r2.loadPacket(0 * Traits::ResPacketSize);
          R1 = r2.loadPacket(1 * Traits::ResPacketSize);
          R2 = r3.loadPacket(0 * Traits::ResPacketSize);
          R3 = r3.loadPacket(1 * Traits::ResPacketSize);
          traits.acc(C2,  alphav, R0);
          traits.acc(C6,  alphav, R1);
          traits.acc(C3,  alphav, R2);
          traits.acc(C7,  alphav, R3);
          r2.storePacket(0 * Traits::ResPacketSize, R0);
          r2.storePacket(1 * Traits::ResPacketSize, R1);
          r3.storePacket(0 * Traits::ResPacketSize, R2);
          r3.storePacket(1 * Traits::ResPacketSize, R3);
          }
        }
      
        // Deal with remaining columns of the rhs
        for(Index j2=packet_cols4; j2<cols; j2++)
        {
          for(Index i=i1; i<actual_panel_end; i+=2*LhsProgress)
          {
          // One column at a time
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(2*Traits::LhsProgress)];
          prefetch(&blA[0]);

        ResScalar* r0 = &res[(j2+0)*resStride + i];
        ResScalar* r1 = r0 + resStride;
        ResScalar* r2 = r1 + resStride;
        ResScalar* r3 = r2 + resStride;

                  R0 = ploadu<ResPacket>(r0);
                  R1 = ploadu<ResPacket>(r1);
        if(nr==4) R2 = ploadu<ResPacket>(r2);
        if(nr==4) R3 = ploadu<ResPacket>(r3);

                  traits.acc(C0, alphav, R0);
                  traits.acc(C1, alphav, R1);
        if(nr==4) traits.acc(C2, alphav, R2);
        if(nr==4) traits.acc(C3, alphav, R3);

                  pstoreu(r0, R0);
                  pstoreu(r1, R1);
        if(nr==4) pstoreu(r2, R2);
        if(nr==4) pstoreu(r3, R3);
      }
      for(Index i=peeled_mc2; i<rows; i++)
      {
        const LhsScalar* blA = &blockA[i*strideA+offsetA];
        prefetch(&blA[0]);

        // gets a 1 x nr res block as registers
        ResScalar C0(0), C1(0), C2(0), C3(0);
        // TODO directly use blockB ???
        const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
        for(Index k=0; k<depth; k++)
        {
          if(nr==2)
          {
            LhsScalar A0;
            RhsScalar B_0, B1;

            A0 = blA[k];
            B_0 = blB[0];
            B1 = blB[1];
            MADD(cj,A0,B_0,C0,B_0);
            MADD(cj,A0,B1,C1,B1);
          }
          else
          {
            LhsScalar A0;
            RhsScalar B_0, B1, B2, B3;

            A0 = blA[k];
            B_0 = blB[0];
            B1 = blB[1];
            B2 = blB[2];
            B3 = blB[3];

            MADD(cj,A0,B_0,C0,B_0);
            MADD(cj,A0,B1,C1,B1);
            MADD(cj,A0,B2,C2,B2);
            MADD(cj,A0,B3,C3,B3);
          }

          R0 = r0.loadPacket(0 * Traits::ResPacketSize);
          R1 = r0.loadPacket(1 * Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          r0.storePacket(0 * Traits::ResPacketSize, R0);
          r0.storePacket(1 * Traits::ResPacketSize, R1);
          }
        }
                  res[(j2+0)*resStride + i] += alpha*C0;
                  res[(j2+1)*resStride + i] += alpha*C1;
        if(nr==4) res[(j2+2)*resStride + i] += alpha*C2;
        if(nr==4) res[(j2+3)*resStride + i] += alpha*C3;
      }
    }
    // process remaining rhs/res columns one at a time
    // => do the same but with nr==1
    for(Index j2=packet_cols; j2<cols; j2++)
    {
      // unpack B
      traits.unpackRhs(depth, &blockB[j2*strideB+offsetB], unpackedB);

      for(Index i=0; i<peeled_mc; i+=mr)
      {
        const LhsScalar* blA = &blockA[i*strideA+offsetA*mr];
        prefetch(&blA[0]);

        // TODO move the res loads to the stores

        // get res block as registers
        AccPacket C0, C4;
        traits.initAcc(C0);
        traits.initAcc(C4);

        const RhsScalar* blB = unpackedB;
        for(Index k=0; k<depth; k++)
        {
          LhsPacket A0, A1;
          RhsPacket B_0;
          RhsPacket T0;

          traits.loadLhs(&blA[0*LhsProgress], A0);
          traits.loadLhs(&blA[1*LhsProgress], A1);
          traits.loadRhs(&blB[0*RhsProgress], B_0);
          traits.madd(A0,B_0,C0,T0);
          traits.madd(A1,B_0,C4,B_0);

          blB += RhsProgress;
          blA += 2*LhsProgress;
        }
        ResPacket R0, R4;
        ResPacket alphav = pset1<ResPacket>(alpha);

        ResScalar* r0 = &res[(j2+0)*resStride + i];

          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gebp micro kernel 1pX1");
            RhsPacket B_0;
        
#define EIGEN_GEBGP_ONESTEP(K) \
            do {                                                                \
              EIGEN_ASM_COMMENT("begin step of gebp micro kernel 1pX1");        \
              EIGEN_ASM_COMMENT("Note: these asm comments work around bug 935!"); \
              traits.loadLhs(&blA[(0+1*K)*LhsProgress], A0);                    \
              traits.loadRhs(&blB[(0+K)*RhsProgress], B_0);                     \
              traits.madd(A0, B_0, C0, B_0);                                    \
              EIGEN_ASM_COMMENT("end step of gebp micro kernel 1pX1");          \
            } while(false);

            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*RhsProgress;
            blA += pk*1*Traits::LhsProgress;

            EIGEN_ASM_COMMENT("end gebp micro kernel 1pX1");
          }

        traits.acc(C0, alphav, R0);
        traits.acc(C4, alphav, R4);

        pstoreu(r0,               R0);
        pstoreu(r0+ResPacketSize, R4);
      }
      if(rows-peeled_mc>=LhsProgress)
      {
        Index i = peeled_mc;
        const LhsScalar* blA = &blockA[i*strideA+offsetA*LhsProgress];
        prefetch(&blA[0]);

        AccPacket C0;
        traits.initAcc(C0);

        const RhsScalar* blB = unpackedB;
        for(Index k=0; k<depth; k++)
        {
          LhsPacket A0;
          RhsPacket B_0;
          traits.loadLhs(blA, A0);
          traits.loadRhs(blB, B_0);
          traits.madd(A0, B_0, C0, B_0);
          blB += RhsProgress;
          blA += LhsProgress;
        }

        ResPacket alphav = pset1<ResPacket>(alpha);
        ResPacket R0 = ploadu<ResPacket>(&res[(j2+0)*resStride + i]);
        traits.acc(C0, alphav, R0);
        pstoreu(&res[(j2+0)*resStride + i], R0);
      }
      for(Index i=peeled_mc2; i<rows; i++)
      {
        const LhsScalar* blA = &blockA[i*strideA+offsetA];
        prefetch(&blA[0]);

        // gets a 1 x 1 res block as registers
        ResScalar C0(0);
        // FIXME directly use blockB ??
        const RhsScalar* blB = &blockB[j2*strideB+offsetB];
        for(Index k=0; k<depth; k++)
        {
          LhsScalar A0 = blA[k];
          RhsScalar B_0 = blB[k];
          MADD(cj, A0, B_0, C0, B_0);
        }
        res[(j2+0)*resStride + i] += alpha*C0;
      }
    }
  }


#undef CJMADD

// pack a block of the lhs
// The traversal is as follow (mr==4):
//   0  4  8 12 ...
//   1  5  9 13 ...
//   2  6 10 14 ...
//   3  7 11 15 ...
//
//  16 20 24 28 ...
//  17 21 25 29 ...
//  18 22 26 30 ...
//  19 23 27 31 ...
//
//  32 33 34 35 ...
//  36 36 38 39 ...
template<typename Scalar, typename Index, int Pack1, int Pack2, int StorageOrder, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs
{
  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const Scalar* EIGEN_RESTRICT _lhs, Index lhsStride, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, int Pack1, int Pack2, int StorageOrder, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_lhs<Scalar, Index, Pack1, Pack2, StorageOrder, Conjugate, PanelMode>
  ::operator()(Scalar* blockA, const Scalar* EIGEN_RESTRICT _lhs, Index lhsStride, Index depth, Index rows, Index stride, Index offset)
{
  typedef typename packet_traits<Scalar>::type Packet;
  enum { PacketSize = packet_traits<Scalar>::size };

  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK LHS");
  EIGEN_UNUSED_VARIABLE(stride)
  EIGEN_UNUSED_VARIABLE(offset)
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  eigen_assert( (StorageOrder==RowMajor) || ((Pack1%PacketSize)==0 && Pack1<=4*PacketSize) );
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  const_blas_data_mapper<Scalar, Index, StorageOrder> lhs(_lhs,lhsStride);
  Index count = 0;
  Index peeled_mc = (rows/Pack1)*Pack1;
  for(Index i=0; i<peeled_mc; i+=Pack1)
  {
    if(PanelMode) count += Pack1 * offset;

    if(StorageOrder==ColMajor)
    {
      for(Index k=0; k<depth; k++)
      {
        Packet A, B, C, D;
        if(Pack1>=1*PacketSize) A = ploadu<Packet>(&lhs(i+0*PacketSize, k));
        if(Pack1>=2*PacketSize) B = ploadu<Packet>(&lhs(i+1*PacketSize, k));
        if(Pack1>=3*PacketSize) C = ploadu<Packet>(&lhs(i+2*PacketSize, k));
        if(Pack1>=4*PacketSize) D = ploadu<Packet>(&lhs(i+3*PacketSize, k));
        if(Pack1>=1*PacketSize) { pstore(blockA+count, cj.pconj(A)); count+=PacketSize; }
        if(Pack1>=2*PacketSize) { pstore(blockA+count, cj.pconj(B)); count+=PacketSize; }
        if(Pack1>=3*PacketSize) { pstore(blockA+count, cj.pconj(C)); count+=PacketSize; }
        if(Pack1>=4*PacketSize) { pstore(blockA+count, cj.pconj(D)); count+=PacketSize; }
      }
    }
    else
    {
      for(Index k=0; k<depth; k++)
      {
        // TODO add a vectorized transpose here
        Index w=0;
        for(; w<Pack1-3; w+=4)
        {
          Scalar a(cj(lhs(i+w+0, k))),
                  b(cj(lhs(i+w+1, k))),
                  c(cj(lhs(i+w+2, k))),
                  d(cj(lhs(i+w+3, k)));
          blockA[count++] = a;
          blockA[count++] = b;
          blockA[count++] = c;
          blockA[count++] = d;
        }
        if(Pack1%4)
          for(;w<Pack1;++w)
            blockA[count++] = cj(lhs(i+w, k));
      }
    }
    if(PanelMode) count += Pack1 * (stride-offset-depth);
  }
  if(rows-peeled_mc>=Pack2)
  {
    if(PanelMode) count += Pack2*offset;
    for(Index k=0; k<depth; k++)
      for(Index w=0; w<Pack2; w++)
        blockA[count++] = cj(lhs(peeled_mc+w, k));
    if(PanelMode) count += Pack2 * (stride-offset-depth);
    peeled_mc += Pack2;
  }
  for(Index i=peeled_mc; i<rows; i++)
  {
    if(PanelMode) count += offset;
    for(Index k=0; k<depth; k++)
      blockA[count++] = cj(lhs(i, k));
    if(PanelMode) count += (stride-offset-depth);
  }
}

// copy a complete panel of the rhs
// this version is optimized for column major matrices
// The traversal order is as follow: (nr==4):
//  0  1  2  3   12 13 14 15   24 27
//  4  5  6  7   16 17 18 19   25 28
//  8  9 10 11   20 21 22 23   26 29
//  .  .  .  .    .  .  .  .    .  .
template<typename Scalar, typename Index, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, nr, ColMajor, Conjugate, PanelMode>
{
  typedef typename packet_traits<Scalar>::type Packet;
  enum { PacketSize = packet_traits<Scalar>::size };
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, int nr, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs<Scalar, Index, nr, ColMajor, Conjugate, PanelMode>
  ::operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Index depth, Index cols, Index stride, Index offset)
{
  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS COLMAJOR");
  EIGEN_UNUSED_VARIABLE(stride)
  EIGEN_UNUSED_VARIABLE(offset)
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  Index packet_cols = (cols/nr) * nr;
  Index count = 0;
  for(Index j2=0; j2<packet_cols; j2+=nr)
  {
    // skip what we have before
    if(PanelMode) count += nr * offset;
    const Scalar* b0 = &rhs[(j2+0)*rhsStride];
    const Scalar* b1 = &rhs[(j2+1)*rhsStride];
    const Scalar* b2 = &rhs[(j2+2)*rhsStride];
    const Scalar* b3 = &rhs[(j2+3)*rhsStride];
    for(Index k=0; k<depth; k++)
    {
      // skip what we have before
      if(PanelMode) count += 4 * offset;
      const LinearMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const LinearMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const LinearMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const LinearMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      Index k=0;
      if((PacketSize%4)==0) // TODO enbale vectorized transposition for PacketSize==2 ??
      {
        for(; k<peeled_k; k+=PacketSize) {
          PacketBlock<Packet,(PacketSize%4)==0?4:PacketSize> kernel;
          kernel.packet[0] = dm0.loadPacket(k);
          kernel.packet[1%PacketSize] = dm1.loadPacket(k);
          kernel.packet[2%PacketSize] = dm2.loadPacket(k);
          kernel.packet[3%PacketSize] = dm3.loadPacket(k);
          ptranspose(kernel);
          pstoreu(blockB+count+0*PacketSize, cj.pconj(kernel.packet[0]));
          pstoreu(blockB+count+1*PacketSize, cj.pconj(kernel.packet[1%PacketSize]));
          pstoreu(blockB+count+2*PacketSize, cj.pconj(kernel.packet[2%PacketSize]));
          pstoreu(blockB+count+3*PacketSize, cj.pconj(kernel.packet[3%PacketSize]));
          count+=4*PacketSize;
        }
      }
      for(; k<depth; k++)
      {
        blockB[count+0] = cj(dm0(k));
        blockB[count+1] = cj(dm1(k));
        blockB[count+2] = cj(dm2(k));
        blockB[count+3] = cj(dm3(k));
        count += 4;
      }
      // skip what we have after
      if(PanelMode) count += 4 * (stride-offset-depth);
    }
    // skip what we have after
    if(PanelMode) count += nr * (stride-offset-depth);
  }

  // copy the remaining columns one at a time (nr==1)
  for(Index j2=packet_cols; j2<cols; ++j2)
  {
    if(PanelMode) count += offset;
    const Scalar* b0 = &rhs[(j2+0)*rhsStride];
    for(Index k=0; k<depth; k++)
    {
      blockB[count] = cj(b0[k]);
      count += 1;
    }
    if(PanelMode) count += (stride-offset-depth);
  }
}

// this version is optimized for row major matrices
template<typename Scalar, typename Index, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, nr, RowMajor, Conjugate, PanelMode>
{
  enum { PacketSize = packet_traits<Scalar>::size };
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, int nr, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs<Scalar, Index, nr, RowMajor, Conjugate, PanelMode>
  ::operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Index depth, Index cols, Index stride, Index offset)
{
  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS ROWMAJOR");
  EIGEN_UNUSED_VARIABLE(stride)
  EIGEN_UNUSED_VARIABLE(offset)
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  Index packet_cols = (cols/nr) * nr;
  Index count = 0;
  for(Index j2=0; j2<packet_cols; j2+=nr)
  {
    // skip what we have before
    if(PanelMode) count += nr * offset;
    for(Index k=0; k<depth; k++)
    {
      const Scalar* b0 = &rhs[k*rhsStride + j2];
                blockB[count+0] = cj(b0[0]);
                blockB[count+1] = cj(b0[1]);
      if(nr==4) blockB[count+2] = cj(b0[2]);
      if(nr==4) blockB[count+3] = cj(b0[3]);
      count += nr;
    }
    // skip what we have after
    if(PanelMode) count += nr * (stride-offset-depth);
  }
  // copy the remaining columns one at a time (nr==1)
  for(Index j2=packet_cols; j2<cols; ++j2)
  {
    if(PanelMode) count += offset;
    const Scalar* b0 = &rhs[j2];
    for(Index k=0; k<depth; k++)
    {
      blockB[count] = cj(b0[k*rhsStride]);
      count += 1;
    }
    if(PanelMode) count += stride-offset-depth;
  }
}

} // end namespace internal

/** \returns the currently set level 1 cpu cache size (in bytes) used to estimate the ideal blocking size parameters.
  * \sa setCpuCacheSize */
inline std::ptrdiff_t l1CacheSize()
{
  std::ptrdiff_t l1, l2;
  internal::manage_caching_sizes(GetAction, &l1, &l2);
  return l1;
}

/** \returns the currently set level 2 cpu cache size (in bytes) used to estimate the ideal blocking size parameters.
  * \sa setCpuCacheSize */
inline std::ptrdiff_t l2CacheSize()
{
  std::ptrdiff_t l1, l2;
  internal::manage_caching_sizes(GetAction, &l1, &l2);
  return l2;
}

/** Set the cpu L1 and L2 cache sizes (in bytes).
  * These values are use to adjust the size of the blocks
  * for the algorithms working per blocks.
  *
  * \sa computeProductBlockingSizes */
inline void setCpuCacheSizes(std::ptrdiff_t l1, std::ptrdiff_t l2)
{
  internal::manage_caching_sizes(SetAction, &l1, &l2);
}

} // end namespace Eigen

#endif // EIGEN_GENERAL_BLOCK_PANEL_H
