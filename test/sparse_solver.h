// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sparse.h"
#include <Eigen/SparseCore>

template<typename Solver, typename Rhs, typename DenseMat, typename DenseRhs>
void check_sparse_solving(Solver& solver, const typename Solver::MatrixType& A, const Rhs& b, const DenseMat& dA, const DenseRhs& db)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;

  DenseRhs refX = dA.householderQr().solve(db);
  {
    Rhs x(A.cols(), b.cols());
    Rhs oldb = b;

    solver.compute(A);
    if (solver.info() != Success)
    {
      std::cerr << "sparse solver testing: factorization failed (check_sparse_solving)\n";
      exit(0);
      return;
    }
    x = solver.solve(b);
    if (solver.info() != Success)
    {
      std::cerr << "sparse solver testing: solving failed\n";
      return;
    }
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");

    VERIFY(x.isApprox(refX,test_precision<Scalar>()));
    x.setZero();
    // test the analyze/factorize API
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Success)
    {
      std::cerr << "sparse solver testing: factorization failed (check_sparse_solving)\n";
      exit(0);
      return;
    }
    x = solver.solve(b);
    if (solver.info() != Success)
    {
      std::cerr << "sparse solver testing: solving failed\n";
      return;
    }
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");

    VERIFY(x.isApprox(refX,test_precision<Scalar>()));
  }
  
  // test dense Block as the result and rhs:
  {
    DenseRhs x(refX.rows(), refX.cols());
    DenseRhs oldb(db);
    x.setZero();
    x.block(0,0,x.rows(),x.cols()) = solver.solve(db.block(0,0,db.rows(),db.cols()));
    VERIFY(oldb.isApprox(db) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x.isApprox(refX,test_precision<Scalar>()));
  }
}

template<typename Solver, typename Rhs>
void check_sparse_solving_real_cases(Solver& solver, const typename Solver::MatrixType& A, const Rhs& b, const Rhs& refX)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef typename Mat::RealScalar RealScalar;
  
  Rhs x(A.cols(), b.cols());
  
  solver.compute(A);
  if (solver.info() != Success)
  {
    std::cerr << "sparse solver testing: factorization failed (check_sparse_solving_real_cases)\n";
    exit(0);
    return;
  }
  x = solver.solve(b);
  if (solver.info() != Success)
  {
    std::cerr << "sparse solver testing: solving failed\n";
    return;
  }
  
  RealScalar res_error;
  // Compute the norm of the relative error
  if(refX.size() != 0)
    res_error = (refX - x).norm()/refX.norm();
  else
  { 
    // Compute the relative residual norm
    res_error = (b - A * x).norm()/b.norm();
  }
  if (res_error > test_precision<Scalar>() ){
    std::cerr << "Test " << g_test_stack.back() << " failed in " EI_PP_MAKE_STRING(__FILE__) 
    << " (" << EI_PP_MAKE_STRING(__LINE__) << ")" << std::endl << std::endl;
    abort();
  }
  
}
template<typename Solver, typename DenseMat>
void check_sparse_determinant(Solver& solver, const typename Solver::MatrixType& A, const DenseMat& dA)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  
  solver.compute(A);
  if (solver.info() != Success)
  {
    std::cerr << "sparse solver testing: factorization failed (check_sparse_determinant)\n";
    return;
  }

  Scalar refDet = dA.determinant();
  VERIFY_IS_APPROX(refDet,solver.determinant());
}
template<typename Solver, typename DenseMat>
void check_sparse_abs_determinant(Solver& solver, const typename Solver::MatrixType& A, const DenseMat& dA)
{
  using std::abs;
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  
  solver.compute(A);
  if (solver.info() != Success)
  {
    std::cerr << "sparse solver testing: factorization failed (check_sparse_abs_determinant)\n";
    return;
  }

  Scalar refDet = abs(dA.determinant());
  VERIFY_IS_APPROX(refDet,solver.absDeterminant());
}

template<typename Solver, typename DenseMat>
int generate_sparse_spd_problem(Solver& , typename Solver::MatrixType& A, typename Solver::MatrixType& halfA, DenseMat& dA, int maxSize = 300)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  int size = internal::random<int>(1,maxSize);
  double density = (std::max)(8./(size*size), 0.01);

  Mat M(size, size);
  DenseMatrix dM(size, size);

  initSparse<Scalar>(density, dM, M, ForceNonZeroDiag);

  A = M * M.adjoint();
  dA = dM * dM.adjoint();
  
  halfA.resize(size,size);
  if(Solver::UpLo==(Lower|Upper))
    halfA = A;
  else
    halfA.template selfadjointView<Solver::UpLo>().rankUpdate(M);
  
  return size;
}


#ifdef TEST_REAL_CASES
template<typename Scalar>
inline std::string get_matrixfolder()
{
  std::string mat_folder = TEST_REAL_CASES; 
  if( internal::is_same<Scalar, std::complex<float> >::value || internal::is_same<Scalar, std::complex<double> >::value )
    mat_folder  = mat_folder + static_cast<std::string>("/complex/");
  else
    mat_folder = mat_folder + static_cast<std::string>("/real/");
  return mat_folder;
}
#endif

template<typename Solver> void check_sparse_spd_solving(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef SparseMatrix<Scalar,ColMajor, typename Mat::StorageIndex> SpMat;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    int size = generate_sparse_spd_problem(solver, A, halfA, dA);

    // generate the right hand sides
    int rhsCols = internal::random<int>(1,16);
    double density = (std::max)(8./(size*rhsCols), 0.1);
    SpMat B(size,rhsCols);
    DenseVector b = DenseVector::Random(size);
    DenseMatrix dB(size,rhsCols);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
  
    check_sparse_solving(solver, A,     b,  dA, b);
    check_sparse_solving(solver, halfA, b,  dA, b);
    check_sparse_solving(solver, A,     dB, dA, dB);
    check_sparse_solving(solver, halfA, dB, dA, dB);
    check_sparse_solving(solver, A,     B,  dA, dB);
    check_sparse_solving(solver, halfA, B,  dA, dB);
    
    // check only once
    if(i==0)
    {
      b = DenseVector::Zero(size);
      check_sparse_solving(solver, A, b, dA, b);
    }
  }
  
  // First, get the folder 
#ifdef TEST_REAL_CASES  
  if (internal::is_same<Scalar, float>::value 
      || internal::is_same<Scalar, std::complex<float> >::value)
    return ;
  
  std::string mat_folder = get_matrixfolder<Scalar>();
  MatrixMarketIterator<Scalar> it(mat_folder);
  for (; it; ++it)
  {
    if (it.sym() == SPD){
      Mat halfA;
      PermutationMatrix<Dynamic, Dynamic, Index> pnull;
      halfA.template selfadjointView<Solver::UpLo>() = it.matrix().template triangularView<Eigen::Lower>().twistedBy(pnull);
      
      std::cout<< " ==== SOLVING WITH MATRIX " << it.matname() << " ==== \n";
      check_sparse_solving_real_cases(solver, it.matrix(), it.rhs(), it.refX());
      check_sparse_solving_real_cases(solver, halfA, it.rhs(), it.refX());
    }
  }
#endif
}

template<typename Solver> void check_sparse_spd_determinant(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  generate_sparse_spd_problem(solver, A, halfA, dA, 30);
  
  for (int i = 0; i < g_repeat; i++) {
    check_sparse_determinant(solver, A,     dA);
    check_sparse_determinant(solver, halfA, dA );
  }
}

template<typename Solver, typename DenseMat>
Index generate_sparse_square_problem(Solver&, typename Solver::MatrixType& A, DenseMat& dA, int maxSize = 300, int options = ForceNonZeroDiag)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;

  Index size = internal::random<int>(1,maxSize);
  double density = (std::max)(8./(size*size), 0.01);
  
  A.resize(size,size);
  dA.resize(size,size);

  initSparse<Scalar>(density, dA, A, options);
  
  return size;
}

template<typename Solver> void check_sparse_square_solving(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef SparseMatrix<Scalar,ColMajor, typename Mat::StorageIndex> SpMat;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  int rhsCols = internal::random<int>(1,16);

  Mat A;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    Index size = generate_sparse_square_problem(solver, A, dA);

    A.makeCompressed();
    DenseVector b = DenseVector::Random(size);
    DenseMatrix dB(size,rhsCols);
    SpMat B(size,rhsCols);
    double density = (std::max)(8./(size*rhsCols), 0.1);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
    B.makeCompressed();
    check_sparse_solving(solver, A, b,  dA, b);
    check_sparse_solving(solver, A, dB, dA, dB);
    check_sparse_solving(solver, A, B,  dA, dB);
    
    // check only once
    if(i==0)
    {
      b = DenseVector::Zero(size);
      check_sparse_solving(solver, A, b, dA, b);
    }
  }
  
  // First, get the folder 
#ifdef TEST_REAL_CASES
  if (internal::is_same<Scalar, float>::value 
      || internal::is_same<Scalar, std::complex<float> >::value)
    return ;
  
  std::string mat_folder = get_matrixfolder<Scalar>();
  MatrixMarketIterator<Scalar> it(mat_folder);
  for (; it; ++it)
  {
    std::cout<< " ==== SOLVING WITH MATRIX " << it.matname() << " ==== \n";
    check_sparse_solving_real_cases(solver, it.matrix(), it.rhs(), it.refX());
  }
#endif

}

template<typename Solver> void check_sparse_square_determinant(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  
  for (int i = 0; i < g_repeat; i++) {
    // generate the problem
    Mat A;
    DenseMatrix dA;
    
    int size = internal::random<int>(1,30);
    dA.setRandom(size,size);
    
    dA = (dA.array().abs()<0.3).select(0,dA);
    dA.diagonal() = (dA.diagonal().array()==0).select(1,dA.diagonal());
    A = dA.sparseView();
    A.makeCompressed();
  
    check_sparse_determinant(solver, A, dA);
  }
}

template<typename Solver> void check_sparse_square_abs_determinant(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  for (int i = 0; i < g_repeat; i++) {
    // generate the problem
    Mat A;
    DenseMatrix dA;
    generate_sparse_square_problem(solver, A, dA, 30);
    A.makeCompressed();
    check_sparse_abs_determinant(solver, A, dA);
  }
}

template<typename Solver, typename DenseMat>
void generate_sparse_leastsquare_problem(Solver&, typename Solver::MatrixType& A, DenseMat& dA, int maxSize = 300, int options = ForceNonZeroDiag)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;

  int rows = internal::random<int>(1,maxSize);
  int cols = internal::random<int>(1,rows);
  double density = (std::max)(8./(rows*cols), 0.01);
  
  A.resize(rows,cols);
  dA.resize(rows,cols);

  initSparse<Scalar>(density, dA, A, options);
}

template<typename Solver> void check_sparse_leastsquare_solving(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef SparseMatrix<Scalar,ColMajor, typename Mat::StorageIndex> SpMat;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  int rhsCols = internal::random<int>(1,16);

  Mat A;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    generate_sparse_leastsquare_problem(solver, A, dA);

    A.makeCompressed();
    DenseVector b = DenseVector::Random(A.rows());
    DenseMatrix dB(A.rows(),rhsCols);
    SpMat B(A.rows(),rhsCols);
    double density = (std::max)(8./(A.rows()*rhsCols), 0.1);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
    B.makeCompressed();
    check_sparse_solving(solver, A, b,  dA, b);
    check_sparse_solving(solver, A, dB, dA, dB);
    check_sparse_solving(solver, A, B,  dA, dB);
    
    // check only once
    if(i==0)
    {
      b = DenseVector::Zero(A.rows());
      check_sparse_solving(solver, A, b, dA, b);
    }
  }
}
