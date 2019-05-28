/*The example of LMS algorithm for Adaptive Filtering
 *Least mean squares (LMS) algorithms are a class of adaptive filter 
 *used to mimic a desired filter by finding the filter coefficients 
 *that relate to producing the least mean square of the error signal 
 *(difference between the desired and the actual signal). It is a 
 *stochastic gradient descent method in that the filter is only 
 *adapted based on the error at the current time.
 *The basic idea behind LMS filter is to approach the optimum filter 
 *weights {displaystyle (R^{-1}P)} (R^{-1}P), by updating the filter 
 *weights in a manner to converge to the optimum filter weight. This 
 *is based on the gradient descent algorithm. The algorithm starts by 
 *assuming small weights (zero in most cases) and, at each step, by 
 *finding the gradient of the mean square error, the weights are updated. 
 *That is, if the MSE-gradient is positive, it implies, the error would 
 *keep increasing positively, if the same weight is used for further 
 *iterations, which means we need to reduce the weights. In the same way, 
 *if the gradient is negative, we need to increase the weights.
 */

#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#define RandomGenerateSystemWeight 1
#define AFSampleNode 1000
#define AFCircleTimes 10
#define AFStepsize 0.14
#define AFSystemOrder 8

using namespace std;
using namespace Eigen;

/*Use FIR One-calss filtering system get the answer signal d*/
VectorXd filter(VectorXd f,int avg,VectorXd x,int sampleN){
	VectorXd d=VectorXd::Zero(sampleN);
	d[0]=f[0]*x[0];
	for(int i=1;i<sampleN;++i){
		for(int j=0;j<=i;++j){
			int fj=(j>=8?7:j);
			d[i]=d[i]+f[fj]*x[i-j];
		}
		d[i]-=d[i-1];
		d[i]/=avg;
	}
	return d;
}

/*Get the input signal from the random signal U*/
VectorXd getFromInput(VectorXd u,int j,const int L){
	VectorXd x=VectorXd::Zero(L);
	for(int i=0;i<L;++i){
		x[i]=u[j-i];
	}
	return x;
}

/*Function of Adaptive Filtering System*/
VectorXd adaptiveFilterSys(int circleTime,int sampleNode,const double stepSize,const int sysOrder=8){
	int i=0,k=0; //For loop

	VectorXd MSE=VectorXd::Zero(sampleNode); //mean least square
	
	#if RandomGenerateSystemWeight
	VectorXd unw=VectorXd::Random(sysOrder); //system-weight
	#else
	VectorXd unw=VectorXd(sysOrder);
    unw<<0.7753,-0.5286,0.6638,-0.3693,0.6873,-0.0742,0.2465,-0.4815;
    #endif
	
	
	VectorXd n=VectorXd::Zero(sysOrder); 
	
	for(k=0;k<circleTime;++k){
		VectorXd w=VectorXd::Zero(sysOrder);        //signal-weight
		VectorXd u=VectorXd::Random(sampleNode);
		VectorXd d=filter(unw,1,u,sampleNode);
		for(i=sysOrder;i<sampleNode;++i){
			VectorXd x=getFromInput(u,i,sysOrder);
			double e=d[i]-x.transpose()*w;
			w=w+stepSize*x*e;                //update function for signal-weight
			MSE[i]=MSE[i]+pow(e,2.0);
		}
	}
	for(k=0;k<sampleNode;++k){
		MSE[k]=MSE[k]/circleTime;
	}
	return MSE;
}
int main()
{
	int K = AFCircleTimes;      //independent run circleTime
	int N = AFSampleNode;       //sample node
	long double sum=0.0;
	const int L=AFSystemOrder;   //system order
    const double mu=AFStepsize;  //step size mu
	VectorXd MSE=VectorXd::Zero(N); //mean least square
    MSE=adaptiveFilterSys(K,N,mu,L);
	
	//calculate the average misalignment of the adaptive system
	for(int i=0;i<N;++i){
		sum=sum+MSE[i];
	}
	cout<<"MSE: "<<MSE<<endl;
	cout<<"avg of the MSE: "<<sum/N<<endl;
}