#pragma once

#include <cxcore.h>

// y := alpha * A * X + beta * y
inline void __dgemv(
	char trans, 
	int m, 
	int n, 
	double alpha, 
	double *A, // n * m
	int lda, 
	double *X, // m('T')
	int incx, 
	double beta, 
	double *y, // n('T')
	int incy
) {
	assert(incx==1 && incy==1);
	if(trans=='T') {
		CvMat A_mat= cvMat(n, m, CV_64FC1, A);
		CvMat X_mat= cvMat(m, 1, CV_64FC1, X);
		CvMat y_mat= cvMat(n, 1, CV_64FC1, y);
		cvGEMM(&A_mat, &X_mat, alpha, &y_mat, beta, &y_mat, 0);
	} else if(trans=='N') {
		CvMat A_mat= cvMat(n, m, CV_64FC1, A);
		CvMat X_mat= cvMat(n, 1, CV_64FC1, X);
		CvMat y_mat= cvMat(m, 1, CV_64FC1, y);
		cvGEMM(&A_mat, &X_mat, alpha, &y_mat, beta, &y_mat, CV_GEMM_A_T);
	} else {
		printf("error in function __dgemv");
		exit(-1);
	}
}