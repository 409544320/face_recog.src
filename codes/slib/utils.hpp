
#pragma once

#include <cxcore.h>
#include <vector>
#include <string>

using namespace std;

// max
inline int max(int a, int b) {
	if (a > b) return a;
	return b;
}

// square of x
#ifndef SQ
#define SQ(x) ((x)*(x))
#endif // SQ

// abs
#define ABS(x) ((x)<0?-(x):(x))

// ALM Stopping Criteria
enum ALMStoppingCriteria {
	ALMSTOPPING_GROUND_TRUTH   = -1,
	ALMSTOPPING_DUALITY_GAP    = 1,
	ALMSTOPPING_SPARSE_SUPPORT = 2,
	ALMSTOPPING_OBJECTIVE_VALUE= 3,
	ALMSTOPPING_SUBGRADIENT    = 4,
	ALMSTOPPING_INCREMENTS     = 5,
	ALMSTOPPING_DEFAULT        = ALMSTOPPING_INCREMENTS
}; // enum ALMStoppingCriteria

// Dual Augmented Lagrange Multiplier (DALM) algorithm
// problem: min ||x||_1 s.t. b = Ax
void FastDALM(
	double *&x, 
	int&  nIter,
	double *b, 
	double *A, 
	double lambda, 
	double tol, 
	int maxIter, 
	int m, 
	int n, 
	ALMStoppingCriteria stop,
	double *xG,
	bool verbose= false
);

// load sample list 
void LoadSampleList(
	const string &list_file,	// sample list file
	vector<string> *sample_list // sample list
);

// load samples, return A,	n*m
// n: sample count
// m: sample_size.w*sample_size.h
CvMat* LoadSamples(
	const vector<string> &sample_list, // sample list, subject by subject
	const CvSize &sample_size // size of sample. if not equal to size of sample, resize to
);

// load sample, return y, 1*m
// m: sample_size.w*sample_size.h
inline CvMat* LoadSample(
	const string &path, // sample path
	const CvSize &sample_size // size of sample. if not equal to size of sample, resize to
) {
	return LoadSamples(vector<string>(1,path), sample_size);
}

// calc delta_i(x)
void DeltaFunction(
	const CvMat *x, // x, 1*n, n=k*n_subject_samples, k= n_subjects
	size_t n_subject_samples, // sample count per subject
	size_t i, // subject id: 0, 1, ..., k-1
	CvMat *delta_i_x // delta_i(x)
);

// calc square of residuals r_i(y)= ||y-A * delta_i(x)||_2
double Residuals(
	const CvMat *y,	// test sample
	const CvMat *A, // train samples
	const CvMat *delta_i_x // delta_i(x)
);

// calc sparsity concentration index
double SCI(
	const CvMat *x,
	size_t n_subject_samples
);

// get identity of y
// -1 returned if not found, else 0, 1, ..., k-1
int Identity(
	const CvMat *A, // train samples
	const CvMat *x, // 
	const CvMat *y, // test sample
	double sci_t, // if SCI(x)<sci_t, return -1
	size_t n_subject_samples,
	vector<double> *r= NULL
);

// get subject ID from subject filename
// "extended Yale Face Database B" file name format:
// eg. yaleB01_P00A+000E+00.pgm
inline string Filename2ID(const string &filename) {
	string::size_type index1= filename.find("yaleB");
	return filename.substr(index1, 7);
}

// get subject names of train samples from sample filenames using function Filename2ID
inline void TrainSubjectNames(
	const vector<string> &sample_list, // sample list, subject by subject
	size_t n_subject_samples, // sample count per subject
	vector<string> *subject_names // len= k
) {
	subject_names->clear();
	size_t k= sample_list.size()/n_subject_samples;
	for(size_t i= 0; i<sample_list.size(); i+= n_subject_samples) {
		subject_names->push_back(Filename2ID(sample_list[i]));
	}
}

/*
 SRC Model
*/

// model of Sparse Representation-based Classificatoin
struct SRCModel {
	size_t n_subject_samples_; // n/k
	CvSize sample_size_; // m= sample_size.w*sample_size.h
	vector<string> subject_names_; // k
	CvMat *A_;	// A, n*m
}; // struct SRCModel

// release SRCModel
inline void ReleaseSRCModel(SRCModel **p) {
	if(p==NULL || *p==NULL) { return; }
	cvReleaseMat(&((*p)->A_));
	delete *p;
	*p= NULL; 
}

// save SRCModel
void SaveSRCModel(const SRCModel *model, const string &path);

// load SRCModel
SRCModel* LoadSRCModel(const string &path);

// train SRCModel
SRCModel* TrainSRCModel(
	const vector<string> &train_sample_list, // train sample list, subject by subject, n
	const CvSize &sample_size, // size of sample. if not equal to size of sample, resize to
	size_t n_subject_samples // sample count per subject, n/k
);

// recognize test sample
string Recognize(
	const SRCModel *model, // SRC model
	const CvMat *y, // test sample
	double sci_t, // if SCI(x)<sci_t, return "Unknown"
	const char *x_file= NULL, // if x saved, not NULL
	const char *r_file= NULL // if residuals r saved, not NULL
); 