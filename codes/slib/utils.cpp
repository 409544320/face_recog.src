
#include "utils.hpp"
#include "blas_wrappers.hpp"
#include <cv.h>
#include <highgui.h>
#include <cmath>
#include <cstdio>
#include <fstream>

using namespace std;

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
	bool verbose
) {
	double const eps = 1e-15;

	int ldA = m;

	// beta = norm(b,1) / m;
	double beta = 0;
	for (long i = 0; i < m; i++) {
		beta += fabs (b[i]);
	}
	beta = beta / m;

	double betaInv = 1 / beta;

	nIter = 0;

	// y = zeros(m,1)
	double *y = new double[m];
	for (long i = 0; i < m; i++)
		y[i] = 0;

	// x = zeros(n,1)
	//x = new double[n];
	for (long i = 0; i < n; i++)
		x[i] = 0;

	// z = zeros (m+n,1);
	double *z = new double[n]; 
	for (long i = 0; i < n; i++)
		z[i] = 0;

	bool converged_main = false;

	// temp = A' * y;
	double *temp = new double [max(m,n)];
	__dgemv ('T', m, n, 1.0, A, ldA, y, 1, 0.0, temp, 1);
 
	double *x_old  = new double[n];
	double *temp1 = new double[max(m,n)];
	double *tmp = new double[max(m,n)];
	double *g = new double[m];

	// f = norm(x,1);  x is 0 at this point
	double f = 0;
	double prev_f = 0;
	double total = 0;
	double nxo, nx, dx;
	
	double *Ag = new double[n];
	
	double dg, dAg, alpha;
	
	do {
		nIter = nIter + 1;
		if(verbose) printf("==== [%d] ====\n", nIter);

		// x_old = x
		for (long i = 0; i < n; i++)
			x_old[i] = x[i];

		// % update z
		// temp1 = temp + x * betaInv
		// z = sign(temp1) .* min(1, abs(temp1));
		for (long i = 0; i < n; i++){
			temp1[i] = temp[i] + x[i] * betaInv;
		}
		for (long i = 0; i < n; i++){
			z[i] = (temp1[i] > 0 ? 1 : -1)
			       * ((fabs(temp1[i]) > 1) ? 1 : fabs(temp1[i]));
		}

		//    %compute A' * y    
		//    g = lambda * y - b + A * (beta * (temp - z) + x);
		for(long i = 0 ; i < n; i++){
			tmp[i] = beta * (temp[i] - z[i]) + x[i];
		}
		for(long i = 0; i < m; i++){
			g[i] = lambda * y[i] - b[i];
		}
		__dgemv('N', m, n, 1.0, A, ldA, tmp, 1, 1, g, 1);
	
		//    %alpha = g' * g / (g' * G * g);
		//    Ag = A' * g;
		__dgemv('T', m, n, 1.0, A, ldA, g, 1, 0.0, Ag, 1);
	
		//    alpha = g' * g / (lambda * g' * g + beta * Ag' * Ag);
		dg = 0;
		dAg = 0;
		for(long i = 0 ; i < n; i++){
			dAg += Ag[i] * Ag[i];
		}
		for(long i = 0 ; i < m; i++){
			dg += g[i] * g[i];
		}
		alpha = dg / (lambda * dg + beta * dAg);
	
		//    y = y - alpha * g;
		for(long i = 0 ; i < m; i++){
			y[i] = y[i] - alpha * g[i];
		}
	
		//    temp = A' * y;
		__dgemv ('T', m, n, 1.0, A, ldA, y, 1, 0.0, temp, 1);

		// % update x
		// x = x - beta * (z - temp);
		for (long i = 0; i < n; i++)
			x[i] = x[i] - beta * (z[i] - temp[i]);

		switch (stop){
		case ALMSTOPPING_GROUND_TRUTH:
		  total = 0;
		  for(int i = 0 ; i < n; i++){
		    total += (xG[i] - x[i])*(xG[i] - x[i]);
		  }
		  if (total < tol * tol)
		    converged_main = true;
		  break;

		case ALMSTOPPING_SUBGRADIENT:
		  printf("Duality gap is not a valid stopping criterion for ALM.");
		  break;

		case ALMSTOPPING_SPARSE_SUPPORT:
		  printf("DALM does not have a support set.");
		  break;

		case ALMSTOPPING_OBJECTIVE_VALUE:
		  prev_f = f;
		  f = 0;
		  for(int i = 0 ; i < n; i++){
		    f += fabs(x[i]);
		  }
		  if (fabs(f-prev_f)/prev_f <= tol){
		    converged_main = true;
		  }
		  break;

		case ALMSTOPPING_DUALITY_GAP:
		  printf("Duality gap is not a valid stopping criterion for ALM.");
		  break;

		case ALMSTOPPING_INCREMENTS:
		  // if norm(x_old - x) < tol * norm(x_old)
		  //     converged_main = true;
		  
		  nxo = 0;
		  for (int i = 0; i < n; i++)
		    nxo = nxo + x_old[i]*x_old[i];
		  nxo = sqrt (nxo);
		  
		  nx = 0;
		  for (int i = 0; i < n; i++)
		    nx = nx + x[i]*x[i];
		  nx = sqrt (nx);
		  
		  dx = 0;
		  for (int i = 0; i < n; i++)
		    dx = dx + (x_old[i]-x[i])*(x_old[i]-x[i]);
		  dx = sqrt(dx);
		  
		  if (dx < tol*nxo)
		    converged_main = true;
		  
		  if (verbose){
		    printf("  ||x|| = %f\n", nx);
		  }
		  
		  if (verbose){
		    if (nIter > 1){
		      printf ("  ||dx|| = %f (= %f * ||x_old||)\n",
				 dx, dx/(nxo+eps));
		    } else {
		      printf ("  ||dx|| = %f\n", dx);
		    }
		  }
		  break;

		default:
		  printf("Undefined stopping criterion.");
		  break;
		}

 		if (nIter >= maxIter){
			if (verbose)
				printf ("Maximum Iterations Reached\n");
			converged_main = true;
		}
	} while (!converged_main);

	if (verbose) printf("==== CONVERGED ==== \n", nIter);

	delete [] tmp;
	delete [] g;
	delete [] Ag;
	delete [] y;
	delete [] z;
	delete [] x_old;
	delete [] temp;
	delete [] temp1;
}

// load sample list 
void LoadSampleList(
	const string &list_file,	// sample list file
	vector<string> *sample_list // sample list
) {
	ifstream fin(list_file.c_str());
	string line;
	while(true) {
		line="";
		getline(fin, line, '\n');
		if(line=="") { break; }
		sample_list->push_back(line);
	}
	fin.close();
}

// load samples, return A,	n*m
// n: sample count
// m: sample_size.w*sample_size.h
CvMat* LoadSamples(
	const vector<string> &sample_list, // sample list, subject by subject
	const CvSize &sample_size // size of sample. if not equal to size of sample, resize to
) {
	CvMat *A= cvCreateMat(sample_list.size(), sample_size.width*sample_size.height, CV_64FC1);
	IplImage *resz= cvCreateImage(sample_size, IPL_DEPTH_8U, 1);
	IplImage *resz_db= cvCreateImage(sample_size, IPL_DEPTH_64F, 1);
	for(size_t i=0; i<sample_list.size(); ++i) {
		IplImage *image= cvLoadImage(sample_list[i].c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		cvResize(image, resz, CV_INTER_LINEAR);
		cvConvertScale(resz, resz_db, 1.0);
		cvNormalize(resz_db, resz_db, 1.0, 0, CV_L2);
		memcpy(cvPtr2D(A,i,0), resz_db->imageData, sizeof(double)*A->cols);
		cvReleaseImage(&image);
	}
	cvReleaseImage(&resz);
	cvReleaseImage(&resz_db);
	return A;
}

// calc delta_i(x)
void DeltaFunction(
	const CvMat *x, // x, 1*n, n=k*n_subject_samples, k= n_subjects
	size_t n_subject_samples, // sample count per subject
	size_t i, // subject id: 0, 1, ..., k-1
	CvMat *delta_i_x // delta_i(x)
) {
	cvSetZero(delta_i_x);
	size_t start= i*n_subject_samples*sizeof(double);
	size_t len= n_subject_samples*sizeof(double);
	memcpy(delta_i_x->data.ptr+start, x->data.ptr+start, len);
}

// calc square of residuals r_i(y)= ||y-A * delta_i(x)||_2
double Residuals(
	const CvMat *y,	// test sample
	const CvMat *A, // train samples, n*m
	const CvMat *delta_i_x // delta_i(x)
) {
	double r= 0;
	int m= y->cols;
	int n= delta_i_x->cols;
	const double *y_data= y->data.db;
	const double *x_data= delta_i_x->data.db;
	for(int i=0; i<m; ++i) {
		double sum= 0;
		const double *A_ptr= A->data.db+i;
		for(int j=0; j<n; ++j) {
			sum+= A_ptr[0]*x_data[j];
			A_ptr+= m;
		}
		r+= SQ(y_data[i]-sum);
	}
	return r;
}

inline double __L1Norm(const double *data, size_t len) {
	double norm= 0;
	for(size_t i=0; i<len; ++i) {
		norm+= ABS(data[i]);
	}
	return norm;
}

// calc sparsity concentration index
double SCI(
	const CvMat *x,
	size_t n_subject_samples
) {
	double max_val= -1.0;
	size_t k= (size_t)(x->cols)/n_subject_samples;
	const double *x_ptr= x->data.db;
	for(size_t i=0; i<k; ++i) {
		double val= __L1Norm(x_ptr, n_subject_samples);
		if(val>max_val) {
			max_val= val;
		}
		x_ptr+= n_subject_samples;
	}
	return (k * max_val / __L1Norm(x->data.db,x->cols) - 1) / ( k -1 );
}

// get identity of y
// -1 returned if not found, else 0, 1, ..., k-1
int Identity(
	const CvMat *A, // train samples
	const CvMat *x, // 
	const CvMat *y, // test sample
	double sci_t, // if SCI(x)<sci_t, return -1
	size_t n_subject_samples,
	vector<double> *rv
) {
	double sci= SCI(x,n_subject_samples);
	printf("%f ", sci);
	if(sci<sci_t) { return -1; }

	if(rv) { rv->clear(); }

	CvMat *delta_i_x= cvCreateMat(1, x->cols, CV_64FC1);
	int k= x->cols/int(n_subject_samples);
	double min_r= FLT_MAX;
	int id= -1;
	for(int i=0; i<k; ++i) {
		DeltaFunction(x, n_subject_samples, i, delta_i_x);
		double r= Residuals(y, A, delta_i_x);
		if(rv) { rv->push_back(r); }
		if(r<min_r) { 
			min_r= r;
			id= i;
		}
	}
	cvReleaseMat(&delta_i_x);
	return id;
}

/*
 SRC Model
*/

inline void __SaveString(const string &s, ostream &out) {
	size_t len= s.size();
	out.write((char*)&len, sizeof(len));
	out.write(s.c_str(), len*sizeof(char));
}

inline void __LoadString(istream &in, string *s) {
	size_t len;
	in.read((char*)&len, sizeof(len));
	s->resize(len);
	in.read(&((*s)[0]), len*sizeof(char));
}

// save SRCModel
void SaveSRCModel(const SRCModel *model, const string &path) {
	ofstream out(path.c_str(), ios_base::binary);
	out.write((char*)&(model->n_subject_samples_), sizeof(model->n_subject_samples_));
	out.write((char*)&(model->sample_size_), sizeof(model->sample_size_));
	size_t k= model->subject_names_.size();
	out.write((char*)&k, sizeof(k));
	for(size_t i=0; i<k; ++i) {
		__SaveString(model->subject_names_[i], out);
	}
	out.write((char*)model->A_->data.ptr, model->A_->rows*model->A_->cols*sizeof(double));
}

// load SRCModel
SRCModel* LoadSRCModel(const string &path) {
	SRCModel *model= new SRCModel;
	ifstream in(path.c_str(), ios_base::binary);
	in.read((char*)&(model->n_subject_samples_), sizeof(model->n_subject_samples_));
	in.read((char*)&(model->sample_size_), sizeof(model->sample_size_));

	size_t k;
	in.read((char*)&k, sizeof(k));
	model->subject_names_.resize(k);
	for(size_t i=0; i<k; ++i) {
		__LoadString(in, &model->subject_names_[i]);
	}

	size_t n= k*model->n_subject_samples_;
	size_t m= model->sample_size_.width * model->sample_size_.height;
	model->A_= cvCreateMat(n, m, CV_64FC1);
	in.read((char*)model->A_->data.ptr, m*n*sizeof(double));

	return model;
}

// train SRCModel
SRCModel* TrainSRCModel(
	const vector<string> &train_sample_list, // train sample list, subject by subject, n
	const CvSize &sample_size, // size of sample. if not equal to size of sample, resize to
	size_t n_subject_samples // sample count per subject, n/k
) {
	SRCModel *model= new SRCModel;
	model->n_subject_samples_= n_subject_samples;
	model->sample_size_= sample_size;
	TrainSubjectNames(train_sample_list, n_subject_samples, &(model->subject_names_));
	model->A_= LoadSamples(train_sample_list, sample_size);
	return model;
}

// recognize test sample
string Recognize(
	const SRCModel *model, // SRC model
	const CvMat *y, // test sample
	double sci_t, // if SCI(x)<sci_t, return "Unknown"
	const char *x_file, // if x saved, not NULL
	const char *r_file // if residuals r saved, not NULL
) {
	CvMat *x= cvCreateMat(1, model->A_->rows, CV_64FC1);

	int max_iters= 1000, iters;
	double lambda= 0.01;
	double tol= 0.0001;
	vector<double> r;
	FastDALM(x->data.db, iters, y->data.db, model->A_->data.db, lambda, tol, max_iters, 
			model->A_->cols, model->A_->rows, ALMSTOPPING_INCREMENTS, NULL, false);
	int id= Identity(model->A_, x, y, sci_t, model->n_subject_samples_, &r);

	if(x_file) {
		ofstream xfout(x_file);
		copy(x->data.db, x->data.db+x->cols, ostream_iterator<double>(xfout, "\n"));
		xfout.close();
	}

	if(r_file) {
		ofstream rfout(r_file);
		copy(r.begin(), r.end(), ostream_iterator<double>(rfout, "\n"));
		rfout.close();
	}

	cvReleaseMat(&x);

	if(id==-1) { 
		return "Unknown";
	}
	return model->subject_names_[id];
}