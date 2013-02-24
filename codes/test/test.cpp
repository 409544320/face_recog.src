#include "../slib/utils.hpp"
#include <iostream>

using namespace std;

// print usage information
inline void Usage(const char *exe) {
	cerr<< "Usage: "<< endl
		<< "  "<< exe<< "SRC-model test-sample-list-file SCI-threshold"<< endl;
}

int main(int ac, char **av) {
	if(ac!=4) {
		Usage(av[0]);
		return -1;
	}

	const string src_model_file= av[1];
	const string test_sample_list_file= av[2];
	const double sci_t= atof(av[3]);

	vector<string> test_sample_list;
	LoadSampleList(test_sample_list_file, &test_sample_list);

	SRCModel *src_model= LoadSRCModel(src_model_file);

	int ok_cnt= 0;
	for(size_t i=0; i<test_sample_list.size(); ++i) {
		CvMat *y= LoadSample(test_sample_list[i], src_model->sample_size_);
		string name= Recognize(src_model, y, sci_t, 
			(test_sample_list[i]+".x").c_str(), (test_sample_list[i]+".r").c_str());
		cout<< test_sample_list[i]<< " "
			<< Filename2ID(test_sample_list[i])<< " "
			<< name<< endl;
		if(Filename2ID(test_sample_list[i]) == name) {
			++ok_cnt;
		}
		cvReleaseMat(&y);
	}

	cout<< "ok count : "<< ok_cnt<< endl
		<< "total count : "<< test_sample_list.size()<< endl
		<< "precision : "<< double(ok_cnt)/test_sample_list.size()<< endl;

	ReleaseSRCModel(&src_model);

	return 0;
}