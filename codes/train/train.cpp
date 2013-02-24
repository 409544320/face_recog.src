#include "../slib/utils.hpp"
#include <iostream>

using namespace std;

// print usage information
inline void Usage(const char *exe) {
	cerr<< "Usage: "<< endl
		<< "  "<< exe<< "train_sample_list_file n_subject_samples "
		<< "sample_width sample_height SRC-model"<< endl;
}

int main(int ac, char **av) {
	if(ac!=6) {
		Usage(av[0]);
		return -1;
	}

	const string train_sample_list_file= av[1];
	const size_t n_subject_samples= atoi(av[2]);
	const int sample_width= atoi(av[3]);
	const int sample_height= atoi(av[4]);
	CvSize sample_size= cvSize(sample_width, sample_height);
	const string src_model_file= av[5];

	vector<string> train_sample_list;
	LoadSampleList(train_sample_list_file, &train_sample_list);
	assert(train_sample_list.size() % n_subject_samples == 0);

	SRCModel *src= TrainSRCModel(train_sample_list, sample_size, n_subject_samples);
	SaveSRCModel(src, src_model_file);
	ReleaseSRCModel(&src);	

	return 0;
}