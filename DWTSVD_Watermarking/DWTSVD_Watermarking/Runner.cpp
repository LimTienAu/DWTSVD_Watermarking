#include "Sequential.h"

int main() {
	std::chrono::milliseconds embed_time, extract_time;
	sequential(&embed_time, &extract_time);

	cout << "Sequential Embed time : " << embed_time.count() << endl;
	cout << "Sequential Extract time : " << extract_time.count() << endl;
	return 0;
}