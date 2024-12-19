#include "Sequential.h"

int main() {

	std::chrono::milliseconds embed_time, extract_time;
	string original_image_path = "home.jpg";
	string watermark_image_path = "mono.png";

	sequential(&embed_time, &extract_time, false, original_image_path, watermark_image_path);

	cout << "Sequential Embed time : " << embed_time.count() << "ms. " << endl;
	cout << "Sequential Extract time : " << extract_time.count() << "ms. " << endl;
	return 0;
}

