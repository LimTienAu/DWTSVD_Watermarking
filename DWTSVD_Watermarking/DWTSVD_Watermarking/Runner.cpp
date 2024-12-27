#include "Sequential.h"
#include "Cuda.h"
#include "json.hpp"
#include "ompwatermark.h"

using json = nlohmann::json;

void append_to_json(const string& file_path, const json& new_data) {
    json existing_data;

    // Attempt to open the existing JSON file
    ifstream input_file(file_path);
    if (input_file.is_open()) {
        try {
            input_file >> existing_data; // Parse existing JSON content
        }
        catch (json::parse_error& e) {
            cerr << "Error parsing existing JSON file: " << e.what() << endl;
            // Handle parsing error (e.g., start with an empty JSON object)
            existing_data = json::array();
        }
        input_file.close();
    }
    else {
        // If the file doesn't exist, initialize as an empty array
        existing_data = json::array();
    }

    // Append new data
    existing_data.push_back(new_data);

    // Write the updated JSON back to the file
    ofstream output_file(file_path);
    if (output_file.is_open()) {
        output_file << existing_data.dump(4); // Pretty print with 4 spaces indentation
        output_file.close();
        cout << "New record has been appended to " << file_path << endl;
    }
    else {
        cerr << "Unable to open file for writing: " << file_path << endl;
    }
}

int main(int argc, char* argv[]) {

    std::chrono::milliseconds execution_time;
    string original_image_path = "apollo-medium.jpg"; //"apollo-medium.jpg"
    string watermark_image_path = "mono.png";
    double ave_execution_time, ave_psnr, psnr=0;
    int watermark_width = 64;
    int watermark_height = 64;
    int type = 0, loop = 3;

    if (argc > 1 && argc != 7) {
        cerr << "Usage: " << argv[0] << " <original_image_path> <watermark_image_path> <watermark_width> <watermark_height> <type> <loop_number>" << endl;
        cerr << "* For type: 0 = Serial, 1 = OMP, 2 = CUDA, 3 = MPI" << endl;
        return 1; // Exit with error code
    }
    else if (argc == 7)
    {
        // Parse command-line arguments
        original_image_path = argv[1];
        watermark_image_path = argv[2];

        // Convert watermark width and height from C-strings to integers
        watermark_width = atoi(argv[3]);
        watermark_height = atoi(argv[4]);
        type = atoi(argv[5]);
        loop = atoi(argv[6]);
    }

    if (watermark_width <= 0 || watermark_height <= 0) {
        cerr << "Error: Watermark width and height must be positive integers." << endl;
        return 1;
    }
    
    vector<long> store_execution_time, store_psnr;
    string type_name = "";
    switch (type) {
    case 1:
        type_name = "OMP";
        break;
    case 2:
        type_name = "CUDA";
        break;    
    case 3:
        type_name = "MPI";
        break;    
    case 0:
    default:
        type_name = "Serial";
    }

    for (int i = 0; i < loop; i++) {
        switch (type) {
        case 1:
            omp(&execution_time,&psnr, false, original_image_path, watermark_image_path, watermark_width, watermark_height);
            break;
        case 2:
            cuda_main(&execution_time, &psnr, false, original_image_path, watermark_image_path);
            break;
        case 3:
            //mpi(&execution_time, false, original_image_path, watermark_image_path);
            break;
        case 0:
        default:
            sequential(&execution_time, &psnr, false, original_image_path, watermark_image_path);
        }
        
        store_execution_time.push_back(execution_time.count());
        store_psnr.push_back(psnr);
        cout << "Loop " << i + 1 << " " << type_name << " execution time: " << execution_time.count() << "ms" << endl;
    }
    long total_execution_time = 0;
    double total_psnr = 0;
    for (int i = 0; i < loop; i++) {
        total_execution_time += store_execution_time[i];
        total_psnr += store_psnr[i];
    }
    // Calculate the average times
    ave_execution_time = static_cast<double>(total_execution_time) / loop;
    ave_psnr = total_psnr / loop;

    cout << endl<< type_name << " Average execution time: " << ave_execution_time << "ms. " << endl;

    json output;
    output["method"] = type_name;
    output["ori_image_path"] = original_image_path;
    output["watermark_image_path"] = watermark_image_path;
    output["psnr"] = ave_psnr;
    output["time"] = ave_execution_time;

    string json_file_path = "time_result.json";

    append_to_json(json_file_path, output);
    return 0;
}

