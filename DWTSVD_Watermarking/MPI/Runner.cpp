#include <mpi.h>
#include "Mpi.h"
#include <fstream>
#include "json.hpp"

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    std::chrono::milliseconds time;
    string original_image_path = "home.jpg"; //"apollo-medium.jpg"
    string watermark_image_path = "mono.png";
    double  psnr = 0;
    int watermark_width = 64;
    int watermark_height = 64;

    if (argc > 1 && argc != 5) {
        cerr << "Usage: " << argv[0] << " <original_image_path> <watermark_image_path> <watermark_width> <watermark_height>" << endl;
        return 1; // Exit with error code
    }
    else if (argc == 5)
    {
        // Parse command-line arguments
        original_image_path = argv[1];
        watermark_image_path = argv[2];

        // Convert watermark width and height from C-strings to integers
        watermark_width = atoi(argv[3]);
        watermark_height = atoi(argv[4]);
    }

    if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
        std::cout << "Starting MPI processing with " << size << " processes..." << std::endl;
    }

    // Perform embedding and extraction
    int mpi_status = mpi(&time, &psnr, false, original_image_path, watermark_image_path, rank, size);
    // Propagate any errors across ranks
    if (mpi_status != 0) {
        cerr << "Rank " << rank << ": MPI processing failed with status " << mpi_status << "." << endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
   if(rank==0) cout << "hahahahaha" << endl;
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);

    // Output results on rank 0
    if (rank == 0) {
        cout << "MPI time: " << time.count() << "ms." << endl;

        json output;
        output["psnr"] = psnr;
        output["time"] = time.count();

        string json_file_path = "mpi_time_result.json";

        append_to_json(json_file_path, output);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
