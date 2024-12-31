# DWTSVD_Watermarking

This project aims to showcase implementation of DWT-SVD based Image Watermarking algorithm using sequential, OMP, CUDA and MPI implementation. 
Visual Studio 2022 is used for building the project. 

#Dependencies
The extra dependencies includes OpenCV, CUDA Toolkit, Eigen library, nlohmann::json and MPI library.
*Note : Ensure it is both release configuration in VS Project
1. Ensure CUDA Toolkit is installed on the device
2. Configure MPI library for project MPI: https://medium.com/geekculture/configuring-mpi-on-windows-10-and-executing-the-hello-world-program-in-visual-studio-code-2019-879776f6493f 
3. Install built opencv 4.10 from https://opencv.org/releases/ and choose Windows version. Create a folder called "Dependencies" and move the opencv folder into the Dependencies folder.
4. Make sure both projects has correct project setting path link to the corresponding dependencies. 
    Mainly check  
    - C++/General/Additional Include Directiories
    - General/C++ Language Standard :  C++ 17
    - C++/Language/OpenMP support :  Yes (For project DWTSVD_Watermarking)
    - Linker/Input
    - Build Events/Post-build events
5. Ensure in both projects, the Nuget packages for Eigen3 is installed

#Task distribution
Sequential - Teh Chong Shin, Lim Tien Au
OMP - Lim Tien Au
CUDA - Teh Chong Shin
MPI - Christopher Wong Jia He 

#Build program
Build both project in Release x64

#Run the program
- Open DSPC_main.ipynb in the same directory as the .sln file.