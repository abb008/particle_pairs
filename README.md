This is how I did it in vscode, Windows 11 (using cl):

1. Developer Command Prompt for Visual Studio.
2. Navigate to the project directory.
3. Compile the source code using:
      cl.exe /arch:AVX /EHsc /O2 /std:c++17 particle_pairs_array.cpp /link /out:particle_pairs_array.exe /STACK:16777216
4. Run the executable:
      particle_pairs_array.exe
