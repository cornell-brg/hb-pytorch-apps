## kernel developer flow
 - 1. identify a kernel to port
 - 2. (possibly) register the kernel with PyTorch
 - 3. add host code, tests, kernel device code
 - 4. test using emulation
 - 5. test using cosim
 - 6. optimize
## workload developer flow
 -  1. identify a workload to port
 -  2. develop workload in pytorch-apps following coding conventions
 -  3. test serial native version (determine kernels not ported to HB)
        + if any kernels not ported yet, goto kernel developer flow
 -  4. test on HB emulation (possibly full workload, maybe one batch)
 -  5. profile parallel native version
 -  6. identify key kernels using native profiling data
 -  7. develop workload kernel file with reduced chunks
 -  8. test workload kernels on emulation
 -  9. test workload kernels on cosim
 - 10. profile workload kernels natively for baseline kernel results
 - 11. profile workload kernels on cosim
 - 12. combine full workload profile data + workload kernel data to
        estimate speedup of full workload on HB vs baseline
