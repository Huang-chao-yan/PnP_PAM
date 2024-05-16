First, to set paths, type

>> startup

Next, to generate synthetic PET or SPECT data, type 

>> SetupSinogram
Enter desired SNR (5, 10, or 20). 10
Enter 1 for PET and 2 for SPECT. 1
For either STARNDARD Tikhonov, negative Laplacian, or total variation 
REGULARIZATION in conjunction with a regularization parameter selection 
method, run, for example,

>> Min_poisson
Enter 0 for Tikhonov; 1 for TV; and 2 for Laplacian. 0
Enter 1 for GCV, 2 for DP, 3 for UPRE, or 4 for no method 1
Enter lower bound for alpha 0
Enter upper bound for alpha 1

For HIERARCHICAL REGULARIZATION, run Min_poisson first with Laplacian 
regularization. Let one of DP, GCV, or UPRE compute a value for alpha.
Then type:

>> Reconstruct

