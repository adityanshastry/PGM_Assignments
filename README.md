# pgm_assignments
Assignments submitted as part of the Porbabilistic Graphical Models course at UMass (Spring 2017).

The topics include:
1) Simple computations of joint,and conditional probabilities for directed graphical models
2) OCR (Optical Character Recognition) for binary images using a trained CRF (Conditional Random Fields) model.
3) Training a CRF to perform OCR for binary images
4) Image denoising using McMC (Markoc chain Monte Carlo),specifically Gibbs' Sampler.
5) Training Bayesian and Variational Inference models for a given dataset

The code is written in Python with util files for common code, the generic code for graphical models, and specific code for 
solving the assignment questions. The libraries used are:
1) Numpy - for numerical analysis/computation
2) Autograd - for automatic differentiation for Bayesian and Variational Inference
3) Sklearn - for the optimization during learning the CRF model for OCR
4) PIL - for reading, and rendering images for image denoising

