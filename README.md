# Text-to-Image-Generation-using-GAN
This project demonstrates a deep learning approach for generating synthetic images from textual descriptions using StackGAN. It focuses on converting fine-grained text embeddings (e.g., bird descriptions) into photo-realistic images, useful for data augmentation and generative AI research.
Gan-Model Training for the conversion of text to image using the Gan model and running mutiple epochs  to reducing the loss parameters and Increasing the accuracy of the images generated using the dataset.


Dataset
-------

This project uses the following datasets:

1. **CUB-200-2011 Birds Dataset**  
   A fine-grained image dataset containing over 11,000 images across 200 bird species.  
    🔗 Download: [CUB-200-2011 Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)

2. **Text Descriptions and Embeddings (cvpr2016_cub)**  
   Text annotations (10 per image) and precomputed char-CNN-RNN text embeddings used in Reed et al.'s CVPR 2016 paper.  
   🔗  Download: [cvpr2016_cub Dataset](https://www.kaggle.com/datasets/kartikvd24/cvpr2016-cub)

3. Text Captions (Natural Language Descriptions) 
   Each image in the CUB dataset has 10 human-annotated natural language descriptions. These are used as input text for training the StackGAN model.  
   🔗 Download: (https://www.kaggle.com/datasets/kartikvd24/birds-captions)

> Note: Ensure all datasets and embeddings are properly formatted and placed in the appropriate directories before training the model.

tools & Frameworks
 -------------------------------
 - Python
 - jyputer Notebook
 - Datasets used:
pandas,torch,transformers,numpy,skcitlearn,PIL,time,keras,matplotlib,pickle,torchvision,pillow,tensorflow,gensim.

note: these modules can be downloaded all together by the command in the command prompt or the terminal:
%pip install numpy pillow pandas tensorflow keras matplotlib torch torchvision transformers gensim


How to Run
----------

1. Install Requirements

if you are using Visual Studio Code download all the required modules for the projects,
in jyputer notebook or google cloud i.e. they already have inbiult modules,some of modules sholud be downloaded for jyputer notebook.

2. Prepare Dataset

Download the CUB dataset,cspr dataset with  text embeddings. Organize them according to the path in the code.

3. Train the Model

Use the file which has been provided, thar can be done in VS Code or Jyputer notebook or google collab and other applications also.

4. Generation of Images

Images are generated inside the notebook itself and it takes some random text embbedings and based on the text embbedings the images are generated


Model Architecture
------------------

The model utilized in this project is Generative Adversarial Networks (GANs) based for the task of generating real bird images given textual descriptions. The process follows a number of organized steps that include data preparation, model training, and adversarial optimization. The most important components and procedures are outlined below:

A. Dataset Preparation
The CUB-200-2011 birds dataset is utilized, which includes images of different birds along with their textual descriptions. Textual descriptions are converted into text embeddings by utilizing pre-trained models like CNN-RNN architectures. Each image is accompanied by its semantic embedding, thus enabling appropriate text-to-image mapping.

B. Text Embedding
The textual information is preprocessed and embedded into continuous vector representations, reflecting the semantic interpretation of each description. These embeddings are used as conditional inputs for both the Generator and the Discriminator networks.
C. Generator Network
The Generator accepts the text embeddings and noise vectors as inputs, generating images using a sequence of fully connected and convolutional layers. It upsamples the data progressively from a low resolution of 4×4 to a final image resolution of 64×64 pixels. As training continues over several epochs, the generated images become increasingly realistic and semantically aligned with the input text.

D. Discriminator Network
The Discriminator is tasked with distinguishing between original and synthetic images during validation for their consistency against the text embeddings. It takes the form of convolutional layers used for extracting features and sends adversarial signals to the Generator to facilitate iteration towards refinement in the synthesized images.

E. Adversarial Training
Both networks are trained in a competitive setup, where the Generator tries to mislead the Discriminator with more realistic images, and the Discriminator tries to correctly label images as real or fake and verify alignment with the text. This adversarial process enhances the performance of the system in subsequent epochs.

F. KL Divergence Loss
To regularize the latent space, Kullback–Leibler (KL) Divergence loss is employed within the CA_NET module. It minimizes the divergence between the posterior distribution (conditioned on text embeddings) and a standard normal prior N(0,1).
              DKL(q(z)∥p(z)) =−0.5×∑(1+log(σ2) −μ2−σ2)
This regularization makes sure the latent variables stay stable and sensible during training.

G. Image Generation
When training has ended, the Generator generates generated bird images given novel text descriptions. The images outputted have resolution 64×64 and show the required features described in the text input.

H. Model Training and Epochs
Training of the model is conducted through multiple epochs, with updates iteratively over the Generator and Discriminator. When epochs move along:
•	The Generator enhances image quality and text-image consistency.
•	The Discriminator strengthens its capacity to identify fakes and evaluate semantic coherence.
•	The overall adversarial loss steadily diminishes, reflecting improvement in performance.

I. Workflow Summary
Figure 1 shows the top-level workflow of the text-to-image GAN system. It starts with an input text file that describes the image, and then preprocessing and text encoding is done. 

![image]https://github.com/user-attachments/assets/83624947-97bf-4490-94fb-92fcda1942d6)-architecture of GAN network

Results
-------
Parameters
Epochs:5
Discriminator Loss: 1.1293970560856004, Generator Loss: 1.1720103672940096 after five Epochs

Example outputs from the model:
Text: "This yellow breasted bird has a black cheek."  
Generated Image: (![Screenshot 2025-05-08 183623](https://github.com/user-attachments/assets/cae98228-25ec-42f4-b83e-d7055d5320f9)

Applications
------------

- Synthetic data generation for training deep models
- Augmented datasets for fine-grained classification
- Creative AI and design automation

References
----------

- StackGAN: Han Zhang et al., "StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks", ICCV 2017.
- Reed et al., "Generative Adversarial Text to Image Synthesis", ICML 2016.




