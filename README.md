## **Cancer Classification Using Weakly Labeled Whole Slide Images (WSIs)**

In this project our team tackled one of the biggest challenges in computational pathology : leveraging massive histopathological images for accurate cancer subtype prediction. This is the full pipeline of all our work which goes in the following sequence:
1. Data Preprocessing : Making patches of whole slide images.
2. Data Cleaning : Filtering out white and faulty patches from our dataset.
3. Feature Extraction : Extracting high quality features using SOTA models such as Conch 1.5.
4. Tissue Classifier : Training a tissue classifier using the CRC100K dataset to cluster the patches and average on the clusters to get a slide level embedding.
5. Slide Classification : Making the final prediction and evaluation.
