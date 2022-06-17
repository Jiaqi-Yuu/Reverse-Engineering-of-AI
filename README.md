# Reverse-Engineering-of-AI
With its commercialization, machine learning as a service (MLaaS) is becoming more and more popular, and providers are paying more attention to the privacy of models and the protection of intellectual property. Generally speaking, the machine learning service deployed on the cloud platform is a black box, where users can only obtain outputs by providing inputs to the model. The attributes of the model such as architecture, training set, training method, are concealed by provider. In this work, we devise a multi-discriminator generative adversarial network(MDGAN) to learn domain invariant features. Based on these features, we can learn a domain-free model to inversely infer the attributes of a target black-box model with unknown training data. The experiments consume a lot of computing resources, so I provide some prepared data. I only show part of this work.

# Requirement
PyTorch == 1.10.2

# Run
```python
python train.py
```

# prepared_data
We pretrained a number of models which are constructed by enumerating all posible attribute values. The details of the attributes and their values are shown in Table1. We sample 5000, 1000 from white-box models as the training set and testing set. 
The 'prepared_outputs_train(cartoon&sketch)' contains 10000 models' outputs(5000 from cartoon, 5000 from sketch). This file is very large, so you need to unzip it.
The inputs is 100 queries listed in 'save_query(cartoon&sketch)'
With the data prepared in advance, the training efficiency of the model is greatly improved!

# PACS
PACS is an image dataset that has been widely used for OOD(out of distribution) learning. We utilize three domains, including Photo(1,670 images), Cartoon(2,344 images) and Sketch(3,929 images), to construct our dataset. Each domain contains 7 categories(Dog, Elephant, Giraffe, Guitar, Horse, House, Person).


![在这里插入图片描述](https://img-blog.csdnimg.cn/09d6adbbc16e4aaeadd44b30525cfe4c.jpeg#pic_center)

# tsne_show
To verify the effectiveness of our proposed method, we utilize t-SNE to visualize samples in the domain invariant feature spave learnt by the generator G in MDGAN. The visualization is carried out on PACS-modelset. As shown in figure, samples from the three different domains are grouped into individual clusters at the 1st epoch. Our method embeds features from the unseen domain and the samples from the target domain also become closer to the source domains at 20th epoch, indicating that our generator is able to generalize an unseen domain into the feature space where the source domains are in.

# future
I will provide more details in the future.

# Reference
[1] Seong Joon Oh, Max Augustin, Bernt Schiele, and Mario Fritz. Towards Reverse-Engineering Black-Box Neural Networks. arXiv:1711.01768 [cs, stat], February 2018. arXiv: 1711.01768.
[2] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.
