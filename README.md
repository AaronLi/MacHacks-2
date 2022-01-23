# MacHacks-2

## Note 
**Data for training and testing were provided from https://github.com/magarveylab/ec_chem_machacks2.git**

## Inspiration
Our team was drawn to this challenge due to the **bioengineering** backgrounds of some of our team members. In many of our chemistry and anatomy classes, we have learned about the **application and importance of enzymes**. Enzymes affect chemical reactions that are happening in our bodies every day. This project gave us the opportunity to further **extend our knowledge** and **use our computer science skills** to solve the challenge.  _This project is just one of the many pieces of the puzzle that come together to help us understand the complexity of living things._

## What it does
Our project takes the reactants and products of a chemical reaction in the form of SMILES and translates it into a machine-readable format. This is done by **tokenizing individual molecules** for relevant chemical features. The tokenization is inspired by **byte-pair-encoding** that is utilized in NLP. It is able to recognize recurring subunits across databases that may be of importance. These tokens are then assigned unique IDs using a dictionary so that they can be input into our model. Each reaction is input as a series of tokens. We used a **Long short-term memory (LSTM)** artificial recurrent neural network (RNN) to train our classifier. 

## How we built it
We used **Google Colab** to work in a collaborative coding setting. To translate the given data to meaningful input to our machine, we used **SMILES Pair Encoding**. [SMILES PE](https://github.com/XinhaoLi74/SmilesPE) learns the vocabulary of SMILES substrings from the **ChEMBL** dataset and uses the vocabulary to tokenize the values. We then created our own library of the tokens and assigned unique IDs to each token. The IDs/tokens present in each reaction were then input into our model. The model was trained using a LSTM RNN. A model was used to predict the class of the enzyme (the first EC number). The training set was then split by the main class of each reaction and 7 subclass models were trained to predict the subclasses. _When evaluating the models, the main model first predicted the class of the reaction; the reaction was then fed into the corresponding submodel to determine the second EC number._

## Challenges we ran into
- Determining a way to represent and quantize the SMILES input 

> Our team looked at several different methods to extract key features from SMILES such as using DeepChem featurizers. Ultimately we chose to use [SMILES PE](https://chemrxiv.org/engage/chemrxiv/article-details/60c74b76ee301cd51cc79e82) as it creates chemically explainable and human-readable SMILES substrings to be used as tokens. 

- Determining good hyperparameter values for our RNN

> Our team experimented with using different batch sizes, epochs and other hyperparameters to see what gave the best results. A particular challenge was balancing the accuracy of the model with both the risk of overfitting, as well as the runtime of fitting the model.

## Accomplishments that we're proud of
- This was everyone on the team's **first time making a deep learning (DL) model**. It was exciting to see everything come together!
- This was also our team's **first time working with NLP and tokenization**! Finding a way to preserve key information and features while quantizing the data was something new and of great importance to us.
- Our model was able to classify reactions for the first EC number with an **accuracy of 79%**! We were surprised and excited about this result! 
- For our team, this was one of the first projects that we worked on to **bridge the gap between chemistry and computer science**. It was interesting to see the intersection between the two fields and the breadth of applications for DL. 

## What we learned
- We learned about processing language and string inputs to _extract useful and quantifiable features_.
- We learned how to build and test deep learning models using **TensorFlow Keras**. 
- We learned about different tools and APIs for working with ML such as **Google Colab**.

> Using Google Colab, we were able to run our model using a _NVIDIA Tesla K80 GPU_. This was incredibly useful, as without the GPU our model would have taken several hours to fit. With the use of Google Colab, we were able to do this in a matter of minutes. 

## What's next for Denatured
- Using a larger training dataset to cover more inputs and EC classifications.
- Considering different DL models to tackle the problem. 
