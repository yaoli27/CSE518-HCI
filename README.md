# CSE518-HCI
 CSE 518 Course Assignment Fall 2019

### Task

Given a dictionary containing 10000 words, use **SHARK2 algorithm** to decode a  user input gesture and output the best decoded word. (Ref: [Microsoft Word - p310-kristensson.doc (pokristensson.com)](http://pokristensson.com/pubs/KristenssonZhaiUIST2004.pdf))

### Demo

![demo!](https://user-images.githubusercontent.com/18680050/152302197-58d115d2-dc80-49a8-80bd-07975fd47b8c.png)

### Function List

#### 1. Sampling 

SHARK2 actually is doing comparations between user input pattern with standard  templates of each word. When we compare different patterns, it is important to make them comparable. No matter how long or how short the gesture is, we uniformly sample 100 points along the pattern.

#### 2. Pruning 

Compute start-to-start and end-to-end distances between a template and the  unknown gesture. Note that the two patterns are all normalized in scale and translation. Normalization is achieved by scaling the largest side of the bounding box to a  parameter L.

![format!](https://user-images.githubusercontent.com/18680050/152302676-ae464078-441e-4463-a206-9fef15857fa1.png)

![pruning lines!](https://user-images.githubusercontent.com/18680050/152302686-73cc7574-8a4f-4291-a966-e8c0c251a6b1.png)

#### 3. Shape Channel 

Relative coordinate, normalized

![normalization!](https://user-images.githubusercontent.com/18680050/152303349-1a18e0b8-f9a8-4c66-81eb-203881130df0.png)

#### 4. Location Channel 

Absolute coordinate, unnormalized

![location channel!](https://user-images.githubusercontent.com/18680050/152303438-b119b3fd-328c-4bae-b805-50d414c00e2f.png)

#### 5. Integration 

Integration score = a * shape score + b * location score 

where a + b = 1 (determined a good (a, b) )

#### 6. Get Best Word 

Select top-N, say, top-3 words with highest integration scores. 

Multiply with their corresponding probabilities. 

For example, integration_score(“too”) == integration_score(“to”), since prob(“too”)  < prob(“to”), integration_score(“too”) * prob(“too”) < integration_score(“to”) *  prob(“to”) 

Hence we choose word “to” and algorithm terminated.