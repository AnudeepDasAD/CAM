# Applying the Model to Non-binary Individuals

Since the model was trained on data which was labelled male or female (binary), we decided to evaluate the model on celebrities who identify as non-binary and analyze the results, both in terms of the prediction accuracy and the output CAM images. We also have well-known public figures as comparison. 

First, for every single input image, the model predicts that the individual is male, regardless of how they actually identify. We knew that the model would not be capable of capturing more nuanced gender identities as it was trained as a binary classifier, but this result shows the model predicts even more poorly than expected.

The CAM images produced are also quite flawed. They show the model as largely ignoring the face and instead focusing on different parts of the background which we would usually expect to play little role in identifying someone's gender. For these cases where the individual is non-binary, the model seems to break down and does an especially poor job. Our cisgender female example shows the model as focusing on important parts of the individual's face, which seems much more reasonable. 

![Non-binary CAM result 1. Model focuses on irrelevant parts of image](/non_binary_test/results/1.jpg)

![Non-binary CAM result 6. Model focuses on irrelevant parts of image](/non_binary_test/results/6.jpg)

![Non-binary CAM result 8. Model focuses on irrelevant parts of image](/non_binary_test/results/8.jpg)

![Non-binary CAM baseline. Model focuses on relevant parts of image](/non_binary_test/results/0.jpg)
