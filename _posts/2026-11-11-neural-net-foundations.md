# Lesson 3.5 Neural net foundations

Following lesson 3 and chapter 4 of the [fastai course](https://course.fast.ai) and [book](https://github.com/fastai/fastbook), I built a linear classifier from scratch and a neural network with one hidden layer using PyTorch and fastai to distinguish between the flags of Luxembourg and the Netherlands. Both models use cross-entropy as the loss function. <br>
I also visualized the weights of the linear learner to look at the features it uses to separate the two classes.

## Loading the data


```python
from pathlib import Path
from fastai.vision.all import *
```


```python
path = Path("deep-learning-notes.github.io/european_countries")
path.exists()
```




    True




```python
luxembourg = (path / "luxembourg").ls()
netherlands = (path / "netherlands").ls()
len(luxembourg), len(netherlands)
```




    (135, 125)




```python
size = (256, 256)

# tensor() is added by fastai to convert PIL images to PyTorch tensors
luxembourg = [tensor(Image.open(img).convert("RGB").resize(size)) for img in luxembourg]
netherlands = [tensor(Image.open(img).convert("RGB").resize(size)) for img in netherlands]
```


```python
show_image(luxembourg[0]);
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_7_0.png)
    


Normalize the images between values of 0 and 1.


```python
# Images remain in channel-last format, so (height, width, channels), instead of PyTorch's (channels, height, width).
luxembourg_stacked = torch.stack(luxembourg).float()/255
netherlands_stacked = torch.stack(netherlands).float()/255

luxembourg_stacked.shape, netherlands_stacked.shape
```




    (torch.Size([135, 256, 256, 3]), torch.Size([125, 256, 256, 3]))



Since the images are in RGB, we have a rank 4 tensor:
- 1st dimension: the number of images
- 2nd dimension: the height of the image
- 3rd dimension: the width of the image
- 4th dimension: RGB channels

## Creating a dataset


```python
all_images = torch.cat([netherlands_stacked, luxembourg_stacked], dim=0)
all_images.shape
```




    torch.Size([260, 256, 256, 3])



Because ``all_images`` begins with all the Netherlands flags (label 0) and ends with all the Luxembourg flags (label 1), labels can be created like this:


```python
labels = torch.cat([torch.zeros(len(netherlands), dtype=torch.long), torch.ones(len(luxembourg),  dtype=torch.long)])
labels.shape
```




    torch.Size([260])




```python
split_idx = int(0.8 * len(labels))
split_idx
```




    208



Randomly shuffle ``all_images`` before splitting it into train and validation set.


```python
torch.manual_seed(42) # for reproducibility

indices = torch.randperm(len(labels))
shuffled_images = all_images[indices]
shuffled_labels = labels[indices]

shuffled_images.shape, shuffled_labels.shape
```




    (torch.Size([260, 256, 256, 3]), torch.Size([260]))




```python
train_images, train_labels = shuffled_images[:split_idx], shuffled_labels[:split_idx]
valid_images, valid_labels = shuffled_images[split_idx:], shuffled_labels[split_idx:]
```

Define a training set.


```python
dset_train = list(zip(train_images, train_labels))

img, label = dset_train[0]
show_image(img); label
```




    tensor(0)




    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_20_1.png)
    


Define a validation set.


```python
dset_valid = list(zip(valid_images, valid_labels))

img, label = dset_valid[0]
show_image(img); label
```




    tensor(0)




    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_22_1.png)
    


As we only have a relatively small dataset, doing full-batch gradient descent makes the gradient account for more images, in turn making it more meaningful. <br>
This makes the training more stable as gradients tend to not jump around as much.

As a downside however, we get fewer weight updates per epoch.


```python
dl_train = DataLoader(dset_train, batch_size=len(dset_train))
dl_valid = DataLoader(dset_valid, batch_size=len(dset_valid))
```

## Establishing a baseline using pixel similarity

Now we try the same approach as in the book, using pixel similarity (the mean of the images).


```python
luxembourg_mask = (train_labels == 1) # luxembourg is labeled as 1 in the dataset
netherlands_mask = (train_labels == 0) # netherlands as 0

mean_luxembourg = train_images[luxembourg_mask].mean(dim=0)
mean_netherlands = train_images[netherlands_mask].mean(dim=0)

show_image(mean_luxembourg); show_image(mean_netherlands);
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_27_0.png)
    



    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_27_1.png)
    


For the distance function, I'll use L1 distance, so we take the absolute values of the differences instead of their squares.


```python
def L1_distance(a,b): return (a - b).abs().mean(dim=(-3, -2, -1)) # average the image sizes and RGB channels (last 3 dimensions) into a single number
```

Let's have the model evaluate against the validation set.


```python
all_means = torch.stack([mean_netherlands, mean_luxembourg], dim=0)
all_means.shape
```




    torch.Size([2, 256, 256, 3])



Now get the overall accuracy, by broadcasting ``valid_images`` to shape: [52, 1, 256, 256, 3] and ``all_means`` to shape: [1, 2, 256, 256, 3], such that each validation image gets subtracted with both means.


```python
distances = L1_distance(valid_images.unsqueeze(1), all_means.unsqueeze(0)) # distance of each image to the mean images
distances.shape
```




    torch.Size([52, 2])




```python
predictions = torch.argmin(distances, dim=1)
baseline_accuracy = (predictions == valid_labels).float().mean()
baseline_accuracy
```




    tensor(0.6154)



An accuracy of about 62% appears better than random guessing, but with just 52 validation images, random guessing has a high variance and a single image can shift the validation accuracy by $\frac{1}{52} \approx 0.019$.
So it is still possible for a score of 62% to occur by chance when randomly guessing. <br>

However the averaged images show certain features that clearly separate both classes from one another.<br>
This is especially visible in their third color stripe and the watermarks which are more prominent in the Netherlands mean than in Luxembourg's. <br>
If these differences don't get drowned out by other, unimportant differences, this approach can actually work quite well.

With this approach, each pixel counts the same, so the logical next step would be to take a weighted sum of the pixels, making areas with features that separate the classes count more and areas with noise count less.

## Trying a linear learner

To achieve this weighted sum, let's use machine learning.

As mentioned in the book, there are 7 steps, that are needed for using machine learning: 

### Initialize the weights

Because there are 256 x 256 x 3 inputs, summing all of them with default ``torch.randn```produces huge initial outputs.

The model could not learn anything until the weights were scaled down, as the initial weights were so big, that the small parameter updates could not overcome the initialization noise.


```python
num_inputs = 256 * 256 * 3
num_outputs = 2

# Initialize the weights very small
weights = (torch.randn((num_inputs, num_outputs))*0.00001).requires_grad_()
bias = (torch.zeros(num_outputs)).requires_grad_()
```

### Make predictions


```python
def linear(x): 
	# Flatten the input tensor to a 2D tensor with shape [batch_size, num_inputs], to make use of matrix multiplication with the weights.
	return (x.flatten(1)@weights) + bias

weights.shape, bias.shape
```




    (torch.Size([196608, 2]), torch.Size([2]))



Quick test run


```python
preds = linear(train_images)
preds.shape
```




    torch.Size([208, 2])




```python
(torch.argmax(preds, dim=1) == train_labels).float().mean()
```




    tensor(0.4760)



This first iteration's accuracy shows that the model is randomly guessing (as expected with untrained weights).

### Calculate the loss

We'll be using the cross entropy loss, mentioned in chapter 5 of the book.


```python
def cross_entropy(preds, labels):
	"""This is a manual implementation of the cross-entropy loss function. In production, you would use the built-in PyTorch function 'F.cross_entropy' instead.""" 
	probs = preds.softmax(dim=-1) # make all predictions combined add up to 1

	correct_class_probs = probs[range(len(labels)), labels] # get only the column of the correct label for each datapoint

	individual_loss = -torch.log(correct_class_probs + 1e-15) # + 1e-15 to avoid NaN

	return individual_loss.mean()
```


```python
loss = cross_entropy(preds, train_labels)
loss
```




    tensor(0.6931, grad_fn=<MeanBackward0>)



### Calculate the gradient


```python
loss.backward()
```

### Update weights and biases


```python
lr = 1e-5
with torch.no_grad():
    weights -= weights.grad * lr
    bias -= bias.grad * lr
    weights.grad.zero_()
    bias.grad.zero_()
```

### Repeat the process & Decide when to Stop


```python
def train_epoch(dl, model, params, lr=1e-5, loss_func=cross_entropy):
	total_loss = 0.0
	total_samples = 0.0
	for x, y in dl:	
		preds = model(x) # make predictions

		loss = loss_func(preds, y) # calculate the loss
		total_loss += loss.item() * len(y)
		total_samples += len(y)

		loss.backward() # calculate the gradients

		with torch.no_grad(): # update the parameters and reset the gradients
			for p in params:
				p -= lr * p.grad
				p.grad.zero_()

	return total_loss / total_samples
```


```python
def validate_epoch(dl, model, loss_func=cross_entropy):
    total_loss = 0.0
    correct_predictions = 0.0
    total_samples = 0.0
    with torch.no_grad():
        for x, y in dl:
            preds = model(x) # make predictions
            
            loss = loss_func(preds, y) # calculate the loss
            total_loss += loss.item() * len(y)

            correct_predictions += (torch.argmax(preds, dim=1) == y).sum().item() # calculate accuracy as a metric
            total_samples += len(y)

    return total_loss / total_samples, correct_predictions / total_samples
```


```python
train_epoch(dl_train, linear, [weights, bias])
```




    0.6862260103225708




```python
validate_epoch(dl_valid, linear)
```




    (0.6933783292770386, 0.5)



The functions appear to work, as the loss is changing and the accuracy is at 0.5, so random guessing, as expected in early epochs.

Also the cross-entropy loss of random guessing would be $-\ln(0.5) \approx 0.693$. So the loss function is working too.


```python
# Reinitialize the weights and bias, as they have been updated during testing.
weights = (torch.randn((num_inputs, num_outputs))*0.00001).requires_grad_()
bias = (torch.zeros(num_outputs)).requires_grad_()
```

Before running the loop, I experimented with different learning rates as described in chapter 5, by stepping through learning rates (e.g. doubling from 1e-6 to 2e-6), until I landed at close to 2e-5. <br>

I found that the learning rate of 2e-5 works best for 150 epochs.


```python
def fit(params: list, epochs: int, lr: float, loss_func=cross_entropy, model=linear):
	history = {"loss": [], "valid_loss": [], "acc": []}
	for epoch in range(epochs):
		loss = train_epoch(dl_train, model, params, lr, loss_func)
		valid_loss, acc = validate_epoch(dl_valid, model, loss_func)

		history["loss"].append(loss)
		history["valid_loss"].append(valid_loss)
		history["acc"].append(acc)

	print(f"Final Epoch {epochs-1}: loss={loss:.4f}, valid_loss={valid_loss:.4f}, acc={acc:.4f}")
	return history
```


```python
history = fit([weights, bias], epochs=150, lr=2e-5)
```

    Final Epoch 149: loss=0.5161, valid_loss=0.6257, acc=0.7308
    


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(history["loss"], label="Train loss")
ax1.plot(history["valid_loss"], label="Valid loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training vs Validation Loss")
ax1.legend()

ax2.plot(history["acc"], label="Valid accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Validation Accuracy")
ax2.legend()

plt.show()
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_66_0.png)
    


150 epochs at a learning rate of 2e-5 appear to be a good place to stop training, as the validation loss is monotonically decreasing, while the accuracy peaked at 73%.

Even across multiple reruns, the convergence stays constant and the model constantly scores a validation loss of 0.62, as the weights are initialized near 0 every run, so the training process is about the same each rerun. <br>
This can be done here, because a linear classifier with cross-entropy loss has a bowl-like convex loss landscape, and because it uses full-batch gradient descent, with weights initialized near 0, which eliminates almost all noise. 
This causes every run to follow virtually the same trajectory and makes the model almost deterministic.

I stopped after the training and validation loss slowed down decreasing, as you need more and mre epochs just to get a small improvement in validation loss, which does not improve accuracy and widens the difference between train and validation loss.

Also if you keep training for say 200 epochs, the accuracy decreases to about 71% again, while the loss still decreases. <br>
This decrease in accuracy, despite the monotonically falling validation loss, could be because the model optimizes for cross-entropy loss.

In this case, the model's accuracy would get worse by about one validation image if we kept training. <br>
So the model wrongly classified one more image, but in turn it became more confident on guessing the correct, non-borderline ones. <br>
This caused its accuracy to drop, despite the falling validation loss.

Regarding performance, the linear model outperformed the baseline accuracy-wise. 

So it appears that weighting each pixel did make a difference, however as we have a small validation set of just 52 images, to properly compare the two models, more validation data would be needed, as this small margin could have also been caused by a lucky validation split, favouring the linear model.
But while the baseline uses average pixel difference, the linear model can discriminate between actual useful differences, by upweighting specific areas and downweighting differences that are unimportant for distinguishing between the two classes. <br>

While it theoretically makes sense for the linear model to beat the baseline, the validation set is too small to prove this statistically.

Let's look at its predictions for the first 20 images from the validation set:


```python
with torch.no_grad():
	preds = linear(valid_images)

print("Raw predictions:\n", preds[:20])
print("Predicted classes:\n", torch.argmax(preds, dim=1)[:20])
print("True classes:\n", valid_labels[:20])
```

    Raw predictions:
     tensor([[ 0.2877, -0.2921],
            [-0.0654,  0.0593],
            [-0.0833,  0.0792],
            [-0.3760,  0.3721],
            [ 0.5071, -0.5136],
            [ 0.1853, -0.1901],
            [-0.4181,  0.4134],
            [ 0.0634, -0.0665],
            [-0.3605,  0.3537],
            [ 0.2268, -0.2331],
            [ 0.0610, -0.0643],
            [-0.4308,  0.4270],
            [ 0.4714, -0.4729],
            [-0.1053,  0.0990],
            [-0.3457,  0.3407],
            [-0.1441,  0.1424],
            [ 0.3795, -0.3832],
            [-0.0616,  0.0575],
            [-0.0134,  0.0071],
            [ 0.5266, -0.5294]])
    Predicted classes:
     tensor([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0])
    True classes:
     tensor([0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])
    

So you can see the first 20 predictions here are mostly correct.

Also the raw predictions are almost the exact negatives of each other. This is because the gradients are the opposites of each other, so the weights for one class get updated in the opposite direction as the other by the same amount. This causes the raw outputs to be almost perfect opposites of each other, because the weights were initialized close to zero.

## Trying a neural network with 1 hidden layer

Now I will use PyTorch to try and beat both the pixel-similarity baseline and the linear model.

Through iterative tuning, I found that 8 neurons give me the best convergence and most stable training.


```python
simple_net = nn.Sequential(
    nn.Flatten(), # flatten the images, so we get input of shape: [batch_size, 256x256x3]
    nn.Linear(256 * 256 * 3, 8), # 8 neurons
    nn.ReLU(), # ReLU activation
    nn.Linear(8, 2) # 2 output neurons for Luxembourg and the Netherlands
)
```


```python
dls = DataLoaders(dl_train, dl_valid).cuda() # put the dataloaders onto the GPU

learner = Learner(
    dls=dls, 
    model=simple_net, 
    loss_func=F.cross_entropy,
    opt_func=SGD, 
    metrics=accuracy
)
```

This time, as I have a fastai learner, I will make use of the learning rate finder.


```python
learner.lr_find(suggest_funcs=(minimum, steep))
```


    SuggestedLRs(minimum=0.00043651582673192023, steep=0.0030199517495930195)




    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_79_2.png)
    


In contrast to the linear learner, there now actually is moderate variance between reruns. <br>

So I will use a learning rate and number of epochs that appear to work well over multiple reruns.
While the learning rate might not be optimal for all runs, it should work acceptably for most.

1e-4 is not located near any sharp jumps or spikes and on a steady downwards slope most of the time.


```python
epochs = 500

with learner.no_logging():
	learner.fit(epochs, 1e-4)

learner.recorder.plot_loss(); f"Final accuracy: {learner.recorder.values[-1][-1]}"
```


    'Final accuracy: 0.7307692766189575'




    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_81_2.png)
    



```python
accuracy_values = [epoch[-1] for epoch in learner.recorder.values]

plt.plot(range(epochs), accuracy_values, label="Validation accuracy")
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_82_0.png)
    


Here I stopped again at the point the loss slowed down decreasing, as with more training there is no gain in accuracy and a risk of overfitting.

Note that I ran the cells above multiple times and that the results plotted here are just one random initialization.

I began with 1,000 epochs and then reduced the number of epochs until the point validation loss stagnated.

Unlike the linear learner's weights, the neural network's weights are not initialized near zero, as PyTorch uses a different initialization. <br>
So gradient descent has to take a different route to a decent local minimum every time. <br>
This causes variance between reruns.

But the most prominent source of variance is dead neurons. <br>
The input pixels are normalized between 0 and 1. If the random weight initialization now causes the weighted sum of a hidden neuron to be negative for all images, ReLU outputs zero for this neuron. 
As this neuron's gradient then is zero, it does not get gradient updates, so it effectively is lost for every image, as there are no negative inputs that could produce a positive weighted sum.

Because this model sometimes yields both accuracies that are better and worse than the linear learner, it would not be fair to just select a random one. What if I for example got lucky, simply because of random weight initialization? 

As a result, we have to train the model multiple times.


```python
import gc

n = 50

final_accuracies = []
final_losses = []
for i in range(n):
	simple_net = nn.Sequential(
		nn.Flatten(), 
		nn.Linear(256 * 256 * 3, 8), 
		nn.ReLU(),
		nn.Linear(8, 2) 
	)
	learner = Learner(
		dls=dls, 
		model=simple_net, 
		loss_func=F.cross_entropy,
		opt_func=SGD, 
		metrics=accuracy
	)
	with learner.no_logging():
		learner.fit(epochs, 1e-4)

	val_acc = learner.recorder.values[-1][-1]
	val_loss = learner.recorder.values[-1][-2]

	final_accuracies.append(val_acc)
	final_losses.append(val_loss)

	del simple_net, learner
	gc.collect()
	torch.cuda.empty_cache()

```



```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.boxplot(final_accuracies, label="1 hidden layer")
ax1.hlines(history["acc"][-1], label="Linear", xmin=0.6, xmax=1.4, linewidth=2) # the linear model has almost no variance
ax1.hlines(y=baseline_accuracy, label="Baseline", xmin=0.6, xmax=1.4, linewidth=2, linestyles="dashed") # the baseline is deterministic and has no variance

ax2.boxplot(final_losses, label="1 hidden layer")
ax2.hlines(history["valid_loss"][-1], label="Linear", xmin=0.6, xmax=1.4, linewidth=2) # the linear model has almost no variance

ax1.set_title("Validation accuracy comparison")
ax2.set_title("Validation loss comparison")

ax1.set_ylabel("Validation accuracy")
ax2.set_ylabel("Validation loss")
ax1.set_xticks([])
ax2.set_xticks([])
ax1.legend()
ax2.legend()

ax1.set_ylim(0.5, 0.8)
ax2.set_ylim(0.5, 0.7)

plt.show()
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_86_0.png)
    


Looking at the accuracy boxplot of 50 training runs, the neural network mostly matched the linear learner on its median and in its upper quartile.

Most runs still either match or fall below the linear learner by a few validation images. <br>
However it is safe to say that the baseline was clearly beaten, as just one outlier performed worse.

As the neural network almost matches the linear learner, their difference is only a few validation images. However as the validation set is quite small at just 52 images, if we were to scale up the validation set, these few images could scale to dozens of images.

Therefore, looking at the loss can reveal whether their performance is equal, or whether one managed to fit better to the validation set.

Looking at the validation loss boxplot, we can see that the neural network beat the linear learner on every run except for two outliers.

Therefore, while the linear learner is more consistent accuracy-wise, the neural network is more confident in its correct guesses.

This likely is caused by the neural network having drastically more parameters, meaning it can better fit to the images, in turn reducing the loss. <br>
It needed:

- 4x the parameters: $256 \cdot 256 \cdot 3 \cdot 8 + 8 \cdot 2 + 8 + 2 = 1,572,890$ for the neural network versus $256 \cdot 256 \cdot 3 \cdot 2 + 2 = 393,218$ for the linear learner
- 3.3x the training epochs: 500 epochs for the neural network versus 150 for the linear learner, as more parameters need longer to train.

just for the neural network to match the linear learner's performance regarding accuracy. So it appears that the task of separating the two classes was already solved decently well linearly. Introducing non-linearity did not really make the significant impact I would have expected.

So in theory, the neural network can match and even outperform the linear learner, as some "lucky runs" did, however plenty still fall below this threshold for the named reasons.

To really compare the two, more validation data would be needed, but the lower loss of the neural network suggests a higher flexibility, so it might outperform the linear learner on a bigger dataset.

## Visualizing

I got curious about what specific features the linear learner uses to distinguish between the two classes.
Because their only difference is the color of the third stripe, I expect the model to assign a high value to the area where the third stripe should be in most images.


```python
with torch.no_grad():
	weights_data = weights.detach().clone()

	weights_netherlands = weights_data[:, 0].reshape(256, 256, 3)
	weights_luxembourg = weights_data[:, 1].reshape(256, 256, 3)

weights_netherlands.shape, weights_luxembourg.shape
```




    (torch.Size([256, 256, 3]), torch.Size([256, 256, 3]))




```python
def to_image(tensor):
	# normalize the tensor to the range [0, 1], making sure weights of 0 are neutral grey
	normalized = (tensor / (2 * tensor.abs().max())) + 0.5
	return (normalized.clamp(0, 1)* 255).byte().cpu().numpy()
```

First, let's look at all the RGB channels combined.


```python
img_netherlands = Image.fromarray(to_image(weights_netherlands), "RGB")
img_luxembourg = Image.fromarray(to_image(weights_luxembourg), "RGB")

display(img_netherlands, img_luxembourg)
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_93_0.png)
    



    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_93_1.png)
    


The model separates the images into three stripes, as we can see here.

As expected, the third stripe is the most prominent. <br>
Also the third stripe has a kind of cyan blue that resembles Luxembourg's blue, while the Netherlands' weights have a more red and purple color for the third stripe.

A noticeable artifact however is how the "pngtree" watermark is burned into the image, like it was on the averaged images from the pixel similarity approach too. 

The fact that the watermark is black for the Luxembourg weights and white for the Netherlands weights means that the model learned that this kind of watermark is a strong indicator for the Netherlands and against Luxembourg. This means my Netherlands subset of the training data has too many of these watermarked images.

This is also visible in the averaged images. There the Netherlands mean has more prominent watermarks of this kind, than the Luxembourg one.

If that is the case, my main [European flag classifier](https://european-flags-classifier.streamlit.app) very likely noticed these too and is actively using them.

Next let's look at the areas that are of relevance for each color.


```python
red_luxembourg = weights_luxembourg[:, :, 0]
green_luxembourg = weights_luxembourg[:, :, 1]
blue_luxembourg = weights_luxembourg[:, :, 2]

red_netherlands = weights_netherlands[:, :, 0]
green_netherlands = weights_netherlands[:, :, 1]
blue_netherlands = weights_netherlands[:, :, 2]

img_netherlands_red = Image.fromarray(to_image(red_netherlands), "L")
img_netherlands_green = Image.fromarray(to_image(green_netherlands), "L")
img_netherlands_blue = Image.fromarray(to_image(blue_netherlands), "L")
img_luxembourg_red = Image.fromarray(to_image(red_luxembourg), "L")
img_luxembourg_green = Image.fromarray(to_image(green_luxembourg), "L")
img_luxembourg_blue = Image.fromarray(to_image(blue_luxembourg), "L")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
(ax_nl_r, ax_nl_g, ax_nl_b), (ax_lx_r, ax_lx_g, ax_lx_b) = axes

ax_nl_r.imshow(img_netherlands_red, cmap="gray")
ax_nl_r.set_title("Netherlands - Red channel")
ax_nl_g.imshow(img_netherlands_green, cmap="gray")
ax_nl_g.set_title("Netherlands - Green channel")
ax_nl_b.imshow(img_netherlands_blue, cmap="gray")
ax_nl_b.set_title("Netherlands - Blue channel")

ax_lx_r.imshow(img_luxembourg_red, cmap="gray")
ax_lx_r.set_title("Luxembourg - Red channel")
ax_lx_g.imshow(img_luxembourg_green, cmap="gray")
ax_lx_g.set_title("Luxembourg - Green channel")
ax_lx_b.imshow(img_luxembourg_blue, cmap="gray")
ax_lx_b.set_title("Luxembourg - Blue channel")

plt.tight_layout()
plt.show()
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3.5/2026-11-11-neural-net-foundations_96_0.png)
    


The bottom stripe has a bright white in the green and blue channels, recognizing Luxembourg's blue. 
There are strong negative weights in the Netherlands' blue and green channel for the same area.

This makes sense, as it is the most prominent feature separating the two classes.

Overall, both images look like the inverted version of the other. <br>
The features that speak for one class speak against the other at about the same absolute value.
This is a result of optimizing for cross-entropy loss, as explained when we looked at the raw model outputs.

The middle stripe surprisingly has strong positive weights in the Netherlands' green and blue channel, with the opposite for Luxembourg's green and blue channel. <br>
Potentially the white of the Netherlands is cleaner, or there are artifacts present in both training and validation sets the model overfit to. <br>
I looked at the training data again but could not find any weird differences between their middle stripes.

The top stripe appears not very tall, as the middle one takes up a big space.
It also has positive weights in Luxembourg's green and blue channels for the top stripe.
Despite their red stripe almost being identical I looked it up and there is indeed a color difference between their red that the model managed to pick up.

Also, especially in the red channel, the flag appears to have rounded corners at the top. <br>
I looked into my dataset and the Netherlands actually has more flags shaped in a circle than Luxembourg. 
The red channel has high positive weights at the upper corners of the image for Luxembourg. 
If there now is a rectangular image, the corners are filled with red, as Luxembourg's red stripe lays over them. <br>
This would be a strong Luxembourg indicator, as the Netherlands' flags are circles more often and the corners would not be filled with much red, instead they'd be filled with background, meaning high green and blue contents. <br>
That is the reason the corners have negative weights for Luxembourg in the blue and green channels. <br>
If the image now is a circle, the edges do not have a high red content, explaining the negative red weights for the Netherlands.

The red channel in general can be largely ignored, as besides a few artifacts, such as the upper corners and positive weights for the Netherlands at the middle stripe, the weights stay small and at neutral grey, because both images largely share the same red contents in their stripes.

In conclusion, the linear model did learn actual features of both classes, but found and heavily utilized leakages, such as the watermarks and the shape of the flag. 

My European flag classifier, as it has a lot more classes, should not be able to pinpoint a country by its watermarks. <br>
However they could bias the model towards a subset of possible classes. For example, if classes one, two and three have strong watermarks and four and five have weak ones, the model is more likely to pick between classes one, two and three, just based on watermarks alone.
