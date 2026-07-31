# Lesson 1: Getting started
For this lesson, you were supposed to experiment around yourself a little bit and come up with ideas, so I decided to make an AI to correctly guess any european country by its flag.
I plan to improve this model with every lesson of the course, to at some point be able to look back and see how it has developed alongside my skills.


```python
# make sure to get the latest fast.ai version
%pip install -U numpy fastai
```

## **Dataset**

I have scraped DuckDuckGo for a total of 8763 images of the following countries:


```python
european_countries = [
    "france",
    "germany",
    "italy",
    "spain",
    "portugal",
    "greece",
    "turkey",
    "austria",
    "switzerland",
    "netherlands",
    "belgium",
    "luxembourg",
    "denmark",
    "norway",
    "sweden",
    "finland",
    "iceland",
    "united kingdom",
    "ireland",
    "czech republic",
    "hungary",
    "poland",
    "romania",
    "bulgaria",
    "slovakia",
    "slovenia",
    "croatia",
    "bosnia and herzegovina",
    "albania",
    "kosovo",
    "montenegro",
    "macedonia",
    "serbia",
    "ukraine",
    "russia",
    "belarus",
    "latvia",
    "lithuania",
    "estonia",
    "malta",
    "cyprus",
    "liechtenstein",
    "monaco",
    "san marino",
    "vatican city",
    "andorra"

]
```

These images are not filtered whatsoever, just directly extracted from the internet.
Because of that, I need to filter out any broken images:


```python
from fastcore.all import *
from fastai.vision.all import *

# since I have already created a dataset of all european countries, I will just use that:
path = Path("../european_countries")
# remove broken images (just use the same code as in the example)
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"Broken images: {len(failed)}")

files = get_image_files(path)
print(f"Total images: {len(files)}")
```

    Broken images: 0
    Total images: 8763
    

It is also important to mention, that each country should have about the same amount of images.

## **Datablock**

Next, I need to design the DataBlock I will be using for my data.

Because the problem is quite similar to the 'is it a bird?' example, I can pretty much just use the same DataBlock:


- First my inputs are images, so I will need an **ImageBlock**.

- My outputs should be probabilities for each possible country, so I need the model to output categories, hence the **CategoryBlock** as output.

- **get_items** should be a function to get the image files saved in my dataset, so I will use the build-in ``get_image_files`` function, just as in the example.

- ``RandomSplitter`` should be fine for my data, because the flag images have no particular order or structure, that the model could overfit to. <br>
I will keep the seed the same, so I can reproduce the training / validation split. This allows me to tweak the item transforms, without introducing additional deviation in performance due to shifting validation sets.

- Due to the structure of my dataset, leaving get_y as the ``parent_label`` function is fine.

- I will play around a bit with the **Resize()** resolution.<br>
I will keep the transformation method the same, because cropping flags will cut off important details and I don't know enough about data augmentation yet (That is something for lesson 2). However I am aware that 'squish' might distort geometry, making it harder to learn for the model.


```python
from fastai.vision.all import *
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(256, method="squish")]
).dataloaders(path)
```

Lets show a batch to see if it worked.

I will use my own GPU, which I already had set up, for this.


```python
dls.show_batch() # this will only show us the first 9 images of the batch, to not flood my screen with flags
```


    
![show_batch output](/deep-learning-notes.github.io/assets/images/lesson_1/output_show_batch.png)
    


Appears about right. Some might be hard to recognize due to text, watermarks, weird shapes or historical flags, but I feel like with 8763 images, the AI should be able to get a decent accuracy.

But most importantly, it appears that the flags have been labeled correctly.

I am kind of unsure of how useful this dataset will be, however, I wil just try it.

## **Learner**

Now that our data is loaded and set up correctly, we can start training the model.

I will stick with resnet18, because I don't really know what other model I should choose that would work better for this specific task. I simply don't have the know-how, but that is what I hope to gain in the course!

I'm using accuracy as a metric, because it is more straightforward. <br>
The goal accuracy I want to go for is at least 90%.


```python
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.remove_cb(ProgressCallback) # I could not get this to work in visual studio code with the progressbar
learn.fine_tune(5)
```

    [0, 2.81087064743042, 1.1838092803955078, 0.6775113940238953, '00:28']
    [0, 0.8393496870994568, 0.4125519394874573, 0.8995434045791626, '00:31']
    [1, 0.3538261353969574, 0.23748676478862762, 0.9446346759796143, '00:30']
    [2, 0.11435025185346603, 0.20426219701766968, 0.9474886059761047, '00:30']
    [3, 0.045558180660009384, 0.1956688016653061, 0.9497717022895813, '00:30']
    [4, 0.024979548528790474, 0.18970417976379395, 0.952625572681427, '00:30']
    

Because the formatting got removed in Visual Studio Code, here is what each value means:

- The first value is the epoch index
- The second value is the training loss
- The third value is the validation loss
- The fourth value is the accuracy
- The fifth value is the time each epoch took

Seems like the pretrained resnet18 was pretty good already at guessing flags. <br> It started off with about 67-70% accuracy, which already is better than random guessing...

After some testing around, I settled for a resolution of 256px.

With <=128px I got a worse accuracy, probably because small details on flags got lost.

With >=256px I got slightly better accuracy, however epoch time got longer, without any major performance improvements (Accuracy hit a ceiling at around 95-96%).


However accuracy might just be that high, because of the quality of my dataset, because it was scraped from the internet. <br>
I know there are duplicate images in the training data, some might also include historical flags, or weirdly shaped ones. <br>
So the model might get an artificially high accuracy, when ones that are in the training set, are also in the validation set...

To be sure, let's test it on some unseen images. <br> Hopefully it is still able to guess some flags correctly.

## **Using the model**


```python
image = PILImage.create("../images/country_flags/germany.jpg") # I have a germany flag image saved, that was not in the dataset

image.show() # open the image first
```




    <Axes: >




    
![Germany Flag](/deep-learning-notes.github.io/assets/images/country_flags/germany.jpg)
    



```python
country, country_index, probs = learn.predict(image) # lets predict!

print(f"This is a picture of a {country} flag.")
print(f"Probability it's a {country} flag: {probs[country_index]:.6f}")
second_index = probs.argsort(descending=True)[1] # sort arguments and take index of second most confident one
second_country = learn.dls.vocab[second_index] # look up what category the index had
print(f"The second most likely country is: {second_country}")
print(f"Probability it's a {second_country} flag: {probs[second_index]:.6f}")
```

    This is a picture of a germany flag.
    Probability it's a germany flag: 0.997555
    The second most likely country is: spain
    Probability it's a spain flag: 0.001052
    

So the model was about 100% sure it was a Germany flag. Very good.


```python
image = PILImage.create("../images/country_flags/liechtenstein.jpg") # Also unseen liechtenstein flag image

image.show()
```




    <Axes: >




    
![Liechtenstein Flag](/deep-learning-notes.github.io/assets/images/country_flags/liechtenstein.jpg)
    



```python
country, country_index, probs = learn.predict(image)

print(f"This is a picture of a {country} flag.")
print(f"Probability it's a {country} flag: {probs[country_index]:.6f}")
second_index = probs.argsort(descending=True)[1]
second_country = learn.dls.vocab[second_index]
print(f"The second most likely country is: {second_country}")
print(f"Probability it's a {second_country} flag: {probs[second_index]:.6f}")
```

    This is a picture of a liechtenstein flag.
    Probability it's a liechtenstein flag: 0.999985
    The second most likely country is: germany
    Probability it's a germany flag: 0.000011
    

about 100% confidence again. <br>
Ok, let's make it harder. How about a San Marino pin?


```python
image = PILImage.create("../images/country_flags/san-marino.jpg") # Also unseen san-marino pin image

image.show()
```




    <Axes: >




    
![San Marino Pin](/deep-learning-notes.github.io/assets/images/country_flags/san-marino.jpg)
    



```python
country, country_index, probs = learn.predict(image)

print(f"This is a picture of a {country} flag.")
print(f"Probability it's a {country} flag: {probs[country_index]:.6f}")
second_index = probs.argsort(descending=True)[1]
second_country = learn.dls.vocab[second_index]
print(f"The second most likely country is: {second_country}")
print(f"Probability it's a {second_country} flag: {probs[second_index]:.6f}")
```

    This is a picture of a san marino flag.
    Probability it's a san marino flag: 0.626695
    The second most likely country is: malta
    Probability it's a malta flag: 0.126816
    

Very good! It got it, but was less confident. <br>
Actually let's also see what happens when you have 2 flags in one image.


```python
image = PILImage.create("../images/country_flags/san-marino-germany.jpg") # Also unseen mixed flag image

image.show()
```




    <Axes: >




    
![San Marino and Germany](/deep-learning-notes.github.io/assets/images/country_flags/san-marino-germany.jpg)
    



```python
country, country_index, probs = learn.predict(image)

print(f"This is a picture of a {country} flag.")
print(f"Probability it's a {country} flag: {probs[country_index]:.6f}")
second_index = probs.argsort(descending=True)[1]
second_country = learn.dls.vocab[second_index]
print(f"The second most likely country is: {second_country}")
print(f"Probability it's a {second_country} flag: {probs[second_index]:.6f}")
```

    This is a picture of a san marino flag.
    Probability it's a san marino flag: 0.347993
    The second most likely country is: czech republic
    Probability it's a czech republic flag: 0.270405
    

Ah, the model chose San Marino correctly, but at about 35%, was not very confident. <br>
It seems confused, because there suddenly are two flags. <br> 
I would have expected Germany to be the second most likely though, not the Czech Republic, but I can see where the model sees similarities. Likely because of the colors white, blue and red, separated by a diagonal line.

Maybe the model also "pays more attention" to the top right or the coat of arms in the middle, because it chose San Marino, not Germany.

With this approach now as the baseline, I can further improve the models accuracy and performance lesson by lesson and measure my new approaches against this one.

And while I still feel more or less sceptic about the result, testing it on unseen data did prove that it at least learned something, despite the data leakage.
