# Lesson 2: Data Cleaning, Augmentation and Deployment
For this post, I want to clean the dataset and improve the European Flags Classifier with the concepts learned in lesson 2, such as using data augmentation.

In addition, I want to deploy the model this time.


```python
# make sure to get the latest fastai version
%pip install -U fastai
```

Let's look at our model's performance so far in more detail.

In my previous post on lesson 1, I just ran the cell where I used ``fine_tune`` a couple of times and then roughly averaged the accuracy. <br>
Now I want to have a more concrete way of saying whether a model has improved or not.


```python
from fastcore.all import *
from fastai.vision.all import *
path = Path("../european_countries_old")
```


```python
def average_final_accuracy(dls: DataLoaders, num_rounds: int = 10, num_epochs: int = 5) -> tuple[list, Learner]:
    """
    Collect the final accuracies of a model trained on the same data multiple times.
    Without a GPU, this may take a long time.
    The function returns a list of the final accuracies and the last trained learner.
    """
    final_accuracies = []
    for i in range(num_rounds):
        print(f"-- round {i+1} --")
        learner = vision_learner(dls, resnet18, metrics=accuracy) # I will use the same model and metrics as last time, to compare the approaches
        learner.remove_cb(ProgressCallback)
        learner.fine_tune(num_epochs)

        valid_loss, final_accuracy = learner.validate()
        final_accuracies.append(final_accuracy)

    return final_accuracies, learner
```

Let's train the model again:


```python
# DataBlock we used last time
flags = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(256, method="squish")]
)
```


```python
dls = flags.dataloaders(path)
final_accuracies, learner = average_final_accuracy(dls)
```

    -- round 1 --
    [0, 2.8226780891418457, 1.1674376726150513, 0.6897143125534058, '00:31']
    [0, 0.8460672497749329, 0.40592560172080994, 0.8897143006324768, '00:33']
    [1, 0.3478294014930725, 0.2343718409538269, 0.9331428408622742, '00:32']
    [2, 0.11846350133419037, 0.17227111756801605, 0.9514285922050476, '00:33']
    [3, 0.04416462406516075, 0.15943317115306854, 0.9588571190834045, '00:32']
    [4, 0.0241212360560894, 0.15707270801067352, 0.9565714001655579, '00:32']
    -- round 2 --
    [0, 2.850627899169922, 1.1799578666687012, 0.6937142610549927, '00:30']
    [0, 0.8373488187789917, 0.3764025568962097, 0.9028571248054504, '00:31']
    [1, 0.3397536873817444, 0.20889681577682495, 0.9462857246398926, '00:31']
    [2, 0.11842575669288635, 0.1693372279405594, 0.9531428813934326, '00:32']
    [3, 0.049824051558971405, 0.15211014449596405, 0.9554286003112793, '00:33']
    [4, 0.024995889514684677, 0.15031394362449646, 0.9554286003112793, '00:31']
    -- round 3 --
    [0, 2.8386690616607666, 1.187479853630066, 0.6977142691612244, '00:36']
    [0, 0.8226386904716492, 0.41815683245658875, 0.8897143006324768, '00:44']
    [1, 0.354287326335907, 0.2255699187517166, 0.9382857084274292, '00:44']
    [2, 0.1182035431265831, 0.18543671071529388, 0.9480000138282776, '00:44']
    [3, 0.042412590235471725, 0.16868549585342407, 0.9537143111228943, '00:44']
    [4, 0.01989520713686943, 0.1653212606906891, 0.954285740852356, '00:43']
    -- round 4 --
    [0, 2.8544037342071533, 1.179818034172058, 0.6977142691612244, '00:42']
    [0, 0.8245506882667542, 0.40816813707351685, 0.8965714573860168, '00:43']
    [1, 0.3330523669719696, 0.23301944136619568, 0.9325714111328125, '00:43']
    [2, 0.11920256167650223, 0.17537976801395416, 0.954285740852356, '00:43']
    [3, 0.047245416790246964, 0.15934611856937408, 0.9554286003112793, '00:44']
    [4, 0.027578921988606453, 0.15585297346115112, 0.9548571705818176, '00:44']
    -- round 5 --
    [0, 2.8749101161956787, 1.1716786623001099, 0.6994285583496094, '00:43']
    [0, 0.82738196849823, 0.3946053385734558, 0.9017142653465271, '00:44']
    [1, 0.33256587386131287, 0.22249025106430054, 0.9354285597801208, '00:44']
    [2, 0.1257244348526001, 0.17023897171020508, 0.9582856893539429, '00:44']
    [3, 0.048274822533130646, 0.1583431214094162, 0.9594285488128662, '00:45']
    [4, 0.024191610515117645, 0.1544111967086792, 0.9599999785423279, '00:44']
    -- round 6 --
    [0, 2.870861768722534, 1.1382558345794678, 0.7102857232093811, '00:42']
    [0, 0.8341187238693237, 0.40767842531204224, 0.8954285979270935, '00:43']
    [1, 0.3420272469520569, 0.21779385209083557, 0.9388571381568909, '00:43']
    [2, 0.12243400514125824, 0.1907033920288086, 0.9474285840988159, '00:44']
    [3, 0.04613418132066727, 0.1674734354019165, 0.9531428813934326, '00:43']
    [4, 0.022634781897068024, 0.16441594064235687, 0.9554286003112793, '00:43']
    -- round 7 --
    [0, 2.8517894744873047, 1.167980432510376, 0.6954285502433777, '00:42']
    [0, 0.8405845761299133, 0.39421454071998596, 0.8954285979270935, '00:44']
    [1, 0.3511265218257904, 0.21982137858867645, 0.9411428570747375, '00:44']
    [2, 0.12888112664222717, 0.18223625421524048, 0.9537143111228943, '00:44']
    [3, 0.05044303834438324, 0.1651441603899002, 0.9582856893539429, '00:44']
    [4, 0.02634851261973381, 0.16647852957248688, 0.9554286003112793, '00:43']
    -- round 8 --
    [0, 2.87068510055542, 1.1589295864105225, 0.6977142691612244, '00:42']
    [0, 0.8212651610374451, 0.4069238305091858, 0.8885714411735535, '00:43']
    [1, 0.34266576170921326, 0.20762310922145844, 0.9457142949104309, '00:43']
    [2, 0.1305905431509018, 0.1881188601255417, 0.9462857246398926, '00:43']
    [3, 0.046180617064237595, 0.15563875436782837, 0.9559999704360962, '00:43']
    [4, 0.02581297606229782, 0.15653052926063538, 0.954285740852356, '00:43']
    -- round 9 --
    [0, 2.8437373638153076, 1.1954222917556763, 0.6845714449882507, '00:41']
    [0, 0.8647134900093079, 0.3955325484275818, 0.8960000276565552, '00:43']
    [1, 0.3503971993923187, 0.23211005330085754, 0.9417142868041992, '00:40']
    [2, 0.11952326446771622, 0.1881890445947647, 0.9480000138282776, '00:43']
    [3, 0.044925156980752945, 0.17244233191013336, 0.9565714001655579, '00:43']
    [4, 0.024806251749396324, 0.1691008359193802, 0.9559999704360962, '00:43']
    -- round 10 --
    [0, 2.809413433074951, 1.2020140886306763, 0.677142858505249, '00:41']
    [0, 0.8408507704734802, 0.3975479304790497, 0.8948571681976318, '00:43']
    [1, 0.36966538429260254, 0.22716771066188812, 0.9434285759925842, '00:43']
    [2, 0.1262463629245758, 0.18397411704063416, 0.952571451663971, '00:43']
    [3, 0.04649239033460617, 0.16927136480808258, 0.9554286003112793, '00:43']
    [4, 0.02338382415473461, 0.16765941679477692, 0.954285740852356, '00:43']
    


```python
import numpy as np
mean_accuracy = np.mean(final_accuracies)
std_accuracy = np.std(final_accuracies, ddof=1)

print(f"The accuracy averages at around: {(mean_accuracy * 100):.2f}% ±{(std_accuracy * 100):.2f}%")
```

    The accuracy averages at around: 95.57% ±0.17%
    

As discussed in my previous post, the model's performance is artificially high, due to data leakage and many duplicate images.

## **Cleaning the Dataset**

Let's look at the images that were the hardest for the model:


```python
interpreter = ClassificationInterpretation.from_learner(learner)
interpreter.plot_top_losses(10, nrows=2)
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_12_0.png)
    


We can already see some major issues in the dataset. There are multiple flags in one image (Germany and Estonia), wrongly labeled ones (e.g., a Turkish flag labeled as Cyprus or the Czech flag labeled as Slovakia) and duplicates (duplicate images of three Germany flags hanging from a pole). 

I wrote a small script to find duplicates in the dataset using image hashing (dhash + colorhash). 
I then manually looked over the dataset, removing unsuitable images and any duplicates the script missed (such as mirrored versions).

It helps me to think about whether a human could still recognize the flag. 

For example, if the same picture were shown to a flag expert and there were no way for them to guess it correctly, or if the image were too confusing, I would remove it.

So I'm keeping images with multiple of the same flags, because a flag expert would still recognize them.


```python
path = Path("../european_countries")
len(get_image_files(path))
```




    6100



In total, 2,663 images got removed... (30.39% of my dataset)

Now that the dataset is cleaned, we can train our model again. I expect the accuracy to go down quite a bit, because the validation set no longer contains duplicates of training samples.


```python
dls = flags.dataloaders(path)
final_accuracies_cleaned, learner = average_final_accuracy(dls)
```

    -- round 1 --
    [0, 3.457838773727417, 1.5650447607040405, 0.5827868580818176, '00:27']
    [0, 1.26643967628479, 0.6213677525520325, 0.8254098296165466, '00:29']
    [1, 0.5628420114517212, 0.3260522186756134, 0.9106557369232178, '00:30']
    [2, 0.22698524594306946, 0.23693610727787018, 0.9368852376937866, '00:30']
    [3, 0.09566570073366165, 0.20997819304466248, 0.9418032765388489, '00:30']
    [4, 0.0509074442088604, 0.21267487108707428, 0.9426229596138, '00:30']
    -- round 2 --
    [0, 3.4425079822540283, 1.532071828842163, 0.5901639461517334, '00:29']
    [0, 1.2116397619247437, 0.6317395567893982, 0.8229508399963379, '00:29']
    [1, 0.5682188868522644, 0.31892234086990356, 0.9131147265434265, '00:29']
    [2, 0.23359312117099762, 0.269927978515625, 0.9254098534584045, '00:30']
    [3, 0.10121003538370132, 0.2464093565940857, 0.9286885261535645, '00:29']
    [4, 0.05464189499616623, 0.23949724435806274, 0.9311475157737732, '00:30']
    -- round 3 --
    [0, 3.431028127670288, 1.4766631126403809, 0.6081967353820801, '00:28']
    [0, 1.191314935684204, 0.6036086678504944, 0.8319672346115112, '00:30']
    [1, 0.543319046497345, 0.3209075629711151, 0.9090163707733154, '00:29']
    [2, 0.22662155330181122, 0.26515433192253113, 0.9262295365333557, '00:30']
    [3, 0.09751983731985092, 0.23496206104755402, 0.935245931148529, '00:30']
    [4, 0.0520564429461956, 0.23159007728099823, 0.9360655546188354, '00:29']
    -- round 4 --
    [0, 3.449531316757202, 1.5462809801101685, 0.5934426188468933, '00:29']
    [0, 1.2401584386825562, 0.6253054141998291, 0.8229508399963379, '00:29']
    [1, 0.5794118642807007, 0.3169268071651459, 0.9106557369232178, '00:30']
    [2, 0.2338590770959854, 0.25450482964515686, 0.9368852376937866, '00:29']
    [3, 0.09815426915884018, 0.22777970135211945, 0.9401639103889465, '00:29']
    [4, 0.05642790347337723, 0.22132796049118042, 0.9450819492340088, '00:30']
    -- round 5 --
    [0, 3.4770469665527344, 1.554725170135498, 0.5786885023117065, '00:28']
    [0, 1.2284810543060303, 0.6323639154434204, 0.8319672346115112, '00:29']
    [1, 0.5669475793838501, 0.337389200925827, 0.9032787084579468, '00:30']
    [2, 0.23250819742679596, 0.26825302839279175, 0.9311475157737732, '00:29']
    [3, 0.09938952326774597, 0.23871394991874695, 0.9368852376937866, '00:29']
    [4, 0.05237910524010658, 0.23708435893058777, 0.9418032765388489, '00:29']
    -- round 6 --
    [0, 3.4546194076538086, 1.5459606647491455, 0.5819672346115112, '00:29']
    [0, 1.2142168283462524, 0.6392275094985962, 0.8245901465415955, '00:29']
    [1, 0.5734072923660278, 0.30973613262176514, 0.911475419998169, '00:29']
    [2, 0.22949783504009247, 0.24624069035053253, 0.9303278923034668, '00:30']
    [3, 0.09706062078475952, 0.22637195885181427, 0.9418032765388489, '00:30']
    [4, 0.054212234914302826, 0.2197759449481964, 0.9434426426887512, '00:29']
    -- round 7 --
    [0, 3.4981234073638916, 1.557814121246338, 0.6000000238418579, '00:28']
    [0, 1.193570852279663, 0.6218485236167908, 0.8139344453811646, '00:29']
    [1, 0.5705550312995911, 0.317889004945755, 0.911475419998169, '00:29']
    [2, 0.2290927767753601, 0.24357947707176208, 0.9311475157737732, '00:30']
    [3, 0.0956994816660881, 0.21748363971710205, 0.9442622661590576, '00:29']
    [4, 0.0567704401910305, 0.21158547699451447, 0.9434426426887512, '00:29']
    -- round 8 --
    [0, 3.42459774017334, 1.548499345779419, 0.58442622423172, '00:29']
    [0, 1.2469948530197144, 0.6275357604026794, 0.83442622423172, '00:29']
    [1, 0.5923246741294861, 0.35699227452278137, 0.9098360538482666, '00:30']
    [2, 0.2383813112974167, 0.2661113142967224, 0.9311475157737732, '00:29']
    [3, 0.09779249131679535, 0.24918420612812042, 0.9336065649986267, '00:29']
    [4, 0.05233815312385559, 0.24674880504608154, 0.9360655546188354, '00:30']
    -- round 9 --
    [0, 3.431314706802368, 1.5042026042938232, 0.6098360419273376, '00:28']
    [0, 1.2198584079742432, 0.6188482642173767, 0.8278688788414001, '00:30']
    [1, 0.5630747079849243, 0.3209867775440216, 0.9237704873085022, '00:30']
    [2, 0.2256600707769394, 0.23877844214439392, 0.9319671988487244, '00:30']
    [3, 0.09631046652793884, 0.2177795171737671, 0.9393442869186401, '00:29']
    [4, 0.05083079636096954, 0.20965784788131714, 0.9401639103889465, '00:29']
    -- round 10 --
    [0, 3.4304988384246826, 1.5645099878311157, 0.5877048969268799, '00:28']
    [0, 1.2044647932052612, 0.6253824234008789, 0.8278688788414001, '00:29']
    [1, 0.571547269821167, 0.3280669152736664, 0.9057376980781555, '00:30']
    [2, 0.22551895678043365, 0.250400573015213, 0.9336065649986267, '00:30']
    [3, 0.09301086515188217, 0.23296606540679932, 0.9377049207687378, '00:29']
    [4, 0.04840100556612015, 0.2245839536190033, 0.9418032765388489, '00:29']
    


```python
mean_accuracy = np.mean(final_accuracies_cleaned)
std_accuracy = np.std(final_accuracies_cleaned, ddof=1)

print(f"The accuracy averages at around: {(mean_accuracy * 100):.2f}% ±{(std_accuracy * 100):.2f}%")
```

    The accuracy averages at around: 94.02% ±0.44%
    

So the accuracy did drop.

While we cannot say for certain whether this drop was purely caused by fixing the data leakage or exposing the model to fewer total images (since 30% of the data was removed), the final accuracy we got is the real baseline performance on unseen data.

## **Data Augmentation**

Next let's start by thinking about possible augmentations that could improve the model's accuracy.

I will try a few different augmentations and resize methods and explain why I chose what.

### **Resize Methods**


```python
dls_viz = ImageDataLoaders.from_folder(
    path, 
    valid_pct=0.2, 
    seed=42, 
    item_tfms=Resize(256, method="squish")
)
dls_viz.valid.show_batch()
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_26_0.png)
    


While this potentially alters the proportions of the flags, they are already not always in the correct proportion, as seen, for example, in the image of Slovenia (Slovenian flag in the shape of Slovenia). <br>
This also forces the model not to rely on the proportions of the flags, but to look at their emblems and colors.


```python
dls_viz = ImageDataLoaders.from_folder(
    path, 
    valid_pct=0.2, 
    seed=42, 
    item_tfms=Resize(256, method="crop")
)
dls_viz.valid.show_batch()
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_28_0.png)
    


 ``crop`` can cut out important details, like emblems and in some cases the entire flag, if it takes up only a small portion of the frame.

For the same reason, I will also not use ``RandomResizedCrop``.


```python
dls_viz = ImageDataLoaders.from_folder(
    path, 
    valid_pct=0.2, 
    seed=42, 
    item_tfms=Resize(256, method="pad", pad_mode="zeros")
)
dls_viz.valid.show_batch()
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_30_0.png)
    



```python
dls_viz = ImageDataLoaders.from_folder(
    path, 
    valid_pct=0.2, 
    seed=42, 
    item_tfms=Resize(256, method="pad", pad_mode="border")
)
dls_viz.valid.show_batch()
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_31_0.png)
    


``border`` and ``reflection`` distort the image too much and possibly teach the model incorrect proportions. (For example with the Lithuanian flag above)

The black bar added by ``zeros`` could be mistaken as an actual black stripe on the flag.
Moreover, it reduces the resolution of the actual flag.

So I will settle for ``squish`` as the best option for my specific problem.

### **Data Augmentation**

Now we could just apply ``aug_transforms``. 
However there are certain augmentations I want to avoid.
Let's take a look at the function signature:


```python
?aug_transforms
```

    [1;31mSignature:[0m
    [0maug_transforms[0m[1;33m([0m[1;33m
    [0m    [0mmult[0m[1;33m:[0m [1;34m'float'[0m [1;33m=[0m [1;36m1.0[0m[1;33m,[0m[1;33m
    [0m    [0mdo_flip[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mTrue[0m[1;33m,[0m[1;33m
    [0m    [0mflip_vert[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mFalse[0m[1;33m,[0m[1;33m
    [0m    [0mmax_rotate[0m[1;33m:[0m [1;34m'float'[0m [1;33m=[0m [1;36m10.0[0m[1;33m,[0m[1;33m
    [0m    [0mmin_zoom[0m[1;33m:[0m [1;34m'float'[0m [1;33m=[0m [1;36m1.0[0m[1;33m,[0m[1;33m
    [0m    [0mmax_zoom[0m[1;33m:[0m [1;34m'float'[0m [1;33m=[0m [1;36m1.1[0m[1;33m,[0m[1;33m
    [0m    [0mmax_lighting[0m[1;33m:[0m [1;34m'float'[0m [1;33m=[0m [1;36m0.2[0m[1;33m,[0m[1;33m
    [0m    [0mmax_warp[0m[1;33m:[0m [1;34m'float'[0m [1;33m=[0m [1;36m0.2[0m[1;33m,[0m[1;33m
    [0m    [0mp_affine[0m[1;33m:[0m [1;34m'float'[0m [1;33m=[0m [1;36m0.75[0m[1;33m,[0m[1;33m
    [0m    [0mp_lighting[0m[1;33m:[0m [1;34m'float'[0m [1;33m=[0m [1;36m0.75[0m[1;33m,[0m[1;33m
    [0m    [0mxtra_tfms[0m[1;33m:[0m [1;34m'list'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0msize[0m[1;33m:[0m [1;34m'int | tuple'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mmode[0m[1;33m:[0m [1;34m'str'[0m [1;33m=[0m [1;34m'bilinear'[0m[1;33m,[0m[1;33m
    [0m    [0mpad_mode[0m[1;33m=[0m[1;34m'reflection'[0m[1;33m,[0m[1;33m
    [0m    [0malign_corners[0m[1;33m=[0m[1;32mTrue[0m[1;33m,[0m[1;33m
    [0m    [0mbatch[0m[1;33m=[0m[1;32mFalse[0m[1;33m,[0m[1;33m
    [0m    [0mmin_scale[0m[1;33m=[0m[1;36m1.0[0m[1;33m,[0m[1;33m
    [0m[1;33m)[0m[1;33m[0m[1;33m[0m[0m
    [1;31mDocstring:[0m Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms.
    [1;31mFile:[0m      c:\projekte\fast_ai\fastai_venv\lib\site-packages\fastai\vision\augment.py
    [1;31mType:[0m      function

I will look at some of these transforms in more detail, because while it's easy to grasp what ``do_flip`` will do, it is not so easy with ``lighting`` or ``warp`` for example.

After roughly testing what each transformation does and how it affects training, we can rule out some of them and set some constraints:

- We do not want to flip or mirror them, because this can change the meaning of a flag drastically and after testing, I found out this degrades model performance.

- We do not want to rotate them too much for the same reason. Some rotation can help though, to imitate a picture from a weird angle or a flag hanging from a pole.

- ``zoom`` appears to be similar to crop, therefore has the same problem of possibly cutting off important details. So I will use a small value.

- I found larger ``warp`` values can heavily distort the flag, so I will use the default value, because the flags are still recognizable and I want to simulate a hanging flag or folds that happen due to wind.


```python
flags = flags.new(item_tfms=Resize(256, method="squish"), # I have already settled for this
        batch_tfms=aug_transforms(
            do_flip=False, # no flipping
            flip_vert=False,
            max_zoom=1.1, # 10% (default) does not cut off too much
            max_rotate=10, # 10° (default) won't distort the flag too much, while imitating flags hanging from a pole slightly or photographed from weird angles
            max_warp=0.2, # 20% (default) To simulate flags waving in the wind, without distorting the flags too much
            max_lighting=0.2 # 20% (default) to simulate shade and strong sunlight, but to not tamper with the color differences between flags
        )
)
dls = flags.dataloaders(path)
dls.train.show_batch(max_n=20, nrows=4)
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_39_0.png)
    


The default values were well balanced already so I did not need any drastic changes for my specific problem.


```python
final_accuracies_augmented, learner = average_final_accuracy(dls)
```

    -- round 1 --
    [0, 3.5063273906707764, 1.623477816581726, 0.5655737519264221, '00:21']
    [0, 1.3215384483337402, 0.7124450206756592, 0.8008196949958801, '00:21']
    [1, 0.651838481426239, 0.29650935530662537, 0.9237704873085022, '00:21']
    [2, 0.3164077699184418, 0.23726634681224823, 0.9344262480735779, '00:21']
    [3, 0.16129913926124573, 0.2006330043077469, 0.9401639103889465, '00:21']
    [4, 0.10565728694200516, 0.2004123479127884, 0.9393442869186401, '00:21']
    -- round 2 --
    [0, 3.4789345264434814, 1.5198485851287842, 0.5950819849967957, '00:19']
    [0, 1.341898798942566, 0.695840060710907, 0.8008196949958801, '00:21']
    [1, 0.6608325242996216, 0.3254829943180084, 0.9163934588432312, '00:21']
    [2, 0.31574204564094543, 0.23869025707244873, 0.9336065649986267, '00:21']
    [3, 0.164557084441185, 0.21041221916675568, 0.9467213153839111, '00:21']
    [4, 0.10463117808103561, 0.20712196826934814, 0.9467213153839111, '00:21']
    -- round 3 --
    [0, 3.4594526290893555, 1.559020757675171, 0.5852459073066711, '00:20']
    [0, 1.2705130577087402, 0.6290283799171448, 0.8336065411567688, '00:21']
    [1, 0.6416240930557251, 0.31661567091941833, 0.9073770642280579, '00:21']
    [2, 0.3084411323070526, 0.24271713197231293, 0.9327868819236755, '00:21']
    [3, 0.16164427995681763, 0.1944335550069809, 0.9442622661590576, '00:21']
    [4, 0.10354796797037125, 0.19023922085762024, 0.9434426426887512, '00:21']
    -- round 4 --
    [0, 3.459892749786377, 1.5931018590927124, 0.5745901465415955, '00:20']
    [0, 1.3310796022415161, 0.6600865721702576, 0.8163934350013733, '00:21']
    [1, 0.6448504328727722, 0.29730841517448425, 0.9131147265434265, '00:21']
    [2, 0.30990391969680786, 0.23253121972084045, 0.9401639103889465, '00:21']
    [3, 0.17240236699581146, 0.20677410066127777, 0.9409835934638977, '00:21']
    [4, 0.10664155334234238, 0.20557910203933716, 0.9409835934638977, '00:21']
    -- round 5 --
    [0, 3.483405351638794, 1.5594865083694458, 0.5860655903816223, '00:20']
    [0, 1.303248643875122, 0.6728401184082031, 0.8139344453811646, '00:21']
    [1, 0.6366692185401917, 0.3107397258281708, 0.9163934588432312, '00:21']
    [2, 0.31312552094459534, 0.2407485395669937, 0.9311475157737732, '00:21']
    [3, 0.15814551711082458, 0.21978147327899933, 0.9368852376937866, '00:21']
    [4, 0.10470609366893768, 0.20677904784679413, 0.9426229596138, '00:21']
    -- round 6 --
    [0, 3.479126214981079, 1.5873124599456787, 0.5655737519264221, '00:20']
    [0, 1.2964069843292236, 0.6573098301887512, 0.8098360896110535, '00:21']
    [1, 0.6387068033218384, 0.3257017433643341, 0.908196747303009, '00:21']
    [2, 0.30241072177886963, 0.2302761822938919, 0.9377049207687378, '00:21']
    [3, 0.1584494709968567, 0.1996086686849594, 0.949999988079071, '00:21']
    [4, 0.10363183170557022, 0.19303865730762482, 0.9516393542289734, '00:21']
    -- round 7 --
    [0, 3.5370678901672363, 1.5848298072814941, 0.5754098296165466, '00:20']
    [0, 1.2960436344146729, 0.6733316779136658, 0.8024590015411377, '00:21']
    [1, 0.6409383416175842, 0.3252964913845062, 0.8991803526878357, '00:22']
    [2, 0.3062860369682312, 0.2403494119644165, 0.9344262480735779, '00:21']
    [3, 0.16115540266036987, 0.22399544715881348, 0.9245901703834534, '00:22']
    [4, 0.09811193495988846, 0.21782571077346802, 0.9360655546188354, '00:21']
    -- round 8 --
    [0, 3.4359848499298096, 1.5618410110473633, 0.5803278684616089, '00:20']
    [0, 1.2792508602142334, 0.6624979972839355, 0.8172131180763245, '00:21']
    [1, 0.6468725800514221, 0.32603365182876587, 0.8967213034629822, '00:22']
    [2, 0.3126054108142853, 0.23844096064567566, 0.9409835934638977, '00:22']
    [3, 0.1684286743402481, 0.2191466987133026, 0.9401639103889465, '00:21']
    [4, 0.10243077576160431, 0.20989060401916504, 0.94590163230896, '00:21']
    -- round 9 --
    [0, 3.505352735519409, 1.5366801023483276, 0.5959016680717468, '00:20']
    [0, 1.2871346473693848, 0.6315518021583557, 0.827049195766449, '00:21']
    [1, 0.6360127925872803, 0.29103562235832214, 0.9204918146133423, '00:21']
    [2, 0.3071564733982086, 0.2442632019519806, 0.9278688430786133, '00:21']
    [3, 0.1641313135623932, 0.22058868408203125, 0.9319671988487244, '00:21']
    [4, 0.10600923001766205, 0.21124432981014252, 0.9393442869186401, '00:21']
    -- round 10 --
    [0, 3.4727203845977783, 1.5872652530670166, 0.5754098296165466, '00:20']
    [0, 1.3061888217926025, 0.6323657631874084, 0.8327868580818176, '00:21']
    [1, 0.6425313353538513, 0.2850382328033447, 0.9270491600036621, '00:21']
    [2, 0.3112106919288635, 0.21633251011371613, 0.9434426426887512, '00:21']
    [3, 0.16109289228916168, 0.1923127919435501, 0.9483606815338135, '00:21']
    [4, 0.10461433976888657, 0.19093583524227142, 0.9516393542289734, '00:21']
    


```python
mean_accuracy = np.mean(final_accuracies_augmented)
std_accuracy = np.std(final_accuracies_augmented, ddof=1)

print(f"The accuracy averages at around: {(mean_accuracy * 100):.2f}% ±{(std_accuracy * 100):.2f}%")
```

    The accuracy averages at around: 94.38% ±0.52%
    

It seems like the data augmentation actually improved the accuracy of our model.

I'm interested to see what the model considers hardest now.


```python
interpreter = ClassificationInterpretation.from_learner(learner)
interpreter.plot_top_losses(10, nrows=2)
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_44_0.png)
    


The model struggles with small flags and images with multiple ones.

This makes sense, because the entire image is fed to the model, and not just the singular flag.
Because the model does not know where the flag is located in the image, it treats the background as part of the flag too, leading to possible confusion, if the flag is not big or prominent enough.

A possible solution would be to train an AI to draw a bounding box around what it believes to be the flag and then have this classifier analyze the image.

Also let's make sure the model does not heavily struggle in picking two countries apart, like Netherlands and Luxembourg, due to their very similar flags.


```python
interpreter.plot_confusion_matrix(figsize=(12, 12))
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_47_0.png)
    


As expected, the hardest countries to pick apart are Luxembourg and Netherlands.

Because the resize method is set to ``squish``, aspect ratio differences are eliminated, leaving the color difference of their blue stripes as their only difference.

## **Using the model**

Regarding the evaluation, during my holiday, I took some pictures of some flags.
They were taken in a situation where you might use the model, if it was running in production.

I just imagined that I would not know the flag and had a flag classifier running, while taking the pictures, to tell me what country it is.


```python
path = Path("../european_countries_test")
```

I will use this separate dataset like a sanity check, to make sure it performs well in the exact situation it should work in, if deployed.


```python
test_images = get_image_files(path)

test_dl = learner.dls.test_dl(test_images, get_y=parent_label, with_labels=True)
test_dl.show_batch(max_n=30, nrows=4) # full test dataset
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_53_0.png)
    



```python
interpreter = ClassificationInterpretation.from_learner(learner, dl=test_dl)
interpreter.plot_confusion_matrix(figsize=(12, 12))
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_2/2026-08-31-data-cleaning-augmentation-and-deployment_54_0.png)
    



```python
probs, targets = learner.get_preds(dl=test_dl)
(probs.argmax(dim=1) == targets).float().mean() # calculate the accuracy of the test dataset
```




    tensor(0.4333)



Oh that's not great. The accuracy is only 43%, when on the validation set it was about 94%.

This is an example of domain shift. 
Because while the training and validation images were sharp, had good lighting and the flags mostly took up a bigger portion of the image, here, it's a zoomed-in phone-camera image with poorer quality.

## **Deployment**

As this experiment shows, the high validation accuracy during training can be misleading regarding real world performance, if the training data differs from the data, the model will be exposed to in deployment.

If I were to seriously deploy such a model, having it analyze just a single image would be too brittle. It would be better to feed each $n$'th frame to the model and then average the predictions.

In my case, running a lightweight model on the phone itself if possible would be the preferred way, because sending many images a second to a server, especially from many phones at once, can be very slow. 
Also if the model would run locally, the recognition would not require internet.


```python
learner.export("european_flags_classifier.pkl") # export the model for deployment
```

Now the model is ready to be deployed.
Despite it struggling with real-world images, it does somewhat work when zooming in and centering the flag, so the model does not get distracted by the background.

For now, I published it using Streamlit here: [european-flag-classifier](https://european-flags-classifier.streamlit.app)
