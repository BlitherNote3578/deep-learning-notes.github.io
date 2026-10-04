# Lesson 3: Finding the best architecture
For this post, I want to find out what different architectures are best suited for my european flags classifier, in the hopes of increasing my production accuracy.


```python
# make sure to get the latest fastai version
%pip install -U fastai
%pip install -U timm
```

Let's set up everything as we had it when we left off last post.


```python
from fastcore.all import *
from fastai.vision.all import *
import timm
path = Path("../european_countries")
```


```python
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(256, method="squish"),
    batch_tfms=aug_transforms(do_flip=False, flip_vert=False)
)
dls.show_batch()
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_4_0.png)
    


When looking at the timm benchmark chart from lesson 3, I searched for architectures that have a similar training time to resnet18, but a higher accuracy. 

I selected these models:
- ``resnet34``, as the next step up from resnet18
- ``swin_tiny``, to see whether attention mechanisms can help the model to ignore the background (and thus score a higher accuracy)
- ``convnext_tiny``, as a competitor against swin_tiny, and a middle ground

I think that the attention of ``swin`` is going to help the model a lot, to focus on the flag itself and not to take the background into consideration as much, as a pure CNN does it.

I read that ``convnext`` was created as a direct competitor to ``swin``, using a CNN architecture inspired by the way transformers like ``swin`` work.

A model with attention could recognize the flag in the image and weigh the chunks of the image, the flag is in more, while weighting the background chunks less. <br>
A standard CNN on the other hand, might mistaken background for foreground or simply not notice the flag, because of pooling layers. <br>
That's why I expect the top losses of the resnets to consist of images where there is much background with a small flag, or possibly where a flag is on an image two times.

I also think ``resnet34`` will beat ``resnet18``. It might start overfitting and taking the background into consideration, because it has more parameters, but it should also be able to better distinguish similar flags and in general make more reasonable guesses, than ``resnet18``, due to its higher capacity.


```python
import gc

def average_final_accuracy(dls: DataLoaders, model=resnet18, num_rounds: int = 10, num_epochs: int = 5) -> tuple[list, Learner]:
	"""
	Collect the final accuracies of a model trained on the same data multiple times.
	Without a GPU, this may take a long time.
	The function returns a list of the final accuracies and the last trained learner.
	"""
	final_accuracies = []

	for i in range(num_rounds):
		print(f"-- round {i+1} --")

		learner = vision_learner(dls, model, metrics=accuracy).to_fp16()
		learner.remove_cb(ProgressCallback)
		learner.fine_tune(num_epochs)

		valid_loss, final_accuracy = learner.validate()
		final_accuracies.append(final_accuracy)

		if i < num_rounds - 1:
			del learner
			gc.collect()
			torch.cuda.empty_cache()

	return final_accuracies, learner
```


```python
final_acc_resnet18, learner_resnet18 = average_final_accuracy(dls, resnet18, num_rounds=5)

final_acc_swin_tiny, learner_swin_tiny = average_final_accuracy(dls, swin_t, num_rounds=5, num_epochs=8)
final_acc_convnext_tiny, learner_convnext_tiny = average_final_accuracy(dls, convnext_tiny, num_rounds=5, num_epochs=8) # needed more training until it converged
final_acc_resnet34, learner_resnet34 = average_final_accuracy(dls, resnet34, num_rounds=5)

learners = {
    "resnet18": (final_acc_resnet18, learner_resnet18),
    "swin_tiny": (final_acc_swin_tiny, learner_swin_tiny),
    "convnext_tiny": (final_acc_convnext_tiny, learner_convnext_tiny),
    "resnet34": (final_acc_resnet34, learner_resnet34)
}
```

    -- round 1 --
    [0, 3.516282081604004, 1.560839295387268, 0.5737704634666443, '00:20']
    [0, 1.2911040782928467, 0.6516265273094177, 0.8221311569213867, '00:20']
    [1, 0.6436280012130737, 0.3048998713493347, 0.9147540926933289, '00:19']
    [2, 0.3165445327758789, 0.2465943843126297, 0.9303278923034668, '00:20']
    [3, 0.15765564143657684, 0.2202591747045517, 0.9409835934638977, '00:20']
    [4, 0.09638651460409164, 0.2188454270362854, 0.9434426426887512, '00:19']
    -- round 2 --
    [0, 3.517235279083252, 1.5623698234558105, 0.5836065411567688, '00:20']
    [0, 1.2746235132217407, 0.6594408750534058, 0.8229508399963379, '00:20']
    [1, 0.6278114914894104, 0.34391045570373535, 0.9106557369232178, '00:20']
    [2, 0.30234983563423157, 0.24911698698997498, 0.9344262480735779, '00:21']
    [3, 0.15463513135910034, 0.22321806848049164, 0.9401639103889465, '00:21']
    [4, 0.10153602808713913, 0.21843203902244568, 0.9401639103889465, '00:20']
    -- round 3 --
    [0, 3.5373077392578125, 1.5188624858856201, 0.5901639461517334, '00:21']
    [0, 1.2866933345794678, 0.6294519901275635, 0.8295081853866577, '00:21']
    [1, 0.6396411657333374, 0.3076989948749542, 0.9065573811531067, '00:20']
    [2, 0.31107407808303833, 0.22823570668697357, 0.9336065649986267, '00:20']
    [3, 0.16584846377372742, 0.20791380107402802, 0.9393442869186401, '00:20']
    [4, 0.10072258859872818, 0.20152492821216583, 0.9409835934638977, '00:20']
    -- round 4 --
    [0, 3.490595817565918, 1.5766412019729614, 0.5893442630767822, '00:21']
    [0, 1.2883398532867432, 0.6537889242172241, 0.8163934350013733, '00:20']
    [1, 0.6630138754844666, 0.30235129594802856, 0.9147540926933289, '00:20']
    [2, 0.3106093406677246, 0.23588038980960846, 0.9377049207687378, '00:20']
    [3, 0.16958199441432953, 0.20464731752872467, 0.9467213153839111, '00:22']
    [4, 0.10854650288820267, 0.20130544900894165, 0.9450819492340088, '00:20']
    -- round 5 --
    [0, 3.4948320388793945, 1.5521448850631714, 0.5877048969268799, '00:20']
    [0, 1.256146788597107, 0.6600156426429749, 0.8024590015411377, '00:20']
    [1, 0.6416286826133728, 0.3041793704032898, 0.9073770642280579, '00:20']
    [2, 0.31312641501426697, 0.2526967227458954, 0.9319671988487244, '00:20']
    [3, 0.16257549822330475, 0.21762984991073608, 0.9434426426887512, '00:21']
    [4, 0.10684821009635925, 0.21434390544891357, 0.9442622661590576, '00:20']
    -- round 1 --
    

    c:\Projekte\Fast_AI\fastai_venv\lib\site-packages\torchvision\models\_utils.py:207: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    c:\Projekte\Fast_AI\fastai_venv\lib\site-packages\torchvision\models\_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=Swin_T_Weights.IMAGENET1K_V1`. You can also use `weights=Swin_T_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)
    

    Downloading: "https://download.pytorch.org/models/swin_t-704ceda3.pth" to C:\Users\Sandführer/.cache\torch\hub\checkpoints\swin_t-704ceda3.pth
    

    100%|██████████| 108M/108M [00:20<00:00, 5.56MB/s] 
    

    [0, 3.7531957626342773, 1.7995035648345947, 0.5426229238510132, '00:34']
    [0, 1.9940341711044312, 1.2429033517837524, 0.6778688430786133, '00:37']
    [1, 1.3921934366226196, 0.6987116932868958, 0.7942622900009155, '00:38']
    [2, 0.9192277789115906, 0.3918939530849457, 0.8860656023025513, '00:38']
    [3, 0.593063473701477, 0.2867469787597656, 0.9147540926933289, '00:38']
    [4, 0.4093044698238373, 0.2410895675420761, 0.9262295365333557, '00:38']
    [5, 0.305006742477417, 0.20913119614124298, 0.9393442869186401, '00:39']
    [6, 0.26414647698402405, 0.20063768327236176, 0.9393442869186401, '00:38']
    [7, 0.24292448163032532, 0.1968269795179367, 0.9409835934638977, '00:38']
    -- round 2 --
    [0, 3.8186957836151123, 1.8687390089035034, 0.5303278565406799, '00:34']
    [0, 2.0369622707366943, 1.276928424835205, 0.658196747303009, '00:38']
    [1, 1.4152038097381592, 0.6961755752563477, 0.7934426069259644, '00:38']
    [2, 0.8779774308204651, 0.4335468113422394, 0.8737704753875732, '00:38']
    [3, 0.5731026530265808, 0.3203870356082916, 0.9065573811531067, '00:37']
    [4, 0.4110200107097626, 0.25454023480415344, 0.9319671988487244, '00:38']
    [5, 0.3083978295326233, 0.23626001179218292, 0.9360655546188354, '00:38']
    [6, 0.2603670358657837, 0.22524254024028778, 0.9360655546188354, '00:38']
    [7, 0.23113232851028442, 0.2240632325410843, 0.9401639103889465, '00:38']
    -- round 3 --
    [0, 3.771198034286499, 1.8556768894195557, 0.5221311450004578, '00:34']
    [0, 1.9980344772338867, 1.297434687614441, 0.6540983319282532, '00:37']
    [1, 1.4389004707336426, 0.670545756816864, 0.7975409626960754, '00:39']
    [2, 0.8844939470291138, 0.3928356468677521, 0.8909835815429688, '00:37']
    [3, 0.5826331377029419, 0.3006559908390045, 0.908196747303009, '00:38']
    [4, 0.39976370334625244, 0.24289478361606598, 0.9278688430786133, '00:38']
    [5, 0.3210870027542114, 0.22831839323043823, 0.935245931148529, '00:39']
    [6, 0.2615555226802826, 0.21662703156471252, 0.938524603843689, '00:38']
    [7, 0.2381361424922943, 0.21460171043872833, 0.9393442869186401, '00:38']
    -- round 4 --
    [0, 3.728455066680908, 1.835395336151123, 0.5311475396156311, '00:34']
    [0, 1.9932152032852173, 1.2652407884597778, 0.6663934588432312, '00:37']
    [1, 1.3878984451293945, 0.6876904368400574, 0.7909836173057556, '00:37']
    [2, 0.890649676322937, 0.41158366203308105, 0.8836065530776978, '00:37']
    [3, 0.5953688025474548, 0.28209710121154785, 0.9131147265434265, '00:37']
    [4, 0.4202256500720978, 0.2437061220407486, 0.9221311211585999, '00:38']
    [5, 0.3088407516479492, 0.21097756922245026, 0.9360655546188354, '00:37']
    [6, 0.2548826336860657, 0.20035502314567566, 0.9368852376937866, '00:37']
    [7, 0.2450791746377945, 0.19826337695121765, 0.938524603843689, '00:37']
    -- round 5 --
    [0, 3.7489404678344727, 1.8005149364471436, 0.5221311450004578, '00:34']
    [0, 1.9901237487792969, 1.2502774000167847, 0.6745901703834534, '00:38']
    [1, 1.4160969257354736, 0.6827254295349121, 0.8081967234611511, '00:38']
    [2, 0.9062009453773499, 0.41889697313308716, 0.8762295246124268, '00:37']
    [3, 0.5900830626487732, 0.30730974674224854, 0.908196747303009, '00:38']
    [4, 0.4224000871181488, 0.2534251809120178, 0.9270491600036621, '00:38']
    [5, 0.3202458918094635, 0.22599239647388458, 0.9327868819236755, '00:38']
    [6, 0.25743919610977173, 0.21535737812519073, 0.9393442869186401, '00:38']
    [7, 0.24625486135482788, 0.21223564445972443, 0.9377049207687378, '00:38']
    -- round 1 --
    

    c:\Projekte\Fast_AI\fastai_venv\lib\site-packages\torchvision\models\_utils.py:222: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1`. You can also use `weights=ConvNeXt_Tiny_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)
    

    [0, 3.4080052375793457, 1.4959934949874878, 0.6073770523071289, '00:26']
    [0, 1.5280356407165527, 1.0145279169082642, 0.7344262003898621, '00:29']
    [1, 1.036551594734192, 0.5772842764854431, 0.8409836292266846, '00:28']
    [2, 0.6098787784576416, 0.30144649744033813, 0.9122951030731201, '00:28']
    [3, 0.3667311668395996, 0.2123037576675415, 0.9393442869186401, '00:28']
    [4, 0.2426314800977707, 0.18526571989059448, 0.9409835934638977, '00:28']
    [5, 0.17187532782554626, 0.15761755406856537, 0.953278660774231, '00:29']
    [6, 0.12923575937747955, 0.15306922793388367, 0.9540983438491821, '00:28']
    [7, 0.11469969898462296, 0.15362340211868286, 0.9565573930740356, '00:28']
    -- round 2 --
    [0, 3.413421630859375, 1.4968663454055786, 0.6122950911521912, '00:25']
    [0, 1.5402493476867676, 0.9983328580856323, 0.7262294888496399, '00:28']
    [1, 1.0415805578231812, 0.5523409843444824, 0.8442623019218445, '00:28']
    [2, 0.6077553033828735, 0.32775041460990906, 0.895901620388031, '00:28']
    [3, 0.3621179461479187, 0.23295044898986816, 0.9311475157737732, '00:28']
    [4, 0.22971521317958832, 0.18755847215652466, 0.9467213153839111, '00:28']
    [5, 0.15472029149532318, 0.17028369009494781, 0.949999988079071, '00:28']
    [6, 0.1285357028245926, 0.1606157124042511, 0.9524590373039246, '00:28']
    [7, 0.11262059211730957, 0.15797632932662964, 0.9549180269241333, '00:28']
    -- round 3 --
    [0, 3.426440954208374, 1.5535619258880615, 0.6065573692321777, '00:24']
    [0, 1.572391390800476, 1.0356229543685913, 0.7377049326896667, '00:28']
    [1, 1.0535902976989746, 0.5923463702201843, 0.8254098296165466, '00:28']
    [2, 0.6579263806343079, 0.318918377161026, 0.9122951030731201, '00:28']
    [3, 0.38535064458847046, 0.2273050993680954, 0.9336065649986267, '00:28']
    [4, 0.24842430651187897, 0.1844858080148697, 0.9450819492340088, '00:29']
    [5, 0.16329725086688995, 0.1671764999628067, 0.949999988079071, '00:28']
    [6, 0.12433359771966934, 0.1521606743335724, 0.9516393542289734, '00:28']
    [7, 0.10741956532001495, 0.1522524207830429, 0.9557377099990845, '00:28']
    -- round 4 --
    [0, 3.4216737747192383, 1.4458988904953003, 0.6180328130722046, '00:26']
    [0, 1.5403759479522705, 1.0129368305206299, 0.7262294888496399, '00:28']
    [1, 1.0381311178207397, 0.5709156394004822, 0.8442623019218445, '00:29']
    [2, 0.630739152431488, 0.33327847719192505, 0.8918032646179199, '00:27']
    [3, 0.37772664427757263, 0.2218792736530304, 0.9336065649986267, '00:27']
    [4, 0.23239530622959137, 0.19165728986263275, 0.9409835934638977, '00:27']
    [5, 0.16820581257343292, 0.16810184717178345, 0.9557377099990845, '00:27']
    [6, 0.12281648069620132, 0.15624043345451355, 0.9540983438491821, '00:27']
    [7, 0.1132880225777626, 0.15294398367404938, 0.9540983438491821, '00:27']
    -- round 5 --
    [0, 3.408923625946045, 1.5451236963272095, 0.6057376861572266, '00:24']
    [0, 1.558010458946228, 1.0317765474319458, 0.7360655665397644, '00:27']
    [1, 1.0655282735824585, 0.5680755376815796, 0.838524580001831, '00:27']
    [2, 0.6395228505134583, 0.3108452260494232, 0.9147540926933289, '00:27']
    [3, 0.3789712190628052, 0.23416705429553986, 0.9295082092285156, '00:27']
    [4, 0.23330754041671753, 0.1886204332113266, 0.94590163230896, '00:27']
    [5, 0.15920597314834595, 0.15682996809482574, 0.9516393542289734, '00:27']
    [6, 0.1352192610502243, 0.15274184942245483, 0.9565573930740356, '00:27']
    [7, 0.10945329070091248, 0.15222807228565216, 0.9549180269241333, '00:27']
    -- round 1 --
    [0, 3.5158603191375732, 1.5248939990997314, 0.588524580001831, '00:20']
    [0, 1.1321468353271484, 0.48510220646858215, 0.868852436542511, '00:22']
    [1, 0.5024252533912659, 0.22952967882156372, 0.9344262480735779, '00:22']
    [2, 0.2420838326215744, 0.2048640251159668, 0.9434426426887512, '00:22']
    [3, 0.1127699464559555, 0.16469235718250275, 0.9549180269241333, '00:22']
    [4, 0.06042410433292389, 0.163430318236351, 0.9557377099990845, '00:22']
    -- round 2 --
    [0, 3.5586395263671875, 1.5024447441101074, 0.6073770523071289, '00:20']
    [0, 1.1448726654052734, 0.4913003444671631, 0.8590164184570312, '00:22']
    [1, 0.5156028270721436, 0.3047734498977661, 0.9122951030731201, '00:22']
    [2, 0.22911271452903748, 0.19020827114582062, 0.94590163230896, '00:21']
    [3, 0.10302159190177917, 0.16499905288219452, 0.9565573930740356, '00:21']
    [4, 0.059756845235824585, 0.1566430926322937, 0.9565573930740356, '00:22']
    -- round 3 --
    [0, 3.5456063747406006, 1.597622275352478, 0.5778688788414001, '00:20']
    [0, 1.11137056350708, 0.4950048625469208, 0.841803252696991, '00:21']
    [1, 0.5190679430961609, 0.24763651192188263, 0.9245901703834534, '00:21']
    [2, 0.23158954083919525, 0.2131621092557907, 0.9409835934638977, '00:22']
    [3, 0.11301952600479126, 0.17738205194473267, 0.9491803050041199, '00:22']
    [4, 0.06565027683973312, 0.17741818726062775, 0.949999988079071, '00:22']
    -- round 4 --
    [0, 3.5202066898345947, 1.5697203874588013, 0.5786885023117065, '00:20']
    [0, 1.158531904220581, 0.5046472549438477, 0.8549180030822754, '00:21']
    [1, 0.4906139075756073, 0.2511407434940338, 0.9311475157737732, '00:21']
    [2, 0.22261524200439453, 0.18839918076992035, 0.9491803050041199, '00:21']
    [3, 0.1109132319688797, 0.16235783696174622, 0.9516393542289734, '00:22']
    [4, 0.06327052414417267, 0.15946324169635773, 0.9508196711540222, '00:22']
    -- round 5 --
    [0, 3.472480535507202, 1.5368143320083618, 0.5942623019218445, '00:20']
    [0, 1.1612613201141357, 0.5141760110855103, 0.8467212915420532, '00:21']
    [1, 0.519769012928009, 0.25173595547676086, 0.9303278923034668, '00:21']
    [2, 0.2387695461511612, 0.19468970596790314, 0.9393442869186401, '00:21']
    [3, 0.11113783717155457, 0.16295409202575684, 0.9508196711540222, '00:21']
    [4, 0.05995533987879753, 0.15645094215869904, 0.9540983438491821, '00:22']
    


```python
import numpy as np

for name, (final_accs, learn) in learners.items():
	mean_accuracy = np.mean(final_accs)
	std_accuracy = np.std(final_accs, ddof=1)

	print(f"The accuracy for model {name} averages at around: {(mean_accuracy * 100):.2f}% ±{(std_accuracy * 100):.2f}%")

	learn.export(f"european_flag_classifier_{name}.pkl")
```

    The accuracy for model resnet18 averages at around: 94.28% ±0.21%
    The accuracy for model swin_tiny averages at around: 93.93% ±0.13%
    The accuracy for model convnext_tiny averages at around: 95.52% ±0.09%
    The accuracy for model resnet34 averages at around: 95.34% ±0.29%
    

I tried getting the best accuracy out of every model, that is why ``convnext_tiny`` and ``swin_tiny`` got fine tuned for 8 epochs and the resnets for 5.

As the accuracy shows, ``resnet34`` and ``convnext_tiny`` won, while ``swin_tiny`` even lost to ``resnet18``.

I thought that `swin`would actually win, however it seems like the CNNs did a better job at this specific dataset.
This might have been because my dataset simply was easier to learn for CNNs, or because ``swin_tiny`` needed some other data augmentations, or resize methods, as it is a completely different architecture.

``resnet34`` beating ``resnet18`` did actually happen as expected.

Although ``convnext_tiny`` performed slightly better than ``resnet34``, also with a lower standard deviation, I would not call ``convnext_tiny`` significantly better than ``resnet34`` for this case, based on just the accuracies.

The standard deviation also shows that ``convnext_tiny``'s results are more stable than ``resnet34``'s.

However when looking at the validation loss, ``convnext_tiny`` amost always reached a slightly lower one than ``resnet34``, meaning it was more confidently correct and more uncertainly incorrect.

One thing that ``resnet34`` wins though, is the training time. <br>
while ``convnext_tiny`` needed 8 epochs to converge, ``resnet34`` needed only 5. <br>
Also the epoch time for ``resnet34`` was lower at about 21.3 seconds versus about 27.4 seconds for ``convnext_tiny``.


```python
interpreters = [ClassificationInterpretation.from_learner(learn) for acc, learn in learners.values()]
for interp in interpreters:
    interp.plot_top_losses(k=10, nrows=2, figsize=(12, 5))
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_12_0.png)
    



    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_12_1.png)
    



    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_12_2.png)
    



    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_12_3.png)
    


There are certain images, that frequently appear in all of the top losses, such as the very distorted flag of Belarus, or the serbian flag without its coat of arms.

In general the models seem to struggle with images, that have unusual coat of arms and that have a solid color as a background.

Something I noticed too, is that especially ``resnet18`` and ``convnext_tiny`` struggle with images that have many flags on the same image, for example the one with the many bulgarian flags.

In general these are mostly reasonably difficult images to guess correctly, however especially ``resnet34`` and ``swin_tiny`` did some mistakes, where guessing the flag correctly should not have been very hard, like with the Malta flags from ``resnet34`` or the Croatia flag from ``swin_tiny``.

Now let's try them on the test dataset, that I used last post too. 

I have added some more images, downloaded from the internet and some taken myself, and changed the zoom on the flags, to make the images more realistic to what the model might encounter in production.


```python
test_path = Path("../european_countries_test")
test_images = get_image_files(test_path)

test_dl = dls.test_dl(test_images, get_y=parent_label, with_labels=True)
test_dl.show_batch(max_n=31, nrows=5)
```


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_15_0.png)
    



```python
for name, (final_accs, learn) in learners.items():
	print(f"{'-'*5} model: {name} {'-'*5}")
	test_interpret = ClassificationInterpretation.from_learner(learn, dl=test_dl)
	test_interpret.plot_confusion_matrix(figsize=(12, 12))
	plt.title(f"Confusion Matrix: {name}")
	plt.show()

	print(f"Top losses for model {name}")
	test_interpret.plot_top_losses(10, nrows=2)
	plt.show()

	probs, targets = learn.get_preds(dl=test_dl)
	test_acc = (probs.argmax(dim=1) == targets).float().mean().item()
	print(f"Accuracy the model {name} scored on test dataset: {(test_acc*100):.2f}%")
```

    ----- model: resnet18 -----
    


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_16_1.png)
    


    Top losses for model resnet18
    


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_16_3.png)
    


    Accuracy the model resnet18 scored on test dataset: 36.36%
    ----- model: swin_tiny -----
    


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_16_5.png)
    


    Top losses for model swin_tiny
    


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_16_7.png)
    


    Accuracy the model swin_tiny scored on test dataset: 54.55%
    ----- model: convnext_tiny -----
    


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_16_9.png)
    


    Top losses for model convnext_tiny
    


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_16_11.png)
    


    Accuracy the model convnext_tiny scored on test dataset: 54.55%
    ----- model: resnet34 -----
    


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_16_13.png)
    


    Top losses for model resnet34
    


    
![png](/deep-learning-notes.github.io/assets/images/lesson_3/2026-10-04-finding-the-best-architecture_16_15.png)
    


    Accuracy the model resnet34 scored on test dataset: 39.39%
    

## Evaluating the results and deciding on an architecture
An important note is that the difference in the models' performances could have likely just happened by chance, just looking at the resulting accuracies, because the dataset is only 31 images big, however there are some patterns that are still relevant.


The resnets fell off massively here. The fact that this happened to both resnets with a similar reduction in performance shows that they probably overfitted to the type of images from my training and validation dataset, due to their architectures.

``swin_tiny`` scored the same accuracy as ``convnext_tiny`` here, while trailing ``resnet18`` during training.
This could be due to the model better being able to filter out background and focus on the flag itself.
It appears, that while it was last place regarding the validation accuracy, it overfitted less than both resnets.

So while ``convnext_tiny`` reached the same accuracy as ``swin_tiny`` on this small dataset, ``convnext_tiny``'s superior performance during training and faster epoch time acts as a sort of "tie-breaker". Because the size of just 31 images is too susceptible to noise, ``convnext_tiny``'s low standard deviation and high accuracy during training outweigh this sanity check.

Something interesting is also how especially ``resnet18`` and ``resnet34`` misclassified images as Estonia a lot. 
Possibly when learning images of the estonian flag, the background had a maritime, cloudy sky or in general a maritime color tone. <br>
These images were taken at the Hamburg harbor, so a more or less maritime environment.

In general the models predict maritime nations, such as Malta or Latvia quite often.

So the models might have to some extent learned correlation between the general color tone or background and the correct flag label.

Looking at the top losses, there are about the same images in every plot, most notably the France flag, whose red column is very compressed. 

Also ``resnet34`` and ``resnet18`` appeared to struggle with Germany a bit, which could explain why they got worse accuracies, because Germany makes up the largest share of the test set.

As a result of training efficiency, validation results and the out-of-domain sanity check, ``convnext_tiny`` appears to be the best suited architecture for this task.

```python

```
