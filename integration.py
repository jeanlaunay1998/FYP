import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import ExtendedKalmanFilter
from extendedKF import ExtendedKalmanFilter
import sys

from system_dynamics import dynamics
from observer import SateliteObserver
from state_model import model
from multi_shooting_MHE import multishooting
from MS_MHE_PE import MS_MHE_PE
from memory import Memory

coeff1 = [5.83292892e+0, 5.21295926e-0, 5.21295926e-0, 5.40161849e-0,0.0510607158,0.0529498483,0.0529498483,0.0539312013,0.0548673287,0.0559484414,0.0587039772,0.0587039772,0.0573881086,0.0568968693,0.0569902988,0.0580912792,0.0580912792,0.0576726129,0.057623361,0.0578517214,0.059202102,0.059202102,0.0593075978,0.0600013316,0.0644128344,0.0641759201,0.0641759201,0.0663482899,0.0675744584,0.0661647401,0.0720831572,0.0720831572,0.0718413774,0.0642912544,0.0638218208,0.0634369391,0.0634369391,0.0648866785,0.0660246926,0.0670873982,0.0669554629,0.0669554629,0.0660651196,0.0655681842,0.0659422187,0.065888182,0.065888182,0.0654235969,0.0654185251,0.0662113703,0.0641108574,0.0641108574,0.0613847585,0.06127789,0.0582663638,0.0587165651,0.0587165651,0.0535345806,0.05305883,0.0563872052,0.0559084738,0.0559084738,0.0555346869,0.0525555983,0.0529613615,0.0516580335,0.0516580335,0.051429982,0.0513845157,0.0498869318,0.0498649671,0.0498649671,0.0511021597,0.0509019616,0.0496816744,0.0499262153,0.0499262153,0.0507795221,0.0510310906,0.0510102035,0.0524616178,0.0524616178,0.0500558332,0.0527926305,0.0533227841,0.0514141488,0.0514141488,0.0519472423,0.0521422037,0.0549580625,0.0549828703,0.0549828703,0.0565853568,0.0570869369,0.0578277662,0.0606412898,0.0606412898,0.0588594543,0.0574568617,0.0566518087,0.0585398983,0.0585398983,0.0577081304,0.0562144517,0.0576411152,0.0569326901,0.0569326901,0.0564136417,0.0612243982,0.0633172838,0.0612538421,0.0612538421,0.0654657693,0.0642159579,0.0647407318,0.0632733464,0.0632733464,0.0585733896,0.0558909744,0.0552333897,0.0552422785,0.0552422785,0.0555330474,0.0564184287,0.053664661,0.0543716556,0.0543716556,0.0543770971,0.0543925923,0.0552896765,0.053461076,0.053461076,0.0538774028,0.0546986136,0.0541672939,0.0540855205,0.0540855205,0.0550637176,0.0547288097,0.0544839957,0.0538172575,0.0538172575,0.0546094713,0.0529776536,0.0549643337,0.0555883475,0.0555883475,0.0555207078,0.0537362561,0.0539948831,0.05814553,0.05814553,0.0584617663,0.0586980814,0.0591515617,0.0594404915,0.0594404915,0.0609046875,0.0606410852,0.0573593835,0.0576882407,0.0576882407,0.0563784261,0.0566225723,0.0544651875,0.0552599193,0.0552599193,0.0558060755,0.0557997158,0.0620335251,0.0614267811,0.0614267811,0.0568000718,0.0562527221,0.0559531287,0.0562808402,0.0562808402,0.0560045157,0.0561077662,0.0560676426,0.0558849258,0.0558849258,0.0560664029,0.0566260582,0.0571358796,0.0600713099,0.0600713099,0.0561977114,0.0574479187,0.0571976697,0.0604066293,0.0604066293,0.0600243718,0.0600236343,0.0573791493,0.05745929,0.05745929,0.0580978696,0.0629989167,0.0639945514,0.0663161766,0.0663161766,0.0567974272,0.0570839336,0.0570835273,0.0563184341,0.0563184341,0.0568279029,0.0565943656,0.057308939,0.0564820102,0.0564820102,0.0545735684,0.0582166967,0.056684105,0.0568818605,0.0568818605,0.056343803,0.0536437447,0.0538208501,0.0556370735,0.0556370735,0.0553376053,0.0546754157,0.0547969559,0.0547980488,0.0547980488,0.0560613106,0.0619829004,0.0616338791,0.0616118691,0.0616118691,0.0628132222,0.0622081781,0.0620131422,0.062082446,0.062082446,0.0632855288,0.0666652854,0.0664953901,0.0697307777,0.0697307777,0.0698171189,0.0694584766,0.0754705749,0.0728841997,0.0728841997,0.0683356472,0.0692467618,0.0697816747,0.0696960939,0.0696960939,0.0673164205,0.0667259239,0.0691014591,0.0687796185,0.0687796185,0.0676500261,0.0644892388,0.0646631461,0.0652945803,0.0652945803,0.0640928388,0.0635636738,0.0632550705,0.0602335508,0.0602335508,0.0570421212,0.056961763,0.0566030303,0.0558770543,0.0558770543,0.0561631218,0.0576323785,0.0568292649,0.0573825207,0.0573825207,0.0570358259,0.0574141902,0.0578841882,0.0586910735,0.0586910735,0.0589584618,0.0589129327,0.0607711519,0.0587584481,0.0587584481,0.0587040066,0.0594434593,0.0581961006,0.0576910898,0.0576910898,0.0547896783,0.056305787,0.0552025392,0.0548942338,0.0548942338,0.056648565,0.0567170228,0.0566477044,0.0574394135,0.0574394135,0.057390646,0.057450192,0.0581075187,0.051248463,0.051248463,0.0481900066,0.0482119095,0.0470392437,0.0471465419,0.0471465419,0.048108372,0.0480048615,0.0480386434,0.0486677259,0.0486677259,0.0492210323,0.0526022655,0.0517476622,0.0526217424,0.0526217424,0.0521489417,0.0502605051,0.050589264,0.0509726008,0.0509726008,0.0509137341,0.0509119465,0.051008829,0.0498472417,0.0498472417,0.0542297565,0.0585337822,0.0587975517,0.0625570777,0.0625570777,0.0622950538,0.0621195742,0.061993885,0.0616594515,0.0616594515,0.0612932515,0.0605722457,0.0580457616,0.0609759693,0.0609759693,0.0600556141,6.28788957e-02]
coeff2 = [0.74338183,0.70064236,0.74836693,2.18017603,39.38943022,58.15673791,64.51592097,71.93272672,68.66833484,69.2241209,69.14345567,61.39241408,61.21981302,61.65039686,61.55644407,61.73631036,61.91550313,63.19284982,62.66177529,62.58587298,67.78399247,64.33115513,63.01185733,64.15649081,65.43964518,72.8852546,68.83476248,64.6614267,65.88670717,69.27083998,68.92817882,65.57576449,76.02377876,76.36789915,69.93033799,70.06129997,70.24954249,69.73750631,69.27267351,69.91319962,69.57334243,71.20456854,68.8241534,69.51296373,71.2242808,71.19686598,77.13427122,71.17082013,75.03341451,73.49307901,74.03024386,73.73364092,77.70056059,81.27373923,76.51308068,85.93200948,103.73903389,87.66972416,96.1436283,90.20763523,85.12810083,86.58999242,86.24570125,85.45102787,88.30779466,98.82937173,89.14912751,95.49408271,90.43607172,98.22671089,92.0301844,92.09739991,91.71239112,93.2511083,85.33486619,85.8840166,85.23682117,85.7124715,87.31043276,80.3806157,78.35036197,82.81422356,78.02183622,75.54007669,74.10104585,71.96067661,71.47814432,71.60358228,72.14987961,71.90501389,71.94993911,82.54916654,73.02535328,97.61413841,68.04179616,90.66793008,117.6736885,90.67158462,105.85949647,82.1180953,89.80879879,73.9532457,73.84618561,81.07733449,80.42037622,85.02383539,87.09301337,86.5504989,89.77650139,88.25849253,90.51444838,84.97678832,94.11432762,89.16772396,91.38381058,92.44547051,93.72017912,93.71054478,95.02317908,93.38669804,94.87878063,97.30984036,91.03292957,92.72058701,91.9719332,92.03411007,91.23479747,88.90660629,84.73043014,83.50915485,83.48279746,75.59936003,75.52549973,72.66895095,74.93980877,74.40760562,74.1770775,73.18202957,72.85876664,72.53994527,71.9850073,71.94899027,71.06603487,71.50594616,70.97658261,68.52246634,67.08685542,67.59920073,67.13203981,69.96390394,65.11615891,65.13633477,69.1480162,69.03325684,72.64106346,70.18139941,69.8702177,69.63361044,71.13433214,68.77324004,70.59210513,71.50661547,70.44926762,71.21492789,71.61484796,67.12370067,61.99879573,62.99763605,63.1946818,63.82850717,64.93084924,68.25657527,67.52802681,66.97664108,67.63092725,68.30784603,65.46362693,64.04071865,64.25139795,64.64219841,65.40504819,65.48759178,64.97898997,64.81735882,63.96915137,64.41152731,68.56927686,81.56255552,81.24776884,80.18825751,77.50804226,77.36766666,76.83602507,76.58008132,77.36193521,75.38289316,71.93981373,81.09382968,86.82785209,87.0717943,85.45621516,85.80231604,86.53487297,86.78659527,84.7504466,86.12092116,86.50771902,82.58797803,82.30632635,81.73131691,82.87349819,86.83001823,85.04343623,84.39409566,84.3839284,85.6222159,80.92086515,73.78225868,75.01425087,74.71371589,73.12059898,73.9629786,74.07678802,73.91871581,73.49831976,74.4434622,74.51721014,74.5048486,77.1494654,77.15560697,76.90533229,76.771812,76.39566291,76.77425724,75.82302635,73.98447015,73.12570413,78.38120873,88.56831012,86.96937643,84.43374675,79.87935511,78.53603302,75.8973666,76.28282301,77.45194309,75.88735846,77.37494905,77.40812737,77.30709922,77.20383017,77.8285287,76.62001006,77.02224568,77.85470174,79.84804083,77.21802613,76.00689075,76.60454438,78.34653643,75.49381097,75.81156769,79.13338754,79.10628093,81.31336897,81.5709245,78.16287902,76.59588155,76.65107059,76.4355115,74.61636679,74.6062477,74.54692577,74.10023586,73.93882962,72.10285071,71.71598653,74.90556797,77.50333194,77.60190957]
coeff3 = [65,65,65,65,65.77784497,70.1691119,71.21306766,69.46995236,70.11549349,70.75832557,69.23983796,68.48508262,68.41626261,71.057017,68.18425869,72.65811559,72.99879172,71.88658144,71.29329262,68.91317537,69.89368785,67.32949994,67.55927383,67.83419093,68.96938021,71.77490085,69.15431477,74.61454561,70.37235069,70.35247742,71.47682933,73.15417903,69.05636183,68.88461192,69.31149874,72.89337249,73.09923516,73.29957497,76.61984867,74.52219754,78.09973517,86.15668634,81.49719272,79.63613365,80.97192465,78.55797567,89.04720911,78.30226676,84.38447257,86.95432783,81.90628505,83.12910722,81.48451566,87.79130485,87.03290287,84.87403793,93.79874447,84.48468799,83.96303889,84.64978196,88.25346936,79.99067237,81.67380136,81.37455449,81.32454836,80.35756544,80.53335301,84.73648385,80.66024477,82.0252783,82.60533267,85.34271733,85.1195181,85.91880748,85.93368456,86.44502971,88.81111856,84.26152635,88.24203556,83.96920617,82.2055072,81.88450251,77.88298739,80.75100115,77.32149786,77.02838256,78.20836904,78.04590552,80.07347186,79.48704938,90.31087316,76.20341537,112.49520447,75.85428029,86.23245581,87.98116756,75.65003492,119.0069601,74.46189708,100.32475611,84.62868503,77.06990217,77.35059602,83.6346159,90.69744575,86.70740004,87.66635958,85.56832919,83.84112545,84.5907851,86.44462703,85.26481633,87.70015591,82.63866123,87.51807811,94.47818053,96.70411707,99.70892531,94.65897009,99.75122862,96.17852035,95.06516704,93.06609319,86.19228149,87.45207514,86.31212894,82.97694758,83.49128871,82.20015727,82.85578902,83.37097296,82.98669942,83.73755661,81.81771453,85.30430663,84.38931674,82.14627383,80.30829371,81.49352033,81.04549573,80.73888753,80.77927704,81.22998621,80.45511262,86.38644665,86.32887913,87.56758601,90.90240888,90.92740947,93.61760191,97.17352488,95.10120174,97.16429388,94.03168376,99.65309364,102.25173634,103.88022963,93.98361889,101.34793738,97.19322434,100.1882356,100.22522207,104.96933508,103.31542118,97.20407883,96.88325296,97.1034831,96.17606317,95.124082,94.83215103,90.60319979,90.34990508,89.67654891,91.41567894,90.6967411,89.18632759,86.24504842,86.99942551,92.83243049,91.01275339,86.71835169,73.61922816,73.07722003,70.96560993,69.85352625,72.42569724,70.63173249,70.79267465,71.05767092,70.37912194,70.63450288,72.8456281,72.50668198,69.61159485,71.76847863,69.90275015,69.76568874,71.7073091,71.19094751,71.53587681,71.19225201,71.1062486,82.29451997,83.30146727,84.07866282,90.1925494,89.21475915,94.34549251,91.03974648,90.66498286,91.17281872,91.13897841,91.93286809,92.31855422,94.01431151,82.73782231,79.32725321,79.88384225,79.79249005,75.21893738,74.70266866,75.79598838,77.51597326,76.20580841,76.32948281,73.77293304,74.41645962,74.86457527,74.72610556,72.86842156,72.94354386,73.61616477,70.76243689,69.85141512,69.58317003,69.64998883,75.13374521,80.3329458,78.11767173,77.40399178,81.18347057,81.04805256,81.49919839,82.71821008,83.65635052,82.54444271,80.1570188,79.69376858,79.77058966,78.40881505,82.05279659,81.91166827,78.49145158,80.80258047,82.035447,81.86940585,83.61133774,82.11370636,79.90279705,83.91061327,83.61884395,85.7978071,86.59188915,83.92837403,83.89011443,83.25336796,86.26808598,89.16298624,84.18428738,82.40667701,82.44304133,83.09869793,78.61524393,80.80273071,81.44748143,80.93182044,79.92427535,77.66535101,78.59950441,80.01065817]
coeff4 = [20,20,20,19.2709197,86.4787682,63.1614339,71.4581781,67.0875976,67.6963585,65.8595097,59.3439438,59.2142397,52.647823,50.1560096,48.5564935,43.7001894,43.5697983,40.6675161,38.2191913,37.2950538,35.8805667,34.4959493,33.187005,32.0840532,31.6907246,30.3958792,28.6832667,28.6514942,27.1203229,26.5588122,26.0712239,24.2125971,23.9727821,23.4773091,23.3889498,22.4558385,21.9563869,21.0649676,20.6127738,20.3830265,19.5010797,19.0859252,17.9101562,17.1852742,17.1406748,15.9284989,15.5844153,14.2238067,13.4499331,12.8197876,11.765669,10.9781569,10.2209015,9.45462187,8.92140646,8.39588015,7.8706278,7.2879177,6.80244782,6.3575761,5.95564021,5.58659623,5.28534341,4.95277266,4.67090959,4.38505748,4.15287384,3.9572463,3.68917409,3.4921594,3.31519352,3.15026222,2.97592453,2.84363717,2.71645791,2.59976729,2.5037755,2.39432512,2.30822187,2.23705392,2.1809745,2.13246408,2.1037376,2.09720614,2.10087132,2.12143361,2.14130378,2.12571329,2.02505947,1.82114384,1.56102293,1.30828383,1.10348457,0.932040725,0.801188994,0.70479323,0.6269022,0.562394653,0.508283097,0.471199951,0.432640002,0.406659917,0.381837652,0.360619059,0.342011664,0.325911739,0.311240132,0.298111196,0.286416908,0.275866019,0.26599539,0.257381401,0.249435085,0.24219919,0.235702659,0.229838506,0.224458233,0.219523311,0.215111734,0.210923442,0.207273898,0.203936468,0.2007064,0.197884373,0.195159814,0.192684979,0.190432083,0.188399511,0.186584327,0.184893478,0.183405617,0.182140865,0.181046674,0.180120639,0.179366492,0.178792795,0.178380221,0.178151225,0.178078185,0.178176982,0.178446818,0.178874395,0.179450199,0.180197779,0.181093403,0.182140555,0.183345402,0.184683409,0.18617867,0.187804764,0.189599446,0.191502127,0.193517378,0.195680028,0.19800841,0.200425696,0.202964058,0.20560991,0.208352759,0.21111871,0.214122044,0.217115145,0.220244215,0.223426386,0.226683212,0.230034619,0.233443616,0.236915036,0.240378637,0.244002121,0.247835028,0.251325317,0.255104353,0.258846777,0.262702024,0.266476184,0.270328386,0.274220001,0.27809576,0.281998747,0.285945534,0.289867691,0.293830166,0.297532823,0.301722417,0.305655427,0.309584267,0.313467752,0.317426978,0.321297044,0.325143985,0.329002305,0.332834321,0.336624107,0.340410439,0.344161572,0.347825634,0.351595426,0.355294517,0.358865059,0.362490526,0.366056214,0.369575074,0.373106032,0.37656468,0.380001036,0.383425326,0.386764645,0.390085038,0.393348588,0.396602771,0.399810423,0.402945253,0.406087793,0.409134421,0.412169513,0.415184196,0.418115761,0.421052696,0.423964702,0.42676861,0.429586208,0.432354398,0.435095898,0.437819254,0.440491906,0.443115098,0.445754716,0.448351445,0.450925566,0.453442501,0.455895148,0.458359512,0.460759945,0.463152612,0.465552348,0.467868597,0.470213593,0.472515397,0.474796563,0.477047734,0.479256456,0.481423736,0.483630309,0.485801671,0.488109342,0.490084678,0.492310623,0.494424846,0.496569517,0.498721915,0.500893321,0.503024727,0.505168024,0.507326029,0.509501462,0.511816533,0.513823535,0.515984652,0.518207433,0.520431009,0.522716818,0.525005079,0.527255042,0.52958075,0.531929274,0.534257082,0.536494039,0.5390533,0.541561492,0.543894304,0.546435715,0.548956431,0.551529289,0.554064804,0.556627859,0.559720583,0.562121092,0.564856941,5.67677838e-01]
coeff5 = [40,40,40,22.592919,32.2837091,31.7209335,33.3970314,31.4194641,32.0906553,29.377503,27.8124557,27.8137845,24.8190028,23.6351975,22.862232,20.6529438,20.5682015,18.9855548,18.1196408,17.6931672,16.7673823,16.4040404,15.8426762,15.3057652,15.1257259,14.3688073,13.7538265,13.682044,13.0418867,12.7800816,12.5711453,11.8486471,11.6178989,11.3949303,11.3651272,10.951166,10.7301771,10.3280226,10.1403085,10.0555703,9.66578754,9.49467503,9.01716488,8.6305088,8.41134558,8.06842594,7.77825525,7.26960249,6.93446484,6.60105599,6.07911428,5.68433739,5.31274344,5.01690716,4.65711753,4.39220667,4.12101801,3.8271293,3.57183482,3.35012905,3.14149068,2.9482805,2.79532974,2.62267017,2.4766534,2.32613799,2.19858517,2.09677392,1.9668574,1.8670515,1.77126188,1.68630716,1.60008836,1.52634319,1.46000062,1.39889929,1.34947217,1.29207425,1.25324624,1.21060226,1.18134858,1.15761671,1.1426029,1.13666621,1.14307872,1.15527325,1.16644954,1.15866737,1.10368011,0.998496611,0.858230813,0.72249064,0.6077612,0.516594329,0.445996012,0.391301072,0.347520124,0.315645203,0.284778924,0.263661691,0.246006276,0.228184219,0.21483486,0.203240213,0.193062056,0.183928105,0.176047485,0.168895025,0.162490756,0.156700912,0.151501973,0.146632378,0.142313246,0.138418481,0.134917768,0.131750685,0.128889243,0.126376406,0.123942423,0.121809335,0.119951423,0.11815322,0.116543773,0.115091936,0.11371312,0.112498463,0.111400718,0.110417655,0.109543971,0.108786808,0.108138012,0.107585748,0.107146029,0.10681992,0.106587761,0.106458798,0.106420754,0.106514559,0.106699362,0.106983739,0.107375833,0.107870323,0.108459195,0.109159516,0.109958172,0.11085676,0.111857863,0.112952991,0.114155523,0.115443094,0.11686176,0.118354858,0.119933232,0.121659461,0.123424859,0.125307353,0.127288208,0.129352606,0.13150128,0.133788141,0.136053371,0.138446769,0.140937392,0.143495446,0.14613471,0.148870337,0.151672348,0.154552133,0.157533597,0.16052941,0.163617942,0.166822039,0.17007488,0.173374082,0.176763011,0.180211235,0.183729953,0.187310404,0.19095441,0.194666959,0.198472808,0.202323238,0.206238204,0.210170851,0.214256912,0.218357753,0.222527592,0.226753486,0.231004714,0.235351653,0.239729851,0.244199021,0.248669598,0.253245988,0.257920979,0.262616871,0.267391047,0.27218758,0.27704887,0.282018235,0.287048291,0.292111515,0.297184223,0.302485157,0.307796234,0.31311591,0.318559781,0.324044796,0.329606241,0.335229331,0.340965462,0.346748272,0.352586032,0.358543865,0.364543208,0.370614569,0.376812505,0.38304398,0.389415244,0.39584619,0.402375195,0.409008332,0.415713844,0.422537856,0.42948046,0.436524582,0.443635707,0.450924948,0.458307328,0.465813001,0.473454267,0.481191233,0.48906451,0.497032166,0.505167501,0.513425455,0.52188235,0.530478638,0.539185553,0.548113472,0.557174347,0.566343158,0.575710417,0.585303201,0.595051747,0.605004118,0.6151211,0.625608597,0.636135908,0.647026064,0.658149996,0.669552396,0.681145616,0.693030685,0.705210115,0.717685856,0.730868514,0.743480377,0.756932769,0.770588319,0.784960091,0.799550912,0.814533078,0.829996277,0.845639837,0.862277044,0.878526597,0.895796012,0.913385609,0.931856178,0.950272223,0.969708061,0.989662677,1.01023772,1.03142773,1.05324451,1.07595125,1.09969244,1.12370779,1.14874857e+00]
coeff6 = [50,50,50,50,59.1161635,52.2056167,54.3938898,50.5846799,51.7763727,45.8076452,44.8019048,44.580211,39.8242636,37.9133272,36.6728349,33.1626753,33.0162937,30.8410877,29.1192297,28.4398357,27.2700096,26.3840625,25.5711378,24.6318209,24.3621264,23.2875436,22.1719641,22.0877833,21.0364411,20.6214787,20.2758751,19.1312806,18.7869624,18.430126,18.3871941,17.7345156,17.3864907,16.7505029,16.4614559,16.3365241,15.7262652,15.4867388,14.6490532,14.0944832,13.7393694,13.2107961,13.3613596,11.9359671,11.4348399,10.864262,10.0172085,9.36756272,8.76764455,8.22214069,7.69966787,7.26689847,6.85041282,6.33910312,5.92882878,5.55437445,5.21161221,4.88537253,4.64141491,4.35675,4.11494019,3.8711512,3.66404442,3.46803892,3.27498219,3.10826511,2.95170596,2.81164317,2.66479643,2.54739664,2.43957128,2.33694346,2.25564848,2.16025222,2.09077944,2.0264283,1.97921389,1.94043564,1.91642762,1.91145732,1.91709219,1.93747673,1.95673851,1.94351103,1.85362194,1.67824032,1.44264371,1.21700499,1.02777248,0.873019614,0.754489568,0.661747184,0.589779929,0.533193663,0.480509278,0.446781473,0.414022948,0.388610166,0.364675071,0.345113576,0.327894902,0.312957569,0.299484893,0.287465345,0.276589263,0.266910709,0.258168562,0.249680221,0.242851122,0.236336978,0.230491649,0.225251615,0.220502112,0.216285723,0.212453091,0.208902258,0.205757979,0.202845722,0.200263581,0.197850022,0.195694587,0.19373798,0.192004576,0.190460663,0.189099466,0.187936701,0.186983991,0.186166057,0.18555881,0.185138129,0.184896031,0.18483663,0.184922381,0.185272198,0.185744292,0.186414841,0.187272664,0.188312999,0.189525326,0.190938754,0.192532447,0.194294718,0.196275725,0.198415549,0.200755349,0.203276059,0.205997344,0.208899809,0.211975489,0.215185403,0.218703296,0.222349048,0.226187189,0.230203552,0.234387874,0.238698535,0.243283018,0.247982286,0.252901035,0.257952016,0.263196806,0.268640646,0.274243538,0.280030967,0.286079984,0.292136678,0.298461657,0.304978855,0.311715205,0.318561012,0.325649214,0.332892872,0.340339425,0.347972186,0.355811366,0.363846987,0.372146987,0.380653559,0.389314767,0.398059351,0.407369476,0.416747264,0.426287448,0.436121195,0.446269706,0.4566029,0.46717179,0.478105818,0.489279408,0.500708542,0.512396061,0.5245231,0.537000133,0.549629225,0.562898323,0.576297847,0.59026534,0.604666492,0.619288323,0.634612337,0.650379623,0.666597703,0.68340261,0.700752246,0.718705269,0.737283579,0.75655078,0.776643884,0.797338521,0.818940538,0.841317327,0.864652005,0.889083897,0.914464355,0.941071046,0.969110593,0.998041168,1.02870797,1.06081184,1.09471244,1.13051804,1.16835562,1.20828687,1.25088147,1.29617276,1.34469769,1.39610985,1.451347,1.51074022,1.57455476,1.64370598,1.7184893,1.80032478,1.88961525,1.98701903,2.09460242,2.21305566,2.34386325,2.48751789,2.64696937,2.82171708,3.01296807,3.21253429,3.4208379,3.61872486,3.78789092,3.9057392,3.95171795,3.92089484,3.82491487,3.68534943,3.52348672,3.35876225,3.19264929,3.04003861,2.9030677,2.77173818,2.65695152,2.55270132,2.4586337,2.37356849,2.29898473,2.2267256,2.16324369,2.10448975,2.05276339,2.00217575,1.95736183,1.91593611,1.87775243,1.84263221,1.80957637,1.77926309,1.75163926,1.72501161,1.70083216e+00]
coeff7 = [361,36.8263684,30,25.9381326,27.0311473,24.6430489,25.4135823,25.0012039,25.9848724,24.5175373,24.2175436,24.5723062,22.5888428,22.287321,22.008223,19.9536972,20.1719272,18.5669337,17.836116,17.3902369,16.4915755,16.2169267,15.2248918,14.8132671,14.7766201,13.9916354,13.2742466,12.9754741,12.3145007,12.0983548,11.580765,10.6591012,10.4213236,10.1052451,9.78725273,9.28616728,8.91059447,8.52404064,8.21552869,7.95053078,7.46619884,7.06151877,6.57289503,6.1940985,5.88333209,5.56727954,5.31807437,4.90084818,4.67846734,4.41384904,4.10033776,3.81173533,3.56616678,3.36862697,3.14553605,2.95345673,2.77428436,2.58920413,2.42374733,2.28386516,2.14014762,2.00012772,1.89982283,1.78737069,1.68677441,1.5932754,1.50647803,1.43899941,1.35678497,1.28837757,1.22309142,1.16687766,1.10914126,1.0573956,1.01214891,0.969788629,0.936184907,0.899968441,0.870694803,0.84398799,0.824116932,0.807240918,0.797591332,0.794781073,0.797687168,0.805964255,0.814638548,0.809446637,0.774067017,0.70071709,0.602368202,0.506492977,0.42759095,0.363248621,0.313794895,0.274933282,0.244827659,0.221492766,0.201470775,0.185407032,0.171349031,0.160592504,0.150973463,0.142672699,0.135445185,0.129003428,0.123301786,0.11818148,0.113560617,0.109351306,0.105530118,0.102139627,0.0990228729,0.0962004536,0.0936246652,0.0913020355,0.0891973088,0.0872398722,0.0855317294,0.0839296741,0.0824638164,0.0810760066,0.0798405334,0.0786500633,0.0776140956,0.076638308,0.0757394697,0.0749343517,0.0741976386,0.0735394797,0.0729522424,0.0724396109,0.0720002143,0.0716283959,0.0713242173,0.0710925147,0.0709246803,0.0708274171,0.0707905502,0.0708246912,0.0709225972,0.0710836178,0.071303676,0.0715912486,0.0719389611,0.0723424404,0.0728121547,0.0733336307,0.0739152809,0.0745547964,0.0752461442,0.0759919531,0.076786852,0.0776590123,0.0785368321,0.0794823499,0.0804793243,0.0815133388,0.0825904744,0.0837107434,0.0848554177,0.08604152,0.0872703317,0.0885229116,0.0898045226,0.0911232698,0.0924648838,0.0938335977,0.0952345103,0.0966375077,0.0981028712,0.0995472767,0.101026718,0.102511151,0.104005369,0.105531713,0.107060251,0.108602452,0.110135687,0.111685792,0.113248268,0.114817963,0.116368821,0.117875019,0.119497925,0.121059692,0.122647358,0.124207978,0.125728427,0.127272076,0.128807575,0.130332454,0.131872112,0.133369702,0.134872188,0.136367097,0.137844566,0.139344678,0.14077701,0.142220407,0.143662349,0.145083258,0.14649443,0.147882488,0.149245764,0.150639721,0.151984274,0.153320198,0.154637383,0.155935283,0.157230894,0.158499661,0.159752496,0.160998832,0.16221579,0.163418231,0.164614723,0.165785302,0.166956569,0.168102036,0.169226882,0.170350275,0.171457558,0.172551315,0.173631216,0.174704557,0.175743677,0.176789215,0.177820636,0.178845468,0.179843859,0.18082615,0.181800878,0.182760403,0.183718853,0.184664137,0.185594814,0.18652244,0.187433348,0.188340155,0.189236899,0.190122623,0.190984016,0.19185753,0.192723196,0.193573948,0.194441341,0.195308958,0.196163851,0.197004324,0.197858429,0.198722524,0.199574664,0.200425606,0.201280233,0.202137855,0.202986027,0.203860836,0.20472027,0.205759391,0.20651004,0.207410363,0.208315676,0.209203972,0.210142207,0.211100989,0.211995825,0.212974072,0.213889773,0.214876152,0.215831853,0.216839559,0.217839423,0.218850495,0.219875171,0.220907794,0.221880013,0.223053652,0.224149967,2.25250647e-01]
coeff8 = [30,30,27.46193044,9.81186016,10.95202542,11.85683926,12.32232295,11.97283677,12.51164441,11.71358243,11.50899777,11.74988537,10.78843316,10.62358766,10.45126409,9.4807085,9.57177227,8.81537464,8.46758837,8.25637029,7.74242768,7.70381352,7.21557895,7.07041113,7.05203533,6.65711074,6.37843311,6.25298058,5.94618602,5.84845951,5.62038207,5.21750834,5.10658929,4.96322905,4.82821749,4.60152473,4.43240815,4.25778594,4.12563466,4.01080815,3.78177844,3.59440948,3.37393237,3.17810631,3.03355635,2.87486304,2.70854605,2.54385949,2.43441334,2.3029988,2.14296519,1.99535223,1.86905968,1.75961115,1.65376708,1.5551036,1.4608659,1.36680848,1.28205828,1.20860946,1.13492882,1.06074924,1.00864753,0.95000519,0.89749378,0.84916434,0.80578216,0.76840386,0.72551129,0.69165854,0.65607351,0.62638852,0.59605913,0.56915102,0.54570004,0.52338182,0.50584982,0.48710251,0.47143542,0.45802763,0.44771118,0.43867101,0.43457349,0.43303493,0.43542562,0.44022375,0.44507787,0.44239602,0.42274238,0.38370819,0.33065657,0.27946935,0.23559594,0.20123095,0.17426289,0.15318302,0.13647137,0.12277565,0.11278042,0.10362241,0.09613517,0.09021553,0.08492939,0.08037546,0.0764259,0.07287284,0.06974207,0.06693126,0.06440314,0.06213996,0.06007204,0.05815563,0.05648755,0.05495343,0.05357998,0.05233402,0.05121327,0.05020038,0.04927928,0.04843392,0.04767023,0.04697787,0.04633711,0.04574517,0.04522191,0.044738,0.04430071,0.04391178,0.04356608,0.04326084,0.04300477,0.0427809,0.04260508,0.04247166,0.04237527,0.04232236,0.04230964,0.04233657,0.0424062,0.04251611,0.0426667,0.04285729,0.04308676,0.04335798,0.04366893,0.04402006,0.04441062,0.04483879,0.04530832,0.04582009,0.04636479,0.04695175,0.0475694,0.04824131,0.04893748,0.04967513,0.05045236,0.05126137,0.05210594,0.0529929,0.0538938,0.05484181,0.05581782,0.05682668,0.05786489,0.05894052,0.06004464,0.06117965,0.06234073,0.06354064,0.06477455,0.06602776,0.06731263,0.06861912,0.06995991,0.07132279,0.07271641,0.07413494,0.07557558,0.07704917,0.07854874,0.08006744,0.08161923,0.08318466,0.08479255,0.08641553,0.08806593,0.08973228,0.09143515,0.09314997,0.09489125,0.09667231,0.09844445,0.1002537,0.10210397,0.10396289,0.10583952,0.10775599,0.10969627,0.11166195,0.11365627,0.11566929,0.11769931,0.11977631,0.12187002,0.12399665,0.1261463,0.12832,0.13052823,0.13275841,0.13501786,0.13731783,0.13963531,0.14199318,0.14436658,0.1467784,0.14923139,0.15170825,0.15423218,0.15679439,0.15937808,0.16199609,0.16466139,0.1673692,0.17011772,0.17289312,0.17573148,0.17861464,0.18154131,0.18452604,0.18754096,0.19060857,0.19372099,0.19688176,0.20011008,0.20338584,0.20672621,0.21013028,0.21358837,0.21710791,0.22069785,0.22434869,0.22805131,0.23183961,0.23570576,0.23964423,0.24367501,0.24779854,0.25200381,0.25627885,0.26068092,0.2651918,0.26978496,0.27448484,0.2792975,0.28423841,0.28923368,0.29445884,0.29978181,0.30536767,0.31086663,0.31662645,0.32255799,0.32862235,0.33487774,0.34143555,0.34786913,0.35464976,0.36165103,0.3688785,0.37626153,0.38393852,0.39181134,0.39993693,0.40832833,0.41701541,0.42597999,0.43514394,0.4447277,0.45456394]
coeff9 = [10,10,10,10.30885784,18.14090608,18.83779582,19.78280425,19.40302405,20.13489895,19.06326764,18.65044243,18.91667522,17.364033,17.0833217,16.79590443,15.23874949,15.37865222,14.1800376,13.60762333,13.26968551,12.60175686,12.38222965,11.68247366,11.37510222,11.34529929,10.8028003,10.28332497,10.0867604,9.59974797,9.44382677,9.07565682,8.42353491,8.27406283,8.05006541,7.84125636,7.48247218,7.21535871,6.9398626,6.73501444,6.55651989,6.18994558,5.88921926,5.4982084,5.22361072,5.01659757,4.73523054,4.50212544,4.19742487,4.00994901,3.80656854,3.54423752,3.3012782,3.09413059,2.932673,2.74063249,2.57843184,2.424641,2.2681518,2.12939239,2.00688903,1.88434904,1.76380142,1.67701056,1.58026207,1.4935239,1.41299089,1.34069247,1.28188982,1.20931465,1.1504437,1.09468424,1.04543745,0.99421389,0.95077871,0.91217526,0.87509607,0.84615867,0.81539906,0.78894384,0.76715694,0.75023861,0.7364078,0.72923736,0.72886873,0.73088342,0.73913558,0.74738577,0.74274949,0.7099064,0.64513583,0.55499357,0.46997224,0.39822442,0.33958013,0.29412461,0.25852989,0.2303369,0.20953479,0.19034808,0.17577833,0.16288441,0.15296385,0.14409804,0.13651882,0.1298404,0.12388436,0.11862913,0.11391772,0.10966806,0.10575843,0.10233875,0.09918877,0.09638584,0.0938388,0.09153994,0.08946902,0.08760686,0.08590969,0.0844229,0.08302179,0.08177014,0.08064888,0.07961827,0.07870449,0.07781278,0.07703957,0.07634333,0.07573354,0.07519767,0.0747321,0.07434256,0.07402239,0.0737819,0.07360568,0.07350288,0.07347506,0.07350789,0.07363064,0.07381458,0.07407522,0.07440653,0.07481011,0.07528376,0.07583262,0.07645564,0.0771479,0.07791726,0.07875508,0.0796697,0.08065741,0.08172101,0.08285623,0.08406285,0.08534728,0.08670127,0.08813023,0.08963594,0.0912109,0.09285586,0.0945632,0.09635277,0.09819903,0.10013622,0.10212933,0.10419034,0.10632922,0.1085397,0.11081994,0.11322025,0.11560084,0.11809586,0.12068101,0.12332913,0.12603898,0.12883501,0.13170135,0.1346495,0.13768495,0.14076518,0.14394597,0.14722659,0.15057799,0.15400549,0.15753615,0.16114088,0.16484956,0.16865663,0.17248534,0.17652877,0.18061812,0.18481259,0.18910894,0.19354151,0.19806357,0.20275503,0.20750446,0.21240436,0.21748553,0.22264389,0.22800778,0.23353707,0.23918752,0.24499137,0.25107277,0.2572946,0.26371585,0.27036464,0.27722667,0.28432859,0.29167474,0.29930586,0.30721084,0.31539666,0.32392886,0.33275989,0.34198597,0.35162059,0.36164923,0.37217046,0.38318324,0.39468968,0.40674063,0.41943472,0.43281781,0.44693379,0.46178451,0.47760184,0.49437846,0.51223051,0.5312887,0.55156928,0.57331783,0.59664324,0.62174874,0.64894503,0.67843326,0.71047829,0.74549467,0.78376162,0.8259009,0.87233875,0.92352341,0.98000139,1.04251446,1.11123003,1.18645382,1.26518933,1.34802288,1.42781329,1.49689044,1.54703083,1.56926834,1.56098293,1.52583227,1.47205028,1.40836794,1.34244932,1.27660978,1.21551561,1.15949967,1.10759845,1.06121584,1.01923556,0.98123686,0.94704992,0.91593103,0.88791089,0.86203962,0.83879442,0.81748591,0.79777768,0.77979854,0.76317389,0.74781621,0.73363085,0.72051799,0.70736631,0.69694397,0.68653964,0.67676773]
coeff10 = [0.51709758,0.4,0.34652715,0.42812393,0.42725902,0.42898717,0.42898175,0.42898421,0.42892151,0.42901525,0.42944463,0.42900289,0.42900022,0.42899942,0.42899842,0.42899852,0.42899886,0.42918206,0.42899931,0.42899607,0.42930181,0.42899881,0.42909543,0.428997,0.42900077,0.42896078,0.42899065,0.42914275,0.42899907,0.42899885,0.42898624,0.42891943,0.42902373,0.42899728,0.4289948,0.42899881,0.42899446,0.42899533,0.4289979,0.42899939,0.42899249,0.42909489,0.42900202,0.42899632,0.42956793,0.42899942,0.43055177,0.42899947,0.4288901,0.42899815,0.428998,0.42896905,0.42899784,0.42940356,0.42899889,0.42899883,0.42926657,0.42899736,0.42914876,0.429029,0.4289826,0.42903837,0.42904423,0.42900965,0.42897694,0.42899799,0.42882425,0.42880066,0.42901834,0.42895977,0.42904664,0.42900028,0.42886093,0.42899727,0.4288373,0.42898815,0.4289542,0.42899396,0.42797892,0.42894865,0.42893311,0.42919689,0.42901294,0.4294779,0.42908284,0.42900782,0.42899732,0.42897046,0.42884863,0.42922148,0.42879708,0.42890098,0.4294846,0.42882616,0.42911815,0.42907442,0.42988126,0.43098456,0.42918499,0.42967128,0.42892985,0.42881782,0.42881155,0.42912821,0.42904108,0.42953799,0.42904582,0.4289163,0.42900702,0.42851412,0.42932611,0.42935057,0.42900181,0.42893981,0.42901768,0.42894459,0.42899505,0.42863462,0.42925994,0.42940511,0.42892491,0.42918319,0.42900569,0.42921779,0.42900404,0.42900417,0.42908932,0.4289745,0.42911257,0.42899732,0.42898917,0.42899485,0.4290056,0.42898542,0.42900746,0.42899884,0.42890759,0.42896766,0.42902188,0.42899819,0.42899814,0.42899859,0.42900037,0.42899824,0.42901183,0.42903919,0.42900049,0.42899897,0.42899829,0.42908505,0.42899606,0.42899555,0.42914467,0.429113,0.42900545,0.42899738,0.42899182,0.42899877,0.42899879,0.42909516,0.42899171,0.42901614,0.42898688,0.42899877,0.42900251,0.42899632,0.4289985,0.42899933,0.42915779,0.42900372,0.42901686,0.42904814,0.42900615,0.42899852,0.42910691,0.42900027,0.42899831,0.4289963,0.42899468,0.42902452,0.42899012,0.42895158,0.42899386,0.42898651,0.42899845,0.42900674,0.42915122,0.42880634,0.42899979,0.42899865,0.42899867,0.42892286,0.42909672,0.42889992,0.42891941,0.42899775,0.42912615,0.42902961,0.42892357,0.4290125,0.42899636,0.42898516,0.4288876,0.42900733,0.42897381,0.42900587,0.42899842,0.42900304,0.42899876,0.42899939,0.42896917,0.42899891,0.42899989,0.42900084,0.42899153,0.42899973,0.42899905,0.42900126,0.42899734,0.42901466,0.42900137,0.42899914,0.42899875,0.42899879,0.42899881,0.42898057,0.42899885,0.42900201,0.42900029,0.42899894,0.42901475,0.42899952,0.42899887,0.42899856,0.42900293,0.42900548,0.42899774,0.42899847,0.42900459,0.42899958,0.42899818,0.42900312,0.42899894,0.42899885,0.42899843,0.42895536,0.42892458,0.42899791,0.42897882,0.42899894,0.42900078,0.42899728,0.42899836,0.42899839,0.42899942,0.42900063,0.42898883,0.42897967,0.42897328,0.42893142,0.42899839,0.42899852,0.42899902,0.42898442,0.42899793,0.42882188,0.42899486,0.42908447,0.42900022,0.42903741,0.42900056,0.42899918,0.42899241,0.42899905,0.42900417,0.42901474,0.42888132,0.42899431,0.42898966,0.42900476]


t_lim = 120
N = [20, 20]  # size of the horizon
measurement_lapse = 0.5  # time lapse between every measurement

t = 0.00
step = int(0)  # number of measurements measurements made
delta = int(0)
height = 80e3

# Initialisation of true dynamics and approximation model
d = dynamics(height, 22, 0, 6000, -5, 60)
o = SateliteObserver(40.24, 3.42)
initialbeta = d.beta[0] + np.random.normal(0, 0.01*d.beta[0], size=1)[0]
m = model(d.r, d.v, initialbeta, measurement_lapse)


# covariance matrices
R = np.array([[50**2, 0, 0], [0, (1e-3)**2, 0], [0, 0, (1e-3)**2]]) # Measurement covariance matrix
P0 = np.zeros((7,7))  # Initial covariance matrix
Q = np.zeros((7,7))  # Process noise covariance matrix
qa = 2 #  Estimated deviation of acceleration between real state and approximated state

for i in range(3):
    P0[i,i] = 500**2
    P0[i+3, i+3] = 1000**2

    Q[i, i] = (qa*measurement_lapse**3)/3
    Q[i, i+3] = (qa*measurement_lapse**2)/2
    Q[i+3, i] = (qa*measurement_lapse**2)/2
    Q[i+3, i+3] = qa*measurement_lapse
P0[6,6] = 20**2
Q[6,6] = 100


# Initialisation of estimators
opt = []
method = ['Newton LS','Newton LS','Newton LS']
measurement_pen =  [1e6, 5e1, 1e1]  # [1e7, 1, 1] #  [1e6, 1e-1, 1e-1] # [0.06, 80, 80] [1, 1e2, 1e3]  #
model_pen =  [1e4, 1e4, 1e4, 1e1, 1e1, 1e1, 1e1]  # [1e6, 1e6, 1e6, 1e1, 1e1, 1e1, 1e-1] #  [3, 3, 3, 1, 1, 1, 0.43] #[1, 1, 1, 1e1, 1e1, 1e1, 1e-1]  #
arrival = [1, 0]
for i in range(len(N)):
    opt.append(multishooting(m, d, o, N[i], measurement_lapse, measurement_pen, model_pen, Q, R, arrival[i], method[i]))
    # opt.append(MS_MHE_PE(m, d, o, N[i], measurement_lapse, measurement_pen, model_pen,[P0, Q, R], opt_method=method[i]))
memory = Memory(o, N, len(N))


# unscented kalman filter
points = MerweScaledSigmaPoints(n=7, alpha=.1, beta=2., kappa=-1)
ukf = UKF(dim_x=7, dim_z=3, fx=m.f, hx=o.h, dt=measurement_lapse, points=points)
ukf.P = P0
ukf.Q = Q
ukf.R = R


# extended kalman filter
ekf = ExtendedKalmanFilter(dim_x=7, dim_z=3, dim_u=0)
ekf.P = P0
ekf.Q = Q
ekf.R = R



time = [0]
y_real = []
y_model = []
real_x = []
y_minus1 = o.h(d.r, 'off')
penalties = []

coeffs = []
while height > 5000 and t < t_lim:
    d.step_update(d.v, d.r)
    delta = delta + 1

    # update stopping criteria
    t = t + d.delta_t
    height = d.h[len(d.h)-1]

    # measurements are only taken every 0.5 seconds (in the interest of time)
    if delta == measurement_lapse/d.delta_t:
        print('time: ', t-t_lim)
        step = step + 1
        delta = int(0)

        if step == 1:
            y_real = [o.h(d.r, 'off')]
            m.reinitialise(y_minus1, y_real[0], o, measurement_lapse)
            y_model = [o.h(m.r)]
            real_x = [[d.r, d.v]]
            real_beta = [d.beta[len(d.beta)-1]]

            # initialisation of the unscented kalman filter
            ukf.x = np.array([m.r[0], m.r[1], m.r[2], m.v[0], m.v[1], m.v[2], m.beta])
            ekf.x = np.array([m.r[0], m.r[1], m.r[2], m.v[0], m.v[1], m.v[2], m.beta])
            UKF_state = [np.copy(ukf.x)]
            EKF_state = [np.copy(ukf.x)]

        else:
            m.step_update()  # the model is updated every 0.5 seconds (problem with discretization)
            y_real.append(o.h(d.r))

            # re-initialise model from taken measurements
            y_model.append(o.h(m.r, 'off'))
            time.append(t)
            real_x.append([d.r, d.v])
            real_beta.append(d.beta[len(d.beta) - 1])

            ukf.predict()
            ekf.F = opt[0].dfdx(ekf.x)
            ekf.predict(fx=m.f)

            ukf.update(y_real[len(y_real)-1])
            ekf.update(z=y_real[len(y_real)-1], HJacobian=opt[0].dh, Hx=o.h)
            UKF_state.append(np.copy(ukf.x))
            EKF_state.append(np.copy(ekf.x))


        for i in range(len(opt)):
            if step >= opt[i].N+1: # MHE is entered only when there exists sufficient measurements over the horizon
                if step==opt[i].N+1:
                    opt[i].estimator_initilisation(step, y_real)
                    # asdfg = 0
                else:
                    opt[i].slide_window(y_real[step-1])
                    # asdfg = asdfg +1

                # coeffs.append(np.copy(opt[i].tuning_MHE(real_x, real_beta, step)))
                # measurementssss = np.array([coeff1[asdfg],coeff2[asdfg],coeff3[asdfg]])
                # modelss = np.array([coeff4[asdfg], coeff4[asdfg],coeff4[asdfg], coeff7[asdfg], coeff7[asdfg], coeff7[asdfg], coeff10[asdfg]])
                opt[i].estimation()
                memory.save_data(t, opt[i].vars, o.h(m.r, 'off'), opt[i].cost(opt[i].vars), i)

memory.make_plots(real_x, real_beta, y_real, m.Sk, UKF_state, EKF_state)
# fig, ax = plt.subplots(5,2)
# for i in range(10):
#     print([np.array(coeffs)[:,i]])
#     ax[i%5,i//5].plot(np.array(coeffs)[:,i])
# plt.show()
# ----------------------------------------------------------------------------------------------------- #
