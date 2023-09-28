from collections import OrderedDict

import numpy as np

from env.husky.husky import HuskyEnv
from env.transform_utils import up_vector_from_quat, Y_vector_from_quat, \
    l2_dist, cos_dist, sample_quat, X_vector_from_quat, alignment_heading_difference, \
    Y_vector_overlapping, movement_heading_difference, get_quaternion_to_next_cpt


# manually input checkpoints
trajectory = [(-17.5, 0), (-17.400008667240787, -0.001140178600320974), (-17.300069332517623, -0.004560121479413734), (-17.200233966826264, -0.010258050180003676), (-17.100554487095916, -0.018231001634255842), (-17.00108272919113, -0.028474829704641314), (-16.90187042095581, -0.040984207340040185), (-16.802969155313452, -0.05575262934593747), (-16.70443036343748, -0.07277241576729485), (-16.60630528800575, -0.09203471588231964), (-16.508644956553034, -0.11352951280506554), (-16.41150015493542, -0.1372456286944692), (-16.314921400920383, -0.16317073056711107), (-16.218958917916225, -0.19129133671068035), (-16.123662608854673, -0.22159282369480876), (-16.029082030240044, -0.254059433975625), (-15.935266366378627, -0.2886742840900813), (-15.848383708870138, -0.3229259781107567), (-15.848383708870138, -0.3229259781107567), (-15.755747893803315, -0.3605903613780476), (-15.663112078736491, -0.3982547446453385), (-15.570476263669669, -0.43591912791262943), (-15.477840448602846, -0.47358351117992037), (-15.385204633536024, -0.5112478944472113), (-15.2925688184692, -0.5489122777145021), (-15.199933003402377, -0.5865766609817931), (-15.107297188335554, -0.624241044249084), (-15.014661373268732, -0.6619054275163749), (-14.92202555820191, -0.6995698107836659), (-14.829389743135085, -0.7372341940509568), (-14.736753928068262, -0.7748985773182477), (-14.64411811300144, -0.8125629605855387), (-14.551482297934616, -0.8502273438528296), (-14.458846482867795, -0.8878917271201203), (-14.36621066780097, -0.9255561103874113), (-14.273574852734146, -0.9632204936547024), (-14.180939037667326, -1.0008848769219931), (-14.088303222600501, -1.038549260189284), (-13.995667407533679, -1.0762136434565752), (-13.903031592466856, -1.1138780267238662), (-13.810395777400032, -1.1515424099911569), (-13.717759962333211, -1.1892067932584478), (-13.625124147266387, -1.2268711765257387), (-13.532488332199563, -1.2645355597930297), (-13.439852517132742, -1.3021999430603206), (-13.347216702065918, -1.3398643263276115), (-13.254580886999095, -1.3775287095949025), (-13.161945071932273, -1.4151930928621934), (-13.06930925686545, -1.4528574761294841), (-12.976673441798628, -1.490521859396775), (-12.884037626731804, -1.528186242664066), (-12.79140181166498, -1.5658506259313572), (-12.698765996598159, -1.603515009198648), (-12.606130181531334, -1.6411793924659388), (-12.513494366464512, -1.6788437757332297), (-12.42085855139769, -1.7165081590005207), (-12.328222736330865, -1.7541725422678116), (-12.235586921264044, -1.7918369255351025), (-12.14295110619722, -1.8295013088023937), (-12.050315291130397, -1.8671656920696844), (-11.957679476063575, -1.9048300753369756), (-11.86504366099675, -1.9424944586042663), (-11.77240784592993, -1.980158841871557), (-11.679772030863106, -2.0178232251388475), (-11.587136215796283, -2.0554876084061386), (-11.49450040072946, -2.09315199167343), (-11.401864585662636, -2.1308163749407205), (-11.309228770595814, -2.1684807582080117), (-11.216592955528991, -2.206145141475303), (-11.123957140462167, -2.2438095247425935), (-11.031321325395345, -2.2814739080098847), (-10.938685510328522, -2.3191382912771754), (-10.8460496952617, -2.356802674544466), (-10.753413880194877, -2.3944670578117573), (-10.660778065128053, -2.432131441079048), (-10.56814225006123, -2.469795824346339), (-10.475506434994408, -2.50746020761363), (-10.382870619927584, -2.545124590880921), (-10.290234804860763, -2.5827889741482113), (-10.197598989793939, -2.6204533574155024), (-10.104963174727116, -2.658117740682793), (-10.012327359660294, -2.6957821239500843), (-9.91969154459347, -2.733446507217375), (-9.827055729526647, -2.771110890484666), (-9.734419914459824, -2.8087752737519573), (-9.641784099393, -2.846439657019248), (-9.549148284326177, -2.884104040286539), (-9.456512469259355, -2.9217684235538295), (-9.363876654192532, -2.9594328068211206), (-9.27124083912571, -2.9970971900884114), (-9.178605024058886, -3.0347615733557025), (-9.085969208992063, -3.0724259566229932), (-8.99333339392524, -3.1100903398902844), (-8.900697578858416, -3.1477547231575755), (-8.808061763791594, -3.1854191064248663), (-8.715425948724771, -3.223083489692157), (-8.622790133657949, -3.260747872959448), (-8.530154318591125, -3.2984122562267393), (-8.437518503524302, -3.3360766394940304), (-8.34488268845748, -3.373741022761321), (-8.252246873390657, -3.411405406028612), (-8.159611058323835, -3.4490697892959026), (-8.06697524325701, -3.486734172563194), (-7.97433942819019, -3.524398555830484), (-7.881703613123365, -3.5620629390977756), (-7.789067798056543, -3.5997273223650663), (-7.69643198298972, -3.637391705632357), (-7.603796167922898, -3.6750560888996477), (-7.511160352856075, -3.7127204721669385), (-7.418524537789251, -3.75038485543423), (-7.3258887227224285, -3.7880492387015208), (-7.233252907655604, -3.8257136219688124), (-7.140617092588782, -3.8633780052361026), (-7.047981277521959, -3.901042388503394), (-6.955345462455135, -3.9387067717706845), (-6.8627096473883125, -3.9763711550379757), (-6.77007383232149, -4.014035538305267), (-6.6774380172546675, -4.051699921572557), (-6.584802202187843, -4.089364304839849), (-6.492166387121021, -4.127028688107139), (-6.399530572054196, -4.164693071374431), (-6.306894756987376, -4.202357454641721), (-6.214258941920551, -4.240021837909013), (-6.121623126853729, -4.277686221176303), (-6.028987311786906, -4.315350604443594), (-5.936351496720084, -4.353014987710885), (-5.843715681653261, -4.390679370978176), (-5.751079866586437, -4.428343754245467), (-5.658444051519615, -4.466008137512758), (-5.565808236452792, -4.503672520780048), (-5.47317242138597, -4.5413369040473395), (-5.380536606319147, -4.57900128731463), (-5.287900791252323, -4.616665670581922), (-5.195264976185502, -4.654330053849211), (-5.102629161118678, -4.691994437116503), (-5.009993346051855, -4.7296588203837935), (-4.917357530985031, -4.7673232036510855), (-4.8247217159182085, -4.804987586918376), (-4.732085900851388, -4.842651970185666), (-4.6394500857845635, -4.880316353452957), (-4.546814270717739, -4.917980736720248), (-4.454178455650915, -4.9556451199875395), (-4.361542640584094, -4.99330950325483), (-4.26890682551727, -5.030973886522121), (-4.1762710104504475, -5.068638269789412), (-4.083635195383623, -5.106302653056703), (-3.9909993803168025, -5.1439670363239935), (-3.89836356524998, -5.181631419591285), (-3.8057277501831557, -5.219295802858576), (-3.713091935116333, -5.256960186125867), (-3.620456120049509, -5.294624569393158), (-3.527820304982688, -5.3322889526604484), (-3.435184489915864, -5.36995333592774), (-3.3425486748490414, -5.407617719195031), (-3.249912859782217, -5.445282102462322), (-3.1572770447153964, -5.482946485729612), (-3.064641229648574, -5.5206108689969025), (-2.9720054145817496, -5.5582752522641945), (-2.879369599514927, -5.595939635531485), (-2.786733784448103, -5.633604018798777), (-2.694097969381282, -5.671268402066066), (-2.601462154314458, -5.708932785333358), (-2.5088263392476353, -5.7465971686006485), (-2.416190524180813, -5.78426155186794), (-2.3235547091139903, -5.82192593513523), (-2.230918894047168, -5.859590318402521), (-2.1382830789803435, -5.897254701669812), (-2.045647263913521, -5.934919084937103), (-1.9530114488466968, -5.972583468204395), (-1.8603756337798742, -6.010247851471685), (-1.7677398187130517, -6.047912234738976), (-1.6751040036462275, -6.085576618006267), (-1.5824681885794067, -6.123241001273557), (-1.4898323735125807, -6.160905384540849), (-1.39719655844576, -6.19856976780814), (-1.3045607433789357, -6.236234151075431), (-1.2119249283121114, -6.273898534342722), (-1.1192891132452907, -6.311562917610012), (-1.0266532981784664, -6.349227300877304), (-0.9340174831116457, -6.386891684144594), (-0.8413816680448214, -6.424556067411886), (-0.7487458529780007, -6.462220450679175), (-0.6561100379111764, -6.499884833946467), (-0.5634742228443557, -6.5375492172137575), (-0.4708384077775314, -6.575213600481049), (-0.3782025927107071, -6.61287798374834), (-0.28556677764388283, -6.650542367015632), (-0.1929309625770621, -6.688206750282921), (-0.10029514751024138, -6.7258711335502115), (-0.0076593324434170995, -6.7635355168175035), (0.08497648262340718, -6.801199900084795), (0.17761229769023146, -6.838864283352086), (0.2702481127570522, -6.876528666619376), (0.36288392782387646, -6.914193049886667), (0.4555197428906972, -6.9518574331539575), (0.5481555579575215, -6.98952181642125), (0.6407913730243422, -7.027186199688539), (0.7334271880911665, -7.064850582955831), (0.8260630031579872, -7.10251496622312), (0.9186988182248115, -7.140179349490412), (1.0113346332916358, -7.177843732757704), (1.10397044835846, -7.215508116024995), (1.1966062634252808, -7.253172499292285), (1.289242078492105, -7.290836882559577), (1.3818778935589293, -7.328501265826868), (1.47451370862575, -7.366165649094157), (1.5671495236925743, -7.403830032361449), (1.659785338759395, -7.44149441562874), (1.7524211538262193, -7.479158798896031), (1.8450569688930436, -7.516823182163321), (1.9376927839598679, -7.5544875654306125), (2.030328599026692, -7.5921519486979046), (2.122964414093513, -7.629816331965195), (2.2156002291603336, -7.667480715232484), (2.308236044227158, -7.705145098499776), (2.400871859293982, -7.742809481767067), (2.493507674360803, -7.780473865034358), (2.586143489427627, -7.818138248301649), (2.6787793044944515, -7.855802631568941), (2.771415119561272, -7.89346701483623), (2.8640509346280965, -7.931131398103522), (2.9566867496949207, -7.9687957813708135), (3.049322564761745, -8.006460164638105), (3.141958379828562, -8.044124547905394), (3.2345941948953865, -8.081788931172685), (3.3272300099622107, -8.119453314439976), (3.419865825029035, -8.15711769770727), (3.5125016400958557, -8.194782080974559), (3.60513745516268, -8.23244646424185), (3.6977732702295043, -8.270110847509141), (3.790409085296325, -8.307775230776432), (3.8830449003631458, -8.345439614043721), (3.97568071542997, -8.383103997311013), (4.068316530496794, -8.420768380578304), (4.160952345563615, -8.458432763845595), (4.253588160630439, -8.496097147112886), (4.346223975697264, -8.533761530380177), (4.438859790764084, -8.571425913647467), (4.531495605830909, -8.60909029691476), (4.624131420897733, -8.64675468018205), (4.716767235964554, -8.68441906344934), (4.809403051031374, -8.72208344671663), (4.902038866098199, -8.759747829983922), (4.994674681165023, -8.797412213251214), (5.087310496231844, -8.835076596518503), (5.179946311298668, -8.872740979785794), (5.272582126365492, -8.910405363053087), (5.3652179414323165, -8.948069746320378), (5.457853756499134, -8.985734129587666), (5.550489571565958, -9.023398512854959), (5.643125386632782, -9.06106289612225), (5.735761201699603, -9.09872727938954), (5.828397016766427, -9.13639166265683), (5.9210328318332515, -9.174056045924122), (6.013668646900076, -9.211720429191415), (6.1063044619668965, -9.249384812458704), (6.198940277033721, -9.287049195725995), (6.2915760921005415, -9.324713578993286), (6.384211907167362, -9.362377962260576), (6.4768477222341865, -9.400042345527867), (6.569483537301011, -9.437706728795158), (6.662119352367835, -9.475371112062449), (6.754755167434659, -9.51303549532974), (6.847390982501484, -9.550699878597031), (6.940026797568308, -9.588364261864323), (7.032662612635129, -9.626028645131614), (7.125298427701949, -9.663693028398903), (7.217934242768774, -9.701357411666194), (7.310570057835598, -9.739021794933485), (7.403205872902419, -9.776686178200777), (7.495841687969243, -9.814350561468068), (7.588477503036067, -9.852014944735359), (7.681113318102891, -9.88967932800265), (7.773749133169712, -9.927343711269941), (7.866384948236533, -9.96500809453723), (7.959020763303357, -10.002672477804522), (8.051656578370178, -10.040336861071813), (8.144292393437002, -10.078001244339104), (8.236928208503826, -10.115665627606395), (8.32956402357065, -10.153330010873686), (8.422199838637471, -10.190994394140978), (8.514835653704296, -10.228658777408269), (8.60747146877112, -10.26632316067556), (8.700107283837937, -10.30398754394285), (8.792743098904761, -10.34165192721014), (8.885378913971586, -10.379316310477432), (8.97801472903841, -10.416980693744723), (9.07065054410523, -10.454645077012012), (9.163286359172055, -10.492309460279305), (9.25592217423888, -10.529973843546596), (9.348557989305704, -10.567638226813887), (9.44119380437252, -10.605302610081177), (9.533829619439345, -10.642966993348468), (9.62646543450617, -10.680631376615759), (9.71910124957299, -10.718295759883048), (9.811737064639814, -10.755960143150341), (9.904372879706639, -10.793624526417632), (9.997008694773463, -10.831288909684924), (10.089644509840284, -10.868953292952213), (10.182280324907108, -10.906617676219504), (10.274916139973932, -10.944282059486797), (10.36755195504075, -10.981946442754085), (10.460187770107574, -11.019610826021376), (10.552823585174398, -11.057275209288669), (10.645459400241222, -11.09493959255596), (10.738095215308043, -11.13260397582325), (10.830731030374867, -11.17026835909054), (10.923366845441691, -11.207932742357832), (11.016002660508512, -11.245597125625123), (11.108638475575333, -11.283261508892412), (11.201274290642157, -11.320925892159703), (11.293910105708981, -11.358590275426996), (11.386545920775802, -11.396254658694286), (11.479181735842626, -11.433919041961577), (11.57181755090945, -11.471583425228868), (11.664453365976275, -11.50924780849616), (11.757089181043096, -11.54691219176345), (11.84972499610992, -11.584576575030741), (11.942360811176744, -11.622240958298033), (12.034996626243565, -11.659905341565322), (12.12763244131039, -11.697569724832613), (12.220268256377214, -11.735234108099904), (12.312904071444034, -11.772898491367195), (12.405539886510859, -11.810562874634487), (12.498175701577683, -11.848227257901778), (12.550786240534965, -11.869617942506393), (12.550786240534965, -11.869617942506393), (12.6438434678706, -11.906222847571271), (12.737711185557263, -11.940696289675316), (12.832340579994487, -11.973020341751893), (12.927682441490507, -12.003178194471843), (13.023687189852577, -12.03115416498477), (13.12030490016992, -12.056933705074506), (13.217485328775954, -12.080503408724574), (13.315177939376273, -12.101851019089642), (13.413331929328812, -12.120965434869397), (13.511896256062506, -12.13783671608151), (13.610819663620727, -12.152456089230675), (13.710050709315691, -12.164815951871049), (13.80953779047995, -12.174909876559717), (13.90922917130111, -12.182732614199129), (14.009073009725753, -12.18828009676676), (14.109017384418635, -12.191549439430592), (14.209010321763092, -12.192538942049291), (14.308999822888644, -12.19124809005633), (14.408933890711737, -12.187677554727575), (14.508760556975545, -12.181829192832206), (14.608427909274774, -12.173706045667142), (14.70788411805146, -12.163312337475505), (14.807077463547628, -12.150653473249887), (14.905956362700891, -12.135736035921633), (15.004469395968975, -12.118567782937538), (15.102565334069162, -12.099157642225784), (15.20019316461881, -12.077515707553186), (15.29730211866314, -12.05365323327618), (15.393841697076333, -12.02758262848827), (15.489761696822384, -11.999317450566984), (15.585012237061953, -11.96887239812369), (15.679543785091667, -11.936263303359949), (15.773307182102386, -11.901507123834365), (15.866253668743042, -11.864621933644223), (15.95833491047673, -11.825626914026502), (16.04950302271591, -11.784542343383137), (16.139710595723585, -11.74138958673574), (16.228910719267603, -11.696191084615247), (16.31705700701515, -11.648970341392266), (16.40410362065483, -11.599751913054206), (16.490005293733752, -11.548561394435547), (16.574717355197222, -11.495425405907865), (16.65819575261885, -11.44037157953657), (16.74039707510893, -11.383428544711526), (16.821278575889195, -11.324625913259037), (16.900798194522217, -11.263994264042948), (16.978914578783943, -11.201565127062862), (17.055587106167877, -11.137370967057723), (17.1307759050098, -11.07144516662335), (17.204441875222102, -11.003822008852618), (17.276546708626753, -10.934536659507382), (17.347052908876606, -10.863625148731394), (17.415923810954368, -10.791124352313709), (17.48312360023943, -10.717071972512326), (17.5486173311323, -10.641506518448072), (17.612370945227255, -10.56446728607887), (17.674351289023562, -10.485994337764842), (17.73452613116614, -10.406128481434866), (17.79286417920669, -10.32491124936541), (17.84933509587654, -10.242384876582701), (17.903909514862846, -10.158592278899421), (17.95655905607977, -10.073577030597416), (18.0, -9.999999999999996)]

checkpoints = trajectory.copy()

checkpoints_backup = checkpoints.copy()


class HuskyTrajectoryEnv(HuskyEnv):
    def __init__(self, **kwargs):
        self.name = 'husky_channel'
        super().__init__('husky_channel.xml', **kwargs)

        # Env info
        self.ob_shape = OrderedDict([("husky_1", 30), ("husky_2", 30),
                                     ("box_1", 9), ("box_2", 9),
                                     ("goal_1", 3), ("goal_2", 3),        # we don't care the actual goal, huskys should only consider checkpoint
                                     ("relative_info_1", 2), ("relative_info_2", 2)])
        
        
        self.action_space.decompose(OrderedDict([("husky_1", 2), ("husky_2", 2)]))

        # Env config
        self._env_config.update({
            'random_husky_pos': 0.01,
            'random_goal_pos': 0.01,
            #'random_goal_pos': 0.5,
            'dist_threshold': 0.3,
            'loose_dist_threshold': 0.5,
            'goal_box_cos_dist_coeff_threshold': 0.95,

            'dist_reward': 10,
            'alignment_reward': 30,
            'goal_dist_reward': 30,
            'goal1_dist_reward': 10,
            'goal2_dist_reward': 10,
            'move_heading_reward': 10,
            'box_linear_vel_reward': 1000,
            'success_reward': 200,
            'bonus_reward': 20,

            'quat_reward': 200, # quat_dist usually between 0.95 ~ 1
            'alive_reward': 10,
            'die_penalty': 50,
            'sparse_reward': 0,
            'init_randomness': 0.01,
            #'max_episode_steps': 400,
            'max_episode_steps': 1000,
        }) 
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        self._husky1_push = False
        self._husky2_push = False
        self._success_count = 0

    def _step(self, a):
        husky1_pos_before = self._get_pos('husky_1_geom')
        husky2_pos_before = self._get_pos('husky_2_geom')

        box1_pos_before = self._get_pos('box_geom1')
        box2_pos_before = self._get_pos('box_geom2')
        box_pos_before = self._get_pos('box')
        box_quat_before = self._get_quat('box')

        self.do_simulation(a)

        husky1_pos = self._get_pos('husky_1_geom')
        husky2_pos = self._get_pos('husky_2_geom')
        box1_pos = self._get_pos('box_geom1')
        box2_pos = self._get_pos('box_geom2')
        box_pos = self._get_pos('box')
        box_quat = self._get_quat('box')

        # goal here are actually checkpoint
        goal1_pos = self._get_pos('cpt_1_geom1')
        goal2_pos = self._get_pos('cpt_1_geom2')
        goal_pos = self._get_pos('cpt_1')
        goal_quat = self._get_quat('cpt_1')

        ob = self._get_obs()
        done = False
        ctrl_reward = self._ctrl_reward(a)

        '''
        goal_forward = forward_vector_from_quat(goal_quat)
        box_forward = forward_vector_from_quat(box_quat)
        box_forward_before = forward_vector_from_quat(box_quat_before)
        '''

        #goal_forward_before = right_vector_from_quat(goal_quat_before)    
        goal_forward = X_vector_from_quat(goal_quat)
        box_forward = X_vector_from_quat(box_quat)
        box_forward_before = X_vector_from_quat(box_quat_before)

        # goal 1
        goal1_dist = l2_dist(goal1_pos, box1_pos)
        goal1_dist_before = l2_dist(goal1_pos, box1_pos_before)

        # goal 2
        goal2_dist = l2_dist(goal2_pos, box2_pos)
        goal2_dist_before = l2_dist(goal2_pos, box2_pos_before)

        husky1_quat = self._get_quat('husky_robot_1')
        husky2_quat = self._get_quat('husky_robot_2')


        '''
            Reward & Penalty
            PART 1, 2, 3: control the formation of huskys
            PART 4, 5: distance between husky and box, box and goal
            PART 6: move heading of huskys
        ''' 
        # PART 1: Forward parallel between two Huskys (checking forward vector)
        husky1_forward_vec = X_vector_from_quat(husky1_quat)
        husky2_forward_vec = X_vector_from_quat(husky2_quat)
        huskys_forward_align_coeff, dir = alignment_heading_difference(husky1_forward_vec, husky2_forward_vec)
        huskys_forward_align_reward = huskys_forward_align_coeff * self._env_config["alignment_reward"]

        # PART 2: Right vector overlapping between two Huskys (checking if two vectors are on the same line and same direction)
        # Actually, if Part 2 is gauranteed, then Part 1 is gauranteedgoal_box_cos_dist
        husky1_right_vec = Y_vector_from_quat(husky1_quat)
        husky2_right_vec = Y_vector_from_quat(husky2_quat)
        huskys_right_align_coeff = Y_vector_overlapping(husky1_right_vec, husky2_right_vec, husky1_pos, husky2_pos)
        huskys_right_align_reward = huskys_right_align_coeff * self._env_config["alignment_reward"]
        
        # PART 3: Distance between two Huskys (to avoid Collision)
        suggested_dist = l2_dist(box1_pos, box2_pos)
        huskys_dist = l2_dist(husky1_pos, husky2_pos)
        huskys_dist_reward = -abs(suggested_dist - huskys_dist) * self._env_config["dist_reward"]

        # PART 4: Linear distance between one husky and one box
        husky1_box_dist = l2_dist(husky1_pos, box1_pos)
        husky2_box_dist = l2_dist(husky2_pos, box2_pos)
        #husky1_box_dist_reward = -husky1_box_dist * self._env_config["dist_reward"]
        #husky2_box_dist_reward = -husky2_box_dist * self._env_config["dist_reward"]
        husky1_box_dist_reward = (5 - husky1_box_dist) * self._env_config["dist_reward"]
        husky2_box_dist_reward = (5 - husky2_box_dist) * self._env_config["dist_reward"]
        huskys_box_dist_reward = husky1_box_dist_reward + husky2_box_dist_reward
        
        # PART 5: Linear distance between box and goal
        goal1_box_dist = l2_dist(goal1_pos, box1_pos)
        goal2_box_dist = l2_dist(goal2_pos, box2_pos)
        #goal1_box_dist_reward = -goal1_box_dist * self._env_config["dist_reward"]
        #goal2_box_dist_reward = -goal2_box_dist * self._env_config["dist_reward"]
        goal1_box_dist_reward = (5 - goal1_box_dist) * self._env_config["goal_dist_reward"]
        goal2_box_dist_reward = (5 - goal2_box_dist) * self._env_config["goal_dist_reward"]
        goal_box_dist_reward = goal1_box_dist_reward + goal2_box_dist_reward

        # PART 6: Movement heading of husky to box
        husky1_move_coeff = movement_heading_difference(box1_pos, husky1_pos, husky1_forward_vec,)
        husky2_move_coeff = movement_heading_difference(box2_pos, husky2_pos, husky2_forward_vec)
        husky1_move_heading_reward = husky1_move_coeff * self._env_config["move_heading_reward"]
        husky2_move_heading_reward = husky2_move_coeff * self._env_config["move_heading_reward"]
        huskys_move_heading_reward = husky1_move_heading_reward + husky2_move_heading_reward

        # PART 7: Box velocity
        box1_linear_vel =  l2_dist(box1_pos, box1_pos_before)
        box2_linear_vel =  l2_dist(box2_pos, box2_pos_before)
        box1_linear_vel_reward = box1_linear_vel * self._env_config["box_linear_vel_reward"]
        box2_linear_vel_reward = box2_linear_vel * self._env_config["box_linear_vel_reward"]
        box_linear_vel_reward = box1_linear_vel_reward + box2_linear_vel_reward

        # PART 8: Cos distance between box and goal
        goal_box_cos_dist_coeff = 1 - cos_dist(goal_forward, box_forward) 
        goal_box_cos_dist_coeff = abs(goal_box_cos_dist_coeff - 0.5) / 0.5      # the larger goal_box_cos_dist_coeff, the better
        goal_box_cos_dist_reward = goal_box_cos_dist_coeff * self._env_config["quat_reward"]

        # PART 9 (Not sure side-effect): huskys velocity control   
        # We want huskys can stop/slow down when the box is close to the goal
        husky1_linear_vel = l2_dist(husky1_pos, husky1_pos_before)
        husky2_linear_vel = l2_dist(husky2_pos, husky2_pos_before)
        '''
        huskys_linear_vel_reward = 0 
        if (goal1_box_dist < 0.3) and husky1_linear_vel <= 0.005:
            huskys_linear_vel_reward += 1000
        if (goal2_box_dist < 0.3) and husky2_linear_vel <= 0.005:
            huskys_linear_vel_reward += 1000
        '''

        # PART 10: Linear distance between goal and huskys
        goal1_husky1_dist = l2_dist(goal1_pos, husky1_pos)
        goal2_husky2_dist = l2_dist(goal2_pos, husky2_pos)
        goal1_husky1_dist_reward = (1 - goal1_husky1_dist) * self._env_config["goal_dist_reward"]
        goal2_husky2_dist_reward = (1 - goal2_husky2_dist) * self._env_config["goal_dist_reward"]
        goal_huskys_dist_reward = goal1_husky1_dist_reward + goal2_husky2_dist_reward


        # Note: goal_quat is the cos_dist between goal and box 
        # NOTE: why "-"? doesn't make sense
        '''
        quat_reward = -self._env_config["quat_reward"] * (1 - goal_quat)
        '''
        quat_reward = self._env_config[("quat_reward")] * (1 - goal_quat)

        reward = 0
        alive_reward = self._env_config["alive_reward"] 
        reward += alive_reward


        '''
            Bonus
        '''
        if husky1_box_dist < 1.5 and husky2_box_dist < 1.5:
            reward += self._env_config['bonus_reward']
        if husky1_move_coeff > 0.92 and husky2_move_coeff > 0.92:
            reward += self._env_config['bonus_reward']
        if huskys_right_align_coeff > 0.92:
            reward += self._env_config['bonus_reward']
        if goal_box_cos_dist_coeff > 0.9:
            reward += (3 * self._env_config['bonus_reward'])

        '''
            Failure Check
        '''
        if huskys_dist < 1 or huskys_dist > 3.0:   # huskys are too close or too far away 
            done = True
        if husky1_box_dist > 6.0 or husky2_box_dist > 6.0: # husky is too far away from box
            done = True
        die_penalty = -self._env_config["die_penalty"] if done else 0

        # give some bonus if pass all failure check
        if done != True:
            reward += self._env_config["alive_reward"]


        '''
            Success update: success pass a checkpoint, update next one
        '''
        success_reward = 0
        if (goal1_dist <= self._env_config["dist_threshold"] and \
            goal2_dist <= self._env_config["dist_threshold"]) or \
            (goal1_dist <= self._env_config["loose_dist_threshold"] and \
             goal2_dist <= self._env_config["loose_dist_threshold"] and \
             goal_box_cos_dist_coeff >= self._env_config["goal_box_cos_dist_coeff_threshold"]):
                # if both goal1 and goal2 suffice, then overall goal should suffice
                #and goal_quat < self._env_config["quat_threshold"]:
            self._success_count += 1
            success_reward = self._env_config["success_reward"]
            reward += success_reward
            if self._success_count >= 1:
                self._success = True
                success_reward = self._env_config["success_reward"]
                reward += success_reward

        # update the checkpoint position
        final_pos = self._get_pos('goal')       # the actual final goal!
        goal_cpt_dist = l2_dist(final_pos, goal_pos)
        goal_box_dist = l2_dist(final_pos, box_pos)

        cpt_box_dist = l2_dist(goal_pos, box_pos)
        cpt_pos = goal_pos

        rx, ry, nrx, nry = 0, 0, 0, 0

        print(len(checkpoints))
        if(len(checkpoints) > 1 and (cpt_box_dist < 1.2 or goal_cpt_dist > goal_box_dist)):
            (rx, ry) = checkpoints.pop(0)
            (nrx, nry) = checkpoints[0]

            #if(cpt_box_dist <= self._env_config['dist_threshold'] or
                #goal_cpt_dist > goal_box_dist):

            while(cpt_box_dist < 1.2):
                if(len(checkpoints) > 1):
                    (rx,ry) = checkpoints.pop(0)
                    (nrx, nry) = checkpoints[0]
                    cpt_pos = np.array([rx, ry, 0.3])
                    box_pos = self._get_pos('box')

                    cpt_box_dist = l2_dist(cpt_pos, box_pos)
                    goal_cpt_dist = l2_dist(final_pos, np.array([rx, ry, 0.3]))

                    pos_1 = np.array([rx, ry, 0])
                    pos_2 = np.array([nrx, nry, 0])

                    quaternion = get_quaternion_to_next_cpt(box_pos, pos_2)
                    self._set_pos('cpt_1', np.array([rx, ry, 0.3]))     # update cpt pos
                    self._set_quat('cpt_1', quaternion)   
                            
                else:
                    break

            pos_1 = np.array([rx, ry, 0])
            pos_2 = np.array([nrx, nry, 0])
            quaternion = get_quaternion_to_next_cpt(pos_1, pos_2)
            self._set_pos('cpt_1', np.array([rx, ry, 0.3]))     # update cpt pos
            self._set_quat('cpt_1', quaternion)                 # update cpt quat

        

        
        '''
            Fail but pass update: the box missed the checkpoint, update next checkpoint
        ''' 
        box_x = box_pos[0]
        box_y = box_pos[1]
        cpt_x = goal_pos[0]
        cpt_y = goal_pos[1]
        


        '''
        # if distance between box and checkpoints is too large and cpt is behine the box, update checkpoint
        if (abs(cpt_box_dist) > 0.2) and (goal_box_dist < goal_cpt_dist):
               if len(checkpoints) > 1:
                (rx, ry) = checkpoints.pop(0)
                (nrx, nry) = checkpoints[0]
                pos_1 = np.array([rx, ry, 0])
                pos_2 = np.array([nrx, nry, 0])
                quaternion = get_quaternion_to_next_cpt(pos_1, pos_2)
                self._set_pos('cpt_1', np.array([rx, ry, 0.3]))     # update cpt pos
                self._set_quat('cpt_1', quaternion)                 # update cpt quat
        '''


        if self._env_config['sparse_reward']:
            self._reward = reward = self._success == 1
        else:

            reward = reward \
                    + huskys_forward_align_reward \
                    + huskys_dist_reward \
                    + huskys_box_dist_reward \
                    + goal_box_dist_reward \
                    + goal_huskys_dist_reward \
                    + huskys_move_heading_reward \
                    + goal_box_cos_dist_reward \
                    #+ huskys_right_align_reward \
                    #+ box_linear_vel_reward
            self._reward = reward


        info = {"success": self._success,
                "Total reward": reward,
                "checkpoint_pos": goal_pos,
                "husky1_pos": husky1_pos,
                "husky2_pos": husky2_pos,
                "huskys_dist": huskys_dist,
                "husky1_linear_vel": husky1_linear_vel,
                "husky2_lienar_vel": husky2_linear_vel,
                "box1_ob": ob['box_1'],
                "box2_ob": ob['box_2'],
                "goal1_ob": ob['goal_1'],
                "goal2_ob": ob['goal_2'],
                }

        return ob, reward, done, info

    def _get_obs(self):
        # husky
        qpos = self.data.qpos
        qvel = self.data.qvel
        qacc = self.data.qacc
        husky_pos1 = self._get_pos('husky_robot_1')
        husky_pos2 = self._get_pos('husky_robot_2')

        # box
        box_pos1 = self._get_pos('box_geom1')
        box_pos2 = self._get_pos('box_geom2')

        #box_quat1 = self._get_quat('box')
        #box_quat2 = self._get_quat('box')

        box_forward1 = self._get_right_vector('box_geom1')
        box_forward2 = self._get_right_vector('box_geom2')

        # goal
        goal_pos1 = self._get_pos('cpt_1_geom1')
        goal_pos2 = self._get_pos('cpt_1_geom2')


        husky1_forward_vec = X_vector_from_quat(self._get_quat("husky_robot_1"))
        husky2_forward_vec = X_vector_from_quat(self._get_quat("husky_robot_2"))
        husky1_move_coeff = movement_heading_difference(box_pos1, husky_pos1, husky1_forward_vec, "forward")
        husky2_move_coeff = movement_heading_difference(box_pos2, husky_pos2, husky2_forward_vec, "forward")
        husky1_align_coeff, direction1 = alignment_heading_difference(box_forward1, husky1_forward_vec)
        husky2_align_coeff, direction2 = alignment_heading_difference(box_forward2, husky2_forward_vec)



        obs = OrderedDict([
            #('ant_1_shared_pos', np.concatenate([qpos[:7], qvel[:6], qacc[:6]])),
            #('ant_1_lower_body', np.concatenate([qpos[7:15], qvel[6:14], qacc[6:14]])),
            #('ant_2_shared_pos', np.concatenate([qpos[15:22], qvel[14:22], qacc[14:22]])),
            #('ant_2_lower_body', np.concatenate([qpos[22:30], qvel[22:30], qacc[22:30]])),
            ('husky_1', np.concatenate([qpos[4:11], qvel[:10], qacc[:10], husky1_forward_vec])),
            ('husky_2', np.concatenate([qpos[15:22], qvel[10:20], qacc[10:20], husky2_forward_vec])),
            ('box_1', np.concatenate([box_pos1 - husky_pos1, goal_pos1 - box_pos1, box_forward1])),
            ('box_2', np.concatenate([box_pos2 - husky_pos2, goal_pos2 - box_pos1, box_forward2])),
            ('goal_1', goal_pos1 - box_pos1),
            ('goal_2', goal_pos2 - box_pos2),
            ('relative_info_1', [husky1_move_coeff, husky1_align_coeff]),
            ('relative_info_2', [husky2_move_coeff, husky2_align_coeff])
        ])

        def ravel(x):
            obs[x] = obs[x].ravel()
        map(ravel, obs.keys())

        return obs

    @property
    def _init_qpos(self):
        # 3 for (x, y, z), 4 for (x, y, z, w), and 2 for each leg
        # If I am correct, one line is for ant1, one line for ant2, the last line is for the box
        '''
        return np.array([0, 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         0, 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         0., 0., 0.8, 1., 0., 0., 0.])
        '''
    
        '''
            The structure of qpose of a single Husky and Box can be found in husky_forward.py _rest()
            
            The first and second 11 elements are for Huskys
            
            The last 7 are for box 
        '''
        return np.array([-2., 1, 0.58, 0., 0., 0., 0., 0., 0, 0., 0.,
                         -2., -1, 0.58, 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0.58, 0., 0., 0., 0.])

    @property
    def _init_qvel(self):
        return np.zeros(self.model.nv)

    def _reset(self):
        '''
        qpos = self._init_qpos + self._init_random(self.model.nq)
        qvel = self._init_qvel + self._init_random(self.model.nv)
        qpos[2] = 0.2
        qpos[13] = 0.2
        qpos[3:7] = [0, 0, 0, 0]
        qpos[14:18] = [0, 0, 0, 0]

        self.set_state(qpos, qvel)
        '''

        self._reset_husky_box()

        self._husky1_push = False
        self._husky2_push = False
        self._success_count = 0

        return self._get_obs()
    


    def _reset_husky_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # Initialize box
        init_box_pos = np.asarray([-18, 0, 0.3])
        #init_box_quat = sample_quat(low=-np.pi/32, high=np.pi/32)
        init_box_quat = sample_quat(low=0, high=0)
        #qpos[22:25] = init_box_pos
        #qpos[25:29] = init_box_quat
        self._set_pos('box', init_box_pos)
        self._set_quat('box', init_box_quat)

        # Initialize husky
        #qpos[0:2] = [-4, 2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        #qpos[15:17] = [-4, -2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        '''
        qpos[0:2] = [-2, 1] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_husky_pos"]
        qpos[11:13] = [-2, -1] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_husky_pos"]
        '''
        husky1_pos = [-20, 1, 0.19]
        husky2_pos = [-20, 1, 0.19] 
        self._set_pos('husky_robot_1', husky1_pos) 
        self._set_pos('husky_robot_2', husky2_pos) 

        #qpos[0:2] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], 2]
        #qpos[15:17] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], -2]


        # Initialize goal
        #x = 4.5 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        '''
        x = 3 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        y = 0 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        z = 0.3
        '''
        x = 18
        y = 0
        z = 0.3
        goal_pos = np.array([x, y, z])
        #goal_quat = sample_quat(low=-np.pi/9, high=np.pi/9)

        self._set_pos('goal', goal_pos)
        self._set_quat('goal', np.array([1, 0, 0, 0]))
        
        # Initialized checkpoint
        self._set_pos('cpt_1', np.array([-17.5, 0, 0.3]))
        self._set_quat('cpt_1', np.array([1, 0, 0, 0]))

        #self.set_state(qpos, qvel)

        # reset all checkpoints
        global checkpoints
        checkpoints = checkpoints_backup.copy()

    def _render_callback(self):

        box_pos = self._get_pos('cpt_1') 
        lookat = box_pos

        #lookat = [0.5, 0, 0]
        #cam_pos = lookat + np.array([-4.5, -12, 20])
        cam_pos = lookat + np.array([-4.5, -12, 20])

        cam_id = self._camera_id
        self._set_camera_position(cam_id, cam_pos)

        #self._set_camera_rotation(cam_id, lookat)
        self._set_camera_rotation(cam_id, lookat)

        self.sim.forward()

