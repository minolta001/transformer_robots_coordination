from collections import OrderedDict

import numpy as np

from env.husky.husky import HuskyEnv
from env.transform_utils import up_vector_from_quat, Y_vector_from_quat, \
    l2_dist, cos_dist, sample_quat, X_vector_from_quat, alignment_heading_difference, \
    Y_vector_overlapping, movement_heading_difference


# manually input checkpoints
checkpoints = [(-18, 0), (-17.9, 0.0), (-17.800004740698924, -0.0009737131562962145), (-17.70002370304514, -0.0029210471472703803), (-17.6000663666386, -0.00584181733843935), (-17.50014220883205, -0.009735746799961024), (-17.400260703832437, -0.014602466332891167), (-17.30043132180263, -0.020441514504188636), (-17.20066352796351, -0.027252337690465672), (-17.100966781696535, -0.035034290130479144), (-17.001350535646864, -0.04378663398635774), (-16.901824234827107, -0.053508539413559306), (-16.802397315721812, -0.06419908463955172), (-16.703079205392747, -0.0758572560512098), (-16.60387932058509, -0.08848194829092002), (-16.504807066834584, -0.10207196436138381), (-16.405871837575766, -0.11662601573910972), (-16.30708301325134, -0.13214272249658335), (-16.208449960422765, -0.14862061343310384), (-16.109982030882197, -0.1660581262142743), (-16.011688560765787, -0.18445360752013282), (-15.913578869668497, -0.20380531320191042), (-15.815662259760467, -0.22411140844740074), (-15.717948014905048, -0.2453699679549258), (-15.620445399778546, -0.26757897611588155), (-15.523163658991818, -0.2907363272058458), (-15.426112016213741, -0.3148398255842303), (-15.329299673296688, -0.33988718590245803), (-15.23273580940406, -0.3658760333206464), (-15.13642958013997, -0.3928039037327751), (-15.040390116681168, -0.42066824400031794), (-14.944626524911278, -0.44946641219431593), (-14.84914788455743, -0.47919567784586925), (-14.753963248329379, -0.5098532222050236), (-14.659081641061176, -0.5414361385080273), (-14.564512058855493, -0.5739414322529333), (-14.470263468230664, -0.6073660214835191), (-14.376344805270534, -0.6417067370815005), (-14.282764974777193, -0.676960323067007), (-14.189532849426676, -0.7131234369072952), (-14.096657268927714, -0.7501926498336668), (-14.004147039183596, -0.7881644471665648), (-13.912010931457258, -0.8270352286488131), (-13.820257681539637, -0.8668013087869721), (-13.728895988921396, -0.9074589172007749), (-13.637934515968096, -0.9490041989806114), (-13.547381887098878, -0.9914332150530283), (-13.457246687968745, -1.0347419425542081), (-13.367537464654532, -1.0789262752113928), (-13.278262722844609, -1.123982023732216), (-13.189430927032426, -1.1699049162019062), (-13.101050499713962, -1.216690598488324), (-13.013129820589151, -1.2643346346547957), (-12.925677225767375, -1.3128325073807001), (-12.838701006977075, -1.3621796183897759), (-12.752209410779592, -1.4123712888861006), (-12.666210637787268, -1.4634027599977073), (-12.580712841885916, -1.515269193227791), (-12.495724129461717, -1.5679656709134673), (-12.411252558632626, -1.6214871966920335), (-12.32730613848434, -1.6758286959746955), (-12.243892828310937, -1.7309850164277087), (-12.161020536860212, -1.786950928460892), (-12.078697121583824, -1.843721125723465), (-11.9969303878923, -1.9012902256071653), (-11.915728088414971, -1.9596527697565944), (-11.835097922264913, -2.018803224586746), (-11.755047534308963, -2.0787359818076685), (-11.675584514442885, -2.139445358956207), (-11.596716396871738, -2.2009255999347817), (-11.518450659395523, -2.263170875557147), (-11.440794722700192, -2.3261752841010783), (-11.363755949654054, -2.389932851867939), (-11.287341644609683, -2.454437533749069), (-11.211559052711353, -2.519683213798948), (-11.136415359208103, -2.5856637058150707), (-11.061917688772473, -2.6523727539244857), (-10.988073104824986, -2.71980403317694), (-10.914888608864436, -2.787951150144573), (-10.84237113980405, -2.8568076435281013), (-10.77052757331358, -2.9263669847694422), (-10.699364721167393, -2.996622578670711), (-10.628889330598625, -3.067567764019538), (-10.559108083659439, -3.139195814220645), (-10.490027596587483, -3.211499937933621), (-10.421654419178573, -3.2844732797168343), (-10.353995034165681, -3.358108920677427), (-10.287055856604283, -3.4323998791273196), (-10.220843233264121, -3.5073391112451757), (-10.155363442027442, -3.5829195117442527), (-10.090622691293762, -3.6591339145460826), (-10.02662711939123, -3.735975093459917), (-9.963382793994626, -3.813435762867871), (-9.900895711550055, -3.8915085784157), (-9.83917179670641, -3.9701861377091507), (-9.778216901753625, -4.049460981015807), (-9.7180368060678, -4.129325591972381), (-9.658637215563234, -4.209772398297367), (-9.600023762151421, -4.290793772508999), (-9.542202003207075, -4.372382032648446), (-9.485177421041206, -4.454529443008164), (-9.428955422381318, -4.537228214865355), (-9.373541337858786, -4.620470507220441), (-9.318940421503427, -4.704248427540505), (-9.26515785024535, -4.788554032507612), (-9.21219872342411, -4.873379328771946), (-9.16006806230522, -4.958716273709689), (-9.10877080960406, -5.044556776185579), (-9.058311829017246, -5.1308926973200615), (-9.008695904761478, -5.217715851260966), (-8.959927741119932, -5.305018005959641), (-8.912011961996226, -5.392790883951469), (-8.864953110476014, -5.481026163140683), (-8.818755648396229, -5.569715477589419), (-8.773423955922048, -5.658850418310925), (-8.728962331131587, -5.7484225340668464), (-8.685374989608382, -5.838423332168528), (-8.642666064041697, -5.928844279282233), (-8.60083960383468, -6.019676802238225), (-8.559899574720431, -6.110912288843626), (-8.51984985838599, -6.202542088698973), (-8.4806942521043, -6.29455751401839), (-8.442436468374172, -6.386949840453321), (-8.405080134568292, -6.479710307919714), (-8.36862879258929, -6.5728301214286), (-8.333085898533918, -6.666300451919985), (-8.298454822365372, -6.760112437099962), (-8.264738847593762, -6.854257182280987), (-8.231941170964793, -6.948725761225214), (-8.200064902156674, -7.04350921699083), (-8.16911306348527, -7.138598562781299), (-8.139088589617547, -7.2339847827974335), (-8.109994327293323, -7.32965883309222), (-8.081833035055364, -7.425611642428311), (-8.054607382987825, -7.521834113138106), (-8.028319952463095, -7.618317121986335), (-8.002973235897047, -7.715051521035073), (-7.978569636512719, -7.812028138511088), (-7.95511146811246, -7.909237779675455), (-7.932600954858545, -8.006671227695348), (-7.911040231062292, -8.10431924451792), (-7.890431340981704, -8.202172571746209), (-7.870776238627641, -8.300221931516946), (-7.852076787578557, -8.398458027380244), (-7.8343347608038005, -8.49687154518102), (-7.817551840495517, -8.595453153942117), (-7.80172961790915, -8.694193506748999), (-7.786869593212574, -8.793083241635978), (-7.77297317534385, -8.892112982473858), (-7.7600416818776425, -8.991273339858921), (-7.748076338900296, -9.090554912003174), (-7.735945972145568, -9.189814273458909), (-7.721555389537694, -9.2887712225146), (-7.704912074545615, -9.387374299040026), (-7.6860246821164075, -9.485572226927683), (-7.664903034174485, -9.583313940757666), (-7.641558114513951, -9.680548612352975), (-7.616002063086746, -9.777225677211474), (-7.588248169689563, -9.87329486080071), (-7.558310867052817, -9.968706204701949), (-7.526205723335254, -10.063410092589825), (-7.491949434028118, -10.157357276034089), (-7.455559813273068, -10.250498900110053), (-7.417055784598371, -10.342786528804384), (-7.376457371078185, -10.434172170203066), (-7.333785684920058, -10.524608301448419), (-7.289062916486032, -10.61404789345219), (-7.2423123227531, -10.702444435351875), (-7.19355821521898, -10.789751958697554), (-7.142825947259525, -10.875925061356646), (-7.090141900944315, -10.960918931124185), (-7.035533473317313, -11.044689369026294), (-6.97902906214971, -11.127192812304783), (-6.920658051172351, -11.20838635707089), (-6.860450794795461, -11.28822778061639), (-6.798438602323567, -11.366675563370482), (-6.734653721673873, -11.443688910491023), (-6.669129322606518, -11.519227773078875), (-6.601899479475458, -11.593252869004363), (-6.532999153508935, -11.665725703334967), (-6.462464174628751, -11.736608588353684), (-6.390331222817796, -11.805864663157582), (-6.316637809045524, -11.873457912826414), (-6.241422255761294, -11.939353187151298), (-6.164723676965726, -12.003516218913713), (-6.086581957870419, -12.06591364170532), (-6.007037734156633, -12.126513007279357), (-5.926132370843698, -12.185282802424538), (-5.843907940778152, -12.24219246535273), (-5.76040720275479, -12.297212401591866), (-5.675673579281008, -12.35031399937581), (-5.58975113399599, -12.40146964452322), (-5.502684548756495, -12.450652734797604), (-5.414519100401158, -12.49783769374118), (-5.32530063720538, -12.542999983975271), (-5.235075555039062, -12.586116119960371), (-5.143890773239567, -12.627163680209222), (-5.051793710212477, -12.666121318946558), (-4.958832258772812, -12.702968777209449), (-4.865054761239554, -12.73768689338248), (-4.770509984296401, -12.77025761316228), (-4.6752470936318655, -12.800663998946206), (-4.579315628371851, -12.82889023864035), (-4.482765475318061, -12.85492165388221), (-4.38564684300558, -12.878744707673818), (-4.288010235593171, -12.900347011421324), (-4.189906426599817, -12.919717331377385), (-4.091386432501209, -12.936845594482989), (-3.9925014861998744, -12.951722893605707), (-3.8933030103827706, -12.964341492171632), (-3.793842590780175, -12.97469482818859), (-3.694171949339795, -12.982777517658537), (-3.5943429173300396, -12.988585357377392), (-3.4944074083864454, -12.992115327120793), (-3.394417391515259, -12.993365591214697), (-3.2944248640682385, -12.992335499489979), (-3.194481824702703, -12.989025587620532), (-3.0946402463409113, -12.983437576844707), (-2.99495204914282, -12.975574373070224), (-2.8954690735062805, -12.965440065363028), (-2.7962430531087117, -12.953039923820864), (-2.697325588004273, -12.938380396832699), (-2.5987681177905246, -12.921469107725393), (-2.500621894858531, -12.90231485079938), (-2.4029379577403045, -12.880927586755394), (-2.305767104567485, -12.857318437514671), (-2.209159866655006, -12.83149968043525), (-2.113166482223532, -12.803484741927448), (-2.0178368702743104, -12.773288190471774), (-1.9232206046300107, -12.740925729042953), (-1.8293668881550866, -12.706414186943983), (-1.7363245271690246, -12.669771511054467), (-1.644141906065828, -12.631016756497779), (-1.552866962152894, -12.590170076731912), (-1.462547160722396, -12.547252713069174), (-1.3732294703681243, -12.502286983630157), (-1.284960338560623, -12.455296271737758), (-1.197785667493327, -12.406305013757244), (-1.1117507902122492, -12.35533868638873), (-1.0269004470416476, -12.302423793418644), (-0.9432787623179193, -12.24758785193708), (-0.8814239725924349, -12.205237534785361), (-0.8814239725924349, -12.205237534785361), (-0.7993969469674695, -12.14803992061203), (-0.7173699213425024, -12.090842306438699), (-0.6353428957175371, -12.03364469226537), (-0.5533158700925709, -11.976447078092038), (-0.47128884446760555, -11.919249463918709), (-0.38926181884263844, -11.862051849745377), (-0.3072347932176731, -11.804854235572048), (-0.2252077675927069, -11.747656621398718), (-0.14318074196774155, -11.690459007225387), (-0.061153716342774445, -11.633261393052056), (0.02087330928219089, -11.576063778878726), (0.10290033490715711, -11.518866164705395), (0.18492736053212244, -11.461668550532064), (0.26695438615708866, -11.404470936358734), (0.3489814117820558, -11.347273322185405), (0.4310084374070211, -11.290075708012074), (0.5130354630319882, -11.232878093838742), (0.5950624886569535, -11.175680479665413), (0.6770895142819189, -11.118482865492084), (0.7591165399068842, -11.061285251318752), (0.8411435655318513, -11.004087637145421), (0.9231705911568167, -10.946890022972092), (1.0051976167817838, -10.88969240879876), (1.087224642406749, -10.83249479462543), (1.1692516680317144, -10.7752971804521), (1.2512786936566815, -10.71809956627877), (1.3333057192816469, -10.660901952105439), (1.4153327449066122, -10.60370433793211), (1.4973597705315793, -10.546506723758778), (1.5793867961565446, -10.489309109585449), (1.6614138217815118, -10.432111495412117), (1.743440847406477, -10.374913881238788), (1.8254678730314442, -10.317716267065457), (1.9074948986564113, -10.260518652892126), (1.9895219242813766, -10.203321038718796), (2.071548949906342, -10.146123424545465), (2.153575975531309, -10.088925810372135), (2.2356030011562726, -10.031728196198804), (2.3176300267812415, -9.974530582025475), (2.399657052406207, -9.917332967852143), (2.481684078031172, -9.860135353678814), (2.5637111036561393, -9.802937739505483), (2.6457381292811046, -9.745740125332153), (2.72776515490607, -9.688542511158822), (2.8097921805310353, -9.631344896985492), (2.8918192061560024, -9.574147282812161), (2.9738462317809677, -9.516949668638832), (3.055873257405935, -9.4597520544655), (3.1379002830309, -9.40255444029217), (3.2199273086558673, -9.34535682611884), (3.3019543342808344, -9.288159211945509), (3.383981359905798, -9.230961597772179), (3.466008385530765, -9.173763983598848), (3.5480354111557304, -9.116566369425518), (3.6300624367806975, -9.059368755252187), (3.712089462405663, -9.002171141078858), (3.794116488030628, -8.944973526905526), (3.8761435136555953, -8.887775912732197), (3.9581705392805624, -8.830578298558866), (4.040197564905526, -8.773380684385536), (4.122224590530493, -8.716183070212205), (4.204251616155458, -8.658985456038875), (4.2862786417804255, -8.601787841865544), (4.368305667405391, -8.544590227692215), (4.450332693030358, -8.487392613518884), (4.532359718655325, -8.430194999345552), (4.614386744280292, -8.372997385172223), (4.696413769905256, -8.315799770998892), (4.778440795530221, -8.258602156825562), (4.860467821155188, -8.20140454265223), (4.9424948467801535, -8.144206928478901), (5.024521872405121, -8.08700931430557), (5.106548898030086, -8.029811700132239), (5.188575923655053, -7.972614085958909), (5.27060294928002, -7.915416471785579), (5.352629974904982, -7.858218857612249), (5.434657000529949, -7.801021243438918), (5.516684026154916, -7.743823629265588), (5.598711051779883, -7.686626015092258), (5.680738077404849, -7.629428400918926), (5.762765103029814, -7.572230786745596), (5.844792128654781, -7.5150331725722666), (5.926819154279748, -7.457835558398936), (6.008846179904712, -7.400637944225605), (6.090873205529677, -7.343440330052276), (6.172900231154644, -7.286242715878945), (6.254927256779611, -7.229045101705614), (6.336954282404577, -7.1718474875322835), (6.418981308029542, -7.114649873358953), (6.501008333654507, -7.057452259185624), (6.583035359279476, -7.000254645012292), (6.66506238490444, -6.943057030838963), (6.747089410529407, -6.885859416665632), (6.82911643615437, -6.828661802492302), (6.911143461779339, -6.771464188318971), (6.993170487404305, -6.71426657414564), (7.07519751302927, -6.65706895997231), (7.157224538654237, -6.599871345798979), (7.239251564279202, -6.54267373162565), (7.321278589904168, -6.485476117452318), (7.403305615529135, -6.428278503278989), (7.485332641154102, -6.371080889105657), (7.567359666779067, -6.313883274932328), (7.649386692404033, -6.256685660758997), (7.731413718028998, -6.1994880465856665), (7.813440743653965, -6.142290432412336), (7.895467769278932, -6.085092818239006), (7.977494794903896, -6.0278952040656755), (8.059521820528863, -5.970697589892345), (8.14154884615383, -5.913499975719015), (8.223575871778795, -5.856302361545684), (8.30560289740376, -5.799104747372354), (8.387629923028728, -5.741907133199024), (8.469656948653691, -5.684709519025692), (8.551683974278655, -5.627511904852363), (8.633710999903625, -5.570314290679033), (8.715738025528593, -5.513116676505702), (8.79776505115356, -5.45591906233237), (8.879792076778527, -5.398721448159041), (8.96181910240349, -5.341523833985711), (9.043846128028454, -5.28432621981238), (9.125873153653417, -5.22712860563905), (9.207900179278392, -5.169930991465717), (9.289927204903352, -5.11273337729239), (9.371954230528319, -5.0555357631190585), (9.453981256153286, -4.998338148945728), (9.536008281778253, -4.941140534772396), (9.618035307403217, -4.883942920599067), (9.700062333028184, -4.826745306425737), (9.78208935865315, -4.769547692252406), (9.864116384278114, -4.7123500780790755), (9.946143409903085, -4.655152463905745), (10.028170435528049, -4.597954849732416), (10.110197461153016, -4.540757235559084), (10.192224486777983, -4.483559621385753), (10.27425151240295, -4.426362007212422), (10.35627853802791, -4.369164393039094), (10.438305563652873, -4.311966778865763), (10.520332589277848, -4.254769164692432), (10.602359614902811, -4.197571550519102), (10.684386640527778, -4.140373936345771), (10.766413666152742, -4.0831763221724415), (10.848440691777709, -4.02597870799911), (10.930467717402676, -3.96878109382578), (11.012494743027643, -3.9115834796524496), (11.094521768652607, -3.854385865479119), (11.17654879427757, -3.797188251305789), (11.258575819902541, -3.7399906371324585), (11.340602845527508, -3.682793022959128), (11.422629871152472, -3.625595408785798), (11.504656896777439, -3.5683977946124674), (11.586683922402402, -3.511200180439138), (11.66871094802737, -3.454002566265806), (11.750737973652333, -3.3968049520924772), (11.832764999277304, -3.339607337919146), (11.914792024902267, -3.2824097237458165), (11.996819050527234, -3.2252121095724853), (12.078846076152201, -3.168014495399154), (12.160873101777165, -3.1108168812258237), (12.242900127402132, -3.0536192670524924), (12.3249271530271, -2.996421652879163), (12.406954178652063, -2.9392240387058317), (12.488981204277026, -2.882026424532503), (12.571008229901997, -2.824828810359172), (12.65303525552696, -2.7676311961858424), (12.735062281151928, -2.710433582012511), (12.817089306776895, -2.65323596783918), (12.899116332401862, -2.5960383536658505), (12.981143358026825, -2.53884073949252), (13.063170383651793, -2.4816431253191915), (13.14519740927676, -2.4244455111458585), (13.227224434901723, -2.367247896972529), (13.30925146052669, -2.310050282799196), (13.391278486151657, -2.2528526686258683), (13.473305511776621, -2.1956550544525353), (13.555332537401588, -2.1384574402792076), (13.637359563026555, -2.0812598261058763), (13.719386588651522, -2.024062211932547), (13.801413614276486, -1.9668645977592156), (13.88344063990145, -1.9096669835858862), (13.96546766552642, -1.852469369412555), (14.047494691151384, -1.7952717552392237), (14.12952171677635, -1.7380741410658924), (14.211548742401318, -1.6808765268925638), (14.293575768026281, -1.6236789127192326), (14.375602793651248, -1.5664812985459022), (14.457629819276212, -1.5092836843725745), (14.539656844901176, -1.4520860701992433), (14.621683870526146, -1.3948884560259112), (14.703710896151113, -1.3376908418525808), (14.785737921776077, -1.2804932276792504), (14.867764947401044, -1.22329561350592), (14.949791973026011, -1.1660979993325897), (15.031818998650978, -1.1089003851592594), (15.113846024275938, -1.0517027709859317), (15.195873049900909, -0.9945051568125987), (15.277900075525876, -0.9373075426392674), (15.359927101150836, -0.8801099284659379), (15.441954126775807, -0.8229123142926067), (15.491834375409182, -0.7881307140277656), (15.491834375409182, -0.7881307140277656), (15.574506446511025, -0.7318733119020351), (15.65843983594675, -0.6775156742414801), (15.743590896219212, -0.6250860684001278), (15.829915346611488, -0.5746117591064248), (15.917368296213928, -0.5261189942848894), (16.005904267268612, -0.4796329914065076), (16.095477218818907, -0.4351779243750311), (16.186040570651954, -0.3927769109559147), (16.27754722752156, -0.3524520007544929), (16.369949603638922, -0.3142241637496266), (16.463199647418442, -0.27811327938878705), (16.55724886646579, -0.24413812625022402), (16.652048352795163, -0.2123163722776411), (16.747548808262692, -0.18266456559240574), (16.843700570202735, -0.15519812588810922), (16.940453637253718, -0.12993133641193744), (17.037757695360106, -0.10687733653700882), (17.135562143937015, -0.08604811492957865), (17.233816122183782, -0.06745450331461811), (17.33246853553289, -0.05110617084304714), (17.431468082220434, -0.03701161906353079), (17.53076327996434, -0.02517817750144502), (17.63030249273646, -0.015611999847362057), (17.73003395761466, -0.008318060756948853), (17.82990581170077, -0.00330015326403732), (17.92986611909069, -0.0005608868081381502), (18.0, 7.105427357601002e-15)]




class HuskyTrajectoryEnv(HuskyEnv):
    def __init__(self, **kwargs):
        self.name = 'husky_trajectory'
        super().__init__('husky_trajectory.xml', **kwargs)

        # Env info
        self.ob_shape = OrderedDict([("husky_1", 31), ("husky_2", 31),
                                     ("box_1", 6), ("box_2", 6),
                                     ("goal_1", 3), ("goal_2", 3),        # we don't care the actual goal, huskys should only consider checkpoint
                                     ("relative_info_1", 2), ("relative_info_2", 2)])
        
        
        self.action_space.decompose(OrderedDict([("husky_1", 2), ("husky_2", 2)]))

        # Env config
        self._env_config.update({
            'random_husky_pos': 0.01,
            'random_goal_pos': 0.01,
            #'random_goal_pos': 0.5,
            'dist_threshold': 0.1,
            'loose_dist_threshold': 0.4,
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
            'max_episode_steps': 100000,
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
            Success Check
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
                "reward: huskys forward align reward": huskys_forward_align_reward,
                "reward: huskys right align reward": huskys_right_align_reward,
                "reward: husky-to-husky dist reward": huskys_dist_reward,
                "reward: husky-to-box dist reward": huskys_box_dist_reward,
                "reward: goal-box dist reward": goal_box_dist_reward,
                "reward: move heading reward": huskys_move_heading_reward,
                "reward: goal_huskys_dist_reward": goal_huskys_dist_reward,
                #"reward: box velocity reward": box_linear_vel_reward,
                "reward: goal-to-box cos dist reward": goal_box_cos_dist_reward,
                #"reward: huskys_vel_control_reward": huskys_linear_vel_reward,
                "coeff: huskys forward align coeff": huskys_forward_align_coeff,
                "coeff: huskys right align coeff": huskys_right_align_coeff,
                "coeff: husky1 move heading coeff": husky1_move_coeff,
                "coeff: husky2 move heading coeff": husky2_move_coeff,
                "die_penalty": die_penalty,
                "reward_success": success_reward,
                "huskys_dist": huskys_dist,
                #"husky1_pos": husky1_pos,
                #"husky2_pos": husky2_pos,
                "husky1_linear_vel": husky1_linear_vel,
                "husky2_lienar_vel": husky2_linear_vel,
                "goal1_husky1_dist": goal1_husky1_dist,
                "goal2_husky2_dist": goal2_husky2_dist,
                "box_goal_cos_dist": goal_box_cos_dist_coeff,
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
            ('husky_1', np.concatenate([qpos[3:11], qvel[:10], qacc[:10], husky1_forward_vec])),
            ('husky_2', np.concatenate([qpos[14:22], qvel[10:20], qacc[10:20], husky2_forward_vec])),
            ('box_1', np.concatenate([box_pos1 - husky_pos1, box_forward1])),
            ('box_2', np.concatenate([box_pos2 - husky_pos2, box_forward2])),
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
        qpos = self._init_qpos + self._init_random(self.model.nq)
        qvel = self._init_qvel + self._init_random(self.model.nv)

        qpos[2] = 0.2
        qpos[13] = 0.2
        qpos[3:7] = [0, 0, 0, 0]
        qpos[14:18] = [0, 0, 0, 0]

        self.set_state(qpos, qvel)

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
        qpos[22:25] = init_box_pos
        qpos[25:29] = init_box_quat


        # Initialize husky
        #qpos[0:2] = [-4, 2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        #qpos[15:17] = [-4, -2] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_ant_pos"]
        '''
        qpos[0:2] = [-2, 1] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_husky_pos"]
        qpos[11:13] = [-2, -1] + np.random.uniform(-1, 1, size=(2,)) * self._env_config["random_husky_pos"]
        '''
        husky1_pos = [-19, 1, 0.2]
        husky2_pos = [-19, 1, 0.2] 
        self._set_pos('husky_robot_1', husky1_pos) 
        self._set_pos('husky_robot_2', husky2_pos) 

        #qpos[0:2] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], 2]
        #qpos[15:17] = [-2 + np.random.uniform(-1, 1) * self._env_config["random_ant_pos"], -2]



        # Initialize goal
        #x = 4.5 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        x = 3 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        y = 0 + np.random.uniform(-1, 1) * self._env_config["random_goal_pos"]
        z = 0.3

        goal_pos = np.asarray([x, y, z])
        #goal_quat = sample_quat(low=-np.pi/9, high=np.pi/9)
        goal_quat = sample_quat(low=-np.pi/6, high=np.pi/6)
        self._set_pos('goal', goal_pos)
        self._set_quat('goal', goal_quat)
        
        
        # Initialized checkpoint
        

        self.set_state(qpos, qvel)

    def _render_callback(self):
        lookat = [0.5, 0, 0]
        cam_pos = lookat + np.array([-4.5, -12, 20])

        cam_id = self._camera_id
        self._set_camera_position(cam_id, cam_pos)
        self._set_camera_rotation(cam_id, lookat)

        self.sim.forward()

