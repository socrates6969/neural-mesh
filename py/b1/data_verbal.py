"""B1 battery: hand-authored lexical/ontology data for the verbal-comprehension proxies. Pools A and B are disjoint (two 'worlds')."""
# (word, synonym, antonym)
VOCAB_A = [("happy","joyful","sad"),("big","large","tiny"),("quick","rapid","slow"),("begin","start","finish"),("brave","courageous","cowardly"),
 ("smart","intelligent","foolish"),("angry","furious","calm"),("silent","quiet","noisy"),("ancient","old","modern"),("difficult","hard","easy"),
 ("rich","wealthy","poor"),("clean","spotless","dirty"),("strong","powerful","weak"),("honest","truthful","deceitful"),("generous","giving","stingy"),
 ("tired","exhausted","energetic"),("empty","vacant","full"),("calm","peaceful","agitated"),("bright","luminous","dim"),("fragile","delicate","sturdy")]
VOCAB_B = [("abundant","plentiful","scarce"),("reluctant","unwilling","eager"),("meticulous","careful","careless"),("benevolent","kind","cruel"),("obsolete","outdated","current"),
 ("verbose","wordy","concise"),("candid","frank","evasive"),("ephemeral","fleeting","permanent"),("diligent","hardworking","lazy"),("hostile","unfriendly","friendly"),
 ("frugal","thrifty","wasteful"),("lucid","clear","confusing"),("modest","humble","arrogant"),("vivid","striking","dull"),("tranquil","serene","turbulent"),
 ("prudent","cautious","reckless"),("novel","original","familiar"),("rigid","stiff","flexible"),("trivial","minor","significant"),("zealous","fervent","apathetic")]
# category -> members
ONT_A = {"fruit":["apple","pear","banana","cherry","grape","plum","mango","peach"],
 "tool":["hammer","screwdriver","wrench","pliers","chisel","saw","drill","spanner"],
 "bird":["sparrow","eagle","owl","pigeon","falcon","robin","crow","swan"],
 "vehicle":["car","truck","bicycle","train","bus","tram","scooter","van"],
 "furniture":["chair","table","sofa","wardrobe","desk","bookshelf","stool","bench"],
 "color":["red","blue","green","yellow","purple","orange","pink","brown"],
 "planet":["mercury","venus","mars","jupiter","saturn","uranus","neptune","earth"],
 "instrument":["violin","piano","flute","trumpet","guitar","drum","cello","harp"],
 "vegetable":["carrot","potato","onion","cabbage","spinach","broccoli","celery","turnip"],
 "metal":["iron","copper","silver","gold","zinc","tin","lead","nickel"]}
ONT_B = {"mammal":["dog","horse","whale","elephant","tiger","rabbit","bat","dolphin"],
 "clothing":["shirt","trousers","jacket","scarf","dress","sweater","gloves","skirt"],
 "profession":["doctor","teacher","engineer","lawyer","plumber","pilot","farmer","nurse"],
 "weather":["rain","snow","hail","fog","thunder","drizzle","sleet","wind"],
 "body part":["elbow","knee","shoulder","ankle","wrist","spine","jaw","hip"],
 "sport":["tennis","football","rugby","golf","cricket","boxing","hockey","archery"],
 "language":["french","german","spanish","italian","russian","japanese","arabic","swahili"],
 "shape":["circle","square","triangle","pentagon","hexagon","oval","rectangle","rhombus"],
 "drink":["water","coffee","tea","juice","milk","lemonade","cocoa","cider"],
 "emotion":["fear","joy","anger","sadness","disgust","surprise","envy","pride"]}
# relation -> (a,b) pairs ; analogy A:B::C:?
REL_A = {"capital":[("france","paris"),("japan","tokyo"),("italy","rome"),("egypt","cairo"),("canada","ottawa"),("kenya","nairobi"),("peru","lima"),("norway","oslo")],
 "young":[("dog","puppy"),("cat","kitten"),("cow","calf"),("sheep","lamb"),("horse","foal"),("goat","kid"),("duck","duckling"),("bear","cub")],
 "tool_use":[("knife","cut"),("pen","write"),("broom","sweep"),("needle","sew"),("hammer","pound"),("brush","paint"),("key","unlock"),("oven","bake")]}
REL_B = {"antonym":[("hot","cold"),("up","down"),("day","night"),("open","closed"),("early","late"),("inside","outside"),("push","pull"),("win","lose")],
 "part_whole":[("wheel","car"),("page","book"),("petal","flower"),("branch","tree"),("finger","hand"),("room","house"),("key","keyboard"),("wing","bird")],
 "worker_place":[("teacher","school"),("doctor","hospital"),("chef","kitchen"),("pilot","cockpit"),("judge","court"),("farmer","field"),("sailor","ship"),("monk","monastery")]}
# sentence completion: (stem with ___, answer, [3 distractors])
CLOZE_A = [("After weeks of drought, the farmers were desperate for ___ to fall.","rain",["sunshine","snow","gold"]),
 ("She was so ___ that she could not stop laughing during the funny film.","amused",["sleepy","furious","hungry"]),
 ("The chef carefully ___ the vegetables before adding them to the soup.","chopped",["painted","sang","flew"]),
 ("He studied hard all semester, so he ___ the exam with high marks.","passed",["lost","ignored","cooked"]),
 ("The ___ in the library was so complete that you could hear a pin drop.","silence",["noise","music","traffic"]),
 ("Because the road was icy, the driver proceeded very ___.","slowly",["quickly","loudly","brightly"]),
 ("The scientist ___ her hypothesis by running a controlled experiment.","tested",["forgot","sold","painted"]),
 ("A good leader must ___ responsibility when things go wrong.","accept",["avoid","ignore","sell"]),
 ("The ___ sun made it impossible to stay outside at midday.","scorching",["freezing","dim","silent"]),
 ("They ___ the old bridge because it was no longer safe for traffic.","demolished",["decorated","borrowed","watered"]),
 ("The audience ___ loudly when the pianist finished her last piece.","applauded",["slept","whispered","vanished"]),
 ("To save money, the family decided to ___ their monthly expenses.","reduce",["inflate","admire","paint"])]
CLOZE_B = [("Although the plan seemed ___ at first, it soon proved unworkable.","plausible",["edible","musical","liquid"]),
 ("The witness gave a ___ account, leaving out no detail of the accident.","thorough",["blurry","tasty","sleepy"]),
 ("Her ___ remarks offended several guests at the dinner party.","tactless",["generous","invisible","frozen"]),
 ("The company's profits ___ sharply after the new product launched.","soared",["whispered","melted","sang"]),
 ("He was ___ about the results, unsure whether to trust the data.","skeptical",["delicious","tall","golden"]),
 ("The treaty was ___ by both nations after years of negotiation.","ratified",["eaten","painted","lost"]),
 ("Climbing the mountain required ___ and careful planning.","endurance",["laughter","furniture","perfume"]),
 ("The ___ of the evidence convinced the jury of his guilt.","weight",["flavour","colour","sound"]),
 ("Rather than argue, she chose to ___ and find a compromise.","negotiate",["evaporate","decorate","hibernate"]),
 ("The old manuscript was so ___ that it crumbled when touched.","brittle",["liquid","fragrant","loud"]),
 ("Scientists ___ that the new vaccine reduces infection by ninety percent.","reported",["danced","baked","borrowed"]),
 ("His ___ attitude made him a pleasure to work with every day.","cheerful",["rotten","metallic","oblong"])]
# general-info fact tables
CAPITALS = [("france","paris"),("germany","berlin"),("spain","madrid"),("italy","rome"),("portugal","lisbon"),("austria","vienna"),("greece","athens"),("poland","warsaw"),
 ("sweden","stockholm"),("norway","oslo"),("finland","helsinki"),("denmark","copenhagen"),("ireland","dublin"),("belgium","brussels"),("hungary","budapest"),("egypt","cairo"),
 ("kenya","nairobi"),("nigeria","abuja"),("ghana","accra"),("morocco","rabat"),("japan","tokyo"),("china","beijing"),("india","new delhi"),("thailand","bangkok"),
 ("vietnam","hanoi"),("indonesia","jakarta"),("canada","ottawa"),("mexico","mexico city"),("brazil","brasilia"),("argentina","buenos aires"),("chile","santiago"),("peru","lima"),
 ("colombia","bogota"),("australia","canberra"),("turkey","ankara"),("iran","tehran")]
ELEMENTS = [("hydrogen","H"),("helium","He"),("lithium","Li"),("carbon","C"),("nitrogen","N"),("oxygen","O"),("sodium","Na"),("magnesium","Mg"),("aluminium","Al"),("silicon","Si"),
 ("phosphorus","P"),("sulfur","S"),("chlorine","Cl"),("potassium","K"),("calcium","Ca"),("iron","Fe"),("copper","Cu"),("zinc","Zn"),("silver","Ag"),("gold","Au"),("mercury","Hg"),("lead","Pb"),("tin","Sn"),("neon","Ne")]
AUTHORS = [("Jane Austen","Pride and Prejudice"),("Charles Dickens","Great Expectations"),("Leo Tolstoy","War and Peace"),("Mark Twain","Adventures of Huckleberry Finn"),
 ("Herman Melville","Moby-Dick"),("George Orwell","Nineteen Eighty-Four"),("Mary Shelley","Frankenstein"),("Homer","the Odyssey"),("Miguel de Cervantes","Don Quixote"),
 ("Fyodor Dostoevsky","Crime and Punishment"),("Victor Hugo","Les Miserables"),("Franz Kafka","The Trial"),("Virginia Woolf","Mrs Dalloway"),("Gabriel Garcia Marquez","One Hundred Years of Solitude"),
 ("Emily Bronte","Wuthering Heights"),("J. R. R. Tolkien","The Hobbit"),("Harper Lee","To Kill a Mockingbird"),("F. Scott Fitzgerald","The Great Gatsby")]
INVENT = [("Alexander Graham Bell","the telephone"),("Thomas Edison","the practical incandescent light bulb"),("Johannes Gutenberg","the movable-type printing press in Europe"),
 ("Tim Berners-Lee","the World Wide Web"),("Alexander Fleming","penicillin"),("Marie Curie","the discovery of polonium and radium"),("Isaac Newton","the law of universal gravitation"),
 ("Charles Darwin","the theory of evolution by natural selection"),("Albert Einstein","the theory of general relativity"),("Dmitri Mendeleev","the periodic table"),
 ("Louis Pasteur","pasteurization"),("Nikola Tesla","the induction motor"),("Wright brothers","the first powered airplane"),("Gregor Mendel","the laws of inheritance")]
GEO = [("the Nile","Egypt and Sudan","river"),("the Amazon","Brazil","river"),("the Danube","central Europe","river"),("Mount Everest","the Himalayas","mountain"),("Mont Blanc","the Alps","mountain"),
 ("Kilimanjaro","Tanzania","mountain"),("the Sahara","North Africa","desert"),("the Gobi","Mongolia and China","desert"),("Lake Victoria","East Africa","lake"),("Lake Baikal","Siberia","lake"),
 ("the Mississippi","the United States","river"),("the Rhine","Germany and the Netherlands","river")]
